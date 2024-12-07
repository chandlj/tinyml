import os
import sys
import argparse
import torch
import torch.nn as nn
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__package__ = "tinyml"

from smoothquant.fake_quant import W8A8Linear, quantize_model
from smoothquant.smooth import smooth_lm
from .run_awq import run_awq
from .auto_scale import apply_scale
from .auto_clip import apply_clip
from awq.quantize.quantizer import pseudo_quantize_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    # OPT Models
    "facebook/opt-125m": {"family": "opt", "act_scales": "opt-125m.pt"},
    "facebook/opt-350m": {"family": "opt", "act_scales": "opt-350m.pt"},
    "facebook/opt-1.3b": {"family": "opt", "act_scales": "opt-1.3b.pt"},
    "facebook/opt-2.7b": {"family": "opt", "act_scales": "opt-2.7b.pt"},
    "facebook/opt-6.7b": {"family": "opt", "act_scales": "opt-6.7b.pt"},
    # Llama Models
    "meta-llama/Llama-2-7b-hf": {"family": "llama", "act_scales": "llama-2-7b.pt"},
    "meta-llama/Llama-2-13b-hf": {"family": "llama", "act_scales": "llama-2-13b.pt"},
    "meta-llama/Llama-2-70b-hf": {"family": "llama", "act_scales": "llama-2-70b.pt"},
}

def get_model_and_tokenizer(model_name):
    """Load appropriate model and tokenizer based on model family."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not supported. Supported models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if config["family"] == "llama" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if config["family"] == "opt":
        model_class = OPTForCausalLM
    elif config["family"] == "llama":
        model_class = LlamaForCausalLM
    else:
        raise ValueError(f"Unknown model family: {config['family']}")
    
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)
    
    return model, tokenizer, config

def test_perplexity(model, tokenizer, seqlen=2048):
    """Test model perplexity on WikiText-2 dataset."""
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = seqlen
    testenc = testenc.input_ids.to(device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []
    
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():.2f}")

    return {"ppl": ppl.item()}

def quantize_and_save_awq(model_name, output_dir, w_bit=4, seqlen=512):
    """Quantize model using SmoothQuant and AWQ, then save results."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer, config = get_model_and_tokenizer(model_name)
    
    # Apply SmoothQuant
    scales_path = os.path.join("./smoothquant/act_scales", config["act_scales"])
    if not os.path.exists(scales_path):
        raise FileNotFoundError(f"Activation scales file not found: {scales_path}")
    
    scales = torch.load(scales_path, map_location=device)
    smooth_lm(model, scales, alpha=0.5)
    model = quantize_model(model)
    
    # Run and save AWQ results
    awq_results = run_awq(
        model,
        tokenizer,
        w_bit=w_bit,
        q_config={"zero_point": True, "q_group_size": 128},
        n_samples=128,
        seqlen=seqlen,
    )

    print("Testing original model perplexity...")
    model = model.to(device)
    test_perplexity(model, tokenizer, seqlen=seqlen)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_id = model_name.split('/')[-1]
    torch.save(awq_results, os.path.join(output_dir, f"{model_id}_awq.pt"))
    print(f"AWQ results saved to {output_dir}")

@torch.no_grad()
def quantize_act_reorder(a, n_bits=4, outlier_ratio=0.5):
    """Quantize activations with reordering for better precision on important values.
    
    Args:
        a: Input activation tensor
        n_bits: Number of bits for quantization
        outlier_ratio: Ratio of values to keep in high precision (0.0 to 1.0)
    """
    importance = a.view(-1, a.shape[-1]).abs().mean(dim=0)

    # Sort the importance and reorder the weight tensor
    indices = torch.argsort(importance, descending=True)
    a = a[..., indices]

    # Quantize the activations
    topk = int(importance.numel() * outlier_ratio)
    a_outliers = a[..., :topk].clone()
    a = pseudo_quantize_tensor(a, n_bit=n_bits, q_group_size=128)
    a[..., :topk] = a_outliers

    # Reorder the weight tensor back
    un_indices = torch.argsort(indices)
    a = a[..., un_indices]
    return a

@torch.no_grad()
def forward_with_activation_quant(self, x, outlier_ratio=0.5):
    """Modified forward pass with activation quantization."""
    q_x = self.act_quant(x)
    q_x = quantize_act_reorder(q_x, n_bits=4, outlier_ratio=outlier_ratio)
    y = torch.functional.F.linear(q_x, self.weight, self.bias)

    q_y = self.output_quant(y)
    q_y = quantize_act_reorder(q_y, n_bits=4, outlier_ratio=outlier_ratio)
    return q_y

def apply_activation_quantization(model, outlier_ratio=0.5):
    """Apply activation quantization to all W8A8Linear layers."""
    import types
    for name, module in model.named_modules():
        if isinstance(module, W8A8Linear):
            # Create a closure to capture the outlier_ratio
            def make_forward(module, outlier_ratio):
                return lambda self, x: forward_with_activation_quant(self, x, outlier_ratio)
            
            module.forward = types.MethodType(make_forward(module, outlier_ratio), module)
    return model

def load_and_test_model(model_name, awq_results_path, seqlen=2048, outlier_ratio=0.5):
    """Load model with AWQ results and test perplexity."""
    
    # Load model and tokenizer
    model, tokenizer, config = get_model_and_tokenizer(model_name)

    print("Testing original model perplexity...")
    test_perplexity(model, tokenizer, seqlen=seqlen)

    print("Applying SmoothQuant...")
    scales_path = os.path.join("./smoothquant/act_scales", config["act_scales"])
    if not os.path.exists(scales_path):
        raise FileNotFoundError(f"Activation scales file not found: {scales_path}")
    
    scales = torch.load(scales_path, map_location=device)
    smooth_lm(model, scales, alpha=0.5)
    model = quantize_model(model)
    
    print("Applying AWQ...")
    # Load AWQ results and apply them
    awq_results = torch.load(awq_results_path, map_location=device)
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])
    model = model.to(device)
    
    print("Testing AWQ model perplexity...")
    results = test_perplexity(model, tokenizer, seqlen=seqlen)
    print(f"Perplexity results after AWQ: {results}")

    print(f"Applying activation quantization (outlier_ratio={outlier_ratio})...")
    model = apply_activation_quantization(model, outlier_ratio=outlier_ratio)
    model = model.to(device)
    
    print("Testing AWQ + activation quantization model perplexity...")
    results = test_perplexity(model, tokenizer, seqlen=seqlen)
    print(f"Perplexity results after AWQ + activation quantization: {results}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Quantize models using AWQ or test perplexity")
    parser.add_argument("--mode", choices=["quantize", "test"], required=True,
                      help="Mode: 'quantize' to run AWQ or 'test' to evaluate perplexity")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                      help="Model name to quantize/test")
    parser.add_argument("--output-dir", default="./awq_results",
                      help="Directory to save AWQ results (for quantize mode)")
    parser.add_argument("--awq-results", 
                      help="Path to AWQ results file (for test mode)")
    parser.add_argument("--seqlen", type=int, default=2048,
                      help="Sequence length for evaluation")
    parser.add_argument("--w-bit", type=int, default=4,
                      help="Number of bits for weight quantization")
    parser.add_argument("--outlier-ratio", type=float, default=0.5,
                      help="Ratio of activation values to keep in high precision (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    if args.mode == "quantize":
        quantize_and_save_awq(
            args.model,
            args.output_dir,
            w_bit=args.w_bit,
            seqlen=args.seqlen
        )
    else:  # test mode
        if not args.awq_results:
            parser.error("--awq-results is required for test mode")
        if not 0.0 <= args.outlier_ratio <= 1.0:
            parser.error("--outlier-ratio must be between 0.0 and 1.0")
        load_and_test_model(
            args.model,
            args.awq_results,
            seqlen=args.seqlen,
            outlier_ratio=args.outlier_ratio
        )

if __name__ == "__main__":
    main()

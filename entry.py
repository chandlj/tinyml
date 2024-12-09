import argparse
import gc
import os
import sys
from functools import partial
import json

import torch
import torch.nn as nn
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__package__ = "tinyml"

from awq.quantize.quantizer import pseudo_quantize_tensor
from awq.utils.calib_data import get_calib_dataset

from smoothquant.fake_quant import (
    W8A8Linear,
    quantize_activation_per_token_absmax,
    quantize_model,
)
from smoothquant.smooth import smooth_lm

from .auto_clip import apply_clip
from .auto_scale import apply_scale
from .run_awq import run_awq

MODEL_CONFIGS = {
    # OPT Models
    "facebook/opt-125m": {"family": "opt", "act_scales": "opt-125m.pt"},
    "facebook/opt-350m": {"family": "opt", "act_scales": "opt-350m.pt"},
    "facebook/opt-1.3b": {"family": "opt", "act_scales": "opt-1.3b.pt"},
    "facebook/opt-2.7b": {"family": "opt", "act_scales": "opt-2.7b.pt"},
    "facebook/opt-6.7b": {"family": "opt", "act_scales": "opt-6.7b.pt"},
    "facebook/opt-13b": {"family": "opt", "act_scales": "opt-13b.pt"},
    "facebook/opt-30b": {"family": "opt", "act_scales": "opt-30b.pt"},
    "facebook/opt-66b": {"family": "opt", "act_scales": "opt-66b.pt"},
    # Llama Models
    "meta-llama/Llama-2-7b-hf": {"family": "llama", "act_scales": "llama-2-7b.pt"},
    "meta-llama/Llama-2-13b-hf": {"family": "llama", "act_scales": "llama-2-13b.pt"},
    "meta-llama/Llama-2-70b-hf": {"family": "llama", "act_scales": "llama-2-70b.pt"},
}


def get_model_and_tokenizer(model_name):
    """Load appropriate model and tokenizer based on model family."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Model {model_name} not supported. Supported models: {list(MODEL_CONFIGS.keys())}"
        )

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
        device_map="auto",
    )

    return model, tokenizer, config


def test_perplexity(model, tokenizer, seqlen=2048):
    """Test model perplexity on WikiText-2 dataset."""
    testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testenc["text"]), return_tensors="pt")
    model.seqlen = seqlen
    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    model = model.eval()
    nlls = []

    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():.2f}")

    return {"ppl": ppl.item()}


@torch.no_grad()
def get_calib_feat(model, tokenizer):
    input_dict = dict()

    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, W8A8Linear):
            hooks.append(
                m.register_forward_hook(partial(stat_input_max_hook, name=name))
            )

    print("Collecting activation scales...")

    samples = get_calib_dataset("pileval", tokenizer, n_samples=128, block_size=512)
    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(model.device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return input_dict


@torch.no_grad()
def quantize_act_reorder(a, n_bits=4, outlier_ratio=0.5, reorder_indices=None, unorder_indices=None):
    """Quantize activations with reordering for better precision on important values.

    Args:
        a: Input activation tensor
        n_bits: Number of bits for quantization
        outlier_ratio: Ratio of values to keep in high precision (0.0 to 1.0)
    """

    # Sort the importance and reorder the weight tensor
    if reorder_indices is None:
        importance = a.view(-1, a.shape[-1]).abs().mean(dim=0)
        reorder_indices = torch.argsort(importance, descending=True)

    a = a[..., reorder_indices]

    # Quantize the activations
    topk = int(reorder_indices.numel() * outlier_ratio)
    a_outliers = a[..., :topk].clone()
    a = pseudo_quantize_tensor(a, n_bit=n_bits, q_group_size=128)
    a[..., :topk] = a_outliers

    # Reorder the weight tensor back
    if unorder_indices is None:
        unorder_indices = torch.argsort(reorder_indices)

    a = a[..., unorder_indices]
    return a


@torch.no_grad()
def forward_with_activation_quant(self, x, reorder_indices, unorder_indices, outlier_ratio=0.5):
    """Modified forward pass with activation quantization."""
    q_x = self.act_quant(x)
    q_x = quantize_act_reorder(q_x, n_bits=4, outlier_ratio=outlier_ratio, reorder_indices=reorder_indices, unorder_indices=unorder_indices)
    y = torch.functional.F.linear(q_x, self.weight, self.bias)

    q_y = self.output_quant(y)
    # q_y = quantize_act_reorder(q_y, n_bits=4, outlier_ratio=outlier_ratio)
    return q_y


def apply_activation_quantization(model, outlier_ratio=0.5, input_feat=None):
    """Apply activation quantization to all W8A8Linear layers."""
    import types

    if input_feat is None:
        raise ValueError("input_feat is required")

    for name, module in model.named_modules():
        if isinstance(module, W8A8Linear):
            importance = sum(input_feat[name]).float()
            reorder_indices = torch.argsort(importance, descending=True)
            unorder_indices = torch.argsort(reorder_indices)
            # Create a closure to capture the outlier_ratio
            def make_forward(module, reorder_indices, unorder_indices, outlier_ratio):
                return lambda self, x: forward_with_activation_quant(
                    self, x, reorder_indices, unorder_indices, outlier_ratio
                )

            module.forward = types.MethodType(
                make_forward(module, reorder_indices, unorder_indices, outlier_ratio), module
            )
    return model


def quantize_and_save_awq(model_name, output_dir, w_bit=4, seqlen=512):
    """Quantize model using SmoothQuant and AWQ, then save results."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Load model and tokenizer
    model, tokenizer, config = get_model_and_tokenizer(model_name)

    print("Testing original model perplexity...")
    results["original"] = test_perplexity(model, tokenizer, seqlen=seqlen)

    # Apply SmoothQuant
    scales_path = os.path.join("./smoothquant/act_scales", config["act_scales"])
    if not os.path.exists(scales_path):
        raise FileNotFoundError(f"Activation scales file not found: {scales_path}")

    scales = torch.load(scales_path, map_location=model.device)
    smooth_lm(model, scales, alpha=0.5)
    model = quantize_model(model)

    print("Testing SmoothQuant model perplexity...")
    results["smoothquant"] = test_perplexity(model, tokenizer, seqlen=seqlen)

    input_feat = get_calib_feat(model, tokenizer)

    for n, m in model.named_modules():
        if isinstance(m, W8A8Linear):
            importance = sum(input_feat[n]).float()
            reorder_indices = torch.argsort(importance, descending=True)
            unorder_indices = torch.argsort(reorder_indices)
            topk = int(reorder_indices.numel() * 0.01)

            # Back up the values of the salient weight channels
            m.weight.data = m.weight.data[:, reorder_indices]
            m.weight.data[:, :topk] *= 2
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, q_group_size=128
            )
            m.weight.data[:, :topk] = m.weight.data[:, :topk] / 2
            m.weight.data = m.weight.data[:, unorder_indices]

    # Test perplexity
    print("Testing SmoothQuant + AWQ model perplexity...")
    results["awq"] = test_perplexity(model, tokenizer, seqlen=seqlen)

    # Save model output
    params_dict = model.state_dict()
    model_id = model_name.split("/")[-1]
    torch.save(params_dict, os.path.join(output_dir, f"{model_id}_awq.pt"))

    with open(os.path.join("./results", f"{model_id}_results.json"), "w") as f:
        json.dump(results, f)


def load_and_test_model(model_name, awq_results_path, seqlen=2048):
    """Load model with AWQ results and test perplexity with different outlier ratios."""
    results = {}

    # Load model and tokenizer
    model, tokenizer, config = get_model_and_tokenizer(model_name)
    # Get original device
    device = model.device
    print(f"Original device: {device}")
    # Make each linear layer W8A8
    model = quantize_model(model)

    print(f"Loading AWQ results from {awq_results_path}")
    # https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html
    params_dict = torch.load(awq_results_path, mmap=True, weights_only=True, map_location="cpu")

    model.load_state_dict(params_dict, assign=True)

    gc.collect()
    torch.cuda.empty_cache()

    model = model.to(device)

    input_feat = get_calib_feat(model, tokenizer)

    # Test different outlier ratios
    outlier_ratios = [0.01, 0.05, 0.1, 0.25]
    for ratio in outlier_ratios:
        print(f"\nTesting outlier ratio {ratio}:")

        # Reload weights - already pseudo quantized
        model.load_state_dict(params_dict, assign=True)

        # Clean up prior weights
        gc.collect()
        torch.cuda.empty_cache()

        # Move model to device
        model = model.to(device)

        print(f"Applying activation quantization (outlier_ratio={ratio})...")
        model = apply_activation_quantization(model, outlier_ratio=ratio, input_feat=input_feat)

        print("Testing AWQ + activation quantization model perplexity...")
        results[f"awq_act_quant_{ratio}"] = test_perplexity(
            model, tokenizer, seqlen=seqlen
        )

    # Save results
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    model_id = model_name.split("/")[-1]

    # Merge results with existing results
    if os.path.exists(os.path.join(output_dir, f"{model_id}_results.json")):
        with open(os.path.join(output_dir, f"{model_id}_results.json"), "r") as f:
            existing_results: dict = json.load(f)

        existing_results.update(results)
        results = existing_results

    with open(os.path.join(output_dir, f"{model_id}_results.json"), "w") as f:
        json.dump(results, f)

    # Print summary of all results
    print("\nSummary of all results:")
    print("=" * 80)
    print(f"Original model perplexity: {results['original']['ppl']:.2f}")
    print(f"SmoothQuant model perplexity: {results['smoothquant']['ppl']:.2f}")
    print(f"AWQ model perplexity: {results['awq']['ppl']:.2f}")
    for ratio in outlier_ratios:
        print(
            f"AWQ + Act Quant (ratio={ratio}) perplexity: {results[f'awq_act_quant_{ratio}']['ppl']:.2f}"
        )
    print("=" * 80)

    return results


def process_all_models(mode, output_dir="./awq_results", w_bit=4, seqlen=2048):
    """Process all models in batch mode."""
    results = {}

    for model_name in MODEL_CONFIGS.keys():
        print(f"\n{'='*80}")
        print(f"Processing {model_name}")
        print(f"{'='*80}\n")

        try:
            if mode == "quantize":
                quantize_and_save_awq(
                    model_name, output_dir, w_bit=w_bit, seqlen=seqlen
                )
            else:  # test mode
                model_id = model_name.split("/")[-1]
                awq_path = os.path.join(output_dir, f"{model_id}_awq.pt")
                if not os.path.exists(awq_path):
                    print(f"Skipping {model_name}: AWQ results not found at {awq_path}")
                    continue

                results[model_name] = load_and_test_model(
                    model_name,
                    awq_path,
                    seqlen=seqlen,
                )
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            continue

    if mode == "test" and results:
        print("\nSummary of Results:")
        print("=" * 80)
        for model_name, result in results.items():
            print(f"{model_name}:")
            print(f"Original: {result['original']['ppl']:.2f}")
            print(f"SmoothQuant: {result['smoothquant']['ppl']:.2f}")
            print(f"AWQ: {result['awq']['ppl']:.2f}")
            for ratio in [0.01, 0.05, 0.1, 0.25]:
                print(
                    f"AWQ + Act Quant (ratio={ratio}) perplexity: {result[f'awq_act_quant_{ratio}']['ppl']:.2f}"
                )
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize models using AWQ or test perplexity"
    )
    parser.add_argument(
        "--mode",
        choices=["quantize", "test"],
        required=True,
        help="Mode: 'quantize' to run AWQ or 'test' to evaluate perplexity",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="Model name to quantize/test, or 'all' for batch processing",
    )
    parser.add_argument(
        "--output-dir",
        default="./awq_results",
        help="Directory to save AWQ results (for quantize mode)",
    )
    parser.add_argument(
        "--awq-results", help="Path to AWQ results file (for test mode)"
    )
    parser.add_argument(
        "--seqlen", type=int, default=2048, help="Sequence length for evaluation"
    )
    parser.add_argument(
        "--w-bit", type=int, default=4, help="Number of bits for weight quantization"
    )
    args = parser.parse_args()

    if args.model == "all":
        process_all_models(
            args.mode,
            output_dir=args.output_dir,
            w_bit=args.w_bit,
            seqlen=args.seqlen,
        )
    else:
        if not args.model:
            parser.error("--model is required")

        if args.mode == "quantize":
            quantize_and_save_awq(
                args.model, args.output_dir, w_bit=args.w_bit, seqlen=args.seqlen
            )
        else:  # test mode
            if not args.awq_results:
                model_id = args.model.split("/")[-1]
                args.awq_results = os.path.join(args.output_dir, f"{model_id}_awq.pt")
                if not os.path.exists(args.awq_results):
                    parser.error(f"AWQ results file not found: {args.awq_results}")

            load_and_test_model(
                args.model,
                args.awq_results,
                seqlen=args.seqlen,
            )


if __name__ == "__main__":
    main()

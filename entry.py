import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__package__ = "tinyml"

import torch
import gc
import functools
from collections import defaultdict
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear, quantize_model
import tqdm
from functools import partial
import torch.nn as nn
from datasets import Dataset, load_dataset
from .auto_scale import auto_scale_block, apply_scale
from .auto_clip import auto_clip_block, apply_clip
from awq.utils.module import append_str_prefix, get_op_name
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.quantizer import pseudo_quantize_tensor

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, W8A8Linear)}


def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers]
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    # elif isinstance(model, LlavaLlamaForCausalLM):
    #     model.model.embed_tokens = model.model.embed_tokens.to(device)
    #     model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    else:
        raise NotImplementedError(type(model))

@torch.no_grad()
def run_awq(
    model,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
    # some configs for ablation study
    calib_data="pileval",
):

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()

        if mse_range:
            clip_list = auto_clip_block(
                layer,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
            )
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(
                clip_list, get_op_name(model, layer) + "."
            )

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")
dataset = load_dataset("lambada", split="validation[:1000]")
evaluator = Evaluator(dataset, tokenizer, "cuda")


model = OPTForCausalLM.from_pretrained(
    "facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto"
)
model = model.to("cuda")
acc_fp16 = evaluator.evaluate(model)
print(f"FP16 acc: {acc_fp16}")

scales = torch.load("./smoothquant/act_scales/opt-1.3b.pt")
smooth_lm(model, scales, alpha=0.5)
model = quantize_model(model)

awq_results = run_awq(
    model,
    tokenizer,
    w_bit=4,
    q_config={"zero_point": True, "q_group_size": 128},
    n_samples=128,
    seqlen=512,
)

# W8A8 Model
model = model.to("cuda")
acc_w4a8 = evaluator.evaluate(model)
print(f"W4A8 acc: {acc_w4a8}")


@torch.no_grad()
def quantize_act_reorder(a, n_bits=4, outlier_ratio = 0.5):
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
def forward(self, x):
    q_x = self.act_quant(x)
    q_x = quantize_act_reorder(q_x, n_bits=4)
    y = torch.functional.F.linear(q_x, self.weight, self.bias)

    q_y = self.output_quant(y)
    q_y = quantize_act_reorder(q_y, n_bits=4)
    return q_y

import types

for name, module in model.named_modules():
    if isinstance(module, W8A8Linear):
        module.forward = types.MethodType(forward, module)

model = model.to("cuda")
acc_w4a4 = evaluator.evaluate(model)
print(f"W4A4 acc: {acc_w4a4}")

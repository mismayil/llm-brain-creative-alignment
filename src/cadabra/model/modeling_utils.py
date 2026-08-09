from typing import Optional
import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
import pathlib
from functools import partial

from cadabra.utils import read_json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def disable_safetensors_auto_conversion():
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

def load_lm(model_path: str, from_pretrained: bool = True, padding_side: str = "left"):
    """
    Load a pre-trained language model and its tokenizer.

    Args:
        model_path (str): The path to the pre-trained model.
        from_pretrained (bool): Whether to load the model from pre-trained weights.
        padding_side (str): The side to use for padding ("left" or "right").

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side=padding_side)
    if from_pretrained:
        print(f"Loading model from pretrained weights at {model_path}")
        disable_safetensors_auto_conversion()
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    else:
        print(f"Loading model from untrained config at {model_path}")
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
        model.to(device)
    return model, tokenizer

def load_classifier(model_path: str, disable_auto_conversion: bool = False):
    """
    Load a reward model and its tokenizer.

    Args:
        model_path (str): The path to the reward model.
        disable_auto_conversion (bool): Whether to disable automatic conversion to safetensors.

    Returns:
        tuple: A tuple containing the reward model and tokenizer.
    """
    if disable_auto_conversion:
        disable_safetensors_auto_conversion()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        num_labels=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    if not model.config.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def prepare_lm_inputs(tokenizer, prompts: list[str], responses: Optional[list[str]] = None):
    """
    Tokenize a list of prompts for model input.

    Args:
        tokenizer: The tokenizer to use.
        prompts (list[str]): A list of text prompts.
        responses (list[str], optional): A list of expected responses (for forced outputs). Defaults to None.
    Returns:
        dict: A dictionary of tokenized inputs.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # check if instruct model and apply chat template if so
    if hasattr(tokenizer, "chat_template") and getattr(tokenizer, "chat_template") is not None:
        # convert prompts to list of messages
        if responses is not None:
            messages = [[{"role": "user", "content": p}, {"role": "assistant", "content": r}] for p, r in zip(prompts, responses)]
        else:
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=not responses, return_dict=True, return_tensors="pt", padding=True, truncation=True)
    else:
        if responses is not None:
            full_texts = [p + tokenizer.eos_token + r for p, r in zip(prompts, responses)]
        else:
            full_texts = prompts
        inputs = tokenizer(full_texts, return_dict=True, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs

def top_k_sampling(logits, top_k=40):
    topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)  # (B, top_k)
    probs = F.softmax(topk_vals, dim=-1)
    next_token_ids = []
    for b in range(logits.shape[0]):
        sampled_idx = torch.multinomial(probs[b], 1)
        next_token_ids.append(topk_idx[b, sampled_idx])
    next_token_ids = torch.stack(next_token_ids).unsqueeze(1)  # (B, 1)
    return next_token_ids

def top_p_sampling(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    for b in range(logits.shape[0]):
        indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
        logits[b, indices_to_remove] = -float('Inf')

    probs = F.softmax(logits, dim=-1)
    next_token_ids = torch.multinomial(probs, num_samples=1)  # (B, 1)
    return next_token_ids

def ancestral_sampling(logits):
    probs = F.softmax(logits, dim=-1)
    next_token_ids = torch.multinomial(probs, num_samples=1)  # (B, 1)
    return next_token_ids

def extract_model_activations(
    model,
    inputs: dict
):
    """
    Extract model activations without generation.
    B = batch size, S = input sequence length,
    L = number of layers, D = model hidden dimension 
    Arguments:
        model: the language model
        tokenizer: the tokenizer
        inputs: dict with "input_ids" and "attention_mask" tensors, shape (B, S)
        
    Returns:
        batch_activations: (B, S, L, D) array of activations.
    """
    input_ids = inputs["input_ids"].to(device)        # (B, S)
    attention_mask = inputs["attention_mask"].to(device) # (B, S)

    B, _ = input_ids.shape

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states   # list of L tensors of shape (B, S, D)

    batch_activations = []  # list of (S, L, D) of length B
    for i in range(B):
        activations = []
        for layer_tensor in hidden_states:
            vec = layer_tensor[i].to(torch.float32).cpu().numpy()  # (S, D)
            activations.append(vec)
        activations = np.stack(activations)  # (L, S, D)
        activations = np.transpose(activations, (1, 0, 2))  # (S, L, D)
        batch_activations.append(activations)
    
    return np.stack(batch_activations)  # (B, S, L, D)

def get_model_name(model_path):
    model_path_parts = model_path.split("/")
    model_name = model_path_parts[-1].lower()

    if model_name.startswith("checkpoint"):
        model_name = model_path_parts[-2].lower()

    return model_name

def load_model_neural_data(model_datapath: str, model_neural_datapath: str) -> np.ndarray:
    """ Load model neural data from` a given path. """
    model_data_dir = pathlib.Path(model_datapath)
    if model_data_dir.is_file():
        model_data_dir = model_data_dir.parent
    neural_data = np.load(model_data_dir / model_neural_datapath, mmap_mode='r')
    return neural_data   # shape (num_tokens, num_layers, model_dim)

def random_network_mask_like(network_mask):
    num_active_units = int(network_mask.sum())
    num_layers, hidden_dim = network_mask.shape
    total_num_units = np.prod(network_mask.shape)
    invnet_mask_indices = np.arange(total_num_units)[(1 - network_mask).flatten().astype(bool)]
    rand_indices = np.random.choice(invnet_mask_indices, size=num_active_units, replace=False)
    random_mask = np.full(total_num_units, 0)
    random_mask[rand_indices] = 1
    assert np.sum(random_mask) == num_active_units
    return random_mask.reshape((num_layers, hidden_dim))

def load_network_mask(network_path, network_type: Optional[str] = None, ignore_first_layer: bool = True):
    network_path = pathlib.Path(network_path)
    network_data = read_json(network_path)
    network_mask_path = network_path.parent / network_data["data"]["network_mask_path"]
    network_mask = np.load(network_mask_path)
    print(f"Loaded network mask from {network_mask_path}, shape: {network_mask.shape}")
    if ignore_first_layer:
        print("Ignoring first layer in network mask")
        network_mask = network_mask[1:]  # ignore first layer (embedding layer)
    if network_type == "random":
        print("Generating random network mask")
        network_mask = random_network_mask_like(network_mask)
    return network_mask

def get_model_layers(model):
    """ Get all layers from the model """
    submodel = getattr(model, "model", None)
    if submodel:
        layers = getattr(submodel, "layers", None)
        if layers:
            return layers
        language_model = getattr(submodel, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers:
            return layers
        raise ValueError("Cannot find layers in model.")
    return model.transformer.h

def get_ablation_hook(layer_idx, ablation_mask):
    """
    Defines a hook function to ablate specific units based on a mask.

    Args:
        layer_idx (int): Layer index.
        ablation_mask (torch.Tensor): Binary mask for ablation for all layers.

    Returns:
        function: A hook function to zero out specified units.
    """
    def hook_ablate(module, input, output):
        mask_layer = ablation_mask[layer_idx]  # (D,)
        unit_indices = mask_layer.nonzero()[0]
        output[:, :, unit_indices] = 0
    return hook_ablate
    
def clear_model_hooks(model):
    for layer in get_model_layers(model):
        layer._forward_hooks.clear()

def register_model_hooks(model, hook):
    for layer_idx, layer in enumerate(get_model_layers(model)):
        layer.register_forward_hook(hook(layer_idx))
    
def ablate_model(model, ablation_mask):
    """
    Register hooks to ablate specific units in the model based on the ablation mask.

    Args:
        model: The language model.
        ablation_mask (torch.Tensor): Binary mask indicating which units to ablate.
    """
    clear_model_hooks(model)
    register_model_hooks(model, partial(get_ablation_hook, ablation_mask=ablation_mask))
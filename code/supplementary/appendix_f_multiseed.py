import os
import sys
from getpass import getpass
from huggingface_hub import login
import re
import json
import math
import time
import random
import requests
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datasets import load_dataset
import transformers
from packaging import version
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, PaliGemmaForConditionalGeneration
HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN and sys.stdin.isatty():
    HF_TOKEN = getpass('Enter your Hugging Face token: ').strip() or None
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

MIN_TRANSFORMERS_VERSION = '4.47.0'
if version.parse(transformers.__version__) < version.parse(MIN_TRANSFORMERS_VERSION):
    raise RuntimeError(f'PaliGemma 2 requires transformers>={MIN_TRANSFORMERS_VERSION}. Current transformers version: {transformers.__version__}. Run: pip install -U transformers accelerate')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
VLM_ID = 'google/paligemma2-3b-mix-224'
LLM_ID = 'google/gemma-2-2b'
if 'paligemma2' not in VLM_ID.lower():
    raise ValueError(f'Wrong VLM_ID for this experiment: {VLM_ID}. Use a PaliGemma 2 checkpoint.')
if 'gemma-2-2b' not in LLM_ID.lower():
    raise ValueError(f'Wrong LLM_ID for the 3B PaliGemma 2 pairing: {LLM_ID}. Use google/gemma-2-2b.')
DATASET_ID = 'tomg-group-umd/pixelprose'
PIXELPROSE_SPLIT = 'commonpool'
NUM_SAMPLES = 100
MAX_TRIES = 3000
DOWNLOAD_TIMEOUT = 10
MAX_CAPTION_WORDS = 80
VLM_BATCH_SIZE = 2
LLM_BATCH_SIZE = 8
TRAIN_NEW_SAES = True
TRAIN_ALL_LAYERS = True
LAYERS_TO_USE = None
SAE_TRAIN_FRACTION = 0.8
TOKENS_PER_LAYER_FOR_TRAIN = 50000
SAE_DIM_MULTIPLIER = 4
SAE_BATCH_SIZE = 2048
SAE_STEPS = 1000
SAE_LR = 0.001
SAE_L1_COEFF = 0.001
ACT_THRESHOLD = 0.5
RECON_EVAL_TOKENS = 20000
RECON_EVAL_BATCH_SIZE = 4096
SAVE_RECON_EVAL = True
TOPK = 200
MODEL_TAG = 'paligemma2-3b-mix-224__gemma-2-2b__pixelprose'
OUTPUT_DIR = os.path.join('./superposition_experiment_outputs', MODEL_TAG)
SAE_ROOT_DIR = os.path.join(OUTPUT_DIR, 'sae_checkpoints')
VLM_SAE_DIR = os.path.join(SAE_ROOT_DIR, 'vlm_paligemma2')
LLM_SAE_DIR = os.path.join(SAE_ROOT_DIR, 'llm_gemma2')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
CHECKPOINT_MANIFEST_CSV = os.path.join(OUTPUT_DIR, 'sae_checkpoint_manifest.csv')
CHECKPOINT_MANIFEST_JSON = os.path.join(OUTPUT_DIR, 'sae_checkpoint_manifest.json')
VLM_TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, 'vlm_sae_train_logs.csv')
LLM_TRAIN_LOG_CSV = os.path.join(OUTPUT_DIR, 'llm_sae_train_logs.csv')
RECON_EVAL_CSV = os.path.join(OUTPUT_DIR, 'sae_reconstruction_eval.csv')
RECON_EVAL_JSON = os.path.join(OUTPUT_DIR, 'sae_reconstruction_eval.json')
for d in [OUTPUT_DIR, SAE_ROOT_DIR, VLM_SAE_DIR, LLM_SAE_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)
print('DEVICE:', DEVICE)
print('DTYPE:', DTYPE)
print('transformers:', transformers.__version__)
print('OUTPUT_DIR:', OUTPUT_DIR)
print('VLM_ID:', VLM_ID)
print('LLM_ID:', LLM_ID)
print('Checkpoint manifest:', CHECKPOINT_MANIFEST_CSV)
print('Reconstruction eval:', RECON_EVAL_CSV)

def shorten_caption_words(text: str, max_words: int=80) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip()
    return ' '.join(words[:max_words]).strip()

def chunk_list(xs, batch_size):
    for i in range(0, len(xs), batch_size):
        yield xs[i:i + batch_size]

def load_image_from_url(url: str, timeout: int=10) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert('RGB')
        return img
    except Exception:
        return None

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def set_plot_style():
    plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 300, 'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.titlesize': 14, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.7, 'lines.linewidth': 2.3, 'lines.markersize': 5, 'legend.frameon': False})
set_plot_style()
stream_ds = load_dataset(DATASET_ID, split=PIXELPROSE_SPLIT, streaming=True)
samples = []
seen = 0
for ex in tqdm(stream_ds, total=MAX_TRIES, desc='Collecting PixelProse samples'):
    seen += 1
    if seen > MAX_TRIES:
        break
    caption = ex.get('vlm_caption', None)
    if caption is None or not isinstance(caption, str) or (not caption.strip()):
        caption = ex.get('original_caption', None)
    url = ex.get('url', None)
    if caption is None or not isinstance(caption, str) or (not caption.strip()):
        continue
    if url is None or not isinstance(url, str) or (not url.strip()):
        continue
    image = load_image_from_url(url, timeout=DOWNLOAD_TIMEOUT)
    if image is None:
        continue
    samples.append({'image': image, 'caption': shorten_caption_words(caption.strip(), MAX_CAPTION_WORDS), 'url': url})
    if len(samples) % 10 == 0:
        print(f'collected={len(samples)} / {NUM_SAMPLES} | seen={seen}')
    if len(samples) >= NUM_SAMPLES:
        break
print('Seen rows:', seen)
print('Collected samples:', len(samples))
if len(samples) == 0:
    raise ValueError('No usable samples collected.')
vlm_processor = AutoProcessor.from_pretrained(VLM_ID, use_fast=False)
vlm_model = PaliGemmaForConditionalGeneration.from_pretrained(VLM_ID, torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True)
vlm_model.to(DEVICE)
vlm_model.eval()
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_ID, use_fast=False)
if llm_tokenizer.pad_token_id is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_tokenizer.padding_side = 'right'
llm_model = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True)
llm_model.to(DEVICE)
llm_model.eval()
llm_model.config.pad_token_id = llm_tokenizer.pad_token_id
if hasattr(vlm_processor, 'tokenizer'):
    if vlm_processor.tokenizer.pad_token_id is None:
        vlm_processor.tokenizer.pad_token = vlm_processor.tokenizer.eos_token
    vlm_processor.tokenizer.padding_side = 'right'
vlm_num_layers = vlm_model.config.text_config.num_hidden_layers
vlm_hidden_size = vlm_model.config.text_config.hidden_size
llm_num_layers = llm_model.config.num_hidden_layers
llm_hidden_size = llm_model.config.hidden_size
if TRAIN_ALL_LAYERS:
    common_num_layers = min(vlm_num_layers, llm_num_layers)
    LAYERS_TO_USE = list(range(common_num_layers))
elif LAYERS_TO_USE is None:
    raise ValueError('Provide LAYERS_TO_USE when TRAIN_ALL_LAYERS=False')
assert vlm_hidden_size == llm_hidden_size, f'Hidden size mismatch: VLM={vlm_hidden_size}, LLM={llm_hidden_size}. This usually means the LLM baseline is not the matching Gemma 2 checkpoint.'
print('VLM:', VLM_ID)
print('VLM layers:', vlm_num_layers, 'hidden:', vlm_hidden_size)
print('LLM:', LLM_ID)
print('LLM layers:', llm_num_layers, 'hidden:', llm_hidden_size)
print('Using layers:', LAYERS_TO_USE[:10], '...' if len(LAYERS_TO_USE) > 10 else '')

def make_paligemma2_prompt(caption: Optional[str]=None) -> str:
    if caption is None or not str(caption).strip():
        return '<image>'
    return f'<image> {str(caption).strip()}'

def make_vlm_text_batch(batch):
    images = [x['image'] for x in batch]
    texts = [make_paligemma2_prompt(x['caption']) for x in batch]
    inputs = vlm_processor(text=texts, images=images, padding=True, truncation=False, return_tensors='pt')
    return {k: v.to(DEVICE) for k, v in inputs.items()}

def make_vlm_image_only_batch(batch):
    images = [x['image'] for x in batch]
    texts = [make_paligemma2_prompt(None)] * len(batch)
    inputs = vlm_processor(text=texts, images=images, padding=True, truncation=False, return_tensors='pt')
    return {k: v.to(DEVICE) for k, v in inputs.items()}

def make_vlm_image_conditioned_batch(batch):
    images = [x['image'] for x in batch]
    captions = [x['caption'] for x in batch]
    perm = np.random.permutation(len(batch))
    swapped_captions = [captions[i] for i in perm]
    texts = [make_paligemma2_prompt(c) for c in swapped_captions]
    inputs = vlm_processor(text=texts, images=images, padding=True, truncation=False, return_tensors='pt')
    return {k: v.to(DEVICE) for k, v in inputs.items()}

def make_llm_text_batch(batch):
    texts = [x['caption'] for x in batch]
    inputs = llm_tokenizer(texts, padding=True, truncation=True, max_length=192, return_tensors='pt')
    return {k: v.to(DEVICE) for k, v in inputs.items()}

@torch.no_grad()
def get_vlm_hidden_states(inputs):
    outputs = vlm_model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    return outputs.hidden_states

@torch.no_grad()
def get_llm_hidden_states(inputs):
    outputs = llm_model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    return outputs.hidden_states
test_vlm = make_vlm_text_batch(samples[:1])
test_llm = make_llm_text_batch(samples[:2])
vlm_hs = get_vlm_hidden_states(test_vlm)
llm_hs = get_llm_hidden_states(test_llm)
print('VLM hidden states:', len(vlm_hs), vlm_hs[1].shape)
print('LLM hidden states:', len(llm_hs), llm_hs[1].shape)

class TrainableSAE(nn.Module):

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.W_enc = nn.Linear(d_in, d_sae)
        self.W_dec = nn.Linear(d_sae, d_in)

    def encode(self, x):
        return F.relu(self.W_enc(x))

    def decode(self, z):
        return self.W_dec(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return (x_hat, z)

class InferenceSAE(nn.Module):

    def __init__(self, d_in, d_sae, W_enc, b_enc, W_dec, b_dec):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.W_enc = nn.Parameter(W_enc, requires_grad=False)
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)
        self.W_dec = nn.Parameter(W_dec, requires_grad=False)
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)

    def encode(self, x):
        z = x @ self.W_enc + self.b_enc
        return torch.relu(z)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return (x_hat, z)

@torch.no_grad()
def collect_layer_activations(samples, layer_idx, batch_builder, hidden_state_fn, target_tokens=50000, batch_size=2):
    collected = []
    total_tokens = 0
    for batch in tqdm(chunk_list(samples, batch_size), desc=f'Collect layer {layer_idx}'):
        inputs = batch_builder(batch)
        hidden_states = hidden_state_fn(inputs)
        hs = hidden_states[layer_idx + 1].float().detach().cpu()
        B, T, D = hs.shape
        flat = hs.reshape(B * T, D)
        collected.append(flat)
        total_tokens += flat.shape[0]
        if total_tokens >= target_tokens:
            break
    if len(collected) == 0:
        raise ValueError(f'No activations collected for layer {layer_idx}.')
    acts = torch.cat(collected, dim=0)[:target_tokens]
    print(f'Layer {layer_idx}: {acts.shape}')
    return acts

def train_sae_for_layer(activations, layer_idx, d_sae, save_dir, steps=1000, batch_size=2048, lr=0.001, l1_coeff=0.001, device='cuda', model_kind=None, model_id=None):
    os.makedirs(save_dir, exist_ok=True)
    d_in = activations.shape[1]
    sae = TrainableSAE(d_in=d_in, d_sae=d_sae).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = activations.shape[0]
    logs = []
    for step in tqdm(range(steps), desc=f'Train SAE layer {layer_idx}'):
        idx = torch.randint(0, n, (min(batch_size, n),))
        x = activations[idx].to(device)
        x_hat, z = sae(x)
        recon_loss = F.mse_loss(x_hat, x)
        l1_loss = z.abs().mean()
        loss = recon_loss + l1_coeff * l1_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            with torch.no_grad():
                l0_mean = (z > 0).float().sum(dim=-1).mean().item()
            logs.append({'model_kind': model_kind, 'model_id': model_id, 'layer': layer_idx, 'step': step, 'loss': float(loss.item()), 'recon_mse_train_batch': float(recon_loss.item()), 'l1': float(l1_loss.item()), 'l0_mean': float(l0_mean)})
    ckpt = {'layer': layer_idx, 'd_in': d_in, 'd_sae': d_sae, 'W_enc': sae.W_enc.weight.detach().T.cpu(), 'b_enc': sae.W_enc.bias.detach().cpu(), 'W_dec': sae.W_dec.weight.detach().T.cpu(), 'b_dec': sae.W_dec.bias.detach().cpu(), 'metadata': {'model_kind': model_kind, 'model_id': model_id, 'transformers_version': transformers.__version__, 'sae_dim_multiplier': SAE_DIM_MULTIPLIER, 'sae_steps': steps, 'sae_lr': lr, 'sae_l1_coeff': l1_coeff, 'tokens_per_layer_for_train': int(activations.shape[0]), 'created_unix_time': time.time()}}
    path = os.path.join(save_dir, f'layer_{layer_idx:02d}.pt')
    torch.save(ckpt, path)
    return (path, logs)

def load_inference_sae(path):
    ckpt = torch.load(path, map_location='cpu')
    sae = InferenceSAE(d_in=ckpt['d_in'], d_sae=ckpt['d_sae'], W_enc=ckpt['W_enc'], b_enc=ckpt['b_enc'], W_dec=ckpt['W_dec'], b_dec=ckpt['b_dec'])
    return sae

def checkpoint_row(model_kind, model_id, layer, path, action):
    ckpt = torch.load(path, map_location='cpu')
    return {'model_kind': model_kind, 'model_id': model_id, 'layer': int(layer), 'checkpoint_path': path, 'action': action, 'exists': bool(os.path.exists(path)), 'file_size_bytes': int(os.path.getsize(path)) if os.path.exists(path) else 0, 'd_in': int(ckpt['d_in']), 'd_sae': int(ckpt['d_sae']), 'transformers_version': ckpt.get('metadata', {}).get('transformers_version', None), 'sae_dim_multiplier': ckpt.get('metadata', {}).get('sae_dim_multiplier', None), 'sae_steps': ckpt.get('metadata', {}).get('sae_steps', None), 'sae_l1_coeff': ckpt.get('metadata', {}).get('sae_l1_coeff', None)}

def train_or_load_sae_bank(samples, layers, batch_builder, hidden_state_fn, hidden_size, save_dir, batch_size, model_kind, model_id):
    d_sae = hidden_size * SAE_DIM_MULTIPLIER
    logs_all = []
    checkpoint_rows = []
    saes = {}
    for layer in layers:
        path = os.path.join(save_dir, f'layer_{layer:02d}.pt')
        action = 'loaded_existing'
        if TRAIN_NEW_SAES or not os.path.exists(path):
            action = 'trained_new'
            acts = collect_layer_activations(samples=samples, layer_idx=layer, batch_builder=batch_builder, hidden_state_fn=hidden_state_fn, target_tokens=TOKENS_PER_LAYER_FOR_TRAIN, batch_size=batch_size)
            path, logs = train_sae_for_layer(activations=acts, layer_idx=layer, d_sae=d_sae, save_dir=save_dir, steps=SAE_STEPS, batch_size=SAE_BATCH_SIZE, lr=SAE_LR, l1_coeff=SAE_L1_COEFF, device=DEVICE, model_kind=model_kind, model_id=model_id)
            logs_all.extend(logs)
        saes[layer] = load_inference_sae(path).to(DEVICE)
        checkpoint_rows.append(checkpoint_row(model_kind, model_id, layer, path, action))
    return (saes, pd.DataFrame(logs_all), pd.DataFrame(checkpoint_rows))

@torch.no_grad()
def evaluate_sae_reconstruction_from_activations(activations, sae, batch_size=4096, device='cuda'):
    sae.eval()
    n = activations.shape[0]
    sse = 0.0
    sae_abs_sum = 0.0
    x_norm2_sum = 0.0
    rel_l2_sum = 0.0
    l0_sum = 0.0
    total = 0
    x_all = activations.float()
    x_mean = x_all.mean(dim=0, keepdim=True)
    centered_ss = float(((x_all - x_mean) ** 2).sum().item())
    for start in range(0, n, batch_size):
        x = activations[start:start + batch_size].to(device).float()
        x_hat, z = sae(x)
        err = x_hat - x
        batch_sse = (err ** 2).sum(dim=1)
        batch_x_norm2 = (x ** 2).sum(dim=1).clamp_min(1e-12)
        sse += float(batch_sse.sum().item())
        x_norm2_sum += float(batch_x_norm2.sum().item())
        rel_l2_sum += float(torch.sqrt(batch_sse / batch_x_norm2).sum().item())
        sae_abs_sum += float(z.abs().sum().item())
        l0_sum += float((z > 0).float().sum(dim=-1).sum().item())
        total += x.shape[0]
    d_in = activations.shape[1]
    mse = sse / max(total * d_in, 1)
    nmse = sse / max(x_norm2_sum, 1e-12)
    explained_variance = 1.0 - sse / max(centered_ss, 1e-12)
    return {'recon_mse': float(mse), 'recon_nmse': float(nmse), 'explained_variance': float(explained_variance), 'relative_l2_mean': float(rel_l2_sum / max(total, 1)), 'mean_abs_feature_activation': float(sae_abs_sum / max(total * sae.d_sae, 1)), 'l0_mean': float(l0_sum / max(total, 1)), 'eval_tokens': int(total), 'd_in': int(d_in), 'd_sae': int(sae.d_sae)}

def evaluate_sae_bank_reconstruction(samples, layers, saes, batch_builder, hidden_state_fn, batch_size, model_kind, model_id, target_tokens=20000):
    rows = []
    for layer in layers:
        acts = collect_layer_activations(samples=samples, layer_idx=layer, batch_builder=batch_builder, hidden_state_fn=hidden_state_fn, target_tokens=target_tokens, batch_size=batch_size)
        metrics = evaluate_sae_reconstruction_from_activations(activations=acts, sae=saes[layer], batch_size=RECON_EVAL_BATCH_SIZE, device=DEVICE)
        rows.append({'model_kind': model_kind, 'model_id': model_id, 'layer': int(layer), **metrics})
    return pd.DataFrame(rows)
if len(samples) >= 10:
    n_train = max(1, int(len(samples) * SAE_TRAIN_FRACTION))
    sae_train_samples = samples[:n_train]
    sae_eval_samples = samples[n_train:]
    if len(sae_eval_samples) == 0:
        sae_eval_samples = samples[-max(1, len(samples) // 5):]
else:
    sae_train_samples = samples
    sae_eval_samples = samples
print(f'SAE train samples: {len(sae_train_samples)}')
print(f'SAE reconstruction-eval samples: {len(sae_eval_samples)}')
vlm_saes, vlm_train_log_df, vlm_ckpt_df = train_or_load_sae_bank(samples=sae_train_samples, layers=LAYERS_TO_USE, batch_builder=make_vlm_text_batch, hidden_state_fn=get_vlm_hidden_states, hidden_size=vlm_hidden_size, save_dir=VLM_SAE_DIR, batch_size=VLM_BATCH_SIZE, model_kind='vlm', model_id=VLM_ID)
llm_saes, llm_train_log_df, llm_ckpt_df = train_or_load_sae_bank(samples=sae_train_samples, layers=LAYERS_TO_USE, batch_builder=make_llm_text_batch, hidden_state_fn=get_llm_hidden_states, hidden_size=llm_hidden_size, save_dir=LLM_SAE_DIR, batch_size=LLM_BATCH_SIZE, model_kind='llm', model_id=LLM_ID)
checkpoint_manifest_df = pd.concat([vlm_ckpt_df, llm_ckpt_df], ignore_index=True)
checkpoint_manifest_df.to_csv(CHECKPOINT_MANIFEST_CSV, index=False)
checkpoint_manifest_df.to_json(CHECKPOINT_MANIFEST_JSON, orient='records', indent=2)
vlm_train_log_df.to_csv(VLM_TRAIN_LOG_CSV, index=False)
llm_train_log_df.to_csv(LLM_TRAIN_LOG_CSV, index=False)
if SAVE_RECON_EVAL:
    vlm_recon_df = evaluate_sae_bank_reconstruction(samples=sae_eval_samples, layers=LAYERS_TO_USE, saes=vlm_saes, batch_builder=make_vlm_text_batch, hidden_state_fn=get_vlm_hidden_states, batch_size=VLM_BATCH_SIZE, model_kind='vlm', model_id=VLM_ID, target_tokens=RECON_EVAL_TOKENS)
    llm_recon_df = evaluate_sae_bank_reconstruction(samples=sae_eval_samples, layers=LAYERS_TO_USE, saes=llm_saes, batch_builder=make_llm_text_batch, hidden_state_fn=get_llm_hidden_states, batch_size=LLM_BATCH_SIZE, model_kind='llm', model_id=LLM_ID, target_tokens=RECON_EVAL_TOKENS)
    recon_eval_df = pd.concat([vlm_recon_df, llm_recon_df], ignore_index=True)
    recon_eval_df.to_csv(RECON_EVAL_CSV, index=False)
    recon_eval_df.to_json(RECON_EVAL_JSON, orient='records', indent=2)
print('Loaded VLM SAE layers:', sorted(vlm_saes.keys())[:10], '...')
print('Loaded LLM SAE layers:', sorted(llm_saes.keys())[:10], '...')
print('VLM train logs:', vlm_train_log_df.shape, '->', VLM_TRAIN_LOG_CSV)
print('LLM train logs:', llm_train_log_df.shape, '->', LLM_TRAIN_LOG_CSV)
print('Checkpoint manifest:', checkpoint_manifest_df.shape, '->', CHECKPOINT_MANIFEST_CSV)
if SAVE_RECON_EVAL:
    print('Reconstruction eval:', recon_eval_df.shape, '->', RECON_EVAL_CSV)
    display_cols = ['model_kind', 'layer', 'recon_mse', 'recon_nmse', 'explained_variance', 'relative_l2_mean', 'l0_mean', 'eval_tokens']
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
SEEDS = [0, 1, 2]
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def chunk_list(xs, batch_size):
    for i in range(0, len(xs), batch_size):
        yield xs[i:i + batch_size]

def make_vlm_text_swapped_batch(batch):
    images = [x['image'] for x in batch]
    captions = [x['caption'] for x in batch]
    perm = np.random.permutation(len(batch))
    swapped_captions = [captions[i] for i in perm]
    texts = [make_paligemma2_prompt(c) for c in swapped_captions]
    inputs = vlm_processor(text=texts, images=images, padding=True, truncation=False, return_tensors='pt')
    return {k: v.to(DEVICE) for k, v in inputs.items()}

def make_vlm_image_swapped_batch(batch):
    captions = [x['caption'] for x in batch]
    images = [x['image'] for x in batch]
    perm = np.random.permutation(len(batch))
    swapped_images = [images[i] for i in perm]
    texts = [make_paligemma2_prompt(c) for c in captions]
    inputs = vlm_processor(text=texts, images=swapped_images, padding=True, truncation=False, return_tensors='pt')
    return {k: v.to(DEVICE) for k, v in inputs.items()}

@torch.no_grad()
def batch_feature_activity(hidden_states, saes, threshold=ACT_THRESHOLD):
    out = {}
    for layer, sae in saes.items():
        hs = hidden_states[layer + 1].float()
        B, T, D = hs.shape
        z = sae.encode(hs.reshape(B * T, D)).reshape(B, T, -1)
        z_mean = z.mean(dim=(0, 1))
        z_max = z.amax(dim=(0, 1))
        active_any = (z > threshold).any(dim=1).float()
        active_rate = active_any.mean(dim=0)
        out[layer] = {'z_mean': z_mean.detach().cpu(), 'z_max': z_max.detach().cpu(), 'active_rate': active_rate.detach().cpu()}
    return out

def init_accumulators(saes):
    acc = {}
    for layer, sae in saes.items():
        F_dim = sae.d_sae
        acc[layer] = {'sum_z_mean': torch.zeros(F_dim), 'sum_z_max': torch.zeros(F_dim), 'sum_active_rate': torch.zeros(F_dim), 'n_batches': 0}
    return acc

def update_accumulators(acc, batch_out):
    for layer in batch_out:
        acc[layer]['sum_z_mean'] += batch_out[layer]['z_mean']
        acc[layer]['sum_z_max'] += batch_out[layer]['z_max']
        acc[layer]['sum_active_rate'] += batch_out[layer]['active_rate']
        acc[layer]['n_batches'] += 1

def finalize_accumulators(acc):
    final = {}
    for layer, stats in acc.items():
        n = max(stats['n_batches'], 1)
        final[layer] = {'z_mean': stats['sum_z_mean'] / n, 'z_max': stats['sum_z_max'] / n, 'active_rate': stats['sum_active_rate'] / n}
    return final

def run_condition(samples, saes, batch_builder, hidden_state_fn, condition_name, batch_size):
    acc = init_accumulators(saes)
    for batch in tqdm(list(chunk_list(samples, batch_size)), desc=condition_name, leave=False):
        inputs = batch_builder(batch)
        hidden_states = hidden_state_fn(inputs)
        batch_out = batch_feature_activity(hidden_states, saes, threshold=ACT_THRESHOLD)
        update_accumulators(acc, batch_out)
    return finalize_accumulators(acc)

def participation_ratio(x, eps=1e-08):
    x = np.asarray(x)
    x = np.maximum(x, 0.0)
    s1 = np.sum(x)
    s2 = np.sum(x ** 2)
    if s2 < eps:
        return 0.0
    return float(s1 ** 2 / s2)

def gini_coefficient(x, eps=1e-08):
    x = np.asarray(x)
    x = np.maximum(x, 0.0) + eps
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * np.sum(x)))
all_condition_stats = {}
activity_rows = []
for seed in SEEDS:
    print(f'\n===== Running seed {seed} =====')
    set_all_seeds(seed)
    llm_text_stats = run_condition(samples, llm_saes, make_llm_text_batch, get_llm_hidden_states, f'llm_text_seed{seed}', LLM_BATCH_SIZE)
    vlm_matched_stats = run_condition(samples, vlm_saes, make_vlm_text_batch, get_vlm_hidden_states, f'vlm_matched_seed{seed}', VLM_BATCH_SIZE)
    vlm_text_swapped_stats = run_condition(samples, vlm_saes, make_vlm_text_swapped_batch, get_vlm_hidden_states, f'vlm_text_swapped_seed{seed}', VLM_BATCH_SIZE)
    vlm_image_swapped_stats = run_condition(samples, vlm_saes, make_vlm_image_swapped_batch, get_vlm_hidden_states, f'vlm_image_swapped_seed{seed}', VLM_BATCH_SIZE)
    vlm_image_only_stats = run_condition(samples, vlm_saes, make_vlm_image_only_batch, get_vlm_hidden_states, f'vlm_image_only_seed{seed}', VLM_BATCH_SIZE)
    seed_stats = {'llm_text': llm_text_stats, 'vlm_matched': vlm_matched_stats, 'vlm_text_swapped': vlm_text_swapped_stats, 'vlm_image_swapped': vlm_image_swapped_stats, 'vlm_image_only': vlm_image_only_stats}
    all_condition_stats[seed] = seed_stats
    for condition_name, stats in seed_stats.items():
        for layer in sorted(stats.keys()):
            a = stats[layer]['active_rate'].numpy()
            activity_rows.append({'seed': seed, 'condition': condition_name, 'layer': layer, 'mean_active_rate': float(np.mean(a)), 'std_active_rate': float(np.std(a)), 'num_active_features': int(np.sum(a > 0)), 'num_features_above_threshold': int(np.sum(a > ACT_THRESHOLD)), 'effective_dimension': participation_ratio(a), 'gini': gini_coefficient(a)})
torch.save(all_condition_stats, os.path.join(OUTPUT_DIR, 'all_condition_stats.pt'))
condition_activity_df = pd.DataFrame(activity_rows)
condition_activity_df.to_csv(os.path.join(OUTPUT_DIR, 'condition_activity_multi_seed.csv'), index=False)
print('\nSaved:')
print('-', os.path.join(OUTPUT_DIR, 'all_condition_stats.pt'))
print('-', os.path.join(OUTPUT_DIR, 'condition_activity_multi_seed.csv'))
print('condition_activity_df:', condition_activity_df.shape)
import os
import numpy as np
import pandas as pd

def topk_indices(x, k):
    k = min(k, len(x))
    return np.argsort(-x)[:k]

def jaccard_of_active_sets(a, b, threshold=0.0):
    A = set(np.where(a > threshold)[0].tolist())
    B = set(np.where(b > threshold)[0].tolist())
    if len(A | B) == 0:
        return 1.0
    return len(A & B) / len(A | B)

def overlap_at_k(a, b, k=TOPK):
    A = set(topk_indices(a, k).tolist())
    B = set(topk_indices(b, k).tolist())
    if len(A | B) == 0:
        return 1.0
    return len(A & B) / len(A | B)

def mean_rank_shift(a, b):
    rank_a = np.argsort(np.argsort(-a))
    rank_b = np.argsort(np.argsort(-b))
    return float(np.mean(np.abs(rank_a - rank_b)))

def cosine_similarity(a, b, eps=1e-08):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def participation_ratio(x, eps=1e-08):
    x = np.asarray(x)
    x = np.maximum(x, 0.0)
    s1 = np.sum(x)
    s2 = np.sum(x ** 2)
    if s2 < eps:
        return 0.0
    return float(s1 ** 2 / s2)

def gini_coefficient(x, eps=1e-08):
    x = np.asarray(x)
    x = np.maximum(x, 0.0) + eps
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * np.sum(x)))
feature_rows = []
layer_summary_rows = []
pair_rows = []
for seed, seed_stats in all_condition_stats.items():
    vlm_matched_stats = seed_stats['vlm_matched']
    vlm_text_swapped_stats = seed_stats['vlm_text_swapped']
    vlm_image_swapped_stats = seed_stats['vlm_image_swapped']
    vlm_image_only_stats = seed_stats['vlm_image_only']
    layers = sorted(vlm_matched_stats.keys())
    seed_feature_rows = []
    for layer in layers:
        matched = vlm_matched_stats[layer]['active_rate'].numpy()
        text_swapped = vlm_text_swapped_stats[layer]['active_rate'].numpy()
        image_swapped = vlm_image_swapped_stats[layer]['active_rate'].numpy()
        image_only = vlm_image_only_stats[layer]['active_rate'].numpy()
        F_dim = len(matched)
        for fid in range(F_dim):
            image_delta = float(matched[fid] - text_swapped[fid])
            text_delta = float(matched[fid] - image_swapped[fid])
            mismatch_total = abs(image_delta) + abs(text_delta)
            seed_feature_rows.append({'seed': seed, 'layer': layer, 'feature_id': fid, 'matched_rate': float(matched[fid]), 'text_swapped_rate': float(text_swapped[fid]), 'image_swapped_rate': float(image_swapped[fid]), 'image_only_rate': float(image_only[fid]), 'image_delta': image_delta, 'text_delta': text_delta, 'mismatch_total': mismatch_total})
    seed_df_features = pd.DataFrame(seed_feature_rows)
    weak_thresh = seed_df_features['mismatch_total'].quantile(0.5)
    strong_thresh = max(seed_df_features['image_delta'].abs().quantile(0.6), seed_df_features['text_delta'].abs().quantile(0.6))

    def classify_feature(row, weak_thresh=weak_thresh, strong_thresh=strong_thresh, ratio_thresh=1.5):
        img = abs(row['image_delta'])
        txt = abs(row['text_delta'])
        total = img + txt
        if total < weak_thresh:
            return 'neither'
        if img >= strong_thresh and txt < strong_thresh / ratio_thresh:
            return 'image'
        if txt >= strong_thresh and img < strong_thresh / ratio_thresh:
            return 'text'
        if img >= strong_thresh and txt >= strong_thresh:
            return 'both'
        return 'uncertain'
    seed_df_features['classification'] = seed_df_features.apply(classify_feature, axis=1)
    feature_rows.extend(seed_df_features.to_dict('records'))
    seed_layer_summary = seed_df_features.groupby(['seed', 'layer', 'classification']).size().unstack(fill_value=0).reset_index()
    for c in ['image', 'text', 'both', 'uncertain', 'neither']:
        if c not in seed_layer_summary.columns:
            seed_layer_summary[c] = 0
    seed_layer_summary['total_features'] = seed_layer_summary['image'] + seed_layer_summary['text'] + seed_layer_summary['both'] + seed_layer_summary['uncertain'] + seed_layer_summary['neither']
    for c in ['image', 'text', 'both', 'uncertain', 'neither']:
        seed_layer_summary[f'{c}_frac'] = seed_layer_summary[c] / seed_layer_summary['total_features'].clip(lower=1)
    layer_summary_rows.extend(seed_layer_summary.to_dict('records'))
    condition_names = list(seed_stats.keys())
    for i in range(len(condition_names)):
        for j in range(i + 1, len(condition_names)):
            c1 = condition_names[i]
            c2 = condition_names[j]
            s1 = seed_stats[c1]
            s2 = seed_stats[c2]
            common_layers = sorted(set(s1.keys()) & set(s2.keys()))
            for layer in common_layers:
                a = s1[layer]['active_rate'].numpy()
                b = s2[layer]['active_rate'].numpy()
                pair_rows.append({'seed': seed, 'layer': layer, 'comparison': f'{c1} vs {c2}', 'jaccard': jaccard_of_active_sets(a, b, threshold=0.0), 'overlap_at_k': overlap_at_k(a, b, k=TOPK), 'mean_rank_shift': mean_rank_shift(a, b), 'active_rate_cosine': cosine_similarity(a, b), 'effective_dimension_a': participation_ratio(a), 'effective_dimension_b': participation_ratio(b), 'effective_dimension_gap': abs(participation_ratio(a) - participation_ratio(b)), 'gini_a': gini_coefficient(a), 'gini_b': gini_coefficient(b), 'gini_gap': abs(gini_coefficient(a) - gini_coefficient(b))})
df_features = pd.DataFrame(feature_rows)
layer_summary = pd.DataFrame(layer_summary_rows)
results_df = pd.DataFrame(pair_rows)

def agg_mean_sem(df, group_cols, value_cols):
    out = df.groupby(group_cols)[value_cols].agg(['mean', 'std', 'count'])
    out.columns = ['_'.join(col).strip() for col in out.columns.values]
    out = out.reset_index()
    for val in value_cols:
        std_col = f'{val}_std'
        count_col = f'{val}_count'
        sem_col = f'{val}_sem'
        out[sem_col] = out[std_col] / np.sqrt(out[count_col].clip(lower=1))
    return out
metric_cols = ['jaccard', 'overlap_at_k', 'mean_rank_shift', 'active_rate_cosine', 'effective_dimension_gap', 'gini_gap']
aggregated_results_df = agg_mean_sem(results_df, group_cols=['layer', 'comparison'], value_cols=metric_cols)
frac_cols = ['image_frac', 'text_frac', 'both_frac', 'uncertain_frac', 'neither_frac']
aggregated_layer_summary = agg_mean_sem(layer_summary, group_cols=['layer'], value_cols=frac_cols)
delta_summary = agg_mean_sem(df_features, group_cols=['layer'], value_cols=['image_delta', 'text_delta', 'mismatch_total'])
topk_rows = []
TOP_SENSITIVE_K = 50
for (seed, layer), sub in df_features.groupby(['seed', 'layer']):
    sub_img = sub.sort_values('image_delta', ascending=False).head(TOP_SENSITIVE_K)
    sub_txt = sub.sort_values('text_delta', ascending=False).head(TOP_SENSITIVE_K)
    topk_rows.append({'seed': seed, 'layer': layer, 'topk_mean_image_delta': float(sub_img['image_delta'].mean()) if len(sub_img) else 0.0, 'topk_mean_text_delta': float(sub_txt['text_delta'].mean()) if len(sub_txt) else 0.0})
topk_df = pd.DataFrame(topk_rows)
aggregated_topk_df = agg_mean_sem(topk_df, group_cols=['layer'], value_cols=['topk_mean_image_delta', 'topk_mean_text_delta'])
df_features.to_csv(os.path.join(OUTPUT_DIR, 'df_features_multi_seed.csv'), index=False)
layer_summary.to_csv(os.path.join(OUTPUT_DIR, 'layer_summary_multi_seed.csv'), index=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'pairwise_superposition_multi_seed.csv'), index=False)
aggregated_results_df.to_csv(os.path.join(OUTPUT_DIR, 'pairwise_superposition_aggregated.csv'), index=False)
aggregated_layer_summary.to_csv(os.path.join(OUTPUT_DIR, 'layer_summary_aggregated.csv'), index=False)
delta_summary.to_csv(os.path.join(OUTPUT_DIR, 'delta_summary_aggregated.csv'), index=False)
aggregated_topk_df.to_csv(os.path.join(OUTPUT_DIR, 'topk_delta_aggregated.csv'), index=False)
print('Saved:')
for fn in ['df_features_multi_seed.csv', 'layer_summary_multi_seed.csv', 'pairwise_superposition_multi_seed.csv', 'pairwise_superposition_aggregated.csv', 'layer_summary_aggregated.csv', 'delta_summary_aggregated.csv', 'topk_delta_aggregated.csv']:
    print('-', os.path.join(OUTPUT_DIR, fn))
print('\ndf_features:', df_features.shape)
print('layer_summary:', layer_summary.shape)
print('results_df:', results_df.shape)
print('aggregated_results_df:', aggregated_results_df.shape)
print('aggregated_layer_summary:', aggregated_layer_summary.shape)
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 300, 'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5, 'figure.titlesize': 13, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.7, 'lines.linewidth': 2.2, 'lines.markersize': 4.5, 'legend.frameon': False})
comparison_order = ['llm_text vs vlm_matched', 'llm_text vs vlm_text_swapped', 'llm_text vs vlm_image_swapped', 'llm_text vs vlm_image_only', 'vlm_matched vs vlm_text_swapped', 'vlm_matched vs vlm_image_swapped', 'vlm_matched vs vlm_image_only', 'vlm_text_swapped vs vlm_image_swapped', 'vlm_text_swapped vs vlm_image_only', 'vlm_image_swapped vs vlm_image_only']
comparison_order = [c for c in comparison_order if c in aggregated_results_df['comparison'].unique()]
comparison_display_map = {'llm_text vs vlm_matched': 'LLM-T vs VLM-M', 'llm_text vs vlm_text_swapped': 'LLM-T vs VLM-Tswap', 'llm_text vs vlm_image_swapped': 'LLM-T vs VLM-Iswap', 'llm_text vs vlm_image_only': 'LLM-T vs VLM-Ionly', 'vlm_matched vs vlm_text_swapped': 'M vs Tswap', 'vlm_matched vs vlm_image_swapped': 'M vs Iswap', 'vlm_matched vs vlm_image_only': 'M vs Ionly', 'vlm_text_swapped vs vlm_image_swapped': 'Tswap vs Iswap', 'vlm_text_swapped vs vlm_image_only': 'Tswap vs Ionly', 'vlm_image_swapped vs vlm_image_only': 'Iswap vs Ionly'}

def pretty_comparison_label(comp):
    return comparison_display_map.get(comp, comp)

def savefig_all(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f'{name}.pdf'), bbox_inches='tight', pad_inches=0.15)
    fig.savefig(os.path.join(FIG_DIR, f'{name}.svg'), bbox_inches='tight', pad_inches=0.15)
    print('Saved:', name)

def add_panel_label(ax, label):
    ax.text(-0.1, 1.03, label, transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='left')

def plot_metric_with_sem(ax, df, metric, ylabel=None, title=None):
    mean_col = f'{metric}_mean'
    sem_col = f'{metric}_sem'
    for comp in comparison_order:
        sub = df[df['comparison'] == comp].sort_values('layer')
        if len(sub) == 0:
            continue
        x = sub['layer'].to_numpy()
        y = sub[mean_col].to_numpy()
        sem = sub[sem_col].fillna(0).to_numpy()
        ax.plot(x, y, marker='o', label=pretty_comparison_label(comp))
        ax.fill_between(x, y - sem, y + sem, alpha=0.16)
    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel if ylabel is not None else metric)
    ax.set_title(title if title is not None else metric.replace('_', ' ').title())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.02)
pivot_jaccard = aggregated_results_df.pivot(index='comparison', columns='layer', values='jaccard_mean').reindex(comparison_order)
act_agg = condition_activity_df.groupby(['condition', 'layer'])[['num_features_above_threshold', 'effective_dimension']].agg(['mean', 'std', 'count'])
act_agg.columns = ['_'.join(c) for c in act_agg.columns]
act_agg = act_agg.reset_index()
for metric in ['num_features_above_threshold', 'effective_dimension']:
    act_agg[f'{metric}_sem'] = act_agg[f'{metric}_std'] / np.sqrt(act_agg[f'{metric}_count'].clip(lower=1))
plot_df = aggregated_layer_summary.sort_values('layer')
x_layers = plot_df['layer'].to_numpy()
stack_order = ['image', 'text', 'both', 'uncertain', 'neither']
delta_sub = delta_summary.sort_values('layer')
topk_sub = aggregated_topk_df.sort_values('layer')
sample_df = df_features.groupby(['seed', 'layer'], group_keys=False).apply(lambda x: x.sample(min(len(x), 400), random_state=0)).reset_index(drop=True)
fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
axes = axes.flatten()
plot_metric_with_sem(axes[0], aggregated_results_df, 'jaccard', ylabel='Jaccard', title='Jaccard overlap')
plot_metric_with_sem(axes[1], aggregated_results_df, 'overlap_at_k', ylabel='Overlap@K', title='Top-K feature overlap')
plot_metric_with_sem(axes[2], aggregated_results_df, 'active_rate_cosine', ylabel='Cosine similarity', title='Feature activity cosine')
plot_metric_with_sem(axes[3], aggregated_results_df, 'mean_rank_shift', ylabel='Mean rank shift', title='Feature rank shift')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.suptitle('Layerwise superposition across LLM and VLM conditions (mean ± SEM)')
savefig_all(fig, 'fig1_main_superposition_grouped')
plt.show()
fig = plt.figure(figsize=(17, 10.5), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
axA = fig.add_subplot(gs[0, 0])
bottom = np.zeros(len(plot_df))
for cls in stack_order:
    vals = plot_df[f'{cls}_frac_mean'].to_numpy()
    axA.bar(x_layers, vals, bottom=bottom, width=0.8, label=cls)
    bottom += vals
axA.set_title('Mismatch feature composition')
axA.set_xlabel('Layer')
axA.set_ylabel('Fraction of SAE features')
axA.set_ylim(0, 1.0)
axA.xaxis.set_major_locator(MaxNLocator(integer=True))
axA.legend(ncol=5, bbox_to_anchor=(0.5, 1.18), loc='upper center', frameon=False)
add_panel_label(axA, 'A')
axB = fig.add_subplot(gs[0, 1])
x = delta_sub['layer'].to_numpy()
y = delta_sub['image_delta_mean'].to_numpy()
sem = delta_sub['image_delta_sem'].fillna(0).to_numpy()
axB.plot(x, y, marker='o', label='mean image delta')
axB.fill_between(x, y - sem, y + sem, alpha=0.16)
y2 = delta_sub['text_delta_mean'].to_numpy()
sem2 = delta_sub['text_delta_sem'].fillna(0).to_numpy()
axB.plot(x, y2, marker='o', label='mean text delta')
axB.fill_between(x, y2 - sem2, y2 + sem2, alpha=0.16)
axB.set_title('Mean mismatch sensitivity')
axB.set_xlabel('Layer')
axB.set_ylabel('Delta')
axB.xaxis.set_major_locator(MaxNLocator(integer=True))
axB.legend()
add_panel_label(axB, 'B')
axC = fig.add_subplot(gs[1, 0])
y3 = delta_sub['mismatch_total_mean'].to_numpy()
sem3 = delta_sub['mismatch_total_sem'].fillna(0).to_numpy()
axC.plot(x, y3, marker='o', label='mismatch total')
axC.fill_between(x, y3 - sem3, y3 + sem3, alpha=0.16)
axC.set_title('Total mismatch sensitivity')
axC.set_xlabel('Layer')
axC.set_ylabel('Delta magnitude')
axC.xaxis.set_major_locator(MaxNLocator(integer=True))
axC.legend()
add_panel_label(axC, 'C')
axD = fig.add_subplot(gs[1, 1])
sc = axD.scatter(sample_df['image_delta'], sample_df['text_delta'], c=sample_df['layer'], s=8, alpha=0.35)
axD.set_title('Feature-level image vs text sensitivity')
axD.set_xlabel('Image delta')
axD.set_ylabel('Text delta')
cbar = fig.colorbar(sc, ax=axD)
cbar.set_label('Layer')
add_panel_label(axD, 'D')
fig.suptitle('Mismatch-based feature analysis')
savefig_all(fig, 'fig2_mismatch_grouped')
plt.show()
fig = plt.figure(figsize=(17, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
axA = fig.add_subplot(gs[0, 0])
for cond, sub in act_agg.groupby('condition'):
    sub = sub.sort_values('layer')
    x = sub['layer'].to_numpy()
    y = sub['num_features_above_threshold_mean'].to_numpy()
    sem = sub['num_features_above_threshold_sem'].fillna(0).to_numpy()
    axA.plot(x, y, marker='o', label=cond)
    axA.fill_between(x, y - sem, y + sem, alpha=0.16)
axA.set_title('Strongly active features by condition')
axA.set_xlabel('Layer')
axA.set_ylabel('Count')
axA.xaxis.set_major_locator(MaxNLocator(integer=True))
axA.legend()
add_panel_label(axA, 'A')
axB = fig.add_subplot(gs[0, 1])
for cond, sub in act_agg.groupby('condition'):
    sub = sub.sort_values('layer')
    x = sub['layer'].to_numpy()
    y = sub['effective_dimension_mean'].to_numpy()
    sem = sub['effective_dimension_sem'].fillna(0).to_numpy()
    axB.plot(x, y, marker='o', label=cond)
    axB.fill_between(x, y - sem, y + sem, alpha=0.16)
axB.set_title('Effective dimension by condition')
axB.set_xlabel('Layer')
axB.set_ylabel('Participation ratio')
axB.xaxis.set_major_locator(MaxNLocator(integer=True))
axB.legend()
add_panel_label(axB, 'B')
axC = fig.add_subplot(gs[1, 0])
x = topk_sub['layer'].to_numpy()
y = topk_sub['topk_mean_image_delta_mean'].to_numpy()
sem = topk_sub['topk_mean_image_delta_sem'].fillna(0).to_numpy()
axC.plot(x, y, marker='o', label='top-50 mean image delta')
axC.fill_between(x, y - sem, y + sem, alpha=0.16)
axC.set_title('Top-K image-sensitive features')
axC.set_xlabel('Layer')
axC.set_ylabel('Mean delta among top features')
axC.xaxis.set_major_locator(MaxNLocator(integer=True))
axC.legend()
add_panel_label(axC, 'C')
axD = fig.add_subplot(gs[1, 1])
y2 = topk_sub['topk_mean_text_delta_mean'].to_numpy()
sem2 = topk_sub['topk_mean_text_delta_sem'].fillna(0).to_numpy()
axD.plot(x, y2, marker='o', label='top-50 mean text delta')
axD.fill_between(x, y2 - sem2, y2 + sem2, alpha=0.16)
axD.set_title('Top-K text-sensitive features')
axD.set_xlabel('Layer')
axD.set_ylabel('Mean delta among top features')
axD.xaxis.set_major_locator(MaxNLocator(integer=True))
axD.legend()
add_panel_label(axD, 'D')
fig.suptitle('Condition-dependent feature activity structure (mean ± SEM)')
savefig_all(fig, 'fig3_activity_grouped')
plt.show()
fig, ax = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
im = ax.imshow(pivot_jaccard.values, aspect='auto', interpolation='nearest')
ax.set_title('Jaccard heatmap across layers and comparisons')
ax.set_xlabel('Layer')
ax.set_ylabel('Comparison')
ax.set_xticks(np.arange(len(pivot_jaccard.columns)))
ax.set_xticklabels(pivot_jaccard.columns)
ax.set_yticks(np.arange(len(pivot_jaccard.index)))
ax.set_yticklabels([pretty_comparison_label(c) for c in pivot_jaccard.index])
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Jaccard')
savefig_all(fig, 'fig4_jaccard_heatmap_grouped')
plt.show()
fig = plt.figure(figsize=(18, 11), constrained_layout=True)
gs = fig.add_gridspec(2, 2)
axA = fig.add_subplot(gs[0, 0])
plot_metric_with_sem(axA, aggregated_results_df, 'jaccard', ylabel='Jaccard', title='Jaccard overlap')
add_panel_label(axA, 'A')
axB = fig.add_subplot(gs[0, 1])
plot_metric_with_sem(axB, aggregated_results_df, 'overlap_at_k', ylabel='Overlap@K', title='Top-K overlap')
add_panel_label(axB, 'B')
axC = fig.add_subplot(gs[1, 0])
bottom = np.zeros(len(plot_df))
for cls in stack_order:
    vals = plot_df[f'{cls}_frac_mean'].to_numpy()
    axC.bar(x_layers, vals, bottom=bottom, width=0.8, label=cls)
    bottom += vals
axC.set_title('Mismatch feature composition')
axC.set_xlabel('Layer')
axC.set_ylabel('Fraction')
axC.set_ylim(0, 1.0)
axC.xaxis.set_major_locator(MaxNLocator(integer=True))
add_panel_label(axC, 'C')
axD = fig.add_subplot(gs[1, 1])
im = axD.imshow(pivot_jaccard.values, aspect='auto', interpolation='nearest')
axD.set_title('Jaccard heatmap')
axD.set_xlabel('Layer')
axD.set_ylabel('Comparison')
axD.set_xticks(np.arange(len(pivot_jaccard.columns)))
axD.set_xticklabels(pivot_jaccard.columns)
axD.set_yticks(np.arange(len(pivot_jaccard.index)))
axD.set_yticklabels([pretty_comparison_label(c) for c in pivot_jaccard.index])
fig.colorbar(im, ax=axD, shrink=0.9)
add_panel_label(axD, 'D')
handles, labels = axA.get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
fig.suptitle('Superposition structure across layers in LLM and VLM representations')
savefig_all(fig, 'multiseed_sparse_mismatch_diagnostic')
plt.show()

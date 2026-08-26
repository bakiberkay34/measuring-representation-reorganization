import os
import sys
from getpass import getpass
from huggingface_hub import login
import gc
import math
import requests
from io import BytesIO
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from sae_lens import SAE
from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration
HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN and sys.stdin.isatty():
    HF_TOKEN = getpass('Enter your Hugging Face token: ').strip() or None
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MODEL_ID = 'google/paligemma2-3b-pt-224'
SAE_RELEASE = 'gemma-scope-2b-pt-res-canonical'
PIXELPROSE_DATASET = 'tomg-group-umd/pixelprose'
PIXELPROSE_SPLIT = 'commonpool'
MIN_SAMPLES = 100
TARGET_SAMPLES = 150
MAX_DOWNLOAD_ATTEMPTS = 5000
OUTPUT_DIR = 'sae_pixelprose_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print('device:', device)
print('dtype:', dtype)
print('output dir:', OUTPUT_DIR)
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map='auto')
model.eval()
text_layers = model.model.language_model.layers
num_layers = len(text_layers)
hidden_size = model.config.text_config.hidden_size
image_token = getattr(processor.tokenizer, 'image_token', None)
image_token_id = None
if image_token is not None:
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)
print('num_layers:', num_layers)
print('hidden_size:', hidden_size)
print('image_token:', image_token)
print('image_token_id:', image_token_id)

def chunked(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]

def pil_from_url(url: str, timeout: int=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert('RGB')
        return img
    except Exception:
        return None

def sample_pixelprose_pairs(target_samples: int=TARGET_SAMPLES, min_samples: int=MIN_SAMPLES, split: str=PIXELPROSE_SPLIT, max_attempts: int=MAX_DOWNLOAD_ATTEMPTS):
    ds = load_dataset(PIXELPROSE_DATASET, split=split, streaming=True)
    pairs = []
    attempts = 0
    for ex in ds:
        attempts += 1
        if attempts > max_attempts:
            break
        url = ex.get('url', None)
        caption = ex.get('vlm_caption', None) or ex.get('original_caption', None)
        if not url or not caption:
            continue
        if not isinstance(caption, str) or len(caption.strip()) == 0:
            continue
        img = pil_from_url(url)
        if img is None:
            continue
        pairs.append({'image': img, 'text': caption.strip(), 'url': url})
        if len(pairs) >= target_samples:
            break
    if len(pairs) < min_samples:
        raise RuntimeError(f'Only collected {len(pairs)} valid PixelProse pairs. Need at least {min_samples}.')
    print(f'Collected {len(pairs)} valid PixelProse samples.')
    return pairs

def load_sae_for_layer(layer_idx: int):
    sae_obj = SAE.from_pretrained(release=SAE_RELEASE, sae_id=f'layer_{layer_idx}/width_16k/canonical', device=device)
    sae = sae_obj[0] if isinstance(sae_obj, tuple) else sae_obj
    sae = sae.to(device)
    sae.eval()
    if sae.cfg.d_in != hidden_size:
        raise ValueError(f'SAE/model mismatch at layer {layer_idx}: sae d_in={sae.cfg.d_in}, hidden_size={hidden_size}')
    return sae

def prepare_multimodal_batch(batch):
    images = [x['image'] for x in batch]
    texts = [f"<image> {x['text']}" for x in batch]
    enc = processor(images=images, text=texts, padding=True, truncation=True, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc

def prepare_text_only_batch(batch):
    texts = [x['text'] for x in batch]
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}
    return enc

def get_text_mask_from_multimodal_inputs(inputs):
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask'].bool()
    if image_token_id is None:
        return attention_mask
    return attention_mask & (input_ids != image_token_id)

def get_text_mask_from_text_inputs(inputs):
    return inputs['attention_mask'].bool()

def find_safe_batch_size(samples, mode='multimodal', trial_sizes=(1, 2, 4, 8, 16)):
    safe_bs = 1
    for bs in trial_sizes:
        trial_batch = samples[:bs]
        if len(trial_batch) < bs:
            break
        try:
            if mode == 'multimodal':
                inputs = prepare_multimodal_batch(trial_batch)
                with torch.no_grad():
                    _ = model(**inputs, use_cache=False)
            elif mode == 'text_only':
                inputs = prepare_text_only_batch(trial_batch)
                with torch.no_grad():
                    _ = model.model.language_model(**inputs, use_cache=False)
            else:
                raise ValueError(f'Unknown mode: {mode}')
            safe_bs = bs
            del inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
    return safe_bs
samples = sample_pixelprose_pairs(target_samples=TARGET_SAMPLES, min_samples=MIN_SAMPLES, split=PIXELPROSE_SPLIT, max_attempts=MAX_DOWNLOAD_ATTEMPTS)
print('First sample caption:')
print(samples[0]['text'][:300])
print('First image size:', samples[0]['image'].size)
multimodal_bs = find_safe_batch_size(samples, mode='multimodal')
text_only_bs = find_safe_batch_size(samples, mode='text_only')
print('multimodal batch size:', multimodal_bs)
print('text-only batch size:', text_only_bs)

def run_layerwise_sae_experiment(samples, experiment_name: str, batch_size: int, mode: str):
    assert mode in {'multimodal', 'text_only'}
    results = []
    for layer_idx in range(num_layers):
        print(f'[{experiment_name}] layer {layer_idx}/{num_layers - 1}')
        sae = load_sae_for_layer(layer_idx)
        captured = {}

        def hook_fn(module, module_input, module_output):
            out = module_output[0] if isinstance(module_output, tuple) else module_output
            captured['act'] = out.detach()
        hook = text_layers[layer_idx].register_forward_hook(hook_fn)
        total_sqerr = 0.0
        total_numel = 0
        total_l0 = 0.0
        total_rows = 0
        used_samples = 0
        for batch in chunked(samples, batch_size):
            try:
                if mode == 'multimodal':
                    inputs = prepare_multimodal_batch(batch)
                    token_mask = get_text_mask_from_multimodal_inputs(inputs)
                    with torch.no_grad():
                        _ = model(**inputs, use_cache=False)
                elif mode == 'text_only':
                    inputs = prepare_text_only_batch(batch)
                    token_mask = get_text_mask_from_text_inputs(inputs)
                    with torch.no_grad():
                        _ = model.model.language_model(**inputs, use_cache=False)
                act = captured['act']
                flat_act = act.reshape(-1, act.shape[-1])
                flat_mask = token_mask.reshape(-1)
                flat_act = flat_act[flat_mask]
                if flat_act.numel() == 0:
                    continue
                with torch.no_grad():
                    encoded = sae.encode(flat_act)
                    decoded = sae.decode(encoded)
                    sqerr = (flat_act - decoded).pow(2)
                    total_sqerr += sqerr.sum().item()
                    total_numel += sqerr.numel()
                    l0_per_token = (encoded > 0).float().sum(dim=-1)
                    total_l0 += l0_per_token.sum().item()
                    total_rows += l0_per_token.numel()
                used_samples += len(batch)
                del inputs, token_mask, act, flat_act, flat_mask
                del encoded, decoded, sqerr, l0_per_token
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f'OOM at layer {layer_idx}; skipping one batch.')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        hook.remove()
        recon_loss = total_sqerr / total_numel if total_numel > 0 else float('nan')
        avg_l0 = total_l0 / total_rows if total_rows > 0 else float('nan')
        results.append({'experiment': experiment_name, 'layer': layer_idx, 'reconstruction_loss': recon_loss, 'avg_l0': avg_l0, 'used_samples': used_samples})
        del sae, captured
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pd.DataFrame(results)
df_multimodal = run_layerwise_sae_experiment(samples=samples, experiment_name='pixelprose_image_text', batch_size=multimodal_bs, mode='multimodal')
multimodal_csv = os.path.join(OUTPUT_DIR, 'sae_pixelprose_image_text.csv')
df_multimodal.to_csv(multimodal_csv, index=False)
print(df_multimodal.head())
print('saved:', multimodal_csv)
df_text_only = run_layerwise_sae_experiment(samples=samples, experiment_name='pixelprose_text_only', batch_size=text_only_bs, mode='text_only')
textonly_csv = os.path.join(OUTPUT_DIR, 'sae_pixelprose_text_only.csv')
df_text_only.to_csv(textonly_csv, index=False)
print(df_text_only.head())
print('saved:', textonly_csv)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(df_multimodal['layer'], df_multimodal['reconstruction_loss'], marker='o', label='image+text')
axes[0].plot(df_text_only['layer'], df_text_only['reconstruction_loss'], marker='o', label='text-only')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Reconstruction loss')
axes[0].legend()
axes[1].plot(df_multimodal['layer'], df_multimodal['avg_l0'], marker='o', label='image+text')
axes[1].plot(df_text_only['layer'], df_text_only['avg_l0'], marker='o', label='text-only')
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('L0')
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'gemmascope_diagnostic.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'gemmascope_diagnostic.svg'), bbox_inches='tight')
plt.close(fig)

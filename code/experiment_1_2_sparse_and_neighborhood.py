import os
import sys
from getpass import getpass
from huggingface_hub import login
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import gc
import io
import re
import json
import random
import warnings
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import orthogonal_procrustes
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN and sys.stdin.isatty():
    HF_TOKEN = getpass('Enter your Hugging Face token: ').strip() or None
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

@dataclass
class CFG:
    results_dir: str = 'results_vlm_feature_geometry_h100'
    resize_side: int = 224
    max_text_length: int = 64
    exp1_max_layers: int = 8
    exp1_batch_size_llm: int = 8
    exp1_batch_size_vlm: int = 2
    exp1_train_examples: int = 1024
    exp1_eval_examples: int = 512
    exp1_sae_multiplier: int = 2
    exp1_sae_steps: int = 1200
    exp1_sae_batch_size: int = 256
    exp1_sae_lr: float = 0.001
    exp1_sae_l1: float = 0.001
    exp2_max_layers: int = 8
    exp2_batch_size_llm: int = 8
    exp2_batch_size_vlm: int = 2
    exp2_concepts: int = 120
    exp2_samples_per_concept: int = 4
    exp2_templates_per_concept: int = 4
    exp2_pca_dim: int = 16
    exp2_topk_list: Tuple[int, ...] = (3, 5, 10, 20)
    max_scan_multiplier: int = 80
    exp2_scan_examples: int = 5000
    overwrite_existing: bool = False
cfg = CFG()
ROOT = Path(cfg.results_dir)
for _sub in ['activations', 'exp1', 'exp2', 'tables', 'figures', 'logs']:
    (ROOT / _sub).mkdir(parents=True, exist_ok=True)
FAMILIES = {'qwen': {'llm_id': 'Qwen/Qwen2.5-3B-Instruct', 'vlm_id': 'Qwen/Qwen2.5-VL-3B-Instruct', 'kind': 'qwen'}, 'smol': {'llm_id': 'HuggingFaceTB/SmolLM2-1.7B-Instruct', 'vlm_id': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct', 'kind': 'smol'}, 'pali': {'llm_id': 'google/gemma-2-2b', 'vlm_id': 'google/paligemma2-3b-pt-224', 'kind': 'paligemma'}}
DATASETS_CFG = {'pixelprose': {'dataset_id': 'tomg-group-umd/pixelprose', 'split': 'commonpool'}, 'flickr30k': {'dataset_id': 'nlphuji/flickr30k', 'split': 'test'}, 'coco_karpathy': {'dataset_id': 'yerevann/coco-karpathy', 'split': 'test'}}
BASE_CONCEPTS = ['cat', 'dog', 'horse', 'cow', 'sheep', 'bird', 'fish', 'elephant', 'bear', 'zebra', 'giraffe', 'duck', 'goose', 'monkey', 'lion', 'tiger', 'deer', 'rabbit', 'mouse', 'squirrel', 'goat', 'pig', 'chicken', 'snake', 'turtle', 'car', 'truck', 'bus', 'train', 'bicycle', 'motorcycle', 'boat', 'airplane', 'van', 'taxi', 'tram', 'subway', 'ship', 'scooter', 'skateboard', 'traffic light', 'stop sign', 'fire hydrant', 'person', 'man', 'woman', 'child', 'boy', 'girl', 'baby', 'people', 'crowd', 'player', 'skier', 'surfer', 'rider', 'driver', 'worker', 'couple', 'family', 'chair', 'table', 'bed', 'sofa', 'couch', 'bench', 'desk', 'shelf', 'cabinet', 'lamp', 'clock', 'mirror', 'door', 'window', 'sign', 'umbrella', 'backpack', 'suitcase', 'handbag', 'book', 'vase', 'pillow', 'blanket', 'box', 'bag', 'phone', 'cell phone', 'laptop', 'computer', 'keyboard', 'television', 'tv', 'monitor', 'camera', 'remote', 'screen', 'tablet', 'headphones', 'microphone', 'apple', 'banana', 'orange', 'bread', 'coffee', 'tea', 'pizza', 'cake', 'sandwich', 'hot dog', 'donut', 'doughnut', 'broccoli', 'carrot', 'bowl', 'cup', 'bottle', 'glass', 'plate', 'fork', 'knife', 'spoon', 'pot', 'pan', 'wine glass', 'sink', 'refrigerator', 'oven', 'toaster', 'ball', 'baseball', 'tennis', 'frisbee', 'ski', 'snowboard', 'surfboard', 'kite', 'bat', 'glove', 'racket', 'helmet', 'field', 'court', 'game', 'shirt', 'pants', 'jacket', 'hat', 'shoe', 'dress', 'tie', 'coat', 'shorts', 'uniform', 'jeans', 'cap', 'glasses', 'street', 'road', 'sidewalk', 'building', 'house', 'room', 'kitchen', 'bathroom', 'beach', 'park', 'forest', 'mountain', 'river', 'lake', 'ocean', 'sky', 'tree', 'flower', 'grass', 'snow', 'water', 'cloud', 'bridge', 'fence', 'wall', 'city', 'yard', 'garden', 'hill', 'path', 'floor', 'ceiling']
TEXT_TEMPLATES = ['A photo of a {concept}.', 'An image of a {concept}.', 'A picture of a {concept}.', 'This is a {concept}.', 'There is a {concept} in the scene.', 'The image shows a {concept}.']

def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def get_single_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')

def load_family_models(family_key: str):
    fam = FAMILIES[family_key]
    single_device = get_single_device()
    tokenizer = AutoTokenizer.from_pretrained(fam['llm_id'], use_fast=True, trust_remote_code=True)
    pad_token_setup(tokenizer)
    llm = AutoModelForCausalLM.from_pretrained(fam['llm_id'], torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation='sdpa').eval()
    if torch.cuda.is_available():
        llm = llm.to(single_device)
    processor = AutoProcessor.from_pretrained(fam['vlm_id'], trust_remote_code=True)
    vlm = AutoModelForImageTextToText.from_pretrained(fam['vlm_id'], torch_dtype=DTYPE, device_map=None, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation='sdpa').eval()
    if torch.cuda.is_available():
        vlm = vlm.to(single_device)
    return (tokenizer, llm, processor, vlm)

def pad_token_setup(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

def normalize_text(s: str) -> str:
    s = '' if s is None else str(s)
    s = s.lower().strip()
    s = re.sub('\\s+', ' ', s)
    return s

def contains_whole_word(text: str, word: str) -> bool:
    return re.search(f'(?<![a-z0-9]){re.escape(word.lower())}(?![a-z0-9])', normalize_text(text)) is not None

def safe_image_from_url(url: str, timeout: int=12):
    try:
        r = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception:
        return None

def resize_image(img: Image.Image, side: int):
    img = img.convert('RGB')
    return img.resize((side, side))

def _extract_image(ex):
    for key in ['image', 'jpg', 'png', 'jpeg']:
        if key in ex and ex[key] is not None:
            obj = ex[key]
            if isinstance(obj, Image.Image):
                return resize_image(obj, cfg.resize_side)
            if isinstance(obj, dict):
                if obj.get('bytes') is not None:
                    try:
                        return resize_image(Image.open(io.BytesIO(obj['bytes'])), cfg.resize_side)
                    except Exception:
                        pass
                if obj.get('path'):
                    try:
                        return resize_image(Image.open(obj['path']), cfg.resize_side)
                    except Exception:
                        pass
    for key in ['url', 'image_url']:
        if isinstance(ex.get(key), str) and ex[key].strip():
            img = safe_image_from_url(ex[key].strip())
            if img is not None:
                return resize_image(img, cfg.resize_side)
    return None

def _extract_captions(ex):
    candidates = []
    for key in ['caption', 'captions', 'sentences', 'text', 'txt', 'vlm_caption', 'original_caption', 'sentence']:
        if key in ex and ex[key] is not None:
            obj = ex[key]
            if isinstance(obj, list):
                candidates.extend([str(x) for x in obj if str(x).strip()])
            elif isinstance(obj, str):
                candidates.append(obj)
            elif isinstance(obj, dict):
                for subkey in ['text', 'caption', 'raw']:
                    if obj.get(subkey):
                        candidates.append(str(obj[subkey]))
    out, seen = ([], set())
    for c in candidates:
        c = normalize_text(c)
        if c and c not in seen:
            seen.add(c)
            out.append(' '.join(c.split()[:48]))
    return [x for x in out if len(x.split()) >= 2]

def load_examples(dataset_key: str, limit: int):
    ds_cfg = DATASETS_CFG[dataset_key]
    try:
        ds = load_dataset(ds_cfg['dataset_id'], split=ds_cfg['split'], streaming=True)
    except Exception:
        ds = load_dataset(ds_cfg['dataset_id'], split=ds_cfg['split'], streaming=False)
    examples = []
    max_scan = max(limit * cfg.max_scan_multiplier, 500)
    for i, ex in enumerate(ds, start=1):
        if i > max_scan:
            break
        img = _extract_image(ex)
        caps = _extract_captions(ex)
        if img is None or not caps:
            continue
        examples.append({'image': img, 'captions': caps, 'primary_caption': caps[0]})
        if len(examples) >= limit:
            break
    if not examples:
        raise RuntimeError(f'No usable examples found for {dataset_key}')
    return examples

def article_for(word: str):
    return 'an' if word[:1].lower() in 'aeiou' else 'a'

def prompt_templates_for(concept: str, max_templates: int):
    out = []
    for t in TEXT_TEMPLATES[:max_templates]:
        out.append(t.replace('a {concept}', f'{article_for(concept)} {{concept}}').format(concept=concept))
    return out

def build_concept_bank(examples, max_concepts, samples_per_concept):
    bank = {}
    for concept in BASE_CONCEPTS:
        matched = []
        for ex in examples:
            if any((contains_whole_word(c, concept) for c in ex['captions'])):
                matched.append(ex)
            if len(matched) >= samples_per_concept:
                break
        if len(matched) >= samples_per_concept:
            bank[concept] = matched[:samples_per_concept]
        if len(bank) >= max_concepts:
            break
    if len(bank) < max(4, max_concepts // 2):
        raise RuntimeError(f'Only {len(bank)} concepts found for concept bank')
    return bank

def choose_layers(n_layers: int, max_layers: int):
    if n_layers <= max_layers:
        return list(range(n_layers))
    idxs = np.linspace(0, n_layers - 1, max_layers, dtype=int).tolist()
    return sorted(set(idxs + [n_layers - 1]))

def llm_num_layers(model):
    return int(model.config.num_hidden_layers)

def vlm_num_layers(model):
    if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'num_hidden_layers'):
        return int(model.config.text_config.num_hidden_layers)
    return int(model.config.num_hidden_layers)

def make_mode_examples(full_batch, mode, text_override=None):
    n = len(full_batch)
    if mode == 'matched':
        return [{'image': full_batch[i]['image'], 'text': full_batch[i]['primary_caption']} for i in range(n)]
    if mode == 'text_swapped':
        return [{'image': full_batch[i]['image'], 'text': full_batch[(i + 1) % n]['primary_caption']} for i in range(n)]
    if mode == 'image_swapped':
        return [{'image': full_batch[(i + 1) % n]['image'], 'text': full_batch[i]['primary_caption']} for i in range(n)]
    if mode == 'image_only':
        return [{'image': full_batch[i]['image'], 'text': 'Describe the image briefly.'} for i in range(n)]
    if mode == 'custom_text':
        if text_override is None or len(text_override) != n:
            raise ValueError('Invalid text_override')
        return [{'image': full_batch[i]['image'], 'text': text_override[i]} for i in range(n)]
    raise ValueError(f'Unsupported mode: {mode}')

def prepare_vlm_inputs_from_pairs(processor, pair_batch, family_kind, model_device=None, model_dtype=None):
    import torch
    if model_dtype is None:
        model_dtype = torch.bfloat16

    def _move_and_cast(x):
        if torch.is_tensor(x):
            if model_device is not None:
                x = x.to(model_device)
            if torch.is_floating_point(x):
                x = x.to(dtype=model_dtype)
            return x
        if isinstance(x, list):
            out = []
            for y in x:
                if torch.is_tensor(y):
                    if model_device is not None:
                        y = y.to(model_device)
                    if torch.is_floating_point(y):
                        y = y.to(dtype=model_dtype)
                out.append(y)
            return out
        return x

    def _pad_2d_tensors(vals, pad_value):
        max_len = max((v.shape[1] for v in vals))
        padded = []
        for v in vals:
            if v.shape[1] == max_len:
                padded.append(v)
            else:
                pad_width = max_len - v.shape[1]
                pad = torch.full((v.shape[0], pad_width), pad_value, dtype=v.dtype, device=v.device)
                padded.append(torch.cat([v, pad], dim=1))
        return torch.cat(padded, dim=0)
    if family_kind == 'qwen':
        conversations = [[{'role': 'user', 'content': [{'type': 'image', 'image': pair['image']}, {'type': 'text', 'text': pair['text']}]}] for pair in pair_batch]
        rendered_texts = [processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in conversations]
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = processor(text=rendered_texts, images=image_inputs, videos=video_inputs, padding=True, truncation=True, return_tensors='pt')
    elif family_kind == 'paligemma':
        texts = [f"<image> {pair['text']}" for pair in pair_batch]
        images = [pair['image'] for pair in pair_batch]
        inputs = processor(images=images, text=texts, padding=True, truncation=True, return_tensors='pt')
    elif family_kind == 'smol':
        messages = [[{'role': 'user', 'content': [{'type': 'image', 'image': pair['image']}, {'type': 'text', 'text': pair['text']}]}] for pair in pair_batch]
        encoded = [processor.apply_chat_template(msg, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors='pt') for msg in messages]
        inputs = {}
        all_keys = sorted(set().union(*[set(e.keys()) for e in encoded]))
        for k in all_keys:
            vals = [e[k] for e in encoded if k in e]
            if len(vals) == 0:
                continue
            if not torch.is_tensor(vals[0]):
                inputs[k] = vals
                continue
            if vals[0].ndim == 2 and k in ['input_ids', 'attention_mask', 'token_type_ids']:
                if k == 'input_ids':
                    pad_token_id = None
                    if hasattr(processor, 'tokenizer'):
                        pad_token_id = getattr(processor.tokenizer, 'pad_token_id', None)
                    if pad_token_id is None:
                        pad_token_id = 0
                    inputs[k] = _pad_2d_tensors(vals, int(pad_token_id))
                elif k == 'attention_mask':
                    inputs[k] = _pad_2d_tensors(vals, 0)
                else:
                    inputs[k] = _pad_2d_tensors(vals, 0)
            elif all((tuple(v.shape[1:]) == tuple(vals[0].shape[1:]) for v in vals)):
                inputs[k] = torch.cat(vals, dim=0)
            elif vals[0].ndim == 2:
                inputs[k] = _pad_2d_tensors(vals, 0)
            else:
                inputs[k] = vals
    else:
        raise ValueError(f'Unsupported family_kind: {family_kind}')
    inputs = {k: _move_and_cast(v) for k, v in inputs.items()}
    return inputs

@torch.no_grad()
def pooled_llm_hidden_states(model, tokenizer, texts, layers, max_length, batch_size):
    model_device = next(model.parameters()).device
    out_batches = []
    for start in range(0, len(texts), batch_size):
        sub = texts[start:start + batch_size]
        inp = tokenizer(sub, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        inp = {k: v.to(model_device) for k, v in inp.items()}
        out = model(**inp, output_hidden_states=True, use_cache=False)
        attn = inp['attention_mask'].detach().cpu().numpy().astype(bool)
        batch_layers = []
        for li in layers:
            hs = out.hidden_states[li + 1].detach().float().cpu().numpy()
            pooled = []
            for i in range(hs.shape[0]):
                valid_idx = np.where(attn[i])[0]
                tail_start = max(0, int(0.75 * len(valid_idx)))
                tail_idx = valid_idx[tail_start:]
                pooled.append(hs[i, tail_idx].mean(axis=0))
            batch_layers.append(np.stack(pooled, axis=0))
        out_batches.append(np.stack(batch_layers, axis=1))
        del inp, out
        clear_mem()
    return np.concatenate(out_batches, axis=0)

@torch.no_grad()
def pooled_vlm_hidden_states(model, processor, batch, layers, family_kind, mode, batch_size, text_override=None):
    model_device = next(model.parameters()).device
    prepared = make_mode_examples(batch, mode, text_override=text_override)
    chunks = [[] for _ in layers]
    for start in range(0, len(prepared), batch_size):
        sub_pairs = prepared[start:start + batch_size]
        inp = prepare_vlm_inputs_from_pairs(processor=processor, pair_batch=sub_pairs, family_kind=family_kind, model_device=model_device)
        out = model(**inp, output_hidden_states=True, use_cache=False)
        attn = inp['attention_mask'].detach().cpu().numpy().astype(bool)
        for j, li in enumerate(layers):
            hs = out.hidden_states[li + 1].detach().float().cpu().numpy()
            pooled = []
            for i in range(hs.shape[0]):
                valid_idx = np.where(attn[i])[0]
                tail_start = max(0, int(0.75 * len(valid_idx)))
                tail_idx = valid_idx[tail_start:]
                pooled.append(hs[i, tail_idx].mean(axis=0))
            chunks[j].append(np.stack(pooled, axis=0))
        del inp, out
        clear_mem()
    stacked = [np.concatenate(x, axis=0) for x in chunks]
    return np.stack(stacked, axis=1)

class TinySAE(nn.Module):

    def __init__(self, d_in, d_sae):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)

    def forward(self, x):
        z = torch.relu(self.encoder(x))
        xhat = self.decoder(z)
        return (xhat, z)

def train_sae_numpy(X, d_sae, steps, batch_size, lr, l1, seed):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    x = torch.tensor(X, dtype=torch.float32, device=dev)
    m = TinySAE(X.shape[1], d_sae).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = x.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, size=(min(batch_size, n),), device=dev)
        xb = x[idx]
        xhat, z = m(xb)
        loss = ((xhat - xb) ** 2).mean() + l1 * z.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()

@torch.no_grad()
def sae_encode_numpy(model, X):
    dev = next(model.parameters()).device
    x = torch.tensor(X, dtype=torch.float32, device=dev)
    _, z = model(x)
    return z.detach().cpu().numpy()

def gini(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.maximum(x, 0)
    if x.size == 0 or np.allclose(x.sum(), 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * x) / (n * np.sum(x) + 1e-12))

def participation_ratio(v):
    v = np.asarray(v, dtype=float)
    return float(v.sum() ** 2 / (np.square(v).sum() + 1e-12)) if v.size else 0.0

def l2_normalize_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def topk_neighbors_from_sim(sim, labels, idx, k):
    order = np.argsort(-sim[idx])
    out = []
    for j in order:
        if j == idx:
            continue
        out.append(labels[j])
        if len(out) >= k:
            break
    return out

def rank_biased_overlap(a, b, p=0.9):
    sa, sb, score = (set(), set(), 0.0)
    depth = max(len(a), len(b))
    for d in range(depth):
        if d < len(a):
            sa.add(a[d])
        if d < len(b):
            sb.add(b[d])
        score += (1 - p) * p ** d * (len(sa & sb) / max(1, d + 1))
    return float(score)
print('Setup complete.')
print('device:', DEVICE)
print('dtype:', DTYPE)
print(json.dumps(asdict(cfg), indent=2))
from pathlib import Path
import os, gc, json, random, itertools, math, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import orthogonal_procrustes
torch.set_float32_matmul_precision('high')
cfg.results_dir = 'results_vlm_feature_geometry_h100'
cfg.resize_side = 224
cfg.max_text_length = 64
cfg.exp1_max_layers = 8
cfg.exp1_batch_size_llm = 16
cfg.exp1_batch_size_vlm = 4
cfg.exp1_train_examples = 1024
cfg.exp1_eval_examples = 512
cfg.exp1_sae_multiplier = 2
cfg.exp1_sae_steps = 1200
cfg.exp1_sae_batch_size = 256
cfg.exp1_sae_lr = 0.001
cfg.exp1_sae_l1 = 0.001
cfg.exp2_max_layers = 8
cfg.exp2_batch_size_llm = 16
cfg.exp2_batch_size_vlm = 4
cfg.exp2_concepts = 120
cfg.exp2_samples_per_concept = 4
cfg.exp2_templates_per_concept = 4
cfg.exp2_pca_dim = 16
cfg.exp2_topk_list = (3, 5, 10, 20)
cfg.exp2_scan_examples = 5000
cfg.max_scan_multiplier = 80
cfg.overwrite_existing = False
ROOT = Path(cfg.results_dir)
for sub in ['activations', 'exp1', 'exp2', 'tables', 'figures', 'logs', ]:
    (ROOT / sub).mkdir(parents=True, exist_ok=True)
FAMILIES_TO_RUN = ['qwen', 'smol', 'pali']
DATASETS_TO_RUN = ['pixelprose', 'flickr30k', 'coco_karpathy']
ROBUST_SEEDS = [11, 23, 37]
ROBUST_WIDTH_FACTORS = [2.0, 4.0]
ROBUST_L1S = [0.0003, 0.001]
ROBUST_NORMALIZE = [False, True]
DOM_MARGINS = [1.1, 1.25, 1.5]
BOOT_N = 500
MAIN_POOLING = 'final_quarter'
POOLING_AUDIT_STRATEGIES = ['all_valid', 'last_token']
POOLING_AUDIT_SEEDS = [11]
POOLING_AUDIT_WIDTH_FACTORS = [2.0]
POOLING_AUDIT_L1S = [0.001]
POOLING_AUDIT_NORMALIZE = [False]
plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 300, 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def clear_mem_strong():
    try:
        clear_mem()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def save_fig(fig, name):
    out = ROOT / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f'{name}.pdf'
    svg = out / f'{name}.svg'
    fig.savefig(pdf, bbox_inches='tight', dpi=300)
    fig.savefig(svg, bbox_inches='tight', dpi=300)
    print('saved:', pdf)
    print('saved:', svg)

def collect_csvs(folder):
    files = sorted(Path(folder).glob('*.csv'))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df):
                dfs.append(df)
        except Exception as e:
            print('could not read', f, repr(e))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def save_table(df, name, float_format='%.3f'):
    out = ROOT / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f'{name}.csv'
    df.to_csv(csv, index=False)
    print('saved:', csv)

def load_examples_disjoint(dataset_key, train_n, eval_n, seed=SEED):
    pool = load_examples(dataset_key, train_n + eval_n)
    if len(pool) < train_n + eval_n:
        raise RuntimeError(f'{dataset_key} returned only {len(pool)} examples; need {train_n + eval_n}.')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    train_idx = order[:train_n]
    eval_idx = order[train_n:train_n + eval_n]
    assert set(train_idx).isdisjoint(set(eval_idx))
    return ([pool[i] for i in train_idx], [pool[i] for i in eval_idx])

def _derangement(n, seed):
    rng = np.random.default_rng(seed)
    if n <= 1:
        return np.arange(n)
    for _ in range(1000):
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p
    return np.roll(np.arange(n), 1)

def make_mode_examples(full_batch, mode, text_override=None):
    n = len(full_batch)
    if mode == 'matched':
        return [{'image': full_batch[i]['image'], 'text': full_batch[i]['primary_caption']} for i in range(n)]
    if mode == 'text_swapped':
        p = _derangement(n, SEED + 101)
        return [{'image': full_batch[i]['image'], 'text': full_batch[p[i]]['primary_caption']} for i in range(n)]
    if mode == 'image_swapped':
        p = _derangement(n, SEED + 202)
        return [{'image': full_batch[p[i]]['image'], 'text': full_batch[i]['primary_caption']} for i in range(n)]
    if mode == 'image_only':
        return [{'image': full_batch[i]['image'], 'text': 'Describe the image briefly.'} for i in range(n)]
    if mode == 'custom_text':
        if text_override is None or len(text_override) != n:
            raise ValueError('custom_text mode requires text_override with same length.')
        return [{'image': full_batch[i]['image'], 'text': text_override[i]} for i in range(n)]
    raise ValueError(f'Unsupported mode: {mode}')

def pool_hidden_state_one(hs_i, valid_idx, strategy=MAIN_POOLING):
    valid_idx = np.asarray(valid_idx)
    if len(valid_idx) == 0:
        raise ValueError('No valid tokens for pooling.')
    if strategy == 'final_quarter':
        tail_start = max(0, int(0.75 * len(valid_idx)))
        idx = valid_idx[tail_start:]
        return hs_i[idx].mean(axis=0)
    if strategy == 'all_valid':
        return hs_i[valid_idx].mean(axis=0)
    if strategy == 'last_token':
        return hs_i[valid_idx[-1]]
    raise ValueError(f'Unknown pooling strategy: {strategy}')

@torch.no_grad()
def pooled_llm_hidden_states_exp(model, tokenizer, texts, layers, max_length, batch_size, pooling=MAIN_POOLING):
    model_device = next(model.parameters()).device
    out_batches = []
    for start in tqdm(range(0, len(texts), batch_size), desc=f'LLM [{pooling}]', leave=False):
        sub = texts[start:start + batch_size]
        inp = tokenizer(sub, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        inp = {k: v.to(model_device) for k, v in inp.items()}
        out = model(**inp, output_hidden_states=True, use_cache=False)
        attn = inp['attention_mask'].detach().cpu().numpy().astype(bool)
        layer_arrays = []
        for li in layers:
            hs = out.hidden_states[li + 1].detach().float().cpu().numpy()
            pooled = []
            for i in range(hs.shape[0]):
                valid_idx = np.where(attn[i])[0]
                pooled.append(pool_hidden_state_one(hs[i], valid_idx, strategy=pooling))
            layer_arrays.append(np.stack(pooled, axis=0))
        out_batches.append(np.stack(layer_arrays, axis=1))
        del inp, out
        clear_mem_strong()
    return np.concatenate(out_batches, axis=0)

@torch.no_grad()
def pooled_vlm_hidden_states_exp(model, processor, examples, layers, family_kind, mode, batch_size, pooling=MAIN_POOLING, text_override=None):
    import inspect
    import numpy as np
    import torch
    from tqdm.auto import tqdm
    model.eval()
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    prepared_pairs = make_mode_examples(examples, mode, text_override=text_override)
    chunks = [[] for _ in layers]

    def _move_and_cast_inputs(obj):
        if torch.is_tensor(obj):
            obj = obj.to(model_device)
            if torch.is_floating_point(obj):
                obj = obj.to(dtype=model_dtype)
            return obj
        if isinstance(obj, dict):
            return {k: _move_and_cast_inputs(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_move_and_cast_inputs(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple((_move_and_cast_inputs(v) for v in obj))
        return obj

    def _prepare_inputs(sub_pairs):
        sig = inspect.signature(prepare_vlm_inputs_from_pairs)
        kwargs = {'processor': processor, 'pair_batch': sub_pairs, 'family_kind': family_kind, 'model_device': model_device}
        if 'model_dtype' in sig.parameters:
            kwargs['model_dtype'] = model_dtype
        inp = prepare_vlm_inputs_from_pairs(**kwargs)
        inp = _move_and_cast_inputs(inp)
        return inp
    for start in tqdm(range(0, len(prepared_pairs), batch_size), desc=f'VLM {mode} [{pooling}]', leave=False):
        sub_pairs = prepared_pairs[start:start + batch_size]
        inp = _prepare_inputs(sub_pairs)
        if 'attention_mask' not in inp:
            raise KeyError('VLM inputs do not contain attention_mask; cannot pool valid tokens.')
        attn = inp['attention_mask']
        if not torch.is_tensor(attn):
            raise TypeError(f'attention_mask must be a tensor, got {type(attn)}')
        attn_np = attn.detach().cpu().numpy().astype(bool)
        with torch.inference_mode():
            out = model(**inp, output_hidden_states=True, use_cache=False, return_dict=True)
        if out.hidden_states is None:
            raise RuntimeError('Model did not return hidden_states. Check output_hidden_states=True.')
        for j, li in enumerate(layers):
            hidden_index = int(li) + 1
            if hidden_index >= len(out.hidden_states):
                raise IndexError(f'Requested layer {li}, hidden_states index {hidden_index}, but model returned only {len(out.hidden_states)} hidden-state tensors.')
            hs = out.hidden_states[hidden_index].detach().float().cpu().numpy()
            if hs.shape[0] != attn_np.shape[0]:
                raise RuntimeError(f'Batch mismatch between hidden states and attention mask: hs batch={hs.shape[0]}, attention batch={attn_np.shape[0]}')
            pooled = []
            for i in range(hs.shape[0]):
                valid_idx = np.where(attn_np[i])[0]
                if len(valid_idx) == 0:
                    valid_idx = np.arange(hs.shape[1])
                valid_idx = valid_idx[valid_idx < hs.shape[1]]
                if len(valid_idx) == 0:
                    valid_idx = np.arange(hs.shape[1])
                pooled.append(pool_hidden_state_one(hs[i], valid_idx, strategy=pooling))
            chunks[j].append(np.stack(pooled, axis=0))
        del inp, out
        clear_mem_strong()
    stacked = []
    for j, x in enumerate(chunks):
        if len(x) == 0:
            raise RuntimeError(f'No VLM chunks collected for layer index {j}.')
        stacked.append(np.concatenate(x, axis=0))
    return np.stack(stacked, axis=1)

def choose_depth_fraction_layers(n_layers, n_points):
    n_layers = int(n_layers)
    n_points = int(n_points)
    if n_layers <= 0:
        raise ValueError(f'n_layers must be positive, got {n_layers}')
    if n_points <= 0:
        raise ValueError(f'n_points must be positive, got {n_points}')
    if n_points == 1:
        fractions = np.array([0.0], dtype=float)
    else:
        fractions = np.linspace(0.0, 1.0, n_points, dtype=float)
    layers = np.array([int(round(float(f) * (n_layers - 1))) for f in fractions], dtype=int)
    unique_layers = []
    unique_fracs = []
    seen = set()
    for f, l in zip(fractions, layers):
        if int(l) not in seen:
            unique_fracs.append(float(f))
            unique_layers.append(int(l))
            seen.add(int(l))
    return (np.array(unique_fracs, dtype=float), np.array(unique_layers, dtype=int))

def choose_matched_depth_layers(llm, vlm, n_points):
    depth_fractions, llm_layers = choose_depth_fraction_layers(llm_num_layers(llm), n_points)
    _, vlm_layers = choose_depth_fraction_layers(vlm_num_layers(vlm), n_points)
    m = min(len(depth_fractions), len(llm_layers), len(vlm_layers))
    return (depth_fractions[:m], llm_layers[:m], vlm_layers[:m])

def exp1_activation_path(family_key, dataset_key, pooling):
    return ROOT / 'activations' / f'exp1_act_{family_key}_{dataset_key}_{pooling}.npz'

def exp1_result_path(family_key, dataset_key, pooling, audit=False):
    tag = 'audit' if audit else 'main'
    return ROOT / 'exp1' / f'exp1_{tag}_{family_key}_{dataset_key}_{pooling}.csv'

def robust_d_sae(d_in, width_factor):
    return int(min(max(512, round(width_factor * d_in)), 4096))

@torch.no_grad()
def sae_forward_numpy(model, X):
    dev = next(model.parameters()).device
    x = torch.tensor(X, dtype=torch.float32, device=dev)
    xhat, z = model(x)
    return (xhat.detach().cpu().numpy(), z.detach().cpu().numpy())

def normalize_by_train(X_train, arrays, do_norm):
    if not do_norm:
        return (X_train, arrays)
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-06
    return ((X_train - mu) / sd, {k: (v - mu) / sd for k, v in arrays.items()})

def recon_stats(model, X):
    xhat, z = sae_forward_numpy(model, X)
    mse = float(np.mean((xhat - X) ** 2))
    r2 = float(1.0 - mse / (np.var(X) + 1e-12))
    active = z > 1e-06
    return {'rec_mse': mse, 'rec_r2': r2, 'l0': float(active.sum(axis=1).mean()), 'active_frac': float(active.mean()), 'dead_frac': float((active.mean(axis=0) < 0.0001).mean()), 'z_abs_mean': float(np.abs(z).mean())}

def boot_ci(x, seed, n_boot=BOOT_N):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    vals = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))

def cache_exp1_activations_one(family_key, dataset_key, pooling=MAIN_POOLING, train_n=None, eval_n=None):
    import time
    import numpy as np
    import torch
    train_n = int(train_n or cfg.exp1_train_examples)
    eval_n = int(eval_n or cfg.exp1_eval_examples)
    out_path = exp1_activation_path(family_key, dataset_key, pooling)
    print('=' * 88, flush=True)
    print(f'Caching Exp1 activations | family={family_key} | dataset={dataset_key} | pooling={pooling} | train_n={train_n} | eval_n={eval_n}', flush=True)
    print('Output cache path:', out_path, flush=True)
    print('=' * 88, flush=True)
    llm_pool_fn = globals().get('pooled_llm_hidden_states_exp', None)
    vlm_pool_fn = globals().get('pooled_vlm_hidden_states_exp', None)
    if llm_pool_fn is None:
        llm_pool_fn = globals().get('pooled_llm_hidden_states', None)
    if vlm_pool_fn is None:
        vlm_pool_fn = globals().get('pooled_vlm_hidden_states', None)
    if llm_pool_fn is None:
        raise NameError('No LLM activation pooling function found. Expected `pooled_llm_hidden_states_exp` or `pooled_llm_hidden_states`.')
    if vlm_pool_fn is None:
        raise NameError('No VLM activation pooling function found. Expected `pooled_vlm_hidden_states_exp` or `pooled_vlm_hidden_states`.')
    print('Using LLM pooling function:', llm_pool_fn.__name__, flush=True)
    print('Using VLM pooling function:', vlm_pool_fn.__name__, flush=True)

    def _to_scalar(x):
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        v = arr.reshape(-1)[0]
        return v.item() if hasattr(v, 'item') else v

    def _choose_depth_fraction_layers(n_layers, n_points):
        n_layers = int(n_layers)
        n_points = int(n_points)
        if n_layers <= 0:
            raise ValueError(f'n_layers must be positive, got {n_layers}')
        if n_points <= 0:
            raise ValueError(f'n_points must be positive, got {n_points}')
        if n_points == 1:
            fractions = np.array([1.0], dtype=float)
        else:
            fractions = np.linspace(0.0, 1.0, n_points, dtype=float)
        layers = np.array([int(round(float(f) * (n_layers - 1))) for f in fractions], dtype=int)
        layers = np.clip(layers, 0, n_layers - 1)
        unique_fractions = []
        unique_layers = []
        seen = set()
        for f, layer in zip(fractions, layers):
            layer = int(layer)
            if layer not in seen:
                unique_fractions.append(float(f))
                unique_layers.append(layer)
                seen.add(layer)
        return (np.array(unique_fractions, dtype=float), np.array(unique_layers, dtype=int))

    def _validate_existing_cache(path):
        required = {'family', 'dataset', 'pooling', 'train_examples', 'eval_examples', 'depth_fractions', 'llm_layers', 'vlm_layers', 'layers', 'llm_train', 'vlm_train', 'vlm_eval_matched', 'vlm_eval_text_swapped', 'vlm_eval_image_swapped', 'vlm_eval_image_only'}
        with np.load(path, allow_pickle=True) as cached:
            files = set(cached.files)
            missing = required - files
            if missing:
                return (False, f'missing fields: {sorted(missing)}')
            cached_family = str(_to_scalar(cached['family']))
            cached_dataset = str(_to_scalar(cached['dataset']))
            cached_pooling = str(_to_scalar(cached['pooling']))
            cached_train_n = int(_to_scalar(cached['train_examples']))
            cached_eval_n = int(_to_scalar(cached['eval_examples']))
            if cached_family != str(family_key):
                return (False, f'family mismatch: {cached_family} vs {family_key}')
            if cached_dataset != str(dataset_key):
                return (False, f'dataset mismatch: {cached_dataset} vs {dataset_key}')
            if cached_pooling != str(pooling):
                return (False, f'pooling mismatch: {cached_pooling} vs {pooling}')
            if cached_train_n != train_n:
                return (False, f'train_n mismatch: {cached_train_n} vs {train_n}')
            if cached_eval_n != eval_n:
                return (False, f'eval_n mismatch: {cached_eval_n} vs {eval_n}')
            depth_fractions = cached['depth_fractions']
            llm_layers = cached['llm_layers']
            vlm_layers = cached['vlm_layers']
            if len(depth_fractions) != len(llm_layers):
                return (False, 'depth_fractions and llm_layers length mismatch')
            if len(depth_fractions) != len(vlm_layers):
                return (False, 'depth_fractions and vlm_layers length mismatch')
            n_depth = len(depth_fractions)
            if cached['llm_train'].shape[0] != train_n:
                return (False, 'llm_train train dimension mismatch')
            if cached['vlm_train'].shape[0] != train_n:
                return (False, 'vlm_train train dimension mismatch')
            if cached['llm_train'].shape[1] != n_depth:
                return (False, 'llm_train depth dimension mismatch')
            if cached['vlm_train'].shape[1] != n_depth:
                return (False, 'vlm_train depth dimension mismatch')
            for key in ['vlm_eval_matched', 'vlm_eval_text_swapped', 'vlm_eval_image_swapped', 'vlm_eval_image_only']:
                if cached[key].shape[0] != eval_n:
                    return (False, f'{key} eval dimension mismatch')
                if cached[key].shape[1] != n_depth:
                    return (False, f'{key} depth dimension mismatch')
        return (True, 'valid')
    if out_path.exists() and (not cfg.overwrite_existing):
        try:
            ok, reason = _validate_existing_cache(out_path)
            if ok:
                print('activation cache exists and is valid:', out_path.name, flush=True)
                return out_path
            print('existing cache invalid; regenerating:', out_path.name, '|', reason, flush=True)
            out_path.unlink()
        except Exception as e:
            print('could not validate existing cache; regenerating:', repr(e), flush=True)
            try:
                out_path.unlink()
            except Exception:
                pass

    def gpu_report(label):
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                reserved = torch.cuda.memory_reserved() / 1024 ** 3
                max_allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
                print(f'[GPU] {label}: allocated={allocated:.2f} GiB | reserved={reserved:.2f} GiB | max_allocated={max_allocated:.2f} GiB', flush=True)
            except Exception as e:
                print(f'[GPU] {label}: could not report GPU memory: {repr(e)}', flush=True)
    t_total = time.time()
    gpu_report('start')
    tokenizer = llm = processor = vlm = None
    train_ex = eval_ex = None
    llm_train = vlm_train = None
    eval_modes = None
    try:
        print('[1/8] Loading model family...', flush=True)
        t0 = time.time()
        tokenizer, llm, processor, vlm = load_family_models(family_key)
        print(f'[1/8] Model family loaded in {(time.time() - t0) / 60:.2f} min.', flush=True)
        gpu_report('after model load')
        family_kind = FAMILIES[family_key]['kind']
        print('family_kind:', family_kind, flush=True)
        print('[2/8] Selecting depth-aligned layers...', flush=True)
        t0 = time.time()
        n_llm = int(llm_num_layers(llm))
        n_vlm = int(vlm_num_layers(vlm))
        n_points = int(cfg.exp1_max_layers)
        depth_fractions, llm_layers = _choose_depth_fraction_layers(n_llm, n_points)
        _, vlm_layers = _choose_depth_fraction_layers(n_vlm, n_points)
        m = min(len(depth_fractions), len(llm_layers), len(vlm_layers))
        depth_fractions = np.asarray(depth_fractions[:m], dtype=float)
        llm_layers = [int(x) for x in llm_layers[:m].tolist()]
        vlm_layers = [int(x) for x in vlm_layers[:m].tolist()]
        if len(depth_fractions) == 0:
            raise RuntimeError('No depth fractions selected.')
        if len(llm_layers) != len(vlm_layers):
            raise RuntimeError('LLM/VLM layer list length mismatch after depth alignment.')
        layers = list(vlm_layers)
        print('LLM number of transformer layers:', n_llm, flush=True)
        print('VLM number of transformer layers:', n_vlm, flush=True)
        print('Depth fractions:', [round(float(x), 4) for x in depth_fractions], flush=True)
        print('LLM depth-aligned layers:', llm_layers, flush=True)
        print('VLM depth-aligned layers:', vlm_layers, flush=True)
        print(f'[2/8] Layer selection done in {time.time() - t0:.2f} sec.', flush=True)
        print('[3/8] Loading disjoint train/eval examples...', flush=True)
        print(f'Requested: train_n={train_n}, eval_n={eval_n}', flush=True)
        t0 = time.time()
        split_seed = SEED + abs(hash((family_key, dataset_key, pooling))) % 10000
        train_ex, eval_ex = load_examples_disjoint(dataset_key, train_n, eval_n, seed=split_seed)
        print(f'[3/8] Examples loaded in {(time.time() - t0) / 60:.2f} min. train={len(train_ex)}, eval={len(eval_ex)}', flush=True)
        if len(train_ex) != train_n:
            raise RuntimeError(f'Expected train_n={train_n}, got {len(train_ex)}.')
        if len(eval_ex) != eval_n:
            raise RuntimeError(f'Expected eval_n={eval_n}, got {len(eval_ex)}.')
        print('[4/8] Extracting LLM train activations...', flush=True)
        print(f'LLM batch size={cfg.exp1_batch_size_llm}, max_text_length={cfg.max_text_length}', flush=True)
        t0 = time.time()
        llm_train = llm_pool_fn(llm, tokenizer, [x['primary_caption'] for x in train_ex], llm_layers, cfg.max_text_length, cfg.exp1_batch_size_llm, pooling=pooling)
        print(f'[4/8] LLM train activations done in {(time.time() - t0) / 60:.2f} min. shape={llm_train.shape}', flush=True)
        gpu_report('after LLM train activations')
        print('[5/8] Extracting VLM train activations: matched...', flush=True)
        print(f'VLM batch size={cfg.exp1_batch_size_vlm}, resize_side={cfg.resize_side}', flush=True)
        t0 = time.time()
        vlm_train = vlm_pool_fn(vlm, processor, train_ex, vlm_layers, family_kind, 'matched', cfg.exp1_batch_size_vlm, pooling=pooling)
        print(f'[5/8] VLM train activations done in {(time.time() - t0) / 60:.2f} min. shape={vlm_train.shape}', flush=True)
        gpu_report('after VLM train activations')
        print('[6/8] Extracting VLM eval activations...', flush=True)
        eval_modes = {}
        for mode in ['matched', 'text_swapped', 'image_swapped', 'image_only']:
            print(f'      mode={mode}...', flush=True)
            t0 = time.time()
            eval_modes[mode] = vlm_pool_fn(vlm, processor, eval_ex, vlm_layers, family_kind, mode, cfg.exp1_batch_size_vlm, pooling=pooling)
            print(f'      mode={mode} done in {(time.time() - t0) / 60:.2f} min. shape={eval_modes[mode].shape}', flush=True)
            gpu_report(f'after VLM eval mode={mode}')
        print('[7/8] Validating shapes...', flush=True)
        expected_depths = len(depth_fractions)
        if llm_train.shape[0] != train_n:
            raise RuntimeError(f'llm_train first dim mismatch: {llm_train.shape[0]} vs train_n={train_n}')
        if vlm_train.shape[0] != train_n:
            raise RuntimeError(f'vlm_train first dim mismatch: {vlm_train.shape[0]} vs train_n={train_n}')
        if llm_train.shape[1] != expected_depths:
            raise RuntimeError(f'llm_train depth dim mismatch: {llm_train.shape[1]} vs {expected_depths}')
        if vlm_train.shape[1] != expected_depths:
            raise RuntimeError(f'vlm_train depth dim mismatch: {vlm_train.shape[1]} vs {expected_depths}')
        for mode, arr in eval_modes.items():
            if arr.shape[0] != eval_n:
                raise RuntimeError(f'{mode} first dim mismatch: {arr.shape[0]} vs eval_n={eval_n}')
            if arr.shape[1] != expected_depths:
                raise RuntimeError(f'{mode} depth dim mismatch: {arr.shape[1]} vs {expected_depths}')
        print('Shape validation passed.', flush=True)
        print('llm_train:', llm_train.shape, flush=True)
        print('vlm_train:', vlm_train.shape, flush=True)
        for mode, arr in eval_modes.items():
            print(f'vlm_eval_{mode}:', arr.shape, flush=True)
        print('[8/8] Saving compressed activation cache...', flush=True)
        t0 = time.time()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, family=np.array([family_key]), dataset=np.array([dataset_key]), pooling=np.array([pooling]), train_examples=np.array([train_n], dtype=int), eval_examples=np.array([eval_n], dtype=int), depth_fractions=np.asarray(depth_fractions, dtype=float), llm_layers=np.asarray(llm_layers, dtype=int), vlm_layers=np.asarray(vlm_layers, dtype=int), layers=np.asarray(layers, dtype=int), llm_train=llm_train, vlm_train=vlm_train, vlm_eval_matched=eval_modes['matched'], vlm_eval_text_swapped=eval_modes['text_swapped'], vlm_eval_image_swapped=eval_modes['image_swapped'], vlm_eval_image_only=eval_modes['image_only'])
        print(f'[8/8] Cache saved in {(time.time() - t0) / 60:.2f} min.', flush=True)
        print('saved activation cache:', out_path, flush=True)
        if not out_path.exists():
            raise RuntimeError(f'Cache file was not created: {out_path}')
        if out_path.stat().st_size == 0:
            raise RuntimeError(f'Cache file is empty: {out_path}')
        print(f'Cache file size: {out_path.stat().st_size / 1024 ** 2:.2f} MiB', flush=True)
        print(f'Finished cache_exp1_activations_one in {(time.time() - t_total) / 60:.2f} min.', flush=True)
        print('=' * 88, flush=True)
        return out_path
    finally:
        print('Cleaning cache_exp1_activations_one memory...', flush=True)
        try:
            del tokenizer, llm, processor, vlm
        except Exception:
            pass
        try:
            del train_ex, eval_ex
        except Exception:
            pass
        try:
            del llm_train, vlm_train, eval_modes
        except Exception:
            pass
        clear_mem_strong()
        gpu_report('after cleanup')

def compute_exp1_metrics_from_cache(family_key, dataset_key, pooling=MAIN_POOLING, seeds=None, width_factors=None, l1s=None, normalizes=None, audit=False):
    seeds = seeds or ROBUST_SEEDS
    width_factors = width_factors or ROBUST_WIDTH_FACTORS
    l1s = l1s or ROBUST_L1S
    normalizes = normalizes or ROBUST_NORMALIZE
    act_path = exp1_activation_path(family_key, dataset_key, pooling)
    if not act_path.exists():
        cache_exp1_activations_one(family_key, dataset_key, pooling)
    data = np.load(act_path, allow_pickle=True)
    required = {'depth_fractions', 'llm_layers', 'vlm_layers'}
    if not required.issubset(set(data.files)):
        raise RuntimeError(f'Activation cache {act_path} does not contain depth-alignment metadata. Delete it or set cfg.overwrite_existing=True and regenerate.')
    depth_fractions = data['depth_fractions'].astype(float)
    llm_layers = data['llm_layers'].astype(int)
    vlm_layers = data['vlm_layers'].astype(int)
    train_examples = int(data['train_examples'][0]) if 'train_examples' in data else cfg.exp1_train_examples
    eval_examples = int(data['eval_examples'][0]) if 'eval_examples' in data else cfg.exp1_eval_examples
    llm_train_all = data['llm_train']
    vlm_train_all = data['vlm_train']
    eval_all = {'matched': data['vlm_eval_matched'], 'text_swapped': data['vlm_eval_text_swapped'], 'image_swapped': data['vlm_eval_image_swapped'], 'image_only': data['vlm_eval_image_only']}
    rows = []
    for li, depth_fraction in enumerate(tqdm(depth_fractions, desc=f'SAE grid {family_key}-{dataset_key}-{pooling}')):
        Xllm_raw = llm_train_all[:, li, :]
        Xvlm_raw = vlm_train_all[:, li, :]
        raw_eval = {k: v[:, li, :] for k, v in eval_all.items()}
        for seed, wf, l1, do_norm in itertools.product(seeds, width_factors, l1s, normalizes):
            Xvlm, evals = normalize_by_train(Xvlm_raw, raw_eval, do_norm)
            Xllm, _ = normalize_by_train(Xllm_raw, {'x': Xllm_raw}, do_norm)
            d_sae = robust_d_sae(Xvlm.shape[1], wf)
            vlm_sae = train_sae_numpy(Xvlm, d_sae, cfg.exp1_sae_steps, cfg.exp1_sae_batch_size, cfg.exp1_sae_lr, l1, seed + 1000 * li)
            llm_sae = train_sae_numpy(Xllm, d_sae, cfg.exp1_sae_steps, cfg.exp1_sae_batch_size, cfg.exp1_sae_lr, l1, seed + 5000 + 1000 * li)
            z_m = sae_encode_numpy(vlm_sae, evals['matched'])
            z_t = sae_encode_numpy(vlm_sae, evals['text_swapped'])
            z_i = sae_encode_numpy(vlm_sae, evals['image_swapped'])
            z_o = sae_encode_numpy(vlm_sae, evals['image_only'])
            z_l = sae_encode_numpy(llm_sae, Xllm)
            text_delta_f = np.mean(np.abs(z_m - z_t), axis=0)
            image_delta_f = np.mean(np.abs(z_m - z_i), axis=0)
            text_delta_e = np.mean(np.abs(z_m - z_t), axis=1)
            image_delta_e = np.mean(np.abs(z_m - z_i), axis=1)
            mfi_e = text_delta_e + image_delta_e
            vlm_train_diag = recon_stats(vlm_sae, Xvlm)
            vlm_eval_diag = recon_stats(vlm_sae, evals['matched'])
            llm_train_diag = recon_stats(llm_sae, Xllm)
            mfi_lo, mfi_hi = boot_ci(mfi_e, seed + 9999 + li)
            for dom_margin in DOM_MARGINS:
                eps = 1e-08
                image_mask = image_delta_f > dom_margin * (text_delta_f + eps)
                text_mask = text_delta_f > dom_margin * (image_delta_f + eps)
                both_mask = ~image_mask & ~text_mask & ((image_delta_f > eps) | (text_delta_f > eps))
                rows.append({'family': family_key, 'dataset': dataset_key, 'pooling': pooling, 'audit': bool(audit), 'layer_idx': int(li), 'depth_fraction': float(depth_fraction), 'llm_layer_number': int(llm_layers[li]), 'vlm_layer_number': int(vlm_layers[li]), 'layer_number': int(vlm_layers[li]), 'seed': int(seed), 'width_factor': float(wf), 'd_sae': int(d_sae), 'l1': float(l1), 'normalize': bool(do_norm), 'dom_margin': float(dom_margin), 'train_examples': int(train_examples), 'eval_examples': int(eval_examples), 'sae_steps': int(cfg.exp1_sae_steps), 'MFI': float(mfi_e.mean()), 'MFI_ci_low': float(mfi_lo), 'MFI_ci_high': float(mfi_hi), 'FMP': float(image_mask.mean() + text_mask.mean() - both_mask.mean()), 'image_driven_frac': float(image_mask.mean()), 'text_driven_frac': float(text_mask.mean()), 'both_driven_frac': float(both_mask.mean()), 'mean_text_delta': float(text_delta_f.mean()), 'mean_image_delta': float(image_delta_f.mean()), 'matched_gini': float(gini(np.mean(np.maximum(z_m, 0.0), axis=0))), 'matched_eff_dim': float(participation_ratio(np.mean(np.maximum(z_m, 0.0), axis=0))), 'image_only_mean_activation': float(np.mean(np.maximum(z_o, 0.0))), 'llm_mean_activation': float(np.mean(np.maximum(z_l, 0.0))), 'vlm_train_rec_r2': float(vlm_train_diag['rec_r2']), 'vlm_eval_rec_r2': float(vlm_eval_diag['rec_r2']), 'llm_train_rec_r2': float(llm_train_diag['rec_r2']), 'vlm_eval_l0': float(vlm_eval_diag['l0']), 'vlm_eval_active_frac': float(vlm_eval_diag['active_frac']), 'vlm_eval_dead_frac': float(vlm_eval_diag['dead_frac']), 'vlm_eval_z_abs_mean': float(vlm_eval_diag['z_abs_mean'])})
            del vlm_sae, llm_sae, z_m, z_t, z_i, z_o, z_l
            clear_mem_strong()
    return pd.DataFrame(rows)

def run_exp1_one(family_key, dataset_key, pooling=MAIN_POOLING, audit=False):
    out_path = exp1_result_path(family_key, dataset_key, pooling, audit=audit)
    if out_path.exists() and (not cfg.overwrite_existing):
        print('skip Exp1:', out_path.name)
        return pd.read_csv(out_path)
    cache_exp1_activations_one(family_key, dataset_key, pooling)
    if audit:
        df = compute_exp1_metrics_from_cache(family_key, dataset_key, pooling, seeds=POOLING_AUDIT_SEEDS, width_factors=POOLING_AUDIT_WIDTH_FACTORS, l1s=POOLING_AUDIT_L1S, normalizes=POOLING_AUDIT_NORMALIZE, audit=True)
    else:
        df = compute_exp1_metrics_from_cache(family_key, dataset_key, pooling, audit=False)
    df.to_csv(out_path, index=False)
    print('saved Exp1:', out_path, df.shape)
    return df

def run_exp1_grid():
    failures, dfs = ([], [])
    for fam in FAMILIES_TO_RUN:
        for ds in DATASETS_TO_RUN:
            try:
                print(f'\nEXP1 MAIN | {fam} | {ds} | {MAIN_POOLING}')
                dfs.append(run_exp1_one(fam, ds, MAIN_POOLING, audit=False))
            except Exception as e:
                print('FAILED EXP1 MAIN:', fam, ds, repr(e))
                failures.append({'stage': 'exp1_main', 'family': fam, 'dataset': ds, 'pooling': MAIN_POOLING, 'error': repr(e)})
                clear_mem_strong()
    for pooling in POOLING_AUDIT_STRATEGIES:
        for fam in FAMILIES_TO_RUN:
            for ds in DATASETS_TO_RUN:
                try:
                    print(f'\nEXP1 POOLING AUDIT | {fam} | {ds} | {pooling}')
                    dfs.append(run_exp1_one(fam, ds, pooling, audit=True))
                except Exception as e:
                    print('FAILED EXP1 AUDIT:', fam, ds, pooling, repr(e))
                    failures.append({'stage': 'exp1_pooling_audit', 'family': fam, 'dataset': ds, 'pooling': pooling, 'error': repr(e)})
                    clear_mem_strong()
    fail_df = pd.DataFrame(failures)
    fail_df.to_csv(ROOT / 'logs' / 'exp1_failures.csv', index=False)
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    save_table(all_df, 'exp1_all_rows')
    return (all_df, fail_df)
CONCEPT_GROUPS = {'animal': ['cat', 'dog', 'horse', 'cow', 'sheep', 'bird', 'fish', 'elephant', 'bear', 'zebra', 'giraffe', 'duck', 'goose', 'monkey', 'lion', 'tiger', 'deer', 'rabbit', 'mouse', 'squirrel', 'goat', 'pig', 'chicken', 'snake', 'turtle'], 'vehicle': ['car', 'truck', 'bus', 'train', 'bicycle', 'motorcycle', 'boat', 'airplane', 'van', 'taxi', 'tram', 'subway', 'ship', 'scooter', 'skateboard', 'traffic light', 'stop sign', 'fire hydrant'], 'person': ['person', 'man', 'woman', 'child', 'boy', 'girl', 'baby', 'people', 'crowd', 'player', 'skier', 'surfer', 'rider', 'driver', 'worker', 'couple', 'family'], 'furniture_object': ['chair', 'table', 'bed', 'sofa', 'couch', 'bench', 'desk', 'shelf', 'cabinet', 'lamp', 'clock', 'mirror', 'door', 'window', 'sign', 'umbrella', 'backpack', 'suitcase', 'handbag', 'book', 'vase', 'pillow', 'blanket', 'box', 'bag'], 'electronics': ['phone', 'cell phone', 'laptop', 'computer', 'keyboard', 'television', 'tv', 'monitor', 'camera', 'remote', 'screen', 'tablet', 'headphones', 'microphone'], 'food_kitchen': ['apple', 'banana', 'orange', 'bread', 'coffee', 'tea', 'pizza', 'cake', 'sandwich', 'hot dog', 'donut', 'doughnut', 'broccoli', 'carrot', 'bowl', 'cup', 'bottle', 'glass', 'plate', 'fork', 'knife', 'spoon', 'pot', 'pan', 'wine glass', 'sink', 'refrigerator', 'oven', 'toaster'], 'sports': ['ball', 'baseball', 'tennis', 'frisbee', 'ski', 'snowboard', 'surfboard', 'kite', 'bat', 'glove', 'racket', 'helmet', 'field', 'court', 'game'], 'clothing': ['shirt', 'pants', 'jacket', 'hat', 'shoe', 'dress', 'tie', 'coat', 'shorts', 'uniform', 'jeans', 'cap', 'glasses'], 'scene_nature': ['street', 'road', 'sidewalk', 'building', 'house', 'room', 'kitchen', 'bathroom', 'beach', 'park', 'forest', 'mountain', 'river', 'lake', 'ocean', 'sky', 'tree', 'flower', 'grass', 'snow', 'water', 'cloud', 'bridge', 'fence', 'wall', 'city', 'yard', 'garden', 'hill', 'path', 'floor', 'ceiling']}
BASE_CONCEPTS = []
CONCEPT_SUPERCLASS = {}
for superclass, concepts in CONCEPT_GROUPS.items():
    for concept in concepts:
        if concept not in BASE_CONCEPTS:
            BASE_CONCEPTS.append(concept)
            CONCEPT_SUPERCLASS[concept] = superclass
CONCEPT_ALIASES = {'person': ['human', 'people'], 'people': ['persons', 'humans'], 'man': ['men'], 'woman': ['women'], 'child': ['children', 'kid', 'kids'], 'boy': ['boys'], 'girl': ['girls'], 'baby': ['babies', 'infant'], 'bicycle': ['bike', 'bikes'], 'motorcycle': ['motorbike', 'motorcycles'], 'airplane': ['plane', 'aircraft', 'jet'], 'television': ['tv', 'televisions'], 'cell phone': ['phone', 'mobile phone', 'smartphone'], 'phone': ['cell phone', 'mobile phone', 'smartphone'], 'sofa': ['couch'], 'couch': ['sofa'], 'hot dog': ['hotdog', 'hot dogs'], 'doughnut': ['donut', 'donuts'], 'donut': ['doughnut', 'doughnuts'], 'wine glass': ['wine glasses'], 'traffic light': ['traffic lights', 'stoplight'], 'stop sign': ['stop signs'], 'fire hydrant': ['fire hydrants', 'hydrant'], 'baseball': ['baseball bat', 'baseball glove'], 'racket': ['racquet', 'tennis racket'], 'tv': ['television']}

def _plural_forms(term):
    forms = {term}
    if ' ' in term:
        forms.add(term + 's')
        return forms
    if term.endswith('y') and len(term) > 2:
        forms.add(term[:-1] + 'ies')
    elif term.endswith(('s', 'x', 'z', 'ch', 'sh')):
        forms.add(term + 'es')
    else:
        forms.add(term + 's')
    return forms

def concept_forms(concept):
    forms = set()
    for x in [concept] + CONCEPT_ALIASES.get(concept, []):
        forms |= _plural_forms(x.lower())
    return sorted(forms, key=len, reverse=True)

def caption_has_concept(captions, concept):
    forms = concept_forms(concept)
    for cap in captions:
        for f in forms:
            if contains_whole_word(cap, f):
                return True
    return False

def build_concept_bank(examples, max_concepts, samples_per_concept, family_key=None, dataset_key=None):
    candidates = []
    for concept in BASE_CONCEPTS:
        matched = []
        for ex in examples:
            if caption_has_concept(ex['captions'], concept):
                matched.append(ex)
            if len(matched) >= samples_per_concept:
                break
        if len(matched) >= samples_per_concept:
            candidates.append({'concept': concept, 'superclass': CONCEPT_SUPERCLASS.get(concept, 'unknown'), 'matched': matched[:samples_per_concept], 'available': len(matched)})
    if not candidates:
        raise RuntimeError('No concepts found for concept bank.')
    suffix = f'{family_key}_{dataset_key}' if family_key and dataset_key else 'latest'
    cand_df = pd.DataFrame([{'concept': c['concept'], 'superclass': c['superclass'], 'available': c['available']} for c in candidates])
    cand_df.to_csv(ROOT / 'tables' / f'concept_candidate_coverage_{suffix}.csv', index=False)
    by_group = {}
    for c in candidates:
        by_group.setdefault(c['superclass'], []).append(c)
    for g in by_group:
        by_group[g] = sorted(by_group[g], key=lambda z: (-z['available'], z['concept']))
    selected = []
    group_names = sorted(by_group.keys())
    while len(selected) < max_concepts:
        added = False
        for g in group_names:
            if by_group[g] and len(selected) < max_concepts:
                selected.append(by_group[g].pop(0))
                added = True
        if not added:
            break
    min_concepts_required = min(90, max(60, int(0.7 * max_concepts)))
    if len(selected) < min_concepts_required:
        raise RuntimeError(f'Only {len(selected)} concepts found; target was {max_concepts}. Increase cfg.exp2_scan_examples or lower cfg.exp2_concepts.')
    selected_df = pd.DataFrame([{'concept': c['concept'], 'superclass': c['superclass'], 'available': c['available']} for c in selected])
    selected_df.to_csv(ROOT / 'tables' / f'concept_bank_selected_{suffix}.csv', index=False)
    print(f'selected concept bank for {suffix}: {len(selected)}/{max_concepts}')
    print(selected_df['superclass'].value_counts().sort_index())
    return {c['concept']: c['matched'] for c in selected}

def exp2_result_path(family_key, dataset_key):
    return ROOT / 'exp2' / f'exp2_main_{family_key}_{dataset_key}.csv'

def concept_prototypes_for_family_dataset(family_key, dataset_key):
    tokenizer, llm, processor, vlm = load_family_models(family_key)
    family_kind = FAMILIES[family_key]['kind']
    depth_fractions, llm_layers, vlm_layers = choose_matched_depth_layers(llm, vlm, cfg.exp2_max_layers)
    if len(depth_fractions) == 0:
        raise RuntimeError('No depth-aligned layers selected for Exp2.')
    layer_info = {'depth_fractions': np.asarray(depth_fractions, dtype=float), 'llm_layers': np.asarray(llm_layers, dtype=int), 'vlm_layers': np.asarray(vlm_layers, dtype=int)}
    pd.DataFrame({'depth_fraction': layer_info['depth_fractions'], 'llm_layer_number': layer_info['llm_layers'], 'vlm_layer_number': layer_info['vlm_layers']}).to_csv(ROOT / 'tables' / f'layer_alignment_exp2_{family_key}_{dataset_key}.csv', index=False)
    scan_n = max(int(cfg.exp2_scan_examples), int(cfg.exp2_concepts * cfg.exp2_samples_per_concept * 10), 1500)
    examples = load_examples(dataset_key, scan_n)
    bank = build_concept_bank(examples, cfg.exp2_concepts, cfg.exp2_samples_per_concept, family_key=family_key, dataset_key=dataset_key)
    pd.DataFrame([{'family': family_key, 'dataset': dataset_key, 'concept': c, 'superclass': CONCEPT_SUPERCLASS.get(c, 'unknown'), 'n_examples': len(exs)} for c, exs in bank.items()]).to_csv(ROOT / 'tables' / f'concept_bank_{family_key}_{dataset_key}.csv', index=False)
    llm_proto, vt_proto, vi_proto, vc_proto = ({}, {}, {}, {})
    for concept, exs in tqdm(bank.items(), desc=f'Concept prototypes {family_key}-{dataset_key}'):
        prompts = prompt_templates_for(concept, cfg.exp2_templates_per_concept)
        llm_arr = pooled_llm_hidden_states_exp(llm, tokenizer, prompts, [int(x) for x in llm_layers], cfg.max_text_length, cfg.exp2_batch_size_llm, pooling=MAIN_POOLING).mean(axis=0)
        vt_arr = pooled_vlm_hidden_states_exp(vlm, processor, exs, [int(x) for x in vlm_layers], family_kind, 'custom_text', cfg.exp2_batch_size_vlm, pooling=MAIN_POOLING, text_override=[exs[i % len(exs)]['primary_caption'] for i in range(len(exs))]).mean(axis=0)
        vi_arr = pooled_vlm_hidden_states_exp(vlm, processor, exs, [int(x) for x in vlm_layers], family_kind, 'image_only', cfg.exp2_batch_size_vlm, pooling=MAIN_POOLING).mean(axis=0)
        vc_arr = pooled_vlm_hidden_states_exp(vlm, processor, exs, [int(x) for x in vlm_layers], family_kind, 'custom_text', cfg.exp2_batch_size_vlm, pooling=MAIN_POOLING, text_override=[prompts[i % len(prompts)] for i in range(len(exs))]).mean(axis=0)
        llm_proto[concept] = llm_arr
        vt_proto[concept] = vt_arr
        vi_proto[concept] = vi_arr
        vc_proto[concept] = vc_arr
        clear_mem_strong()
    del tokenizer, llm, processor, vlm
    clear_mem_strong()
    return (layer_info, llm_proto, vt_proto, vi_proto, vc_proto)

def compute_exp2_metrics(llm_proto, vlm_text_proto, vlm_img_proto, vlm_cond_proto, layer_info):
    labels = list(llm_proto.keys())
    if len(labels) < 20:
        raise RuntimeError(f'Too few concepts: {len(labels)}')
    if isinstance(layer_info, dict):
        depth_fractions = np.asarray(layer_info['depth_fractions'], dtype=float)
        llm_layers = np.asarray(layer_info['llm_layers'], dtype=int)
        vlm_layers = np.asarray(layer_info['vlm_layers'], dtype=int)
    else:
        vlm_layers = np.asarray(layer_info, dtype=int)
        llm_layers = np.asarray(layer_info, dtype=int)
        depth_fractions = np.linspace(0.0, 1.0, len(vlm_layers), dtype=float)
    rows = []
    for li, depth_fraction in enumerate(depth_fractions):
        X = np.stack([llm_proto[c][li] for c in labels])
        Yt = np.stack([vlm_text_proto[c][li] for c in labels])
        Yi = np.stack([vlm_img_proto[c][li] for c in labels])
        Yc = np.stack([vlm_cond_proto[c][li] for c in labels])
        n_comp = min(cfg.exp2_pca_dim, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_comp, random_state=SEED)
        Xp = pca.fit_transform(X)
        Ytp = pca.transform(Yt)
        Yip = pca.transform(Yi)
        Ycp = pca.transform(Yc)
        R, _ = orthogonal_procrustes(Ytp, Xp)
        Yta = Ytp @ R
        Yia = Yip @ R
        Yca = Ycp @ R
        simX = cosine_similarity(l2_normalize_rows(Xp))
        simYt = cosine_similarity(l2_normalize_rows(Yta))
        simYi = cosine_similarity(l2_normalize_rows(Yia))
        simYc = cosine_similarity(l2_normalize_rows(Yca))
        for k in cfg.exp2_topk_list:
            if k >= len(labels):
                continue
            jac_t, jac_i, jac_c = ([], [], [])
            rbo_t, rbo_i, rbo_c = ([], [], [])
            gain_c, shared_c, cross_gain_c = ([], [], [])
            for idx, c in enumerate(labels):
                nx = topk_neighbors_from_sim(simX, labels, idx, k)
                nt = topk_neighbors_from_sim(simYt, labels, idx, k)
                ni = topk_neighbors_from_sim(simYi, labels, idx, k)
                nc = topk_neighbors_from_sim(simYc, labels, idx, k)
                sx, st, si, sc = (set(nx), set(nt), set(ni), set(nc))
                jac_t.append(len(sx & st) / max(1, len(sx | st)))
                jac_i.append(len(sx & si) / max(1, len(sx | si)))
                jac_c.append(len(sx & sc) / max(1, len(sx | sc)))
                rbo_t.append(rank_biased_overlap(nx, nt))
                rbo_i.append(rank_biased_overlap(nx, ni))
                rbo_c.append(rank_biased_overlap(nx, nc))
                gained = list(sc - sx)
                gain_c.append(len(gained) / k)
                shared_c.append(len(sx & sc) / k)
                if gained:
                    base_super = CONCEPT_SUPERCLASS.get(c, 'unknown')
                    cross = [g for g in gained if CONCEPT_SUPERCLASS.get(g, 'unknown') != base_super]
                    cross_gain_c.append(len(cross) / len(gained))
            rows.append({'layer_idx': int(li), 'depth_fraction': float(depth_fraction), 'llm_layer_number': int(llm_layers[li]), 'vlm_layer_number': int(vlm_layers[li]), 'layer_number': int(vlm_layers[li]), 'K': int(k), 'n_concepts': int(len(labels)), 'n_pca_components': int(n_comp), 'LPS_text': float((np.mean(jac_t) + np.mean(rbo_t)) / 2), 'LPS_image_only': float((np.mean(jac_i) + np.mean(rbo_i)) / 2), 'LPS_image_conditioned': float((np.mean(jac_c) + np.mean(rbo_c)) / 2), 'NRI_image_conditioned': float(np.mean(gain_c)), 'shared_neighbor_ratio': float(np.mean(shared_c)), 'cross_superclass_gained_frac': float(np.mean(cross_gain_c)) if cross_gain_c else np.nan, 'random_shared_baseline': float(k / max(1, len(labels) - 1)), 'align_fit_proxy': float(1 - np.linalg.norm(Xp - Yta) / (np.linalg.norm(Xp) + 1e-12))})
    return pd.DataFrame(rows)

def run_exp2_one(family_key, dataset_key):
    out_path = exp2_result_path(family_key, dataset_key)
    if out_path.exists() and (not cfg.overwrite_existing):
        print('skip Exp2:', out_path.name)
        return pd.read_csv(out_path)
    print(f'\nEXP2 | {family_key} | {dataset_key}')
    layer_info, llm_proto, vt_proto, vi_proto, vc_proto = concept_prototypes_for_family_dataset(family_key, dataset_key)
    print(f'concepts used for {family_key}-{dataset_key}: {len(llm_proto)}')
    print('depth fractions:', [round(float(x), 4) for x in layer_info['depth_fractions']])
    print('LLM layers:', [int(x) for x in layer_info['llm_layers']])
    print('VLM layers:', [int(x) for x in layer_info['vlm_layers']])
    df = compute_exp2_metrics(llm_proto, vt_proto, vi_proto, vc_proto, layer_info)
    df.insert(0, 'dataset', dataset_key)
    df.insert(0, 'family', family_key)
    df.to_csv(out_path, index=False)
    print('saved Exp2:', out_path, df.shape)
    del llm_proto, vt_proto, vi_proto, vc_proto
    clear_mem_strong()
    return df

def run_exp2_grid():
    failures, dfs = ([], [])
    for fam in FAMILIES_TO_RUN:
        for ds in DATASETS_TO_RUN:
            try:
                dfs.append(run_exp2_one(fam, ds))
            except Exception as e:
                print('FAILED EXP2:', fam, ds, repr(e))
                failures.append({'family': fam, 'dataset': ds, 'error': repr(e)})
                clear_mem_strong()
    fail_df = pd.DataFrame(failures)
    fail_df.to_csv(ROOT / 'logs' / 'exp2_failures.csv', index=False)
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    save_table(all_df, 'exp2_all_rows')
    return (all_df, fail_df)

def pretty_family(x):
    return {'qwen': 'Qwen2.5-VL', 'smol': 'SmolVLM2', 'pali': 'PaliGemma2'}.get(x, x)

def pretty_dataset(x):
    return {'pixelprose': 'PixelProse', 'flickr30k': 'Flickr30k', 'coco_karpathy': 'COCO'}.get(x, x)

def export_all_paper_outputs():
    exp1_df = collect_csvs(ROOT / 'exp1')
    exp2_df = collect_csvs(ROOT / 'exp2')
    if len(exp1_df) == 0:
        raise RuntimeError('No Exp1 rows found.')
    if len(exp2_df) == 0:
        raise RuntimeError('No Exp2 rows found.')
    exp1_main = exp1_df[(exp1_df['pooling'] == MAIN_POOLING) & (exp1_df['audit'] == False) & np.isclose(exp1_df['dom_margin'].astype(float), 1.25)].copy()
    exp1_all_margin = exp1_df[(exp1_df['pooling'] == MAIN_POOLING) & (exp1_df['audit'] == False)].copy()
    exp1_audit = exp1_df[(exp1_df['audit'] == True) & np.isclose(exp1_df['dom_margin'].astype(float), 1.25)].copy()
    table1_family = exp1_main.groupby('family').agg(summary_type=('family', lambda x: 'family'), group=('family', 'first'), MFI_mean=('MFI', 'mean'), MFI_low=('MFI', lambda x: float(np.quantile(x, 0.025))), MFI_high=('MFI', lambda x: float(np.quantile(x, 0.975))), FMP_mean=('FMP', 'mean'), L0_mean=('vlm_eval_l0', 'mean'), rec_R2_mean=('vlm_eval_rec_r2', 'mean'), dead_frac_mean=('vlm_eval_dead_frac', 'mean')).reset_index(drop=True)
    table1_dataset = exp1_main.groupby('dataset').agg(summary_type=('dataset', lambda x: 'dataset'), group=('dataset', 'first'), MFI_mean=('MFI', 'mean'), MFI_low=('MFI', lambda x: float(np.quantile(x, 0.025))), MFI_high=('MFI', lambda x: float(np.quantile(x, 0.975))), FMP_mean=('FMP', 'mean'), L0_mean=('vlm_eval_l0', 'mean'), rec_R2_mean=('vlm_eval_rec_r2', 'mean'), dead_frac_mean=('vlm_eval_dead_frac', 'mean')).reset_index(drop=True)
    table1 = pd.concat([table1_family, table1_dataset], ignore_index=True)
    save_table(table1, 'Table_1_family_dataset_summary')
    appA_sparse = exp1_main.groupby(['family', 'dataset']).agg(MFI_mean=('MFI', 'mean'), MFI_std=('MFI', 'std'), MFI_low=('MFI', lambda x: float(np.quantile(x, 0.025))), MFI_high=('MFI', lambda x: float(np.quantile(x, 0.975))), FMP_mean=('FMP', 'mean'), text_delta_mean=('mean_text_delta', 'mean'), image_delta_mean=('mean_image_delta', 'mean'), rec_R2_mean=('vlm_eval_rec_r2', 'mean'), L0_mean=('vlm_eval_l0', 'mean'), active_frac_mean=('vlm_eval_active_frac', 'mean'), dead_frac_mean=('vlm_eval_dead_frac', 'mean'), train_examples=('train_examples', 'max'), eval_examples=('eval_examples', 'max'), sae_steps=('sae_steps', 'max')).reset_index()
    save_table(appA_sparse, 'Appendix_A_Table_A1_full_sparse_3x3_grid')
    appA_threshold = exp1_all_margin.groupby(['family', 'dataset', 'dom_margin']).agg(MFI_mean=('MFI', 'mean'), FMP_mean=('FMP', 'mean'), image_driven_frac=('image_driven_frac', 'mean'), text_driven_frac=('text_driven_frac', 'mean'), both_driven_frac=('both_driven_frac', 'mean')).reset_index()
    save_table(appA_threshold, 'Appendix_A_Table_A2_threshold_sensitivity')
    appA_sae_quality = exp1_main.groupby(['family', 'dataset', 'width_factor', 'l1', 'normalize']).agg(rec_R2_mean=('vlm_eval_rec_r2', 'mean'), rec_R2_min=('vlm_eval_rec_r2', 'min'), L0_mean=('vlm_eval_l0', 'mean'), active_frac_mean=('vlm_eval_active_frac', 'mean'), dead_frac_mean=('vlm_eval_dead_frac', 'mean')).reset_index()
    save_table(appA_sae_quality, 'Appendix_A_Table_A3_SAE_quality_by_hparam')
    if len(exp1_audit):
        appA_pooling = pd.concat([exp1_main.assign(audit=False), exp1_audit], ignore_index=True).groupby(['family', 'dataset', 'pooling']).agg(MFI_mean=('MFI', 'mean'), FMP_mean=('FMP', 'mean'), rec_R2_mean=('vlm_eval_rec_r2', 'mean'), L0_mean=('vlm_eval_l0', 'mean'), dead_frac_mean=('vlm_eval_dead_frac', 'mean')).reset_index()
        save_table(appA_pooling, 'Appendix_A_Table_A4_pooling_audit')
    exp2_k5 = exp2_df[exp2_df['K'] == 5].copy()
    table2 = exp2_k5.groupby(['family', 'dataset']).agg(n_concepts=('n_concepts', 'max'), LPS_text=('LPS_text', 'mean'), LPS_image_only=('LPS_image_only', 'mean'), LPS_image_conditioned=('LPS_image_conditioned', 'mean'), NRI_image_conditioned=('NRI_image_conditioned', 'mean'), shared_neighbor_ratio=('shared_neighbor_ratio', 'mean'), cross_superclass_gained_frac=('cross_superclass_gained_frac', 'mean'), random_shared_baseline=('random_shared_baseline', 'mean')).reset_index()
    save_table(table2, 'Table_2_neighborhood_K5_grid')
    appB_full = exp2_df.groupby(['family', 'dataset', 'K']).agg(n_concepts=('n_concepts', 'max'), LPS_text=('LPS_text', 'mean'), LPS_image_only=('LPS_image_only', 'mean'), LPS_image_conditioned=('LPS_image_conditioned', 'mean'), NRI_image_conditioned=('NRI_image_conditioned', 'mean'), shared_neighbor_ratio=('shared_neighbor_ratio', 'mean'), cross_superclass_gained_frac=('cross_superclass_gained_frac', 'mean'), random_shared_baseline=('random_shared_baseline', 'mean'), align_fit_proxy=('align_fit_proxy', 'mean')).reset_index()
    save_table(appB_full, 'Appendix_B_Table_B1_full_neighborhood_3x3_grid')
    concept_coverage = exp2_df.groupby(['family', 'dataset']).agg(n_concepts=('n_concepts', 'max'), min_concepts=('n_concepts', 'min'), max_concepts=('n_concepts', 'max')).reset_index()
    save_table(concept_coverage, 'Appendix_B_Table_B2_concept_coverage')
    draw_cols = ['seed', 'width_factor', 'l1', 'normalize']
    fam_draw = exp1_main.groupby(draw_cols + ['family'])['MFI'].mean().reset_index()
    ds_draw = exp1_main.groupby(draw_cols + ['dataset'])['MFI'].mean().reset_index()
    family_rank_rows, dataset_rank_rows = ([], [])
    for draw, g in fam_draw.groupby(draw_cols):
        family_rank_rows.append({'draw': str(draw), 'max_family': g.sort_values('MFI', ascending=False).iloc[0]['family'], 'min_family': g.sort_values('MFI', ascending=True).iloc[0]['family']})
    for draw, g in ds_draw.groupby(draw_cols):
        dataset_rank_rows.append({'draw': str(draw), 'max_dataset': g.sort_values('MFI', ascending=False).iloc[0]['dataset'], 'min_dataset': g.sort_values('MFI', ascending=True).iloc[0]['dataset']})
    family_rank = pd.DataFrame(family_rank_rows)
    dataset_rank = pd.DataFrame(dataset_rank_rows)
    save_table(family_rank, 'Appendix_A_Table_A5_family_rank_stability')
    save_table(dataset_rank, 'Appendix_A_Table_A6_dataset_rank_stability')
    rank_stability = {'qwen_highest_family_fraction': float((family_rank['max_family'] == 'qwen').mean()) if len(family_rank) else np.nan, 'smol_lowest_family_fraction': float((family_rank['min_family'] == 'smol').mean()) if len(family_rank) else np.nan, 'pixelprose_highest_dataset_fraction': float((dataset_rank['max_dataset'] == 'pixelprose').mean()) if len(dataset_rank) else np.nan}
    with open(ROOT / 'tables' / 'rank_stability.json', 'w') as f:
        json.dump(rank_stability, f, indent=2)
    exp1_plot = exp1_main.copy()
    if 'depth_fraction' not in exp1_plot.columns:
        exp1_plot['depth_fraction'] = exp1_plot['layer_idx'].astype(float)
    layer_family = exp1_plot.groupby(['family', 'depth_fraction']).agg(MFI=('MFI', 'mean'), FMP=('FMP', 'mean'), rec_R2=('vlm_eval_rec_r2', 'mean')).reset_index()
    heat = appA_sparse.pivot(index='family', columns='dataset', values='MFI_mean')
    heat = heat.loc[[x for x in ['pali', 'qwen', 'smol'] if x in heat.index], [x for x in ['coco_karpathy', 'flickr30k', 'pixelprose'] if x in heat.columns]]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    ax = axes[0, 0]
    for fam, g in layer_family.groupby('family'):
        g = g.sort_values('depth_fraction')
        ax.plot(g['depth_fraction'], g['MFI'], marker='o', label=pretty_family(fam))
    ax.set_title('A. Sparse mismatch sensitivity')
    ax.set_xlabel('Depth fraction')
    ax.set_ylabel('MFI')
    ax.legend(frameon=False)
    ax = axes[0, 1]
    for fam, g in layer_family.groupby('family'):
        g = g.sort_values('depth_fraction')
        ax.plot(g['depth_fraction'], g['FMP'], marker='o', label=pretty_family(fam))
    ax.set_title('B. Feature-modality partition')
    ax.set_xlabel('Depth fraction')
    ax.set_ylabel('FMP')
    ax = axes[1, 0]
    im = ax.imshow(heat.values, aspect='auto')
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels([pretty_dataset(c) for c in heat.columns], rotation=35, ha='right')
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels([pretty_family(r) for r in heat.index])
    ax.set_title('C. 3×3 MFI grid')
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f'{heat.values[i, j]:.3f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[1, 1]
    for fam, g in layer_family.groupby('family'):
        g = g.sort_values('depth_fraction')
        ax.plot(g['depth_fraction'], g['rec_R2'], marker='o', label=pretty_family(fam))
    ax.set_title('D. SAE reconstruction validation')
    ax.set_xlabel('Depth fraction')
    ax.set_ylabel('Eval reconstruction $R^2$')
    fig.tight_layout()
    save_fig(fig, 'Figure_1_sparse_perturbation')
    plt.close(fig)
    exp2_plot = exp2_k5.copy()
    if 'depth_fraction' not in exp2_plot.columns:
        exp2_plot['depth_fraction'] = exp2_plot['layer_idx'].astype(float)
    exp2_layer = exp2_plot.groupby(['family', 'depth_fraction']).agg(shared_neighbor_ratio=('shared_neighbor_ratio', 'mean'), NRI_image_conditioned=('NRI_image_conditioned', 'mean'), cross_superclass_gained_frac=('cross_superclass_gained_frac', 'mean')).reset_index()
    n_heat = table2.pivot(index='family', columns='dataset', values='NRI_image_conditioned')
    n_heat = n_heat.loc[[x for x in ['pali', 'qwen', 'smol'] if x in n_heat.index], [x for x in ['coco_karpathy', 'flickr30k', 'pixelprose'] if x in n_heat.columns]]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    ax = axes[0, 0]
    for fam, g in exp2_layer.groupby('family'):
        g = g.sort_values('depth_fraction')
        ax.plot(g['depth_fraction'], g['shared_neighbor_ratio'], marker='o', label=pretty_family(fam))
    ax.set_title('A. Shared-neighbor preservation')
    ax.set_xlabel('Depth fraction')
    ax.set_ylabel('Shared-neighbor ratio')
    ax.legend(frameon=False)
    ax = axes[0, 1]
    for fam, g in exp2_layer.groupby('family'):
        g = g.sort_values('depth_fraction')
        ax.plot(g['depth_fraction'], g['NRI_image_conditioned'], marker='o', label=pretty_family(fam))
    ax.set_title('B. Neighborhood rewriting')
    ax.set_xlabel('Depth fraction')
    ax.set_ylabel('NRI')
    ax = axes[1, 0]
    for fam, g in exp2_layer.groupby('family'):
        g = g.sort_values('depth_fraction')
        ax.plot(g['depth_fraction'], g['cross_superclass_gained_frac'], marker='o', label=pretty_family(fam))
    ax.set_title('C. Cross-superclass gained neighbors')
    ax.set_xlabel('Depth fraction')
    ax.set_ylabel('Cross-superclass fraction')
    ax = axes[1, 1]
    im = ax.imshow(n_heat.values, aspect='auto')
    ax.set_xticks(np.arange(n_heat.shape[1]))
    ax.set_xticklabels([pretty_dataset(c) for c in n_heat.columns], rotation=35, ha='right')
    ax.set_yticks(np.arange(n_heat.shape[0]))
    ax.set_yticklabels([pretty_family(r) for r in n_heat.index])
    ax.set_title('D. 3×3 NRI grid')
    for i in range(n_heat.shape[0]):
        for j in range(n_heat.shape[1]):
            ax.text(j, i, f'{n_heat.values[i, j]:.3f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_fig(fig, 'Figure_2_neighborhood_preservation_rewriting')
    plt.close(fig)
    q = appA_sparse.copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    ax = axes[0]
    ax.bar(np.arange(len(q)), q['rec_R2_mean'])
    ax.set_xticks(np.arange(len(q)))
    ax.set_xticklabels([f'{r.family}-{r.dataset}' for r in q.itertuples()], rotation=75, ha='right', fontsize=6)
    ax.set_title('A. Reconstruction $R^2$')
    ax.set_ylabel('$R^2$')
    ax = axes[1]
    ax.bar(np.arange(len(q)), q['L0_mean'])
    ax.set_xticks(np.arange(len(q)))
    ax.set_xticklabels([f'{r.family}-{r.dataset}' for r in q.itertuples()], rotation=75, ha='right', fontsize=6)
    ax.set_title('B. Active latents')
    ax.set_ylabel('L0')
    ax = axes[2]
    ax.bar(np.arange(len(q)), q['dead_frac_mean'])
    ax.set_xticks(np.arange(len(q)))
    ax.set_xticklabels([f'{r.family}-{r.dataset}' for r in q.itertuples()], rotation=75, ha='right', fontsize=6)
    ax.set_title('C. Dead latent fraction')
    ax.set_ylabel('Fraction')
    fig.tight_layout()
    save_fig(fig, 'Appendix_A_Figure_SAE_quality_diagnostics')
    plt.close(fig)
    print('All required outputs saved under:', ROOT)
    return {'Table_1': table1, 'Table_2': table2, 'Appendix_A_sparse': appA_sparse, 'Appendix_B_full': appB_full, 'concept_coverage': concept_coverage, 'rank_stability': rank_stability}
exp1_df, exp1_failures = run_exp1_grid()
exp2_df, exp2_failures = run_exp2_grid()
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _as_path(x):
    try:
        return Path(x)
    except Exception:
        return None
root_candidates = []
for name in ['ROOT', 'OUT_DIR', 'RESULTS_DIR']:
    if name in globals():
        p = _as_path(globals()[name])
        if p is not None:
            root_candidates.append(p)
root_candidates += [Path('results_vlm_feature_geometry_h100'), Path('./results_vlm_feature_geometry_h100')]
ROOT = None
for p in root_candidates:
    if p.exists() and ((p / 'tables').exists() or (p / 'exp1').exists() or (p / 'exp2').exists()):
        ROOT = p
        break
if ROOT is None:
    ROOT = Path('results_vlm_feature_geometry_h100')
EXP1_DIR = ROOT / 'exp1'
EXP2_DIR = ROOT / 'exp2'
TABLE_DIR = ROOT / 'tables'
FIG_DIR = ROOT / 'figures'
OUT_TABLE = TABLE_DIR / 'final_validated_tables'
OUT_FIG = FIG_DIR / 'final_validated_figures'
OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
print('ROOT:', ROOT.resolve())
print('OUT_TABLE:', OUT_TABLE.resolve())
print('OUT_FIG:', OUT_FIG.resolve())
plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 300, 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
FAMILY_ORDER = ['Pali', 'Qwen', 'Smol']
DATASET_ORDER = ['COCO', 'Flickr30k', 'PixelProse']

def _norm(s):
    return re.sub('[^a-z0-9]+', '_', str(s).strip().lower()).strip('_')

def clean_columns(df):
    out = df.copy()
    out.columns = [_norm(c) for c in out.columns]
    return out

def first_col(df, aliases, required=False):
    if df is None or len(df) == 0:
        if required:
            raise KeyError(f'Empty dataframe while looking for {aliases}')
        return None
    cols = set(df.columns)
    for a in aliases:
        aa = _norm(a)
        if aa in cols:
            return aa
    if required:
        raise KeyError(f'Missing {aliases}. Available columns: {list(df.columns)}')
    return None

def to_num(x):
    return pd.to_numeric(x, errors='coerce')

def canonical_family(x):
    s = str(x).lower()
    if 'qwen' in s:
        return 'Qwen'
    if 'pali' in s or 'gemma' in s:
        return 'Pali'
    if 'smol' in s:
        return 'Smol'
    return str(x)

def canonical_dataset(x):
    s = str(x).lower()
    if 'coco' in s:
        return 'COCO'
    if 'flickr' in s:
        return 'Flickr30k'
    if 'pixel' in s:
        return 'PixelProse'
    return str(x)

def canonicalize(df):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = clean_columns(df)
    fcol = first_col(out, ['family', 'model', 'model_family'], required=False)
    dcol = first_col(out, ['dataset', 'dataset_key', 'data'], required=False)
    if fcol is not None:
        out['family'] = out[fcol].map(canonical_family)
    if dcol is not None:
        out['dataset'] = out[dcol].map(canonical_dataset)
    return out

def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['source_file'] = path.name
    return canonicalize(df)

def concat_csvs(folder):
    folder = Path(folder)
    if not folder.exists():
        return pd.DataFrame()
    dfs = []
    for p in sorted(folder.glob('*.csv')):
        try:
            df = pd.read_csv(p)
            if len(df):
                df['source_file'] = p.name
                dfs.append(df)
        except Exception as e:
            print('Could not read:', p, repr(e))
    if not dfs:
        return pd.DataFrame()
    return canonicalize(pd.concat(dfs, ignore_index=True, sort=False))

def load_exp1():
    p = TABLE_DIR / 'exp1_all_rows.csv'
    if p.exists():
        print('Loading Exp1:', p)
        return read_csv(p)
    print('Loading Exp1 from per-cell CSVs:', EXP1_DIR)
    return concat_csvs(EXP1_DIR)

def load_exp2():
    p = TABLE_DIR / 'exp2_all_rows.csv'
    if p.exists():
        print('Loading Exp2:', p)
        return read_csv(p)
    print('Loading Exp2 from per-cell CSVs:', EXP2_DIR)
    return concat_csvs(EXP2_DIR)

def order_fd(df):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    sort_cols = []
    if 'family' in out.columns:
        out['family'] = pd.Categorical(out['family'], FAMILY_ORDER, ordered=True)
        sort_cols.append('family')
    if 'dataset' in out.columns:
        out['dataset'] = pd.Categorical(out['dataset'], DATASET_ORDER, ordered=True)
        sort_cols.append('dataset')
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    for c in ['family', 'dataset']:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out

def interval_summary(values):
    x = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy()
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, 0)
    mean = float(np.mean(x))
    if len(x) == 1:
        return (mean, np.nan, np.nan, 1)
    lo = float(np.quantile(x, 0.025))
    hi = float(np.quantile(x, 0.975))
    return (mean, lo, hi, int(len(x)))

def grouped_summary(df, group_cols, metric_cols):
    metric_cols = [m for m in metric_cols if m is not None and m in df.columns]
    if df is None or len(df) == 0 or (not metric_cols):
        return pd.DataFrame()
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {group_cols[i]: key[i] for i in range(len(group_cols))}
        for m in metric_cols:
            mean, lo, hi, n = interval_summary(g[m])
            row[m] = mean
            row[f'{m}_lo'] = lo
            row[f'{m}_hi'] = hi
            row[f'{m}_n'] = n
        rows.append(row)
    return order_fd(pd.DataFrame(rows))

def save_table(df, stem, digits=3, latex=True):
    if df is None or len(df) == 0:
        print('[skip empty table]', stem)
        return pd.DataFrame()
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(digits)
    csv_path = OUT_TABLE / f'{stem}.csv'
    out.to_csv(csv_path, index=False)
    print('[saved]', csv_path)
    return out

def save_interval_table(df, group_cols, metric_cols, stem):
    if df is None or len(df) == 0:
        print('[skip empty interval table]', stem)
        return pd.DataFrame()
    out = df[group_cols].copy()
    for m in metric_cols:
        if m not in df.columns:
            continue
        vals = []
        for _, r in df.iterrows():
            mean = r.get(m, np.nan)
            lo = r.get(f'{m}_lo', np.nan)
            hi = r.get(f'{m}_hi', np.nan)
            if pd.isna(mean):
                vals.append('')
            elif pd.isna(lo) or pd.isna(hi):
                vals.append(f'{mean:.3f}')
            else:
                vals.append(f'{mean:.3f} [{lo:.3f}, {hi:.3f}]')
        out[m] = vals
    return save_table(out, stem, digits=3)

def save_fig(fig, stem, main_alias=None):
    pdf = OUT_FIG / f'{stem}.pdf'
    svg = OUT_FIG / f'{stem}.svg'
    fig.savefig(pdf, bbox_inches='tight', dpi=300)
    fig.savefig(svg, bbox_inches='tight')
    print('[saved]', pdf)
    print('[saved]', svg)
    if main_alias is not None:
        fig.savefig(FIG_DIR / f'{main_alias}.pdf', bbox_inches='tight', dpi=300)
        fig.savefig(FIG_DIR / f'{main_alias}.svg', bbox_inches='tight')
        print('[saved main replacement]', FIG_DIR / f'{main_alias}.pdf')

def adaptive_text_color(value, vmin, vmax, cmap_name):
    if not np.isfinite(value) or vmax <= vmin:
        return 'black'
    normed = (value - vmin) / (vmax - vmin)
    rgba = plt.get_cmap(cmap_name)(normed)
    r, g, b = rgba[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'black' if luminance > 0.55 else 'white'

def plot_heatmap(ax, df, metric, title, fmt='{:.3f}', cmap='viridis', fixed_vmin=None, fixed_vmax=None):
    if df is None or len(df) == 0 or metric is None or (metric not in df.columns):
        ax.axis('off')
        ax.text(0.5, 0.5, 'missing data', ha='center', va='center')
        ax.set_title(title)
        return
    mat = df.pivot_table(index='family', columns='dataset', values=metric, aggfunc='mean', observed=False)
    rows = [r for r in FAMILY_ORDER if r in mat.index] + [r for r in mat.index if r not in FAMILY_ORDER]
    cols = [c for c in DATASET_ORDER if c in mat.columns] + [c for c in mat.columns if c not in DATASET_ORDER]
    mat = mat.loc[rows, cols]
    arr = mat.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        ax.axis('off')
        ax.text(0.5, 0.5, 'no finite values', ha='center', va='center')
        ax.set_title(title)
        return
    vmin = np.nanmin(arr) if fixed_vmin is None else fixed_vmin
    vmax = np.nanmax(arr) if fixed_vmax is None else fixed_vmax
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1.0)
        vmin = vmin - pad
        vmax = vmax + pad
    im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, fmt.format(arr[i, j]), ha='center', va='center', fontsize=8, color=adaptive_text_color(arr[i, j], vmin, vmax, cmap))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_trajectory(ax, df, xcol, ycol, title, xlabel, ylabel, group='family'):
    if df is None or len(df) == 0 or xcol is None or (ycol is None) or (xcol not in df.columns) or (ycol not in df.columns):
        ax.axis('off')
        ax.text(0.5, 0.5, 'missing trajectory', ha='center', va='center')
        ax.set_title(title)
        return
    tmp = df.copy()
    tmp[xcol] = to_num(tmp[xcol])
    tmp[ycol] = to_num(tmp[ycol])
    tmp = tmp.dropna(subset=[xcol, ycol])
    if len(tmp) == 0:
        ax.axis('off')
        ax.text(0.5, 0.5, 'no finite data', ha='center', va='center')
        ax.set_title(title)
        return
    groups = tmp.groupby(group, dropna=False) if group in tmp.columns else [('all', tmp)]
    for name, g in groups:
        s = grouped_summary(g, [xcol], [ycol]).sort_values(xcol)
        if len(s) == 0:
            continue
        x = to_num(s[xcol]).to_numpy()
        y = to_num(s[ycol]).to_numpy()
        ax.plot(x, y, marker='o', linewidth=1.5, label=str(name))
        lo = to_num(s.get(f'{ycol}_lo', pd.Series([np.nan] * len(s)))).to_numpy()
        hi = to_num(s.get(f'{ycol}_hi', pd.Series([np.nan] * len(s)))).to_numpy()
        if np.isfinite(lo).any() and np.isfinite(hi).any():
            ax.fill_between(x, lo, hi, alpha=0.14)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

def plot_scatter(ax, df, xcol, ycol, title, xlabel, ylabel):
    if df is None or len(df) == 0 or xcol is None or (ycol is None) or (xcol not in df.columns) or (ycol not in df.columns):
        ax.axis('off')
        ax.text(0.5, 0.5, 'missing scatter', ha='center', va='center')
        ax.set_title(title)
        return
    tmp = df.copy()
    tmp[xcol] = to_num(tmp[xcol])
    tmp[ycol] = to_num(tmp[ycol])
    tmp = tmp.dropna(subset=[xcol, ycol])
    if len(tmp) == 0:
        ax.axis('off')
        ax.text(0.5, 0.5, 'no finite data', ha='center', va='center')
        ax.set_title(title)
        return
    if 'family' in tmp.columns:
        for fam, g in tmp.groupby('family', dropna=False):
            ax.scatter(g[xcol], g[ycol], alpha=0.72, label=str(fam), s=24)
        ax.legend(frameon=False)
    else:
        ax.scatter(tmp[xcol], tmp[ycol], alpha=0.72, s=24)
    if len(tmp) >= 3:
        try:
            r = np.corrcoef(tmp[xcol].to_numpy(), tmp[ycol].to_numpy())[0, 1]
            ax.text(0.04, 0.96, f'r = {r:.2f}', transform=ax.transAxes, va='top')
            coef = np.polyfit(tmp[xcol].to_numpy(), tmp[ycol].to_numpy(), 1)
            xs = np.linspace(tmp[xcol].min(), tmp[xcol].max(), 100)
            ax.plot(xs, coef[0] * xs + coef[1], linewidth=1.0)
        except Exception:
            pass
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def filtered_corr(df, metrics, min_unique=2):
    metrics = [m for m in metrics if m is not None and m in df.columns]
    if df is None or len(df) == 0 or len(metrics) < 2:
        return pd.DataFrame()
    tmp = df[metrics].apply(pd.to_numeric, errors='coerce')
    keep = []
    for c in tmp.columns:
        s = tmp[c].dropna()
        if len(s) >= 3 and s.nunique() >= min_unique and (float(s.std(ddof=0)) > 0):
            keep.append(c)
    if len(keep) < 2:
        return pd.DataFrame()
    return tmp[keep].corr()

def plot_corr(ax, df, metrics, title, cmap='coolwarm'):
    C = filtered_corr(df, metrics)
    if C is None or len(C) == 0 or C.shape[0] < 2:
        ax.axis('off')
        ax.text(0.5, 0.5, 'insufficient nonconstant metrics', ha='center', va='center')
        ax.set_title(title)
        return
    arr = C.to_numpy(dtype=float)
    im = ax.imshow(arr, vmin=-1, vmax=1, aspect='auto', cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(C.columns)))
    ax.set_xticklabels(C.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(C.index)))
    ax.set_yticklabels(C.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f'{arr[i, j]:.2f}', ha='center', va='center', fontsize=7, color=adaptive_text_color(arr[i, j], -1, 1, cmap))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_category(ax, df, cat, metric, title, xlabel, ylabel):
    if df is None or len(df) == 0 or cat is None or (metric is None) or (cat not in df.columns) or (metric not in df.columns):
        ax.axis('off')
        ax.text(0.5, 0.5, 'missing category data', ha='center', va='center')
        ax.set_title(title)
        return
    tmp = df.copy()
    tmp[metric] = to_num(tmp[metric])
    s = tmp.groupby(cat, dropna=False)[metric].mean().reset_index()
    ax.bar(s[cat].astype(str), s[metric])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=25)
exp1 = load_exp1()
exp2 = load_exp2()
if len(exp1) == 0:
    raise RuntimeError('No Exp1 rows found.')
if len(exp2) == 0:
    raise RuntimeError('No Exp2 rows found.')
print('Exp1:', exp1.shape)
print('Exp2:', exp2.shape)
mfi = first_col(exp1, ['mfi'], required=True)
fmp = first_col(exp1, ['fmp'], required=True)
mean_text_delta = first_col(exp1, ['mean_text_delta'], required=False)
mean_image_delta = first_col(exp1, ['mean_image_delta'], required=False)
image_driven_frac = first_col(exp1, ['image_driven_frac'], required=False)
text_driven_frac = first_col(exp1, ['text_driven_frac'], required=False)
both_driven_frac = first_col(exp1, ['both_driven_frac'], required=False)
depth1 = first_col(exp1, ['depth_fraction', 'layer_idx', 'layer_number'], required=True)
pooling_col = first_col(exp1, ['pooling'], required=False)
audit_col = first_col(exp1, ['audit'], required=False)
dom_margin = first_col(exp1, ['dom_margin', 'dominance_margin'], required=False)
seed_col = first_col(exp1, ['seed'], required=False)
width_col = first_col(exp1, ['width_factor'], required=False)
l1_col = first_col(exp1, ['l1'], required=False)
normalize_col = first_col(exp1, ['normalize'], required=False)
rec_r2 = first_col(exp1, ['vlm_eval_rec_r2', 'reconstruction_r2', 'rec_r2'], required=False)
l0_col = first_col(exp1, ['vlm_eval_l0', 'l0'], required=False)
active_col = first_col(exp1, ['vlm_eval_active_frac', 'active_frac'], required=False)
dead_col = first_col(exp1, ['vlm_eval_dead_frac', 'dead_frac'], required=False)
zabs_col = first_col(exp1, ['vlm_eval_z_abs_mean', 'z_abs_mean'], required=False)
lps_text = first_col(exp2, ['lps_text'], required=False)
lps_img_only = first_col(exp2, ['lps_image_only'], required=False)
lps_img = first_col(exp2, ['lps_image_conditioned'], required=True)
nri_img = first_col(exp2, ['nri_image_conditioned'], required=True)
shared_ratio = first_col(exp2, ['shared_neighbor_ratio'], required=False)
cross_super = first_col(exp2, ['cross_superclass_gained_frac'], required=False)
random_baseline = first_col(exp2, ['random_shared_baseline'], required=False)
align_proxy = first_col(exp2, ['align_fit_proxy'], required=False)
k_col = first_col(exp2, ['k'], required=True)
depth2 = first_col(exp2, ['depth_fraction', 'layer_idx', 'layer_number'], required=True)
n_concepts = first_col(exp2, ['n_concepts'], required=False)
print('Resolved columns:')
for k, v in {'mfi': mfi, 'fmp': fmp, 'mean_text_delta': mean_text_delta, 'mean_image_delta': mean_image_delta, 'rec_r2': rec_r2, 'l0_col': l0_col, 'dead_col': dead_col, 'lps_text': lps_text, 'lps_img_only': lps_img_only, 'lps_img': lps_img, 'nri_img': nri_img, 'shared_ratio': shared_ratio, 'cross_super': cross_super, 'k_col': k_col, 'depth2': depth2}.items():
    print(f'  {k}: {v}')
exp1_main = exp1.copy()
if audit_col is not None:
    audit_s = exp1_main[audit_col].astype(str).str.lower()
    keep = ~audit_s.isin(['true', '1', 'yes'])
    if keep.sum() > 0:
        exp1_main = exp1_main[keep].copy()
if pooling_col is not None:
    p_s = exp1_main[pooling_col].astype(str)
    if (p_s == 'final_quarter').any():
        exp1_main = exp1_main[p_s == 'final_quarter'].copy()
if dom_margin is not None:
    exp1_main[dom_margin] = to_num(exp1_main[dom_margin])
    if np.isclose(exp1_main[dom_margin], 1.25).any():
        exp1_main = exp1_main[np.isclose(exp1_main[dom_margin], 1.25)].copy()
exp2_k5 = exp2.copy()
exp2_k5[k_col] = to_num(exp2_k5[k_col])
if (exp2_k5[k_col] == 5).any():
    exp2_k5 = exp2_k5[exp2_k5[k_col] == 5].copy()
print('After filters:')
print('  exp1_main:', exp1_main.shape)
print('  exp2_k5:', exp2_k5.shape)
sparse_metrics = [mfi, fmp, mean_text_delta, mean_image_delta, image_driven_frac, text_driven_frac, both_driven_frac, rec_r2, l0_col, active_col, dead_col, zabs_col]
sparse_metrics = [c for c in sparse_metrics if c is not None and c in exp1_main.columns]
neigh_metrics = [lps_text, lps_img_only, lps_img, nri_img, shared_ratio, cross_super, random_baseline, align_proxy, n_concepts]
neigh_metrics = [c for c in neigh_metrics if c is not None and c in exp2_k5.columns]
A1 = grouped_summary(exp1_main, ['family', 'dataset'], sparse_metrics)
B1 = grouped_summary(exp2_k5, ['family', 'dataset'], neigh_metrics)
save_table(A1, 'Appendix_A1_full_sparse_grid_numeric')
save_interval_table(A1, ['family', 'dataset'], sparse_metrics, 'Appendix_A1_full_sparse_grid_interval')
save_table(B1, 'Appendix_B1_full_neighborhood_grid_K5_numeric')
save_interval_table(B1, ['family', 'dataset'], neigh_metrics, 'Appendix_B1_full_neighborhood_grid_K5_interval')
A1_core = A1[['family', 'dataset', mfi, fmp]].copy()
B1_core = B1[['family', 'dataset', lps_img, nri_img]].copy()
T1_cells = A1_core.merge(B1_core, on=['family', 'dataset'], how='outer')
T1_family = order_fd(T1_cells.groupby('family', dropna=False).mean(numeric_only=True).reset_index())
T1_dataset = order_fd(T1_cells.groupby('dataset', dropna=False).mean(numeric_only=True).reset_index())
save_table(T1_family, 'Table_1a_family_summary_numeric')
save_table(T1_dataset, 'Table_1b_dataset_summary_numeric')
T1_combined = pd.concat([T1_family.rename(columns={'family': 'group'}).assign(summary_type='family'), T1_dataset.rename(columns={'dataset': 'group'}).assign(summary_type='dataset')], ignore_index=True, sort=False)
T1_combined = T1_combined[['summary_type', 'group'] + [c for c in T1_combined.columns if c not in ['summary_type', 'group']]]
save_table(T1_combined, 'Table_1_family_and_dataset_summary_numeric')
T2_cols = ['family', 'dataset'] + [c for c in [lps_text, lps_img_only, lps_img, nri_img, n_concepts] if c is not None and c in B1.columns]
T2 = B1[T2_cols].copy()
save_table(T2, 'Table_2_neighborhood_K5_grid_numeric')
save_interval_table(B1, ['family', 'dataset'], [c for c in [lps_text, lps_img_only, lps_img, nri_img, n_concepts] if c is not None], 'Table_2_neighborhood_K5_grid_interval')
sparse_validation_cols = [mfi, fmp, mean_text_delta, mean_image_delta, image_driven_frac, text_driven_frac, both_driven_frac]
sparse_validation_cols = [c for c in sparse_validation_cols if c is not None and c in exp1_main.columns]
sparse_corr = filtered_corr(exp1_main, sparse_validation_cols)
if len(sparse_corr) > 0:
    save_table(sparse_corr.reset_index().rename(columns={'index': 'metric'}), 'Appendix_A2_sparse_metric_correlation')
quality_cols = [rec_r2, l0_col, active_col, dead_col, zabs_col]
quality_cols = [c for c in quality_cols if c is not None and c in exp1_main.columns]
if quality_cols:
    A3 = grouped_summary(exp1_main, ['family', 'dataset'], quality_cols)
    save_table(A3, 'Appendix_A3_SAE_quality_by_cell_numeric')
    save_interval_table(A3, ['family', 'dataset'], quality_cols, 'Appendix_A3_SAE_quality_by_cell_interval')
hparam_group = [c for c in [width_col, l1_col, normalize_col] if c is not None and c in exp1_main.columns]
hparam_metrics = [c for c in [mfi, fmp, rec_r2, l0_col, active_col, dead_col] if c is not None and c in exp1_main.columns]
if hparam_group and hparam_metrics:
    A4 = grouped_summary(exp1_main, hparam_group, hparam_metrics)
    save_table(A4, 'Appendix_A4_SAE_hyperparameter_robustness_numeric')
    save_interval_table(A4, hparam_group, hparam_metrics, 'Appendix_A4_SAE_hyperparameter_robustness_interval')
if seed_col is not None:
    seed_metrics = [c for c in [mfi, fmp, rec_r2, l0_col, dead_col] if c is not None and c in exp1_main.columns]
    A5 = grouped_summary(exp1_main, [seed_col], seed_metrics)
    save_table(A5, 'Appendix_A5_seed_robustness_numeric')
    save_interval_table(A5, [seed_col], seed_metrics, 'Appendix_A5_seed_robustness_interval')
if dom_margin is not None:
    threshold_metrics = [c for c in [mfi, fmp, image_driven_frac, text_driven_frac, both_driven_frac] if c is not None and c in exp1.columns]
    A6 = grouped_summary(exp1, ['family', 'dataset', dom_margin], threshold_metrics)
    save_table(A6, 'Appendix_A6_threshold_sensitivity_numeric')
    save_interval_table(A6, ['family', 'dataset', dom_margin], threshold_metrics, 'Appendix_A6_threshold_sensitivity_interval')
if pooling_col is not None:
    pooling_metrics = [c for c in [mfi, fmp, mean_text_delta, mean_image_delta, rec_r2, l0_col, dead_col] if c is not None and c in exp1.columns]
    A7 = grouped_summary(exp1, ['family', 'dataset', pooling_col], pooling_metrics)
    save_table(A7, 'Appendix_A7_pooling_audit_numeric')
    save_interval_table(A7, ['family', 'dataset', pooling_col], pooling_metrics, 'Appendix_A7_pooling_audit_interval')
B2 = grouped_summary(exp2, ['family', 'dataset', k_col], neigh_metrics)
save_table(B2, 'Appendix_B2_K_sensitivity_numeric')
save_interval_table(B2, ['family', 'dataset', k_col], neigh_metrics, 'Appendix_B2_K_sensitivity_interval')
if n_concepts is not None:
    B3 = grouped_summary(exp2, ['family', 'dataset'], [n_concepts])
    save_table(B3, 'Appendix_B3_concept_coverage_numeric')
    save_interval_table(B3, ['family', 'dataset'], [n_concepts], 'Appendix_B3_concept_coverage_interval')
neigh_validation_cols = [lps_text, lps_img_only, lps_img, nri_img, shared_ratio, cross_super, align_proxy]
neigh_validation_cols = [c for c in neigh_validation_cols if c is not None and c in exp2_k5.columns]
neigh_corr = filtered_corr(exp2_k5, neigh_validation_cols)
if len(neigh_corr) > 0:
    save_table(neigh_corr.reset_index().rename(columns={'index': 'metric'}), 'Appendix_B4_neighborhood_metric_correlation')
fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8))
plot_heatmap(axes[0, 0], A1, mfi, 'A. MFI, 3×3 grid', cmap='viridis')
plot_heatmap(axes[0, 1], A1, fmp, 'B. FMP, 3×3 grid', cmap='viridis')
plot_trajectory(axes[0, 2], exp1_main, depth1, mfi, 'C. MFI across relative depth', 'Relative depth', 'MFI')
plot_trajectory(axes[1, 0], exp1_main, depth1, fmp, 'D. FMP across relative depth', 'Relative depth', 'FMP')
if mean_text_delta is not None:
    plot_scatter(axes[1, 1], exp1_main, mean_text_delta, mfi, 'E. MFI validation', 'Mean text delta', 'MFI')
elif mean_image_delta is not None:
    plot_scatter(axes[1, 1], exp1_main, mean_image_delta, mfi, 'E. MFI validation', 'Mean image delta', 'MFI')
else:
    plot_scatter(axes[1, 1], exp1_main, fmp, mfi, 'E. Sparse metric relation', 'FMP', 'MFI')
if rec_r2 is not None and l0_col is not None:
    plot_scatter(axes[1, 2], exp1_main, l0_col, rec_r2, 'F. SAE validation', 'Mean active latents', 'Reconstruction R²')
elif rec_r2 is not None and dead_col is not None:
    plot_scatter(axes[1, 2], exp1_main, dead_col, rec_r2, 'F. SAE validation', 'Dead-latent fraction', 'Reconstruction R²')
else:
    plot_corr(axes[1, 2], exp1_main, sparse_validation_cols, 'F. Sparse metric correlations')
fig.tight_layout()
save_fig(fig, 'Figure_1_sparse_perturbation_and_SAE_validation', main_alias='Figure_1_sparse_perturbation')
plt.close(fig)
fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8))
plot_heatmap(axes[0, 0], B1, lps_img, 'A. Image-conditioned LPS, K=5', cmap='viridis')
plot_heatmap(axes[0, 1], B1, nri_img, 'B. Image-conditioned NRI, K=5', cmap='viridis')
plot_trajectory(axes[0, 2], exp2_k5, depth2, lps_img, 'C. LPS across relative depth', 'Relative depth', 'LPS')
if shared_ratio is not None:
    plot_trajectory(axes[1, 0], exp2_k5, depth2, shared_ratio, 'D. Shared-neighbor ratio', 'Relative depth', 'Shared ratio')
    if random_baseline is not None:
        tmp = exp2_k5.copy()
        tmp[depth2] = to_num(tmp[depth2])
        tmp[random_baseline] = to_num(tmp[random_baseline])
        rb = tmp.groupby(depth2, dropna=False)[random_baseline].mean().reset_index().sort_values(depth2)
        axes[1, 0].plot(rb[depth2], rb[random_baseline], linestyle='--', linewidth=1.2, label='random baseline')
        axes[1, 0].legend(frameon=False)
else:
    plot_trajectory(axes[1, 0], exp2_k5, depth2, nri_img, 'D. NRI across relative depth', 'Relative depth', 'NRI')
if cross_super is not None:
    plot_trajectory(axes[1, 1], exp2_k5, depth2, cross_super, 'E. Cross-superclass gained neighbors', 'Relative depth', 'Fraction')
else:
    plot_scatter(axes[1, 1], exp2_k5, lps_img, nri_img, 'E. LPS–NRI relation', 'LPS', 'NRI')
tmp = exp2.copy()
tmp[k_col] = to_num(tmp[k_col])
tmp[lps_img] = to_num(tmp[lps_img])
for fam, g in tmp.dropna(subset=[k_col, lps_img]).groupby('family', dropna=False):
    s = grouped_summary(g, [k_col], [lps_img]).sort_values(k_col)
    axes[1, 2].plot(s[k_col], s[lps_img], marker='o', linewidth=1.5, label=str(fam))
axes[1, 2].set_title('F. K-sensitivity of LPS')
axes[1, 2].set_xlabel('K')
axes[1, 2].set_ylabel('LPS')
axes[1, 2].legend(frameon=False)
fig.tight_layout()
save_fig(fig, 'Figure_2_neighborhood_preservation_and_validation', main_alias='Figure_2_neighborhood_preservation_rewriting')
plt.close(fig)
fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8))
plot_heatmap(axes[0, 0], A1, rec_r2, 'A. Reconstruction R²', cmap='viridis')
plot_heatmap(axes[0, 1], A1, dead_col, 'B. Dead-latent fraction', cmap='viridis')
if rec_r2 is not None and l0_col is not None:
    plot_scatter(axes[0, 2], exp1_main, l0_col, rec_r2, 'C. Reconstruction vs sparsity', 'Mean active latents', 'Reconstruction R²')
elif rec_r2 is not None and active_col is not None:
    plot_scatter(axes[0, 2], exp1_main, active_col, rec_r2, 'C. Reconstruction vs activity', 'Active fraction', 'Reconstruction R²')
else:
    plot_corr(axes[0, 2], exp1_main, quality_cols, 'C. SAE quality correlations')
if width_col is not None and rec_r2 is not None:
    plot_trajectory(axes[1, 0], exp1_main, width_col, rec_r2, 'D. Reconstruction by width', 'Width factor', 'Reconstruction R²')
else:
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'width/R² unavailable', ha='center', va='center')
if l1_col is not None and l0_col is not None:
    plot_trajectory(axes[1, 1], exp1_main, l1_col, l0_col, 'E. Sparsity by L1', 'L1', 'Mean active latents')
else:
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.5, 'L1/L0 unavailable', ha='center', va='center')
if pooling_col is not None:
    plot_category(axes[1, 2], exp1, pooling_col, mfi, 'F. Pooling audit', 'Pooling', 'MFI')
else:
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, 'pooling unavailable', ha='center', va='center')
fig.tight_layout()
save_fig(fig, 'Appendix_A_SAE_and_sparse_metric_validation')
plt.close(fig)
fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.8))
plot_heatmap(axes[0, 0], B1, shared_ratio, 'A. Shared-neighbor ratio', cmap='viridis')
plot_heatmap(axes[0, 1], B1, cross_super, 'B. Cross-superclass gained neighbors', cmap='viridis')
plot_scatter(axes[0, 2], exp2_k5, lps_img, nri_img, 'C. LPS–NRI consistency', 'LPS', 'NRI')
tmp = exp2.copy()
tmp[k_col] = to_num(tmp[k_col])
tmp[lps_img] = to_num(tmp[lps_img])
tmp[nri_img] = to_num(tmp[nri_img])
for fam, g in tmp.dropna(subset=[k_col, lps_img]).groupby('family', dropna=False):
    s = grouped_summary(g, [k_col], [lps_img]).sort_values(k_col)
    axes[1, 0].plot(s[k_col], s[lps_img], marker='o', linewidth=1.5, label=str(fam))
axes[1, 0].set_title('D. K-sensitivity of LPS')
axes[1, 0].set_xlabel('K')
axes[1, 0].set_ylabel('LPS')
axes[1, 0].legend(frameon=False)
for fam, g in tmp.dropna(subset=[k_col, nri_img]).groupby('family', dropna=False):
    s = grouped_summary(g, [k_col], [nri_img]).sort_values(k_col)
    axes[1, 1].plot(s[k_col], s[nri_img], marker='o', linewidth=1.5, label=str(fam))
axes[1, 1].set_title('E. K-sensitivity of NRI')
axes[1, 1].set_xlabel('K')
axes[1, 1].set_ylabel('NRI')
axes[1, 1].legend(frameon=False)
plot_corr(axes[1, 2], exp2_k5, neigh_validation_cols, 'F. Filtered metric correlations')
fig.tight_layout()
save_fig(fig, 'Appendix_B_neighborhood_metric_validation')
plt.close(fig)
manifest = {'root': str(ROOT.resolve()), 'out_table': str(OUT_TABLE.resolve()), 'out_fig': str(OUT_FIG.resolve()), 'exp1_shape': tuple(exp1.shape), 'exp2_shape': tuple(exp2.shape), 'main_exp1_shape': tuple(exp1_main.shape), 'main_exp2_k5_shape': tuple(exp2_k5.shape), 'columns': {'mfi': mfi, 'fmp': fmp, 'mean_text_delta': mean_text_delta, 'mean_image_delta': mean_image_delta, 'rec_r2': rec_r2, 'l0': l0_col, 'dead': dead_col, 'lps_img': lps_img, 'nri_img': nri_img, 'shared_ratio': shared_ratio, 'cross_super': cross_super, 'k': k_col, 'n_concepts': n_concepts}, 'tables': sorted([p.name for p in OUT_TABLE.glob('*')]), 'figures': sorted([p.name for p in OUT_FIG.glob('*')])}
manifest_path = OUT_TABLE / 'final_figure_table_manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print('\nDONE.')
print('Tables:', OUT_TABLE.resolve())
print('Figures:', OUT_FIG.resolve())
print('Main replacements:', FIG_DIR.resolve())
print('Manifest:', manifest_path.resolve())

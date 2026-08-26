import os
import sys
from getpass import getpass
import re
import gc
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None
    print('Warning: AutoModelForVision2Seq is not available in this transformers version.')
    print('The notebook will use AutoModelForImageTextToText where possible.')
warnings.filterwarnings('ignore')
HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN and sys.stdin.isatty():
    HF_TOKEN = getpass('Enter your Hugging Face token: ').strip() or None
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
PROJECT_DIR = Path('./supplementary_pca_outputs')
FIG_DIR = PROJECT_DIR / 'figures'
TABLE_DIR = PROJECT_DIR / 'tables'
EMB_DIR = PROJECT_DIR / 'embeddings'
for path in [PROJECT_DIR, FIG_DIR, TABLE_DIR, EMB_DIR]:
    path.mkdir(parents=True, exist_ok=True)
print(f'Device: {DEVICE}')
print(f'Default dtype: {DTYPE}')
print(f'Output directory: {PROJECT_DIR.resolve()}')
MAX_SAMPLES_PER_DATASET = 250
BATCH_SIZE = 1
MAX_TEXT_LENGTH = 512
TARGET_LAYER_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_PCA_COMPONENTS = 64
NORMALIZE_ACTIVATIONS_BEFORE_PCA = True
MIN_SAMPLES_FOR_PCA = 12
MODEL_SPECS = {'paligemma2': {'model_id': 'google/paligemma2-3b-mix-224', 'loader': 'image_text_to_text', 'family': 'PaliGemma2-mix'}, 'qwen2_vl': {'model_id': 'Qwen/Qwen2-VL-2B-Instruct', 'loader': 'image_text_to_text', 'family': 'Qwen2-VL-2B'}, 'smolvlm': {'model_id': 'HuggingFaceTB/SmolVLM-500M-Instruct', 'loader': 'image_text_to_text', 'family': 'SmolVLM-500M'}}
DATASET_SPECS = {'pixelprose': {'hf_name': 'tomg-group-umd/pixelprose', 'split_candidates': ['train'], 'image_candidates': ['image', 'jpg', 'url', 'image_url'], 'caption_candidates': ['caption', 'text', 'vlm_caption', 'dense_caption', 'description', 'txt']}, 'flickr30k': {'hf_name': 'lmms-lab/flickr30k', 'split_candidates': ['test', 'validation', 'train'], 'image_candidates': ['image', 'jpg', 'img'], 'caption_candidates': ['caption', 'captions', 'sentences', 'sentence', 'text']}, 'coco_karpathy': {'hf_name': 'yerevann/coco-karpathy', 'split_candidates': ['test', 'val', 'validation', 'train'], 'image_candidates': ['image', 'url', 'image_url', 'filepath', 'filename'], 'caption_candidates': ['sentences', 'caption', 'captions', 'text']}}
CONDITIONS = {'image_caption_query': {'use_image': True, 'prompt_template': 'Describe this image in one concise sentence.'}, 'image_caption_alignment': {'use_image': True, 'prompt_template': 'Does this image match the following caption? Caption: {caption}'}, 'text_caption_only': {'use_image': False, 'prompt_template': 'Caption: {caption}'}}
print(json.dumps({'max_samples_per_dataset': MAX_SAMPLES_PER_DATASET, 'batch_size': BATCH_SIZE, 'target_layer_fractions': TARGET_LAYER_FRACTIONS, 'models': MODEL_SPECS, 'datasets': list(DATASET_SPECS.keys()), 'conditions': list(CONDITIONS.keys())}, indent=2))
import requests
from io import BytesIO
from itertools import islice
DATASET_ALIASES = {'pixelprose': ['tomg-group-umd/pixelprose'], 'flickr30k': ['lmms-lab/flickr30k', 'Mozilla/flickr30k-transformed-captions', 'nlphuji/flickr30k'], 'coco_karpathy': ['yerevann/coco-karpathy']}

def find_first_existing_key(example: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in example:
            return key
    return None

def extract_caption(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        if len(value) == 0:
            return ''
        first = value[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for key in ['raw', 'caption', 'text', 'sentence', 'description']:
                if key in first and first[key] is not None:
                    return str(first[key]).strip()
        return str(first).strip()
    if isinstance(value, dict):
        for key in ['raw', 'caption', 'text', 'sentence', 'description']:
            if key in value and value[key] is not None:
                return str(value[key]).strip()
    return str(value).strip()

def download_image_from_url(url: str, timeout: int=15) -> Optional[Image.Image]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception:
        return None

def ensure_pil_image_from_example(example: Dict[str, Any], image_key: str) -> Optional[Image.Image]:
    value = example.get(image_key)
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    if isinstance(value, dict):
        if 'bytes' in value and value['bytes'] is not None:
            try:
                return Image.open(BytesIO(value['bytes'])).convert('RGB')
            except Exception:
                return None
        for key in ['path', 'url']:
            if key in value and value[key]:
                value = str(value[key])
                if value.startswith('http'):
                    return download_image_from_url(value)
                try:
                    return Image.open(value).convert('RGB')
                except Exception:
                    return None
    if isinstance(value, str):
        if value.startswith('http'):
            return download_image_from_url(value)
        try:
            return Image.open(value).convert('RGB')
        except Exception:
            return None
    return None

def try_load_dataset_any_mode(hf_names: List[str], split_candidates: List[str]):
    errors = []
    for hf_name in hf_names:
        for split in split_candidates:
            try:
                ds = load_dataset(hf_name, split=split, streaming=True)
                return (ds, split, hf_name, True)
            except Exception as error:
                errors.append(f'streaming {hf_name}/{split}: {repr(error)}')
        for split in split_candidates:
            try:
                ds = load_dataset(hf_name, split=split)
                return (ds, split, hf_name, False)
            except Exception as error:
                errors.append(f'non-streaming {hf_name}/{split}: {repr(error)}')
        try:
            dsdict = load_dataset(hf_name)
            split = list(dsdict.keys())[0]
            return (dsdict[split], split, hf_name, False)
        except Exception as error:
            errors.append(f'datasetdict {hf_name}: {repr(error)}')
    raise RuntimeError('Could not load dataset. Errors:\n' + '\n'.join(errors[-8:]))

def iter_dataset_examples(dataset, is_streaming: bool, scan_limit: int):
    if is_streaming:
        try:
            dataset = dataset.shuffle(buffer_size=2000, seed=SEED)
        except Exception:
            pass
        for i, ex in enumerate(islice(dataset, scan_limit)):
            yield (i, ex)
    else:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for i, idx in enumerate(indices[:scan_limit]):
            yield (int(idx), dataset[int(idx)])

def load_caption_dataset_subset(dataset_name: str, spec: Dict[str, Any], max_samples: int, scan_limit: int=5000) -> List[Dict[str, Any]]:
    hf_names = DATASET_ALIASES.get(dataset_name, [spec['hf_name']])
    dataset, used_split, used_hf_name, is_streaming = try_load_dataset_any_mode(hf_names=hf_names, split_candidates=spec['split_candidates'])
    records = []
    skipped = 0
    image_key = None
    caption_key = None
    iterator = iter_dataset_examples(dataset, is_streaming=is_streaming, scan_limit=scan_limit)
    for scanned, ex in tqdm(iterator, total=scan_limit, desc=f'Loading {dataset_name}'):
        if len(records) >= max_samples:
            break
        if image_key is None:
            image_key = find_first_existing_key(ex, spec['image_candidates'])
            caption_key = find_first_existing_key(ex, spec['caption_candidates'])
            if image_key is None:
                raise KeyError(f'No image column found for {dataset_name}. Available keys: {list(ex.keys())}')
            if caption_key is None:
                raise KeyError(f'No caption column found for {dataset_name}. Available keys: {list(ex.keys())}')
            print(f'{dataset_name}: hf_name={used_hf_name}, split={used_split}, streaming={is_streaming}, image_key={image_key}, caption_key={caption_key}')
        image = ensure_pil_image_from_example(ex, image_key)
        if image is None:
            skipped += 1
            continue
        caption = extract_caption(ex.get(caption_key))
        if len(caption) < 5:
            skipped += 1
            continue
        records.append({'dataset': dataset_name, 'dataset_source': used_hf_name, 'dataset_split': used_split, 'dataset_index': int(scanned), 'sample_id': f'{dataset_name}_{len(records):05d}', 'image': image, 'caption': caption})
    if len(records) == 0:
        raise RuntimeError(f'No valid records loaded for {dataset_name}. Skipped={skipped}. Increase scan_limit or check columns.')
    print(f'{dataset_name}: loaded {len(records)} samples from {used_hf_name}/{used_split}; skipped={skipped}')
    return records
all_dataset_records = {}
for dataset_name, spec in DATASET_SPECS.items():
    records = load_caption_dataset_subset(dataset_name=dataset_name, spec=spec, max_samples=MAX_SAMPLES_PER_DATASET, scan_limit=5000)
    all_dataset_records[dataset_name] = records
total_samples = sum((len(v) for v in all_dataset_records.values()))
print(f'Total base image-caption samples: {total_samples}')

def build_condition_records(all_records: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    condition_records = []
    for dataset_name, records in all_records.items():
        for record in records:
            for condition_name, condition_spec in CONDITIONS.items():
                prompt = condition_spec['prompt_template'].format(caption=record['caption'])
                condition_records.append({'dataset': dataset_name, 'dataset_split': record['dataset_split'], 'sample_id': record['sample_id'], 'dataset_index': record['dataset_index'], 'caption': record['caption'], 'label': record['caption'], 'condition': condition_name, 'prompt': prompt, 'image': record['image'] if condition_spec['use_image'] else None, 'use_image': condition_spec['use_image']})
    return condition_records
condition_records = build_condition_records(all_dataset_records)
condition_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'image'} for r in condition_records])
print(condition_df.groupby(['dataset', 'condition']).size())
from transformers import AutoProcessor, AutoModelForImageTextToText
try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

def load_model_and_processor(model_key: str, spec: Dict[str, Any]):
    model_id = spec['model_id']
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    common_kwargs = {'device_map': 'auto' if torch.cuda.is_available() else None, 'low_cpu_mem_usage': True, 'trust_remote_code': True}
    common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}
    loaders = [AutoModelForImageTextToText]
    if AutoModelForVision2Seq is not None:
        loaders.append(AutoModelForVision2Seq)
    errors = []
    for loader in loaders:
        for dtype_key in ['dtype', 'torch_dtype']:
            try:
                kwargs = dict(common_kwargs)
                kwargs[dtype_key] = DTYPE
                model = loader.from_pretrained(model_id, **kwargs)
                model.eval()
                return (model, processor)
            except Exception as error:
                errors.append(f'{loader.__name__} with {dtype_key}: {repr(error)}')
    raise RuntimeError(f'Could not load model {model_key} / {model_id}.\n' + '\n'.join(errors[-6:]))

def release_model(model=None, processor=None):
    try:
        del model
    except Exception:
        pass
    try:
        del processor
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def format_model_input_text(model_key: str, prompt: str, use_image: bool) -> str:
    if model_key == 'paligemma2':
        return f'<image> {prompt}'
    if model_key == 'qwen2_vl':
        if use_image:
            return f'<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
        return f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
    if model_key == 'smolvlm':
        if use_image:
            return f'<image> {prompt}'
        return prompt
    return f'<image> {prompt}' if use_image else prompt

def make_blank_image(size: int=224) -> Image.Image:
    return Image.new('RGB', (size, size), color=(255, 255, 255))

def prepare_inputs(model_key: str, processor, records: List[Dict[str, Any]]):
    texts = [format_model_input_text(model_key=model_key, prompt=r['prompt'], use_image=r['use_image']) for r in records]
    if model_key == 'paligemma2':
        images = [r['image'] if r['use_image'] else make_blank_image() for r in records]
        return processor(text=texts, images=images, return_tensors='pt', padding=True)
    any_image = any((r['use_image'] for r in records))
    if any_image:
        images = [r['image'] if r['use_image'] else make_blank_image() for r in records]
        return processor(text=texts, images=images, return_tensors='pt', padding=True)
    return processor(text=texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_TEXT_LENGTH)

def move_inputs_to_model(inputs, model):
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device(DEVICE)
    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved

def get_hidden_states_from_output(outputs):
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
        return outputs.hidden_states
    if isinstance(outputs, dict) and 'hidden_states' in outputs:
        return outputs['hidden_states']
    raise RuntimeError('Model output does not contain hidden_states. The forward pass may not support output_hidden_states=True for this class.')

def mean_pool_hidden_state(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    hidden = hidden.float()
    if hidden.ndim != 3:
        hidden = hidden.reshape(hidden.shape[0], -1)
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.float().to(hidden.device)
    if mask.ndim == 2 and mask.shape[1] == hidden.shape[1]:
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask).sum(dim=1) / denom
    return hidden.mean(dim=1)

def select_layer_indices(num_hidden_states: int, fractions: List[float]) -> List[int]:
    max_idx = num_hidden_states - 1
    indices = []
    for frac in fractions:
        idx = int(round(frac * max_idx))
        idx = max(0, min(idx, max_idx))
        indices.append(idx)
    return sorted(list(set(indices)))

def extract_model_activations(model_key: str, model_spec: Dict[str, Any], records: List[Dict[str, Any]]) -> pd.DataFrame:
    print(f"\nLoading model: {model_key} | {model_spec['model_id']}")
    model, processor = load_model_and_processor(model_key, model_spec)
    rows = []
    failed_batches = []
    for start in tqdm(range(0, len(records), BATCH_SIZE), desc=f'Extracting {model_key}'):
        batch_records = records[start:start + BATCH_SIZE]
        try:
            inputs = prepare_inputs(model_key, processor, batch_records)
            inputs = move_inputs_to_model(inputs, model)
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = get_hidden_states_from_output(outputs)
            selected_indices = select_layer_indices(num_hidden_states=len(hidden_states), fractions=TARGET_LAYER_FRACTIONS)
            attention_mask = inputs.get('attention_mask', None)
            for layer_index in selected_indices:
                hidden = hidden_states[layer_index]
                pooled = mean_pool_hidden_state(hidden, attention_mask=attention_mask)
                pooled = pooled.detach().cpu().numpy().astype(np.float32)
                for i, record in enumerate(batch_records):
                    if i >= pooled.shape[0]:
                        continue
                    rows.append({'model_key': model_key, 'model_family': model_spec['family'], 'model_id': model_spec['model_id'], 'dataset': record['dataset'], 'dataset_source': record.get('dataset_source', ''), 'dataset_split': record.get('dataset_split', ''), 'sample_id': record['sample_id'], 'dataset_index': record['dataset_index'], 'caption': record['caption'], 'label': record.get('label', record['caption']), 'condition': record['condition'], 'use_image': record['use_image'], 'prompt': record['prompt'], 'layer_index': int(layer_index), 'layer_name': f'hidden_states[{layer_index}]', 'activation': pooled[i]})
        except Exception as error:
            failed_batches.append({'start': int(start), 'end': int(start + len(batch_records)), 'error': repr(error), 'example_prompt': batch_records[0]['prompt'] if len(batch_records) else '', 'example_condition': batch_records[0]['condition'] if len(batch_records) else '', 'example_dataset': batch_records[0]['dataset'] if len(batch_records) else ''})
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failed_batches:
        fail_path = TABLE_DIR / f'{model_key}_failed_batches.json'
        with open(fail_path, 'w') as f:
            json.dump(failed_batches, f, indent=2)
        print(f'Failed batches saved to: {fail_path}')
        print('First failed batch:')
        print(json.dumps(failed_batches[0], indent=2)[:2000])
    df = pd.DataFrame(rows)
    output_path = EMB_DIR / f'{model_key}_activations.pkl'
    df.to_pickle(output_path)
    print(f'Saved activations: {output_path}')
    print(f'Rows: {len(df)}')
    if len(df) > 0:
        print(df.groupby(['dataset', 'condition', 'layer_index']).size().head(20))
    release_model(model, processor)
    return df
all_activation_dfs = []
for model_key, model_spec in MODEL_SPECS.items():
    try:
        model_df = extract_model_activations(model_key=model_key, model_spec=model_spec, records=condition_records)
        if model_df is not None and len(model_df) > 0:
            all_activation_dfs.append(model_df)
        else:
            print(f'Warning: {model_key} produced 0 activation rows.')
    except Exception as error:
        print(f'\nModel failed completely: {model_key}')
        print(repr(error))
if len(all_activation_dfs) == 0:
    raise RuntimeError('No model produced activations. Do not continue to PCA.')
activation_df = pd.concat(all_activation_dfs, ignore_index=True)
activation_df.to_pickle(EMB_DIR / 'all_activations.pkl')
summary = activation_df.groupby(['model_key', 'dataset', 'condition', 'layer_index']).size().reset_index(name='n').sort_values(['model_key', 'dataset', 'condition', 'layer_index'])
summary.to_csv(TABLE_DIR / 'activation_counts.csv', index=False)
print(f'Total activation rows: {len(activation_df)}')
print(f"Saved combined activations to: {EMB_DIR / 'all_activations.pkl'}")
min_n = summary['n'].min()
max_n = summary['n'].max()
print(f'Minimum n per group: {min_n}')
print(f'Maximum n per group: {max_n}')
if min_n < 50:
    raise RuntimeError('Some model/dataset/condition/layer groups have too few samples. Do not continue to PCA until extraction is balanced.')

def stack_activations(df: pd.DataFrame) -> np.ndarray:
    X = np.stack(df['activation'].to_numpy()).astype(np.float64)
    if NORMALIZE_ACTIVATIONS_BEFORE_PCA:
        X = normalize(X, norm='l2', axis=1)
    return X

def participation_ratio(eigenvalues: np.ndarray) -> float:
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    denom = np.sum(eigenvalues ** 2)
    if denom <= 0:
        return float('nan')
    return float(np.sum(eigenvalues) ** 2 / denom)

def components_needed(cumulative: np.ndarray, threshold: float) -> int:
    hits = np.where(cumulative >= threshold)[0]
    if len(hits) == 0:
        return int(len(cumulative))
    return int(hits[0] + 1)

def fit_pca_for_group(group_df: pd.DataFrame, max_components: int=MAX_PCA_COMPONENTS):
    X = stack_activations(group_df)
    n_samples, n_features = X.shape
    n_components = min(max_components, n_samples - 1, n_features)
    if n_samples < MIN_SAMPLES_FOR_PCA or n_components < 2:
        return (None, None)
    pca = PCA(n_components=n_components, random_state=SEED)
    Z = pca.fit_transform(X)
    return (pca, Z)

def compute_pca_metrics(activation_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple, PCA]]:
    metric_rows = []
    pca_objects = {}
    group_cols = ['model_key', 'model_family', 'model_id', 'dataset', 'condition', 'layer_index', 'layer_name']
    for group_key, group_df in tqdm(activation_df.groupby(group_cols), desc='Computing PCA metrics'):
        pca, Z = fit_pca_for_group(group_df)
        if pca is None:
            continue
        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        row = dict(zip(group_cols, group_key))
        row.update({'n_samples': int(len(group_df)), 'hidden_dim': int(stack_activations(group_df).shape[1]), 'n_components_fit': int(len(explained)), 'top1_variance_ratio': float(explained[0]), 'top3_variance_ratio': float(np.sum(explained[:3])), 'top5_variance_ratio': float(np.sum(explained[:5])), 'pc90': components_needed(cumulative, 0.9), 'pc95': components_needed(cumulative, 0.95), 'effective_dimension': participation_ratio(pca.explained_variance_), 'variance_entropy': float(-np.sum(explained * np.log(explained + 1e-12)))})
        metric_rows.append(row)
        pca_key = (row['model_key'], row['dataset'], row['condition'], row['layer_index'])
        pca_objects[pca_key] = pca
    return (pd.DataFrame(metric_rows), pca_objects)
pca_metric_df, pca_objects = compute_pca_metrics(activation_df)
pca_metric_df.to_csv(TABLE_DIR / 'pca_metrics.csv', index=False)
print(f"Saved PCA metrics: {TABLE_DIR / 'pca_metrics.csv'}")

def orthonormal_rows_to_columns(components: np.ndarray, k: int) -> np.ndarray:
    A = components[:k].T
    Q, _ = np.linalg.qr(A)
    return Q[:, :k]

def subspace_alignment_score(pca_a: PCA, pca_b: PCA, k: int) -> Dict[str, float]:
    k = min(k, pca_a.components_.shape[0], pca_b.components_.shape[0])
    Qa = orthonormal_rows_to_columns(pca_a.components_, k)
    Qb = orthonormal_rows_to_columns(pca_b.components_, k)
    singular_values = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles = np.arccos(singular_values)
    return {'k': int(k), 'mean_cos2': float(np.mean(singular_values ** 2)), 'min_cos': float(np.min(singular_values)), 'max_angle_degrees': float(np.max(angles) * 180.0 / np.pi), 'mean_angle_degrees': float(np.mean(angles) * 180.0 / np.pi)}

def compute_pairwise_subspace_metrics(pca_objects: Dict[Tuple, PCA], k_values: List[int]=[2, 5, 10, 20]) -> pd.DataFrame:
    rows = []
    keys = list(pca_objects.keys())
    for model_key in sorted(set((k[0] for k in keys))):
        for dataset in sorted(set((k[1] for k in keys if k[0] == model_key))):
            for layer_index in sorted(set((k[3] for k in keys if k[0] == model_key and k[1] == dataset))):
                relevant = [k for k in keys if k[0] == model_key and k[1] == dataset and (k[3] == layer_index)]
                for i in range(len(relevant)):
                    for j in range(i + 1, len(relevant)):
                        key_a = relevant[i]
                        key_b = relevant[j]
                        condition_a = key_a[2]
                        condition_b = key_b[2]
                        for k_dim in k_values:
                            result = subspace_alignment_score(pca_objects[key_a], pca_objects[key_b], k_dim)
                            rows.append({'model_key': model_key, 'dataset': dataset, 'layer_index': int(layer_index), 'condition_a': condition_a, 'condition_b': condition_b, **result})
    return pd.DataFrame(rows)
subspace_df = compute_pairwise_subspace_metrics(pca_objects)
subspace_df.to_csv(TABLE_DIR / 'subspace_alignment_metrics.csv', index=False)
print(f"Saved subspace metrics: {TABLE_DIR / 'subspace_alignment_metrics.csv'}")

def compute_condition_separability(activation_df: pd.DataFrame, max_components: int=32) -> pd.DataFrame:
    rows = []
    group_cols = ['model_key', 'model_family', 'dataset', 'layer_index', 'layer_name']
    for group_key, group_df in tqdm(activation_df.groupby(group_cols), desc='Computing condition separability'):
        condition_counts = group_df['condition'].value_counts()
        valid_conditions = condition_counts[condition_counts >= 5].index.tolist()
        local_df = group_df[group_df['condition'].isin(valid_conditions)].copy()
        if local_df['condition'].nunique() < 2 or len(local_df) < 20:
            continue
        X = stack_activations(local_df)
        y = local_df['condition'].astype(str).to_numpy()
        n_components = min(max_components, X.shape[0] - 1, X.shape[1])
        if n_components < 2:
            continue
        pca = PCA(n_components=n_components, random_state=SEED)
        Z = pca.fit_transform(X)
        min_class_count = min(pd.Series(y).value_counts())
        n_splits = min(5, int(min_class_count))
        if n_splits < 2:
            continue
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        scores = cross_val_score(clf, Z, y, cv=cv, scoring='balanced_accuracy')
        row = dict(zip(group_cols, group_key))
        row.update({'n_samples': int(len(local_df)), 'n_conditions': int(local_df['condition'].nunique()), 'pca_components': int(n_components), 'balanced_accuracy_mean': float(np.mean(scores)), 'balanced_accuracy_std': float(np.std(scores))})
        rows.append(row)
    return pd.DataFrame(rows)
separability_df = compute_condition_separability(activation_df)
separability_df.to_csv(TABLE_DIR / 'condition_separability_metrics.csv', index=False)
print(f"Saved separability metrics: {TABLE_DIR / 'condition_separability_metrics.csv'}")

def compute_joint_pca_projection(activation_df: pd.DataFrame, n_components: int=2) -> pd.DataFrame:
    rows = []
    group_cols = ['model_key', 'model_family', 'dataset', 'layer_index', 'layer_name']
    for group_key, group_df in tqdm(activation_df.groupby(group_cols), desc='Computing 2D projections'):
        if len(group_df) < MIN_SAMPLES_FOR_PCA:
            continue
        X = stack_activations(group_df)
        if min(X.shape[0] - 1, X.shape[1]) < n_components:
            continue
        pca = PCA(n_components=n_components, random_state=SEED)
        Z = pca.fit_transform(X)
        for i, (_, row) in enumerate(group_df.reset_index(drop=True).iterrows()):
            rows.append({'model_key': row['model_key'], 'model_family': row['model_family'], 'dataset': row['dataset'], 'layer_index': int(row['layer_index']), 'layer_name': row['layer_name'], 'condition': row['condition'], 'sample_id': row['sample_id'], 'label': row['label'], 'pc1': float(Z[i, 0]), 'pc2': float(Z[i, 1]), 'pc1_var': float(pca.explained_variance_ratio_[0]), 'pc2_var': float(pca.explained_variance_ratio_[1])})
    return pd.DataFrame(rows)
projection_df = compute_joint_pca_projection(activation_df)
projection_df.to_csv(TABLE_DIR / 'joint_pca_2d_projection.csv', index=False)
print(f"Saved 2D projections: {TABLE_DIR / 'joint_pca_2d_projection.csv'}")

def save_current_figure(filename: str):
    path = FIG_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.show()
    print(f'Saved figure: {path}')

def safe_filename(text: str) -> str:
    return re.sub('[^a-zA-Z0-9_.-]+', '_', str(text)).strip('_')

def plot_metric_by_layer(metric_df: pd.DataFrame, metric_name: str, ylabel: str):
    for model_key in sorted(metric_df['model_key'].unique()):
        for dataset in sorted(metric_df['dataset'].unique()):
            local = metric_df[(metric_df['model_key'] == model_key) & (metric_df['dataset'] == dataset)].copy()
            if local.empty:
                continue
            plt.figure(figsize=(8, 5))
            for condition in sorted(local['condition'].unique()):
                sub = local[local['condition'] == condition].sort_values('layer_index')
                plt.plot(sub['layer_index'], sub[metric_name], marker='o', label=condition)
            plt.xlabel('Layer index')
            plt.ylabel(ylabel)
            plt.title(f'{model_key} | {dataset} | {metric_name}')
            plt.legend()
            save_current_figure(f'layer_metric_{safe_filename(model_key)}_{safe_filename(dataset)}_{safe_filename(metric_name)}.pdf')
plot_metric_by_layer(pca_metric_df, metric_name='effective_dimension', ylabel='Effective dimension')
plot_metric_by_layer(pca_metric_df, metric_name='pc90', ylabel='Components needed for 90% variance')
plot_metric_by_layer(pca_metric_df, metric_name='top5_variance_ratio', ylabel='Top-5 explained variance ratio')

def plot_joint_pca_scatter(projection_df: pd.DataFrame):
    for model_key in sorted(projection_df['model_key'].unique()):
        for dataset in sorted(projection_df['dataset'].unique()):
            local_md = projection_df[(projection_df['model_key'] == model_key) & (projection_df['dataset'] == dataset)]
            for layer_index in sorted(local_md['layer_index'].unique()):
                local = local_md[local_md['layer_index'] == layer_index]
                if local.empty:
                    continue
                plt.figure(figsize=(7, 6))
                for condition in sorted(local['condition'].unique()):
                    sub = local[local['condition'] == condition]
                    plt.scatter(sub['pc1'], sub['pc2'], s=28, alpha=0.75, label=condition)
                pc1_var = local['pc1_var'].iloc[0]
                pc2_var = local['pc2_var'].iloc[0]
                plt.xlabel(f'PC1 ({pc1_var:.2%})')
                plt.ylabel(f'PC2 ({pc2_var:.2%})')
                plt.title(f'{model_key} | {dataset} | layer {layer_index}')
                plt.legend()
                save_current_figure(f'joint_pca_scatter_{safe_filename(model_key)}_{safe_filename(dataset)}_layer_{layer_index}.pdf')
plot_joint_pca_scatter(projection_df)

def plot_subspace_alignment_heatmaps(subspace_df: pd.DataFrame, k_dim: int=10):
    local_k = subspace_df[subspace_df['k'] == k_dim].copy()
    if local_k.empty:
        print(f'No subspace rows for k={k_dim}')
        return
    for model_key in sorted(local_k['model_key'].unique()):
        for dataset in sorted(local_k['dataset'].unique()):
            for layer_index in sorted(local_k['layer_index'].unique()):
                local = local_k[(local_k['model_key'] == model_key) & (local_k['dataset'] == dataset) & (local_k['layer_index'] == layer_index)]
                if local.empty:
                    continue
                conditions = sorted(set(local['condition_a']).union(set(local['condition_b'])))
                matrix = pd.DataFrame(np.eye(len(conditions)), index=conditions, columns=conditions)
                for _, row in local.iterrows():
                    a = row['condition_a']
                    b = row['condition_b']
                    score = row['mean_cos2']
                    matrix.loc[a, b] = score
                    matrix.loc[b, a] = score
                plt.figure(figsize=(6, 5))
                plt.imshow(matrix.values, aspect='auto', vmin=0, vmax=1)
                plt.colorbar(label='Mean squared cosine')
                plt.xticks(range(len(conditions)), conditions, rotation=45, ha='right')
                plt.yticks(range(len(conditions)), conditions)
                plt.title(f'{model_key} | {dataset} | layer {layer_index} | k={k_dim}')
                for i in range(len(conditions)):
                    for j in range(len(conditions)):
                        plt.text(j, i, f'{matrix.values[i, j]:.2f}', ha='center', va='center')
                save_current_figure(f'subspace_alignment_{safe_filename(model_key)}_{safe_filename(dataset)}_layer_{layer_index}_k{k_dim}.pdf')
plot_subspace_alignment_heatmaps(subspace_df, k_dim=10)

def plot_condition_separability(separability_df: pd.DataFrame):
    if separability_df.empty:
        print('No separability metrics available.')
        return
    for model_key in sorted(separability_df['model_key'].unique()):
        local_model = separability_df[separability_df['model_key'] == model_key]
        plt.figure(figsize=(9, 5))
        for dataset in sorted(local_model['dataset'].unique()):
            sub = local_model[local_model['dataset'] == dataset].sort_values('layer_index')
            plt.errorbar(sub['layer_index'], sub['balanced_accuracy_mean'], yerr=sub['balanced_accuracy_std'], marker='o', capsize=3, label=dataset)
        plt.xlabel('Layer index')
        plt.ylabel('Balanced accuracy')
        plt.title(f'{model_key} | condition separability from PCA space')
        plt.legend()
        save_current_figure(f'condition_separability_{safe_filename(model_key)}.pdf')
plot_condition_separability(separability_df)

def summarize_results(pca_metric_df: pd.DataFrame, subspace_df: pd.DataFrame, separability_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    summary_tables = {}
    pca_summary = pca_metric_df.groupby(['model_key', 'dataset', 'condition'], as_index=False).agg({'effective_dimension': 'mean', 'pc90': 'mean', 'pc95': 'mean', 'top1_variance_ratio': 'mean', 'top5_variance_ratio': 'mean', 'variance_entropy': 'mean'}).sort_values(['model_key', 'dataset', 'condition'])
    summary_tables['pca_summary'] = pca_summary
    if not subspace_df.empty:
        subspace_summary = subspace_df[subspace_df['k'] == 10].groupby(['model_key', 'dataset', 'condition_a', 'condition_b'], as_index=False).agg({'mean_cos2': 'mean', 'mean_angle_degrees': 'mean', 'max_angle_degrees': 'mean'}).sort_values(['model_key', 'dataset', 'condition_a', 'condition_b'])
    else:
        subspace_summary = pd.DataFrame()
    summary_tables['subspace_summary'] = subspace_summary
    if not separability_df.empty:
        separability_summary = separability_df.groupby(['model_key', 'dataset'], as_index=False).agg({'balanced_accuracy_mean': 'mean', 'balanced_accuracy_std': 'mean'}).sort_values(['model_key', 'dataset'])
    else:
        separability_summary = pd.DataFrame()
    summary_tables['separability_summary'] = separability_summary
    return summary_tables
summary_tables = summarize_results(pca_metric_df, subspace_df, separability_df)
for name, table in summary_tables.items():
    path = TABLE_DIR / f'{name}.csv'
    table.to_csv(path, index=False)
    print(f'Saved: {path}')

def dataframe_to_records(df: pd.DataFrame, max_rows: int=200):
    if df is None or df.empty:
        return []
    return df.head(max_rows).replace({np.nan: None}).to_dict(orient='records')
final_summary = {'configuration': {'max_samples_per_dataset': MAX_SAMPLES_PER_DATASET, 'batch_size': BATCH_SIZE, 'target_layer_fractions': TARGET_LAYER_FRACTIONS, 'max_pca_components': MAX_PCA_COMPONENTS, 'normalize_activations_before_pca': NORMALIZE_ACTIVATIONS_BEFORE_PCA, 'models': MODEL_SPECS, 'datasets': DATASET_SPECS, 'conditions': CONDITIONS}, 'tables': {'pca_summary': dataframe_to_records(summary_tables['pca_summary']), 'subspace_summary': dataframe_to_records(summary_tables['subspace_summary']), 'separability_summary': dataframe_to_records(summary_tables['separability_summary'])}, 'output_files': {'pca_metrics': str(TABLE_DIR / 'pca_metrics.csv'), 'subspace_alignment_metrics': str(TABLE_DIR / 'subspace_alignment_metrics.csv'), 'condition_separability_metrics': str(TABLE_DIR / 'condition_separability_metrics.csv'), 'joint_pca_2d_projection': str(TABLE_DIR / 'joint_pca_2d_projection.csv'), 'figures_directory': str(FIG_DIR)}}
summary_path = PROJECT_DIR / 'compact_summary.json'
with open(summary_path, 'w') as f:
    json.dump(final_summary, f, indent=2)
print(f'Saved compact summary: {summary_path}')
SELECTED_DIR = PROJECT_DIR / 'selected_outputs'
SELECTED_FIG_DIR = SELECTED_DIR / 'figures'
SELECTED_TABLE_DIR = SELECTED_DIR / 'tables'
SELECTED_FIG_DIR.mkdir(parents=True, exist_ok=True)
SELECTED_TABLE_DIR.mkdir(parents=True, exist_ok=True)

def save_selected_fig(name):
    path = SELECTED_FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=350, bbox_inches='tight')
    plt.show()
    print(f'Saved: {path}')

def choose_representative_layer(df):
    layers = sorted(df['layer_index'].unique())
    return layers[int(0.75 * (len(layers) - 1))]
subspace_k10 = subspace_df[subspace_df['k'] == 10].copy()
subspace_compact = subspace_k10.groupby(['model_key', 'dataset'], as_index=False).agg(mean_subspace_alignment=('mean_cos2', 'mean'), mean_subspace_angle=('mean_angle_degrees', 'mean'))
pca_compact = pca_metric_df.groupby(['model_key', 'dataset'], as_index=False).agg(effective_dimension=('effective_dimension', 'mean'), pc90=('pc90', 'mean'), top5_variance_ratio=('top5_variance_ratio', 'mean'), variance_entropy=('variance_entropy', 'mean'))
sep_compact = separability_df.groupby(['model_key', 'dataset'], as_index=False).agg(condition_separability=('balanced_accuracy_mean', 'mean'))
main_table = pca_compact.merge(subspace_compact, on=['model_key', 'dataset'], how='left').merge(sep_compact, on=['model_key', 'dataset'], how='left').sort_values(['model_key', 'dataset'])
main_table.to_csv(SELECTED_TABLE_DIR / 'main_table_model_dataset_summary.csv', index=False)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
metrics = [('effective_dimension', 'Effective dimension'), ('pc90', 'PC90'), ('top5_variance_ratio', 'Top-5 variance ratio'), ('condition_separability', 'Condition separability')]
plot_df = main_table.copy()
for ax, (metric, title) in zip(axes, metrics):
    for model_key in sorted(plot_df['model_key'].unique()):
        sub = plot_df[plot_df['model_key'] == model_key]
        ax.plot(sub['dataset'], sub[metric], marker='o', label=model_key)
    ax.set_title(title)
    ax.set_xlabel('Dataset')
    ax.set_ylabel(title)
    ax.tick_params(axis='x', rotation=30)
    ax.legend(fontsize=8)
fig.suptitle('Representation geometry summary', y=1.02)
save_selected_fig('supplementary_pca_geometry_summary.pdf')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
models = sorted(projection_df['model_key'].unique())
for ax, model_key in zip(axes, models):
    local_model = projection_df[projection_df['model_key'] == model_key]
    dataset = 'pixelprose' if 'pixelprose' in set(local_model['dataset']) else sorted(local_model['dataset'].unique())[0]
    local_dataset = local_model[local_model['dataset'] == dataset]
    layer = choose_representative_layer(local_dataset)
    local = local_dataset[local_dataset['layer_index'] == layer]
    for condition in sorted(local['condition'].unique()):
        sub = local[local['condition'] == condition]
        ax.scatter(sub['pc1'], sub['pc2'], s=16, alpha=0.65, label=condition)
    pc1_var = local['pc1_var'].iloc[0]
    pc2_var = local['pc2_var'].iloc[0]
    ax.set_title(f'{model_key} | {dataset} | layer {layer}')
    ax.set_xlabel(f'PC1 ({pc1_var:.1%})')
    ax.set_ylabel(f'PC2 ({pc2_var:.1%})')
    ax.legend(fontsize=7)
fig.suptitle('Condition geometry in PCA space', y=1.05)
save_selected_fig('supplementary_pca_condition_scatter.pdf')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, model_key in zip(axes, sorted(subspace_k10['model_key'].unique())):
    local_model = subspace_k10[subspace_k10['model_key'] == model_key]
    dataset = 'pixelprose' if 'pixelprose' in set(local_model['dataset']) else sorted(local_model['dataset'].unique())[0]
    local_dataset = local_model[local_model['dataset'] == dataset]
    layer = choose_representative_layer(local_dataset)
    local = local_dataset[local_dataset['layer_index'] == layer]
    conditions = sorted(set(local['condition_a']).union(set(local['condition_b'])))
    matrix = pd.DataFrame(np.eye(len(conditions)), index=conditions, columns=conditions)
    for _, row in local.iterrows():
        matrix.loc[row['condition_a'], row['condition_b']] = row['mean_cos2']
        matrix.loc[row['condition_b'], row['condition_a']] = row['mean_cos2']
    im = ax.imshow(matrix.values, vmin=0, vmax=1, aspect='auto')
    ax.set_title(f'{model_key} | {dataset} | layer {layer}')
    ax.set_xticks(range(len(conditions)))
    ax.set_yticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(conditions, fontsize=7)
    for i in range(len(conditions)):
        for j in range(len(conditions)):
            ax.text(j, i, f'{matrix.values[i, j]:.2f}', ha='center', va='center', fontsize=8)
fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label='Mean squared cosine')
fig.suptitle(' PCA subspace alignment across conditions', y=1.05)
save_selected_fig('supplementary_pca_subspace_alignment.pdf')
condition_table = pca_metric_df.groupby(['model_key', 'dataset', 'condition'], as_index=False).agg(effective_dimension=('effective_dimension', 'mean'), pc90=('pc90', 'mean'), top1_variance_ratio=('top1_variance_ratio', 'mean'), top5_variance_ratio=('top5_variance_ratio', 'mean'), variance_entropy=('variance_entropy', 'mean')).sort_values(['model_key', 'dataset', 'condition'])
condition_table.to_csv(SELECTED_TABLE_DIR / 'supplementary_condition_level_pca_summary.csv', index=False)
print(f'Selected outputs saved to: {SELECTED_DIR.resolve()}')

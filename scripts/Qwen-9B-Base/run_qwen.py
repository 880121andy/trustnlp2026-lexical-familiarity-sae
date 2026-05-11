"""
SAE feature analysis for Qwen3.5-9B-Base with the Top-K residual SAE
(sae_lens release: qwen-scope-3.5-9b-base-w64k-l100, K=100, width 65,536,
HF: Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100).

Differences from the original Gemma script:
  - Loads Qwen/Qwen3.5-9B (HF token not required).
  - Layer path is model.model.layers.
  - Iterates over ALL datasets in a single run so the model is loaded once.

"""

# =============================================================================
# 1. IMPORTS AND SETUP
# =============================================================================

import argparse
import ast
import gc
import json
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from sae_lens import SAE
from scipy import stats
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

if os.environ.get('DISPLAY') is None and os.name != 'nt':
    matplotlib.use('Agg')

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

@dataclass
class SlangPair:
    slang_text: str
    literal_text: str


@dataclass
class SlangPairWithPosition:
    slang_text: str
    literal_text: str
    slang_positions: List[Tuple[int, int]]
    literal_positions: List[Tuple[int, int]]
    slang_segments: str
    literal_segments: str


# =============================================================================
# 3. DATASET LOADERS  (identical to scripts/run.py)
# =============================================================================

def load_semantic_shift_pairs(df: pd.DataFrame) -> List[SlangPairWithPosition]:
    pairs = []
    for _, row in df.iterrows():
        try:
            pairs.append(SlangPairWithPosition(
                slang_text=str(row['gen_z']),
                literal_text=str(row['normal']),
                slang_positions=ast.literal_eval(row['slang_positions']),
                literal_positions=ast.literal_eval(row['literal_positions']),
                slang_segments=str(row.get('slang_segments', '')),
                literal_segments=str(row.get('literal_segments', '')),
            ))
        except Exception:
            continue
    return pairs


def load_neologism_pairs(df: pd.DataFrame) -> List[SlangPairWithPosition]:
    pairs = []
    for _, row in df.iterrows():
        try:
            pairs.append(SlangPairWithPosition(
                slang_text=str(row['gen_z']),
                literal_text=str(row['normal']),
                slang_positions=ast.literal_eval(row['slang_positions']),
                literal_positions=ast.literal_eval(row['literal_positions']),
                slang_segments=str(row.get('slang_word', row.get('term', ''))),
                literal_segments=str(row.get('literal_replacement', '')),
            ))
        except Exception:
            continue
    return pairs


def load_metaphor_pairs(df: pd.DataFrame) -> List[SlangPairWithPosition]:
    pairs = []
    for _, row in df.iterrows():
        try:
            pairs.append(SlangPairWithPosition(
                slang_text=str(row['metaphorical']),
                literal_text=str(row['normal']),
                slang_positions=ast.literal_eval(row['metaphor_positions']),
                literal_positions=ast.literal_eval(row['literal_positions']),
                slang_segments=str(row.get('metaphor_segments', '')),
                literal_segments=str(row.get('literal_segments', '')),
            ))
        except Exception:
            continue
    return pairs


def load_idiom_pairs(df: pd.DataFrame) -> List[SlangPairWithPosition]:
    pairs = []
    for _, row in df.iterrows():
        try:
            pairs.append(SlangPairWithPosition(
                slang_text=str(row['idiomatic']),
                literal_text=str(row['normal']),
                slang_positions=ast.literal_eval(row['idiom_positions']),
                literal_positions=ast.literal_eval(row['literal_positions']),
                slang_segments=str(row.get('idiom_segments', '')),
                literal_segments=str(row.get('literal_segments', '')),
            ))
        except Exception:
            continue
    return pairs


def load_construction_pairs(df: pd.DataFrame) -> List[SlangPairWithPosition]:
    pairs = []
    for _, row in df.iterrows():
        try:
            pairs.append(SlangPairWithPosition(
                slang_text=str(row['gen_z']),
                literal_text=str(row['normal']),
                slang_positions=ast.literal_eval(row['slang_positions']),
                literal_positions=ast.literal_eval(row['literal_positions']),
                slang_segments=str(row.get('slang_in_sentence', row.get('construction', ''))),
                literal_segments=str(row.get('literal_replacement', '')),
            ))
        except Exception:
            continue
    return pairs


def load_literal_paraphrase_pairs(df: pd.DataFrame) -> List[SlangPairWithPosition]:
    pairs = []
    for _, row in df.iterrows():
        try:
            pairs.append(SlangPairWithPosition(
                slang_text=str(row['paraphrase']),
                literal_text=str(row['normal']),
                slang_positions=ast.literal_eval(row['paraphrase_positions']),
                literal_positions=ast.literal_eval(row['literal_positions']),
                slang_segments=str(row.get('paraphrase_segments', '')),
                literal_segments=str(row.get('literal_segments', '')),
            ))
        except Exception:
            continue
    return pairs


def load_identical_pairs(df: pd.DataFrame) -> List[SlangPairWithPosition]:
    pairs = []
    for _, row in df.iterrows():
        try:
            pairs.append(SlangPairWithPosition(
                slang_text=str(row['identical']),
                literal_text=str(row['normal']),
                slang_positions=ast.literal_eval(row['identical_positions']),
                literal_positions=ast.literal_eval(row['literal_positions']),
                slang_segments=str(row.get('identical_segments', '')),
                literal_segments=str(row.get('literal_segments', '')),
            ))
        except Exception:
            continue
    return pairs


# =============================================================================
# 4. MODEL LOADING
# =============================================================================

def load_model(model_id: str = "Qwen/Qwen3.5-9B"):
    """Load Qwen base model and tokenizer."""
    print(f"Loading {model_id} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    print(f"\nModel loaded!")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden size: {hidden_size}")
    if torch.cuda.is_available():
        print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, tokenizer, num_layers, hidden_size


# =============================================================================
# 5. ACTIVATION CACHE
# =============================================================================

class ActivationCache:
    """Captures layer activations via forward hooks (Qwen layer path)."""

    def __init__(self, model):
        self.model = model
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks: List = []

    def _hook(self, layer_idx: int):
        def fn(module, input, output):
            if isinstance(output, tuple):
                self.activations[layer_idx] = output[0].detach()
            else:
                self.activations[layer_idx] = output.detach()
        return fn

    def _get_layers(self):
        return self.model.model.layers

    def register(self, layers: Optional[List[int]] = None):
        self.clear()
        model_layers = self._get_layers()
        layers = layers or list(range(len(model_layers)))
        for idx in layers:
            hook = model_layers[idx].register_forward_hook(self._hook(idx))
            self.hooks.append(hook)
        return self

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}

    def get(self) -> Dict[int, torch.Tensor]:
        return self.activations


# =============================================================================
# 6. SAE LOADING UTILITIES
# =============================================================================

DEFAULT_SAE_RELEASE = "qwen-scope-3.5-9b-base-w64k-l100"


def qwen_sae_id(layer: int) -> str:
    """sae_id format used by the Qwen-scope release (differs from Gemma)."""
    return f"layer{layer}"


def load_sae(layer: int, release: str = DEFAULT_SAE_RELEASE) -> Optional[SAE]:
    """Load a single Qwen-scope SAE for the given layer via sae_lens."""
    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=release,
            sae_id=qwen_sae_id(layer),
        )
        return sae.to(device)
    except Exception as e:
        print(f"Could not load SAE for layer {layer}: {e}")
        return None


def load_all_saes(num_layers: int, release: str = DEFAULT_SAE_RELEASE) -> Dict[int, SAE]:
    """Load SAEs for all layers (only used by extract_sae_features / steering)."""
    saes: Dict[int, SAE] = {}
    print(f"Loading SAEs from release '{release}' for {num_layers} layers...")
    for layer in tqdm(range(num_layers), desc="Loading SAEs"):
        sae = load_sae(layer, release)
        if sae is not None:
            saes[layer] = sae
    print(f"\nLoaded {len(saes)}/{num_layers} SAEs")
    if saes:
        sample = saes[next(iter(saes))]
        print(f"  d_in: {sample.cfg.d_in}")
        print(f"  d_sae: {sample.cfg.d_sae}")
    return saes


# =============================================================================
# 7. TOKEN POSITION MAPPING 
# =============================================================================

def word_to_token_position(text: str, word_positions: List[Tuple[int, int]], tokenizer) -> List[int]:
    words = text.split()
    word_spans = []
    current_char = 0
    for word in words:
        start = text.find(word, current_char)
        if start == -1:
            start = current_char
        end = start + len(word)
        word_spans.append((start, end))
        current_char = end

    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding['offset_mapping']

    target_token_indices = set()
    for start_word_idx, end_word_idx in word_positions:
        if start_word_idx >= len(word_spans):
            continue
        valid_end_idx = min(end_word_idx, len(word_spans))
        if start_word_idx >= valid_end_idx:
            continue
        char_start = word_spans[start_word_idx][0]
        char_end = word_spans[valid_end_idx - 1][1]
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue
            if (tok_start >= char_start and tok_start < char_end) or \
               (tok_end > char_start and tok_end <= char_end):
                target_token_indices.add(i)

    return sorted(list(target_token_indices))


def get_first_diff_token_position(text: str, word_positions: List[Tuple[int, int]], tokenizer) -> int:
    if not word_positions:
        tokens = tokenizer(text, return_tensors="pt")['input_ids']
        return tokens.shape[1] // 2
    token_indices = word_to_token_position(text, word_positions[:1], tokenizer)
    if token_indices:
        return min(token_indices)
    tokens = tokenizer(text, return_tensors="pt")['input_ids']
    return tokens.shape[1] // 2


# =============================================================================
# 8. SAE FEATURE EXTRACTION
# =============================================================================

def extract_sae_features(text: str, saes: Dict[int, SAE], model, tokenizer,
                         pooling: str = 'last') -> Dict[int, torch.Tensor]:
    layers = list(saes.keys())
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    cache = ActivationCache(model)
    cache.register(layers)
    with torch.no_grad():
        model(**inputs)

    result = {}
    for layer_idx, acts in cache.get().items():
        acts = acts.squeeze(0)
        if pooling == 'last':
            pooled = acts[-1]
        elif pooling == 'mean':
            pooled = acts.mean(dim=0)
        else:
            pooled = acts.max(dim=0)[0]

        sae = saes[layer_idx]
        features = sae.encode(pooled.unsqueeze(0).to(sae.W_enc.dtype))
        result[layer_idx] = features.squeeze(0).cpu()

    cache.clear()
    return result


# =============================================================================
# 9. TOKEN-LEVEL ANALYSIS (SINGLE LAYER)
# =============================================================================

def analyze_pairs_single_layer_token_level(
    pairs: List[SlangPairWithPosition],
    layer: int,
    model,
    tokenizer,
    sae_release: str = DEFAULT_SAE_RELEASE,
    max_pairs: Optional[int] = None,
    pooling: str = "first_diff",
) -> pd.DataFrame:
    if max_pairs:
        pairs = pairs[:max_pairs]

    print(f"\nLoading SAE for layer {layer}...")
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=qwen_sae_id(layer))
    sae = sae.to(device)
    print(f"  Loaded (d_sae={sae.cfg.d_sae})")

    metrics = {
        'l1_dist': [], 'l2_dist': [], 'cosine_dist': [],
        'slang_only_features': [], 'literal_only_features': [], 'shared_features': [],
        'slang_active': [], 'literal_active': []
    }

    skipped = 0
    for pair in tqdm(pairs, desc=f"Layer {layer}"):
        try:
            # === SLANG ===
            slang_inputs = tokenizer(pair.slang_text, return_tensors="pt",
                                     truncation=True, max_length=512)
            slang_inputs = {k_: v.to(device) for k_, v in slang_inputs.items()}
            seq_len_slang = slang_inputs['input_ids'].shape[1]

            cache = ActivationCache(model)
            cache.register([layer])
            with torch.no_grad():
                model(**slang_inputs)
            slang_acts_full = cache.get()[layer].squeeze(0)
            cache.clear()

            if pooling == "first_diff":
                pos = get_first_diff_token_position(pair.slang_text, pair.slang_positions, tokenizer)
                pos = min(pos, seq_len_slang - 1)
                slang_acts = slang_acts_full[pos]
            elif pooling == "all_diff":
                token_indices = word_to_token_position(pair.slang_text, pair.slang_positions, tokenizer)
                if token_indices:
                    token_indices = [min(t, seq_len_slang - 1) for t in token_indices]
                    slang_acts = slang_acts_full[token_indices].mean(dim=0)
                else:
                    slang_acts = slang_acts_full[-1]
            else:
                slang_acts = slang_acts_full[-1]

            sf = sae.encode(slang_acts.unsqueeze(0).to(sae.W_enc.dtype)).squeeze(0).cpu()

            # === LITERAL ===
            literal_inputs = tokenizer(pair.literal_text, return_tensors="pt",
                                       truncation=True, max_length=512)
            literal_inputs = {k_: v.to(device) for k_, v in literal_inputs.items()}
            seq_len_literal = literal_inputs['input_ids'].shape[1]

            cache = ActivationCache(model)
            cache.register([layer])
            with torch.no_grad():
                model(**literal_inputs)
            literal_acts_full = cache.get()[layer].squeeze(0)
            cache.clear()

            if pooling == "first_diff":
                pos = get_first_diff_token_position(pair.literal_text, pair.literal_positions, tokenizer)
                pos = min(pos, seq_len_literal - 1)
                literal_acts = literal_acts_full[pos]
            elif pooling == "all_diff":
                token_indices = word_to_token_position(pair.literal_text, pair.literal_positions, tokenizer)
                if token_indices:
                    token_indices = [min(t, seq_len_literal - 1) for t in token_indices]
                    literal_acts = literal_acts_full[token_indices].mean(dim=0)
                else:
                    literal_acts = literal_acts_full[-1]
            else:
                literal_acts = literal_acts_full[-1]

            lf = sae.encode(literal_acts.unsqueeze(0).to(sae.W_enc.dtype)).squeeze(0).cpu()

            # === METRICS ===
            metrics['l1_dist'].append(torch.norm(sf - lf, p=1).item())
            metrics['l2_dist'].append(torch.norm(sf - lf, p=2).item())

            if sf.norm() > 0 and lf.norm() > 0:
                cos = 1 - F.cosine_similarity(sf.unsqueeze(0), lf.unsqueeze(0)).item()
            else:
                cos = 1.0
            metrics['cosine_dist'].append(cos)

            slang_active = sf > 0
            literal_active = lf > 0
            metrics['slang_only_features'].append((slang_active & ~literal_active).sum().item())
            metrics['literal_only_features'].append((~slang_active & literal_active).sum().item())
            metrics['shared_features'].append((slang_active & literal_active).sum().item())
            metrics['slang_active'].append(slang_active.sum().item())
            metrics['literal_active'].append(literal_active.sum().item())

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"Error: {e}")
            continue

    del sae
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Layer {layer} complete, SAE freed (skipped {skipped} pairs)")

    rows = []
    for metric_name, values in metrics.items():
        if len(values) > 0:
            rows.append({
                'layer': layer,
                'metric': metric_name,
                'mean': np.mean(values),
                'std': np.std(values),
                'n': len(values),
            })
    return pd.DataFrame(rows)


# =============================================================================
# 10. FULL LAYER ANALYSIS
# =============================================================================

def analyze_all_layers_sequential_token_level(
    pairs: List[SlangPairWithPosition],
    model,
    tokenizer,
    sae_release: str,
    num_layers: int,
    start_layer: int = 0,
    max_pairs: Optional[int] = None,
    pooling: str = "all_diff",
    checkpoint_dir: Path = Path("."),
    checkpoint_prefix: str = "checkpoint",
    resume: bool = True,
) -> pd.DataFrame:
    checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_prefix}_progress.csv"

    all_results: List[pd.DataFrame] = []
    actual_start = start_layer

    if resume and checkpoint_path.exists():
        try:
            prior_df = pd.read_csv(checkpoint_path)
        except Exception as e:
            print(f"  [resume] Failed to read {checkpoint_path}: {e}; starting fresh.")
            prior_df = pd.DataFrame()
        if not prior_df.empty and 'layer' in prior_df.columns:
            done_layers = set(prior_df['layer'].astype(int).unique())
            next_layer = start_layer
            while next_layer in done_layers and next_layer < num_layers:
                next_layer += 1
            actual_start = next_layer
            kept = prior_df[(prior_df['layer'] >= start_layer) &
                            (prior_df['layer'] < actual_start)]
            if not kept.empty:
                all_results.append(kept)
                print(f"  [resume] Loaded layers {start_layer}-{actual_start - 1} "
                      f"from {checkpoint_path.name} ({len(kept)} rows); "
                      f"resuming at layer {actual_start}")

    if actual_start >= num_layers:
        print(f"  [resume] All {num_layers} layers already complete; skipping compute.")
        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

    for layer in range(actual_start, num_layers):
        layer_df = analyze_pairs_single_layer_token_level(
            pairs, layer, model, tokenizer,
            sae_release=sae_release, max_pairs=max_pairs, pooling=pooling,
        )
        all_results.append(layer_df)

        checkpoint_df = pd.concat(all_results, ignore_index=True)
        tmp_path = checkpoint_path.with_name(checkpoint_path.name + '.tmp')
        checkpoint_df.to_csv(tmp_path, index=False)
        tmp_path.replace(checkpoint_path)

        if (layer + 1) % 10 == 0 or layer == num_layers - 1:
            print(f"  Checkpoint saved: {checkpoint_path} (through layer {layer})")

    return pd.concat(all_results, ignore_index=True)


# =============================================================================
# 11. VISUALIZATION  (identical logic to scripts/run.py)
# =============================================================================

def plot_layer_differences(df: pd.DataFrame, metric: str = 'l2_dist',
                           title_prefix: str = "", save_path: Optional[Path] = None):
    metric_df = df[df['metric'] == metric].sort_values('layer')
    if len(metric_df) == 0:
        print(f"No data for metric: {metric}")
        return None

    num_layers = int(metric_df['layer'].max()) + 1
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.fill_between(metric_df['layer'],
                    metric_df['mean'] - metric_df['std'],
                    metric_df['mean'] + metric_df['std'],
                    alpha=0.3, color='#e74c3c')
    ax.plot(metric_df['layer'], metric_df['mean'], 'o-',
            color='#e74c3c', linewidth=2, markersize=5)

    peak_idx = metric_df['mean'].idxmax()
    peak = metric_df.loc[peak_idx]
    ax.axvline(x=peak['layer'], color='gray', linestyle='--', alpha=0.7)
    ax.annotate(
        f"Peak: Layer {int(peak['layer'])} ({100 * peak['layer'] / (num_layers - 1):.0f}% depth)",
        xy=(peak['layer'], peak['mean']), xytext=(15, 15),
        textcoords='offset points', fontsize=11, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='gray'),
    )
    ax.axvspan(0, num_layers * 0.33, alpha=0.08, color='blue', label='Early (0-33%)')
    ax.axvspan(num_layers * 0.33, num_layers * 0.66, alpha=0.08, color='green', label='Middle (33-66%)')
    ax.axvspan(num_layers * 0.66, num_layers, alpha=0.08, color='red', label='Deep (66-100%)')
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=14)
    ax.set_title(
        f'{title_prefix}TOKEN-LEVEL SAE Feature Differences (Qwen3.5-9B)\n'
        f'Comparing activations at SPECIFIC differing token positions',
        fontsize=16, fontweight='bold',
    )
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, num_layers)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return peak


def plot_all_metrics(df: pd.DataFrame, save_path: Optional[Path] = None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = ['l2_dist', 'cosine_dist', 'slang_only_features', 'literal_only_features']
    titles = ['L2 Distance', 'Cosine Distance', 'Slang-Only Features', 'Literal-Only Features']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    for ax, metric, title, color in zip(axes.flat, metrics, titles, colors):
        metric_df = df[df['metric'] == metric].sort_values('layer')
        if len(metric_df) == 0:
            ax.set_title(f"{title}\nNo data", fontsize=12)
            continue
        num_layers = int(metric_df['layer'].max()) + 1
        ax.fill_between(metric_df['layer'],
                        metric_df['mean'] - metric_df['std'],
                        metric_df['mean'] + metric_df['std'],
                        alpha=0.3, color=color)
        ax.plot(metric_df['layer'], metric_df['mean'], 'o-',
                color=color, linewidth=2, markersize=3)
        peak_idx = metric_df['mean'].idxmax()
        peak = metric_df.loc[peak_idx]
        ax.axvline(x=peak['layer'], color='gray', linestyle='--', alpha=0.5)
        ax.set_title(
            f"{title}\nPeak: Layer {int(peak['layer'])} "
            f"({100 * peak['layer'] / (num_layers - 1):.0f}%)",
            fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer')
        ax.grid(True, alpha=0.3)

    plt.suptitle('TOKEN-LEVEL SAE Feature Analysis (Qwen3.5-9B)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_feature_activation_comparison(df: pd.DataFrame, save_path: Optional[Path] = None):
    fig, ax = plt.subplots(figsize=(16, 6))
    slang_df = df[df['metric'] == 'slang_active'].sort_values('layer')
    literal_df = df[df['metric'] == 'literal_active'].sort_values('layer')
    shared_df = df[df['metric'] == 'shared_features'].sort_values('layer')

    if len(slang_df) == 0:
        print("No activation data available")
        return

    ax.plot(slang_df['layer'], slang_df['mean'], 'o-', label='Slang Active',
            linewidth=2, markersize=4)
    ax.plot(literal_df['layer'], literal_df['mean'], 's-', label='Literal Active',
            linewidth=2, markersize=4)
    ax.plot(shared_df['layer'], shared_df['mean'], '^-', label='Shared',
            linewidth=2, markersize=4)
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Mean # Active Features', fontsize=14)
    ax.set_title('Feature Activation Patterns: Slang vs Literal (Qwen3.5-9B)',
                 fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


# =============================================================================
# 12. STATISTICAL ANALYSIS
# =============================================================================

def analyze_results(df: pd.DataFrame, metric: str = 'l2_dist', dataset_name: str = ""):
    metric_df = df[df['metric'] == metric].sort_values('layer')
    if len(metric_df) == 0:
        print(f"No data for metric: {metric}")
        return None

    num_layers = int(metric_df['layer'].max()) + 1
    early_end = int(num_layers * 0.33)
    middle_end = int(num_layers * 0.66)
    early = list(range(0, early_end))
    middle = list(range(early_end, middle_end))
    deep = list(range(middle_end, num_layers))

    print("=" * 70)
    print(f"TOKEN-LEVEL SAE FEATURE ANALYSIS RESULTS  ({dataset_name})")
    print("=" * 70)
    print(f"\nModel: Qwen3.5-9B-Base")
    print(f"SAE: Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100 (Top-K, K=100)")
    print(f"Dataset: {int(metric_df['n'].iloc[0])} pairs")
    print(f"Layers analyzed: {num_layers}")

    peak_idx = metric_df['mean'].idxmax()
    peak = metric_df.loc[peak_idx]
    peak_layer = int(peak['layer'])
    peak_pct = peak_layer / (num_layers - 1) * 100

    print(f"\nPeak Layer: {peak_layer} / {num_layers - 1} ({peak_pct:.1f}% depth)")
    print(f"Peak Value: {peak['mean']:.4f} +/- {peak['std']:.4f}")

    top5 = metric_df.nlargest(5, 'mean')
    print(f"Top 5 Layers: {top5['layer'].astype(int).tolist()}")

    early_df = metric_df[metric_df['layer'].isin(early)]
    middle_df = metric_df[metric_df['layer'].isin(middle)]
    deep_df = metric_df[metric_df['layer'].isin(deep)]

    early_mean = early_df['mean'].mean() if len(early_df) > 0 else np.nan
    middle_mean = middle_df['mean'].mean() if len(middle_df) > 0 else np.nan
    deep_mean = deep_df['mean'].mean() if len(deep_df) > 0 else np.nan

    print(f"  Early  (0-{early_end - 1}):  {early_mean:.4f}")
    print(f"  Middle ({early_end}-{middle_end - 1}): {middle_mean:.4f}")
    print(f"  Deep   ({middle_end}-{num_layers - 1}): {deep_mean:.4f}")

    results_dict = {
        'dataset': dataset_name,
        'peak_layer': peak_layer,
        'peak_pct': float(peak_pct),
        'peak_value': float(peak['mean']),
        'early_mean': float(early_mean) if not np.isnan(early_mean) else None,
        'middle_mean': float(middle_mean) if not np.isnan(middle_mean) else None,
        'deep_mean': float(deep_mean) if not np.isnan(deep_mean) else None,
    }

    early_vals = early_df['mean'].values
    middle_vals = middle_df['mean'].values
    deep_vals = deep_df['mean'].values
    if len(early_vals) > 1 and len(middle_vals) > 1 and len(deep_vals) > 1:
        f_stat, p_val = stats.f_oneway(early_vals, middle_vals, deep_vals)
        print(f"  ANOVA: F={f_stat:.2f}, p={p_val:.6f}")
        results_dict['anova_f'] = float(f_stat)
        results_dict['anova_p'] = float(p_val)

    return results_dict


# =============================================================================
# 13. ACTIVATION STEERING (kept for completeness; not invoked by main())
# =============================================================================

class ActivationSteering:
    """Steer model by adding direction vector at specific layer."""

    def __init__(self, model, tokenizer, saes: Dict[int, SAE]):
        self.model = model
        self.tokenizer = tokenizer
        self.saes = saes
        self.steering_vector: Optional[torch.Tensor] = None
        self.target_layer: Optional[int] = None
        self.strength: float = 1.0
        self.hook = None

    def compute_steering_vector(self, pairs: List[SlangPair], layer: int, n_samples: int = 50):
        slang_feats = []
        literal_feats = []
        for pair in tqdm(pairs[:n_samples], desc="Computing steering vector"):
            sf = extract_sae_features(pair.slang_text, self.saes, self.model, self.tokenizer)[layer]
            lf = extract_sae_features(pair.literal_text, self.saes, self.model, self.tokenizer)[layer]
            slang_feats.append(sf)
            literal_feats.append(lf)
        mean_slang = torch.stack(slang_feats).mean(dim=0)
        mean_literal = torch.stack(literal_feats).mean(dim=0)
        feature_diff = mean_slang - mean_literal

        sae = self.saes[layer]
        self.steering_vector = sae.decode(feature_diff.unsqueeze(0).to(device)).squeeze(0)
        self.target_layer = layer
        print(f"Steering vector computed for layer {layer} (norm={self.steering_vector.norm().item():.4f})")

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            h[:, -1, :] += self.strength * self.steering_vector.to(h.dtype)
            return (h,) + output[1:]
        else:
            output[:, -1, :] += self.strength * self.steering_vector.to(output.dtype)
            return output

    def generate(self, prompt: str, strength: float = 1.0, max_tokens: int = 50) -> str:
        self.strength = strength
        layers = self.model.model.layers
        self.hook = layers[self.target_layer].register_forward_hook(self._hook_fn)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.hook.remove()
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# =============================================================================
# 14. MAIN EXECUTION
# =============================================================================

DATASET_CONFIGS = {
    'idiom':              {'file': 'idiom_baseline.csv',             'loader': load_idiom_pairs},
    'metaphor':           {'file': 'metaphor_baseline.csv',          'loader': load_metaphor_pairs},
    'semantic_shift':     {'file': 'genz_dataset_tagged.csv',        'loader': load_semantic_shift_pairs},
    'neologism':          {'file': 'neologism_tagged.csv',           'loader': load_neologism_pairs},
    'construction':       {'file': 'constructions_tagged.csv',       'loader': load_construction_pairs},
    'literal_paraphrase': {'file': 'literal_paraphrase_baseline.csv','loader': load_literal_paraphrase_pairs},
    'identical':          {'file': 'identical_baseline.csv',         'loader': load_identical_pairs},
}


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run Qwen SAE analysis across all datasets.")
    parser.add_argument('--sae-release', default=DEFAULT_SAE_RELEASE,
                        help='sae_lens release name (default: %(default)s). The L=50 sibling '
                             'release "qwen-scope-3.5-9b-base-w64k-l50" can be used as a '
                             'sparsity-ablation control.')
    parser.add_argument('--data-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/data'),
                        help='Directory containing the *_baseline.csv / *_tagged.csv files.')
    parser.add_argument('--output-dir', type=Path, default=Path('/home/tyleryeh47/sae_slang/results/qwen'),
                        help='Where to write CSVs / plots / analysis JSON.')
    parser.add_argument('--model-id', default='Qwen/Qwen3.5-9B',
                        help='Hugging Face model id of the Qwen base model.')
    parser.add_argument('--datasets', nargs='+', default=['all'],
                        choices=['all'] + list(DATASET_CONFIGS.keys()),
                        help='Subset of datasets to run (default: all).')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Cap pairs per dataset (smoke-test only).')
    parser.add_argument('--pooling', default='all_diff', choices=['first_diff', 'all_diff', 'last'],
                        help='Token pooling strategy at the divergence span.')
    parser.add_argument('--start-layer', type=int, default=0)
    parser.add_argument('--end-layer', type=int, default=None,
                        help='Exclusive upper bound on layers; defaults to num_hidden_layers.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                        help='Resume from {prefix}_progress.csv if present (default: enabled). '
                             'Use --no-resume to force a fresh run.')
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Qwen3.5-9B SAE feature analysis (camera-ready cross-model run)")
    print("=" * 70)
    print(f"  Model:        {args.model_id}")
    print(f"  SAE release:  {args.sae_release}")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Out dir:      {args.output_dir}")

    model, tokenizer, num_layers, hidden_size = load_model(args.model_id)

    end_layer = args.end_layer if args.end_layer is not None else num_layers

    target_datasets = list(DATASET_CONFIGS.keys()) if args.datasets == ['all'] else args.datasets

    all_summaries = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for ds_name in target_datasets:
        cfg = DATASET_CONFIGS[ds_name]
        ds_path = args.data_dir / cfg['file']
        if not ds_path.exists():
            print(f"\n[skip] {ds_name}: {ds_path} not found")
            continue

        print("\n" + "#" * 70)
        print(f"### DATASET: {ds_name}  ({ds_path.name})")
        print("#" * 70)

        df_ds = pd.read_csv(ds_path)
        pairs = cfg['loader'](df_ds)
        print(f"Loaded {len(pairs)} pairs")
        if len(pairs) == 0:
            print(f"[skip] {ds_name}: no pairs parsed")
            continue

        results_df = analyze_all_layers_sequential_token_level(
            pairs, model, tokenizer,
            sae_release=args.sae_release,
            num_layers=end_layer,
            start_layer=args.start_layer,
            max_pairs=args.max_pairs,
            pooling=args.pooling,
            checkpoint_dir=args.output_dir,
            checkpoint_prefix=f"checkpoint_{ds_name}_qwen",
            resume=args.resume,
        )

        results_csv = args.output_dir / f"{ds_name}_qwen_{timestamp}.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"\nSaved results: {results_csv}")

        plot_layer_differences(
            results_df, 'l2_dist',
            title_prefix=f"[{ds_name}] ",
            save_path=args.output_dir / f"{ds_name}_qwen_l2_{timestamp}.png",
        )
        plot_all_metrics(
            results_df,
            save_path=args.output_dir / f"{ds_name}_qwen_metrics_{timestamp}.png",
        )
        plot_feature_activation_comparison(
            results_df,
            save_path=args.output_dir / f"{ds_name}_qwen_activations_{timestamp}.png",
        )

        summary = analyze_results(results_df, metric='l2_dist', dataset_name=ds_name)
        cosine_summary = analyze_results(results_df, metric='cosine_dist', dataset_name=ds_name)
        all_summaries[ds_name] = {
            'l2_dist': summary,
            'cosine_dist': cosine_summary,
        }

    summary_path = args.output_dir / f"analysis_summary_qwen_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll datasets done. Summary: {summary_path}")


if __name__ == "__main__":
    main()

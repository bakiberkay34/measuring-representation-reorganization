from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def canonical_family(value):
    value = str(value).strip().lower()
    if value.startswith('pali'):
        return 'pali'
    if value.startswith('qwen'):
        return 'qwen'
    if value.startswith('smol'):
        return 'smol'
    return value


def canonical_dataset(value):
    value = str(value).strip().lower().replace('-', '').replace('_', '')
    if value in {'coco', 'cocokarpathy'}:
        return 'coco'
    if value == 'flickr30k':
        return 'flickr30k'
    if value == 'pixelprose':
        return 'pixelprose'
    return value


def zscore(values):
    values = np.asarray(values, dtype=float)
    std = values.std(ddof=0)
    if std == 0:
        raise ValueError('Cannot z-score a constant array.')
    return (values - values.mean()) / std


def normalized_auc(frame, value):
    rows = []
    for keys, group in frame.groupby(['family', 'dataset'], sort=True):
        group = group.sort_values('depth_fraction')
        x = group['depth_fraction'].to_numpy(float)
        y = group[value].to_numpy(float)
        span = x[-1] - x[0]
        if span <= 0:
            raise ValueError(f'Invalid depth span for {keys}.')
        rows.append({'family': keys[0], 'dataset': keys[1], value: float(np.trapz(y, x) / span)})
    return pd.DataFrame(rows)


def final_depth(frame, value):
    idx = frame.groupby(['family', 'dataset'])['depth_fraction'].idxmax()
    return frame.loc[idx, ['family', 'dataset', value]].reset_index(drop=True)


def spearman_rho(x, y):
    rx = rankdata(np.asarray(x, dtype=float), method='average')
    ry = rankdata(np.asarray(y, dtype=float), method='average')
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.dot(rx, ry) / denom)


def exact_permutation_p(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = abs(spearman_rho(x, y))
    count = 0
    total = 0
    for perm in itertools.permutations(y.tolist()):
        total += 1
        if abs(spearman_rho(x, perm)) >= observed - 1e-12:
            count += 1
    return count / total


def restricted_permutation_p(x, y, strata):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    strata = np.asarray(strata)
    observed = abs(spearman_rho(x, y))
    groups = [np.where(strata == level)[0] for level in pd.unique(strata)]
    group_permutations = [list(itertools.permutations(indices.tolist())) for indices in groups]
    count = 0
    total = 0
    for combination in itertools.product(*group_permutations):
        permuted = y.copy()
        for original, permuted_indices in zip(groups, combination):
            permuted[original] = y[np.asarray(permuted_indices)]
        total += 1
        if abs(spearman_rho(x, permuted)) >= observed - 1e-12:
            count += 1
    return count / total


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for position, index in enumerate(order):
        candidate = min(1.0, (m - position) * p[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def sign_flip_p(differences):
    d = np.asarray(differences, dtype=float)
    observed = abs(d.mean())
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(d)):
        total += 1
        if abs(np.mean(d * np.asarray(signs))) >= observed - 1e-12:
            count += 1
    return count / total


def load_probe_grid(tables_dir, pca_path):
    sparse = pd.read_csv(tables_dir / 'Appendix_A1_full_sparse_grid_numeric.csv')
    neigh = pd.read_csv(tables_dir / 'Appendix_B1_full_neighborhood_grid_K5_numeric.csv')
    pca = pd.read_csv(pca_path)
    sparse = sparse[['family', 'dataset', 'mfi']].copy()
    neigh = neigh[['family', 'dataset', 'lps_image_conditioned', 'nri_image_conditioned']].copy()
    pca = pca[['model', 'dataset', 'separability']].rename(columns={'model': 'family'})
    for df in (sparse, neigh, pca):
        df['family'] = df['family'].map(canonical_family)
        df['dataset'] = df['dataset'].map(canonical_dataset)
    probes = sparse.merge(neigh, on=['family', 'dataset'], validate='one_to_one').merge(pca, on=['family', 'dataset'], validate='one_to_one')
    probes['neighborhood_axis'] = 0.5 * (zscore(probes['lps_image_conditioned']) + zscore(-probes['nri_image_conditioned']))
    return probes.sort_values(['family', 'dataset']).reset_index(drop=True)


def normalize_result_keys(df):
    df = df.copy()
    df['family'] = df['family'].map(canonical_family)
    if 'dataset' in df.columns:
        df['dataset'] = df['dataset'].map(canonical_dataset)
    if 'source_dataset' in df.columns:
        df['source_dataset'] = df['source_dataset'].map(canonical_dataset)
    if 'target_dataset' in df.columns:
        df['target_dataset'] = df['target_dataset'].map(canonical_dataset)
    return df


def baseline_auc_within(retrieval):
    rows = []
    for baseline, group in retrieval.groupby('baseline'):
        frame = normalized_auc(group[['family', 'dataset', 'depth_fraction', 'mrr']], 'mrr')
        frame['baseline'] = baseline
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def baseline_auc_cross(cross):
    averaged = cross.groupby(['family', 'source_dataset', 'depth_fraction', 'baseline'], as_index=False)['mrr'].mean().rename(columns={'source_dataset': 'dataset'})
    rows = []
    for baseline, group in averaged.groupby('baseline'):
        frame = normalized_auc(group[['family', 'dataset', 'depth_fraction', 'mrr']], 'mrr')
        frame['baseline'] = baseline
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def contrast_table(auc, setting):
    mapping = {'direct': 'no_alignment', 'procrustes': 'aligned', 'shuffled': 'shuffled_alignment'}
    wide = auc.pivot(index=['family', 'dataset'], columns='baseline', values='mrr')
    rows = []
    for first, second in [('direct', 'procrustes'), ('direct', 'shuffled'), ('procrustes', 'shuffled')]:
        a = wide[mapping[first]].to_numpy(float)
        b = wide[mapping[second]].to_numpy(float)
        diff = a - b
        rows.append({'setting': setting, 'contrast': f'{first}_minus_{second}', 'first_mean': a.mean(), 'second_mean': b.mean(), 'mean_paired_difference': diff.mean(), 'positive_cell_fraction': np.mean(diff > 0), 'exact_sign_flip_p': sign_flip_p(diff)})
    return pd.DataFrame(rows)


def build_endpoint_grid(retrieval, cross, classification, mismatch):
    within_direct = retrieval[retrieval['baseline'] == 'no_alignment'][['family', 'dataset', 'depth_fraction', 'mrr']]
    cross_direct = cross[cross['baseline'] == 'no_alignment'].groupby(['family', 'source_dataset', 'depth_fraction'], as_index=False)['mrr'].mean().rename(columns={'source_dataset': 'dataset'})
    cls = classification[['family', 'dataset', 'depth_fraction', 'macro_f1']]
    dense = mismatch[['family', 'dataset', 'depth_fraction', 'cosine_combined_mismatch']]
    auc = normalized_auc(within_direct, 'mrr').rename(columns={'mrr': 'within_mrr_auc'})
    auc = auc.merge(normalized_auc(cross_direct, 'mrr').rename(columns={'mrr': 'cross_mrr_auc'}), on=['family', 'dataset'])
    auc = auc.merge(normalized_auc(cls, 'macro_f1').rename(columns={'macro_f1': 'classification_auc'}), on=['family', 'dataset'])
    auc = auc.merge(normalized_auc(dense, 'cosine_combined_mismatch').rename(columns={'cosine_combined_mismatch': 'dense_mismatch_auc'}), on=['family', 'dataset'])
    final = final_depth(within_direct, 'mrr').rename(columns={'mrr': 'within_mrr_final'})
    final = final.merge(final_depth(cross_direct, 'mrr').rename(columns={'mrr': 'cross_mrr_final'}), on=['family', 'dataset'])
    final = final.merge(final_depth(cls, 'macro_f1').rename(columns={'macro_f1': 'classification_final'}), on=['family', 'dataset'])
    final = final.merge(final_depth(dense, 'cosine_combined_mismatch').rename(columns={'cosine_combined_mismatch': 'dense_mismatch_final'}), on=['family', 'dataset'])
    return auc.merge(final, on=['family', 'dataset'])


def test_rows(probes, endpoints, summary):
    merged = probes.merge(endpoints, on=['family', 'dataset'], validate='one_to_one')
    if summary == 'depth_integrated_auc':
        specs = [('P1', 'mfi', 'dense_mismatch_auc'), ('P2', 'neighborhood_axis', 'within_mrr_auc'), ('P3', 'neighborhood_axis', 'cross_mrr_auc'), ('P4', 'separability', 'classification_auc')]
    else:
        specs = [('P1', 'mfi', 'dense_mismatch_final'), ('P2', 'neighborhood_axis', 'within_mrr_final'), ('P3', 'neighborhood_axis', 'cross_mrr_final'), ('P4', 'separability', 'classification_final')]
    rows = []
    for hypothesis, probe, endpoint in specs:
        x = merged[probe].to_numpy(float)
        y = merged[endpoint].to_numpy(float)
        rows.append({'hypothesis': hypothesis, 'depth_summary': summary, 'probe': probe, 'endpoint': endpoint, 'spearman_rho': spearman_rho(x, y), 'global_exact_p': exact_permutation_p(x, y), 'within_family_exact_p': restricted_permutation_p(x, y, merged['family']), 'within_dataset_exact_p': restricted_permutation_p(x, y, merged['dataset'])})
    return pd.DataFrame(rows)


def selection_regret(probes, endpoints):
    merged = probes.merge(endpoints[['family', 'dataset', 'cross_mrr_auc']], on=['family', 'dataset'], validate='one_to_one')
    scores = merged['cross_mrr_auc'].to_numpy(float)
    maximum = scores.max()
    minimum = scores.min()
    rank = merged['cross_mrr_auc'].rank(method='min', ascending=False).astype(int)
    rows = []
    for label, column in [('Neighborhood preservation axis', 'neighborhood_axis'), ('MFI', 'mfi')]:
        index = merged[column].idxmax()
        chosen = merged.loc[index]
        rows.append({'probe': label, 'family': chosen['family'], 'dataset': chosen['dataset'], 'retrieval_rank': int(rank.loc[index]), 'normalized_regret': float((maximum - chosen['cross_mrr_auc']) / (maximum - minimum))})
    return pd.DataFrame(rows)



def neighborhood_cross_dataset_leave_one_out(probes, endpoints):
    merged = probes.merge(
        endpoints[['family', 'dataset', 'cross_mrr_auc']],
        on=['family', 'dataset'],
        validate='one_to_one',
    )
    rows = []
    for group_column in ('family', 'dataset'):
        for held_out in sorted(merged[group_column].unique()):
            subset = merged[merged[group_column] != held_out]
            rows.append({
                'group': group_column,
                'held_out': held_out,
                'spearman_rho': spearman_rho(
                    subset['neighborhood_axis'].to_numpy(float),
                    subset['cross_mrr_auc'].to_numpy(float),
                ),
            })
    return pd.DataFrame(rows)


def neighborhood_cross_dataset_depthwise(probes, cross):
    direct = cross[cross['baseline'] == 'no_alignment']
    direct = (
        direct
        .groupby(['family', 'source_dataset', 'depth_fraction'], as_index=False)['mrr']
        .mean()
        .rename(columns={'source_dataset': 'dataset'})
    )
    rows = []
    for depth_fraction in sorted(direct['depth_fraction'].unique()):
        depth = direct[np.isclose(direct['depth_fraction'], depth_fraction)]
        merged = probes[['family', 'dataset', 'neighborhood_axis']].merge(
            depth[['family', 'dataset', 'mrr']],
            on=['family', 'dataset'],
            validate='one_to_one',
        )
        rows.append({
            'depth_fraction': float(depth_fraction),
            'spearman_rho': spearman_rho(
                merged['neighborhood_axis'].to_numpy(float),
                merged['mrr'].to_numpy(float),
            ),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-dir', default='additional_evaluation_outputs/results/primary_validation')
    parser.add_argument('--tables-dir', default='results/tables')
    parser.add_argument('--pca-summary', default='results/supplementary/pca_geometry/reported_pca_summary.csv')
    parser.add_argument('--output-dir', default='results/additional_evaluations')
    args = parser.parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    tables_dir = Path(args.tables_dir)
    pca_path = Path(args.pca_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieval = normalize_result_keys(pd.read_csv(evaluation_dir / 'paired_retrieval_summary.csv'))
    cross = normalize_result_keys(pd.read_csv(evaluation_dir / 'cross_dataset_retrieval_summary.csv'))
    classification = normalize_result_keys(pd.read_csv(evaluation_dir / 'condition_classification_summary.csv'))
    mismatch = normalize_result_keys(pd.read_csv(evaluation_dir / 'dense_mismatch_summary.csv'))
    probes = load_probe_grid(tables_dir, pca_path)
    endpoints = build_endpoint_grid(retrieval, cross, classification, mismatch)
    auc_tests = test_rows(probes, endpoints, 'depth_integrated_auc')
    final_tests = test_rows(probes, endpoints, 'final_depth')
    joint = pd.concat([auc_tests, final_tests], ignore_index=True)
    joint['holm_adjusted_p_across_8'] = holm_adjust(joint['global_exact_p'])
    joint['significant_after_joint_holm'] = joint['holm_adjusted_p_across_8'] < 0.05
    within_auc = baseline_auc_within(retrieval)
    cross_auc = baseline_auc_cross(cross)
    controls = pd.concat([contrast_table(within_auc, 'within_dataset_auc'), contrast_table(cross_auc, 'cross_dataset_auc')], ignore_index=True)
    regret = selection_regret(probes, endpoints)
    target_depth = classification.iloc[(classification['depth_fraction'] - 0.75).abs().argsort()[:1]]['depth_fraction'].iloc[0]
    null_rows = classification[np.isclose(classification['depth_fraction'], target_depth)]
    classification_control = pd.DataFrame([{'depth_fraction': target_depth, 'macro_f1': null_rows['macro_f1'].mean(), 'random_label_macro_f1': null_rows['null_macro_f1'].dropna().mean(), 'chance': 0.25}])
    leave_one_out = neighborhood_cross_dataset_leave_one_out(probes, endpoints)
    depthwise = neighborhood_cross_dataset_depthwise(probes, cross)
    probes.to_csv(output_dir / 'probe_grid.csv', index=False)
    endpoints.to_csv(output_dir / 'endpoint_grid.csv', index=False)
    joint.to_csv(output_dir / 'joint_eight_test_holm_audit.csv', index=False)
    controls.to_csv(output_dir / 'retrieval_baseline_comparisons.csv', index=False)
    regret.to_csv(output_dir / 'cross_dataset_selection_regret.csv', index=False)
    classification_control.to_csv(output_dir / 'condition_classification_control.csv', index=False)
    leave_one_out.to_csv(output_dir / 'neighborhood_cross_dataset_leave_one_out.csv', index=False)
    depthwise.to_csv(output_dir / 'neighborhood_cross_dataset_depthwise.csv', index=False)
    p3_auc = joint[(joint['hypothesis'] == 'P3') & (joint['depth_summary'] == 'depth_integrated_auc')].iloc[0]
    p3_final = joint[(joint['hypothesis'] == 'P3') & (joint['depth_summary'] == 'final_depth')].iloc[0]
    print(f"P3 AUC rho={p3_auc['spearman_rho']:.3f}, exact p={p3_auc['global_exact_p']:.4f}, Holm-8 p={p3_auc['holm_adjusted_p_across_8']:.4f}")
    print(f"P3 final rho={p3_final['spearman_rho']:.3f}, exact p={p3_final['global_exact_p']:.4f}, Holm-8 p={p3_final['holm_adjusted_p_across_8']:.4f}")


if __name__ == '__main__':
    main()

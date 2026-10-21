# -*- coding: utf-8 -*-
"""Master evaluation script: Vector (Polar) vs Director (Nematic) on ARCADE 200 Test Images.

Follows the exact mathematical and statistical protocol of the pre-registered
axis_prior_analysis on Neuro, Neurite, and SY5Y.
Outputs pure quantitative numerical metrics:
  - point_p_median_error vs point_q_median_error
  - point_p_acc15 vs point_q_acc15
  - point_p_acc30 vs point_q_acc30
  - img_p_mean_error vs img_q_mean_error
  - paired_diff_mean with 10,000 bootstrap 95% CIs
  - Wilcoxon signed-rank test p-value
"""

import os
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import correlate, convolve, gaussian_filter, binary_dilation
from scipy import stats
from skimage.morphology import skeletonize

SOBEL_KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64) / 8.0
SOBEL_KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64) / 8.0

SIGMAS = [1.0, 0.5, 2.0]


def compute_fields_for_sigmas(img, sigmas=[1.0, 0.5, 2.0]):
    if img.ndim == 3:
        if img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
            base_img = np.mean(img, axis=0)
        else:
            base_img = np.mean(img, axis=2)
    else:
        base_img = img.astype(np.float64)

    # Normalize [0, 1]
    b_min, b_max = base_img.min(), base_img.max()
    if b_max > b_min:
        base_img = (base_img - b_min) / (b_max - b_min)

    gx = correlate(base_img, SOBEL_KX, mode='nearest')
    gy = correlate(base_img, SOBEL_KY, mode='nearest')

    w = np.sqrt(gx**2 + gy**2)
    w_safe = np.maximum(w, 1e-6)

    # Tangent unit vector (-gy, gx)
    nx = -gy / w_safe
    ny = gx / w_safe

    # 2nd-order nematic tensor components
    q11 = 2 * nx * nx - 1.0
    q12 = 2 * nx * ny

    w_nx = w * nx
    w_ny = w * ny
    w_q11 = w * q11
    w_q12 = w * q12

    reps = {}
    for sigma in sigmas:
        smooth_w = gaussian_filter(w, sigma=sigma, mode='nearest')
        denom = np.maximum(smooth_w, 1e-6)

        # 1st-order Polar Vector field
        px = gaussian_filter(w_nx, sigma=sigma, mode='nearest') / denom
        py = gaussian_filter(w_ny, sigma=sigma, mode='nearest') / denom
        r_p = np.sqrt(px**2 + py**2)
        tx_p = px / np.maximum(r_p, 1e-6)
        ty_p = py / np.maximum(r_p, 1e-6)

        # 2nd-order Nematic Director field
        smooth_q11 = gaussian_filter(w_q11, sigma=sigma, mode='nearest') / denom
        smooth_q12 = gaussian_filter(w_q12, sigma=sigma, mode='nearest') / denom
        s = np.sqrt(smooth_q11**2 + smooth_q12**2)
        theta_q = 0.5 * np.arctan2(smooth_q12, smooth_q11)
        tx_q = np.cos(theta_q)
        ty_q = np.sin(theta_q)

        reps[sigma] = {
            'smooth_w': smooth_w,
            'tx_p': tx_p, 'ty_p': ty_p, 'r_p': r_p,
            'tx_q': tx_q, 'ty_q': ty_q, 's': s,
        }
    return reps

def analyze_skeleton_points_with_exclusions(lbl, win_r=3, min_linearity=0.65, prune_r=2):
    skel = skeletonize(lbl > 0)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    n_neighbors = convolve(skel.astype(np.uint8), kernel, mode='constant') * skel

    junction_mask = (skel == 1) & (n_neighbors > 2)
    endpoint_mask = (skel == 1) & (n_neighbors == 1)

    struct = np.ones((2 * prune_r + 1, 2 * prune_r + 1), dtype=bool)
    excl_junction_dilated = binary_dilation(junction_mask, structure=struct)
    excl_endpoint_dilated = binary_dilation(endpoint_mask, structure=struct)

    ys, xs = np.where(skel == 1)
    h, w = lbl.shape
    point_records = []

    for y, x in zip(ys, xs):
        if endpoint_mask[y, x] or excl_endpoint_dilated[y, x]:
            continue
        if junction_mask[y, x] or excl_junction_dilated[y, x]:
            continue
        if n_neighbors[y, x] != 2:
            continue

        y0, y1 = max(0, y - win_r), min(h, y + win_r + 1)
        x0, x1 = max(0, x - win_r), min(w, x + win_r + 1)
        sub_skel = skel[y0:y1, x0:x1]
        sy, sx = np.where(sub_skel)
        if len(sy) < 3:
            continue

        dy = sy - np.mean(sy)
        dx = sx - np.mean(sx)
        cov = np.cov(dx, dy)
        if cov.shape != (2, 2):
            continue

        eigvals, eigvecs = np.linalg.eigh(cov)
        lin = (eigvals[1] - eigvals[0]) / max(eigvals[1] + eigvals[0], 1e-8)
        if lin < min_linearity:
            continue

        v = eigvecs[:, -1]
        point_records.append((y, x, float(v[0]), float(v[1])))

    return skel, point_records

def process_single_image(args):
    img_path, mask_path = args
    img_name = Path(img_path).name

    img = np.array(Image.open(img_path))
    lbl = np.array(Image.open(mask_path))
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]

    if lbl.sum() < 20:
        return None

    skel, point_records = analyze_skeleton_points_with_exclusions(lbl)
    if len(point_records) < 3:
        return None

    reps = compute_fields_for_sigmas(img, sigmas=SIGMAS)
    results_by_sigma = {}

    for sigma in SIGMAS:
        rep = reps[sigma]
        mut_p_errs = []
        mut_q_errs = []

        for (y, x, ref_tx, ref_ty) in point_records:
            energy = float(rep['smooth_w'][y, x])
            rp = float(rep['r_p'][y, x])
            s_val = float(rep['s'][y, x])

            p_val = (energy >= 1e-4 and rp >= 0.05)
            q_val = (energy >= 1e-4 and s_val >= 0.05)

            if p_val and q_val:
                tp = np.array([rep['tx_p'][y, x], rep['ty_p'][y, x]])
                tq = np.array([rep['tx_q'][y, x], rep['ty_q'][y, x]])
                tgt = np.array([ref_tx, ref_ty])

                ep = float(np.arccos(np.clip(np.abs(np.dot(tp, tgt)), 0.0, 1.0)) * 180.0 / np.pi)
                eq = float(np.arccos(np.clip(np.abs(np.dot(tq, tgt)), 0.0, 1.0)) * 180.0 / np.pi)

                mut_p_errs.append(ep)
                mut_q_errs.append(eq)

        if len(mut_p_errs) > 0:
            arr_p = np.array(mut_p_errs)
            arr_q = np.array(mut_q_errs)
            results_by_sigma[sigma] = {
                'img_name': img_name,
                'n_pts': len(mut_p_errs),
                'p_mean_err': float(np.mean(arr_p)),
                'p_med_err': float(np.median(arr_p)),
                'q_mean_err': float(np.mean(arr_q)),
                'q_med_err': float(np.median(arr_q)),
                'diff_mean': float(np.mean(arr_q - arr_p)),
                'p_acc_15': float(np.mean(arr_p < 15.0)),
                'q_acc_15': float(np.mean(arr_q < 15.0)),
                'p_acc_30': float(np.mean(arr_p < 30.0)),
                'q_acc_30': float(np.mean(arr_q < 30.0)),
                'all_p_errs': mut_p_errs,
                'all_q_errs': mut_q_errs,
            }

    return results_by_sigma

def run_bootstrap_ci(data_array, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(data_array)
    boot_means = np.mean(rng.choice(data_array, size=(n_boot, n), replace=True), axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

def main():
    img_dir = '/home/wangshengping/DataSet/ARCADE/test/images'
    mask_dir = '/home/wangshengping/DataSet/ARCADE/test/masks'
    out_dir = './Outputs/Ablation'
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    tasks = []
    for ip in img_files:
        mp = os.path.join(mask_dir, os.path.basename(ip))
        if os.path.exists(mp):
            tasks.append((ip, mp))

    print(f'Starting ARCADE Vector vs Director study on {len(tasks)} test images across 16 CPU workers...')
    with ProcessPoolExecutor(max_workers=16) as executor:
        all_res = list(executor.map(process_single_image, tasks))

    valid_res = [r for r in all_res if r is not None]
    print(f'Completed processing: {len(valid_res)}/{len(tasks)} images with valid skeletal points.')

    records = []
    for sigma in SIGMAS:
        sigma_imgs = [r[sigma] for r in valid_res if sigma in r]
        if not sigma_imgs:
            continue

        n_imgs = len(sigma_imgs)
        all_p = [pt for item in sigma_imgs for pt in item['all_p_errs']]
        all_q = [pt for item in sigma_imgs for pt in item['all_q_errs']]
        total_pts = len(all_p)

        p_pt_med = float(np.median(all_p))
        q_pt_med = float(np.median(all_q))
        p_pt_acc15 = float(np.mean(np.array(all_p) < 15.0)) * 100
        q_pt_acc15 = float(np.mean(np.array(all_q) < 15.0)) * 100
        p_pt_acc30 = float(np.mean(np.array(all_p) < 30.0)) * 100
        q_pt_acc30 = float(np.mean(np.array(all_q) < 30.0)) * 100

        img_p_means = np.array([x['p_mean_err'] for x in sigma_imgs])
        img_q_means = np.array([x['q_mean_err'] for x in sigma_imgs])
        diff_means = img_q_means - img_p_means

        p_mean = float(np.mean(img_p_means))
        q_mean = float(np.mean(img_q_means))
        diff_mean = float(np.mean(diff_means))
        ci_low, ci_high = run_bootstrap_ci(diff_means)

        # Wilcoxon signed-rank test
        stat_val, p_val = stats.wilcoxon(img_q_means, img_p_means)

        records.append({
            'dataset': 'ARCADE (Angiography)',
            'split': 'test (200 samples)',
            'sigma': sigma,
            'total_images': len(tasks),
            'valid_images': n_imgs,
            'total_skeleton_points': total_pts,
            'point_p_median_error': f'{p_pt_med:.2f}°',
            'point_q_median_error': f'{q_pt_med:.2f}°',
            'point_p_acc15': f'{p_pt_acc15:.2f}%',
            'point_q_acc15': f'{q_pt_acc15:.2f}%',
            'point_p_acc30': f'{p_pt_acc30:.2f}%',
            'point_q_acc30': f'{q_pt_acc30:.2f}%',
            'img_p_mean_error': f'{p_mean:.2f}°',
            'img_q_mean_error': f'{q_mean:.2f}°',
            'paired_diff_mean': f'{diff_mean:.2f}°',
            'paired_diff_95ci': f'[{ci_low:.2f}°, {ci_high:.2f}°]',
            'wilcoxon_p_value': f'{p_val:.3e}' if p_val >= 1e-4 else 'p < 1e-4',
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, 'arcade_vector_vs_director_metrics.csv')
    md_path = os.path.join(out_dir, 'arcade_vector_vs_director_metrics.md')

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# ARCADE Angiography Test Set (200 images) Vector vs Director Geometric Prior Numerical Analysis\n\n')
        f.write(df.to_markdown(index=False))
        f.write('\n\n### Metric definitions:\n')
        f.write('- **point_p_median_error / point_q_median_error**: median angular error between skeleton-point direction and ground-truth vessel tangent (lower is better)\n')
        f.write('- **point_p_acc15 / point_q_acc15**: angular accuracy within 15 degrees (higher is better)\n')
        f.write('- **point_p_acc30 / point_q_acc30**: angular accuracy within 30 degrees (higher is better)\n')
        f.write('- **img_p_mean_error / img_q_mean_error**: image-level mean angular error\n')
        f.write('- **paired_diff_mean**: paired angular-error difference (Director Q - Polar Vector P; negative means Director error is significantly lower)\n')
        f.write('- **wilcoxon_p_value**: statistical significance of the paired Wilcoxon signed-rank test\n')

    print(f'\n[SUCCESS] Reports successfully saved to:\n  - {csv_path}\n  - {md_path}\n')
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()

import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

REPO_TVNN = Path(__file__).resolve().parents[1] / 'repo_tvnn' / 'TVNN'
sys.path.insert(0, str(REPO_TVNN))
from read_dataset import data_from_name  # noqa: E402
from model import TVNN  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / 'experiments' / 'results'
FIG_DIR = Path(__file__).resolve().parents[1] / 'experiments' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def original_filter(x: np.ndarray, alpha: float):
    xf = np.fft.fft(x, axis=0)
    mask = np.ones((xf.shape[0], xf.shape[1]))
    mask[int((1 - alpha) * xf.shape[0]) :, :] = 0
    x_l = np.fft.ifft(xf * mask, axis=0).real
    x_g = x - x_l
    return x_g, x_l


def symmetric_lowpass_filter(x: np.ndarray, alpha: float):
    xf = np.fft.fft(x, axis=0)
    n = xf.shape[0]
    k = max(1, int(alpha * n / 2))
    mask = np.zeros((n, xf.shape[1]), dtype=float)
    mask[:k, :] = 1.0
    mask[-k:, :] = 1.0
    x_low = np.fft.ifft(xf * mask, axis=0).real
    x_high = x - x_low
    return x_high, x_low


def build_tensors(dataset: str, noise: float, M: int, L: int, M_S: int, alpha: float, filter_fn):
    Z = data_from_name(dataset, noise)
    Z = Z[: M + L]
    zmax, zmin = np.max(Z), np.min(Z)
    Z = 2 * ((Z - zmin) / (zmax - zmin) - 0.5)
    Z_g, Z_l = filter_fn(Z, alpha)
    Ztrain_g, Ztrain_l = Z_g[:M], Z_l[:M]
    Ztest = Z[M : M + L]
    S = M // M_S
    Xtrain_g = np.array([Ztrain_g[i * M_S : (i + 1) * M_S] for i in range(S)])
    Xtrain_l = np.array([Ztrain_l[i * M_S : (i + 1) * M_S] for i in range(S)])
    Xtrain_g = torch.from_numpy(Xtrain_g).float().contiguous()
    Xtrain_l = torch.from_numpy(Xtrain_l).float().contiguous()
    train_data = torch.utils.data.TensorDataset(
        Xtrain_g[: S - 2], Xtrain_l[: S - 2], Xtrain_g[1 : S - 1], Xtrain_l[1 : S - 1], Xtrain_g[2:S], Xtrain_l[2:S]
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=8, shuffle=False)
    return Z, Xtrain_g, Xtrain_l, Ztest, train_loader


def train_and_eval(dataset: str, seed: int, filter_name: str, filter_fn, *, noise=0.1, M=1024, L=64, M_S=8, alpha=0.3, epochs=40):
    set_seed(seed)
    Z, Xtrain_g, Xtrain_l, Ztest, train_loader = build_tensors(dataset, noise, M, L, M_S, alpha, filter_fn)
    D = Z.shape[1]
    model = TVNN(D, alpha)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, weight_decay=1e-5)
    criterion = nn.MSELoss()
    lr_milestones = {20, 30, 35}
    for epoch in range(epochs):
        model.train()
        for X_g, X_l, Y_g, Y_l, Z_g, Z_l in train_loader:
            ix_g, ny_g, iy_g, nz_g, ix_l, iy_l, nz_l, n_z = model(X_g, X_l, Y_g, Y_l)
            loss_g_id = (criterion(ix_g, X_g) + criterion(iy_g, Y_g)) / 2.0
            loss_g = (criterion(ny_g, Y_g) + criterion(nz_g, Z_g)) / 2.0
            loss_l_id = (criterion(ix_l, X_l) + criterion(iy_l, Y_l)) / 2.0
            loss_l = criterion(nz_l, Z_l)
            loss_con = criterion(n_z, Z_g + Z_l)
            loss = 0.1 * loss_g_id + 0.1 * loss_g + 0.3 * loss_l_id + 0.4 * loss_l + 0.1 * loss_con
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-7)
            optimizer.step()
        if epoch in lr_milestones:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5

    model.eval()
    S = Xtrain_g.shape[0]
    X_fir_g = Xtrain_g[S - 2]
    X_fir_l = Xtrain_l[S - 2]
    X_sec_g = Xtrain_g[S - 1]
    X_sec_l = Xtrain_l[S - 1]
    Xpred = []
    with torch.no_grad():
        for _ in range(L // M_S):
            _, nY_g = model.Global_module(X_sec_g)
            _, _, nY_l = model.TV_module(X_fir_l, X_sec_l)
            X_fir_g = X_sec_g
            X_fir_l = X_sec_l
            X_sec_g = nY_g
            X_sec_l = nY_l
            Xpred.append((X_sec_g + X_sec_l).detach().numpy())
    Xpred = np.array(Xpred).reshape(-1, D)
    def rmse(k):
        return float(np.sqrt(np.sum(np.sum((Xpred[:k] - Ztest[:k]) ** 2, axis=1)) / k))
    return {'rmse16': rmse(16), 'rmse32': rmse(32), 'rmse48': rmse(48), 'rmse64': rmse(64)}


def summarize_runs(runs):
    summary = {}
    for dataset in sorted(set(r['dataset'] for r in runs)):
        for alpha in sorted(set(r['alpha'] for r in runs if r['dataset'] == dataset)):
            for filter_name in sorted(set(r['filter'] for r in runs if r['dataset'] == dataset)):
                rows = [r for r in runs if r['dataset'] == dataset and r['alpha'] == alpha and r['filter'] == filter_name]
                key = f'{dataset}::alpha={alpha:.1f}::{filter_name}'
                summary[key] = {}
                for metric in ['rmse16', 'rmse32', 'rmse48', 'rmse64']:
                    vals = [r[metric] for r in rows]
                    summary[key][metric] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'min': float(np.min(vals)),
                        'max': float(np.max(vals)),
                    }
    return summary


def make_plots(runs):
    horizons = [16, 32, 48, 64]
    metrics = ['rmse16', 'rmse32', 'rmse48', 'rmse64']
    filters = ['original', 'symmetric_lowpass']
    labels = {'original': 'Original FFT mask', 'symmetric_lowpass': 'Corrected symmetric low-pass'}
    colors = {'original': '#d95f02', 'symmetric_lowpass': '#1b9e77'}

    for dataset in sorted(set(r['dataset'] for r in runs)):
        alphas = sorted(set(r['alpha'] for r in runs if r['dataset'] == dataset))

        # line plots: one panel per alpha
        fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 4), sharey=True)
        if len(alphas) == 1:
            axes = [axes]
        for ax, alpha in zip(axes, alphas):
            for filter_name in filters:
                rows = [r for r in runs if r['dataset'] == dataset and r['alpha'] == alpha and r['filter'] == filter_name]
                means = [np.mean([row[m] for row in rows]) for m in metrics]
                stds = [np.std([row[m] for row in rows]) for m in metrics]
                ax.errorbar(horizons, means, yerr=stds, marker='o', capsize=3, color=colors[filter_name], label=labels[filter_name])
            ax.set_title(f'{dataset}, alpha={alpha:.1f}')
            ax.set_xlabel('Forecast horizon')
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel('RMSE')
        handles, leglabels = axes[0].get_legend_handles_labels()
        fig.legend(handles, leglabels, loc='upper center', ncol=2)
        fig.suptitle(f'TVNN forecasting robustness by Fourier decomposition ({dataset})', y=1.05)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f'tvnn_{dataset}_lineplots.png', dpi=180, bbox_inches='tight')
        plt.close(fig)

        # improvement heatmap and std ratio heatmap
        improvement = np.zeros((len(alphas), len(horizons)))
        stability = np.zeros((len(alphas), len(horizons)))
        for i, alpha in enumerate(alphas):
            orig_rows = [r for r in runs if r['dataset'] == dataset and r['alpha'] == alpha and r['filter'] == 'original']
            corr_rows = [r for r in runs if r['dataset'] == dataset and r['alpha'] == alpha and r['filter'] == 'symmetric_lowpass']
            for j, metric in enumerate(metrics):
                orig_vals = np.array([r[metric] for r in orig_rows])
                corr_vals = np.array([r[metric] for r in corr_rows])
                improvement[i, j] = orig_vals.mean() - corr_vals.mean()
                stability[i, j] = orig_vals.std() - corr_vals.std()

        for matrix, title, fname, cmap in [
            (improvement, 'RMSE improvement: original - corrected', f'tvnn_{dataset}_improvement_heatmap.png', 'RdYlGn'),
            (stability, 'Stability gain: std(original) - std(corrected)', f'tvnn_{dataset}_stability_heatmap.png', 'PuOr'),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            im = ax.imshow(matrix, aspect='auto', cmap=cmap)
            ax.set_xticks(range(len(horizons)))
            ax.set_xticklabels(horizons)
            ax.set_yticks(range(len(alphas)))
            ax.set_yticklabels([f'{a:.1f}' for a in alphas])
            ax.set_xlabel('Forecast horizon')
            ax.set_ylabel('alpha')
            ax.set_title(f'{title} ({dataset})')
            for i in range(len(alphas)):
                for j in range(len(horizons)):
                    ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', color='black', fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.85)
            fig.tight_layout()
            fig.savefig(FIG_DIR / fname, dpi=180, bbox_inches='tight')
            plt.close(fig)


def main():
    filters = {'original': original_filter, 'symmetric_lowpass': symmetric_lowpass_filter}
    datasets = ['pendulum', 'lorenz']
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    seeds = [0, 1, 2, 3, 4]
    runs = []
    for dataset in datasets:
        for alpha in alphas:
            for seed in seeds:
                for filter_name, filter_fn in filters.items():
                    print(f'RUN dataset={dataset} alpha={alpha:.1f} seed={seed} filter={filter_name}', flush=True)
                    metrics = train_and_eval(dataset, seed, filter_name, filter_fn, alpha=alpha)
                    runs.append({'dataset': dataset, 'alpha': alpha, 'seed': seed, 'filter': filter_name, **metrics})
    payload = {'runs': runs, 'summary': summarize_runs(runs)}
    out_path = OUT_DIR / 'tvnn_fourier_robustness_atlas.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    make_plots(runs)
    print(f'SAVED {out_path}')
    print(f'FIGURES {FIG_DIR}')


if __name__ == '__main__':
    main()

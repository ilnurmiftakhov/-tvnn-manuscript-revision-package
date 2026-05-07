import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import sys

REPO_TVNN = Path(__file__).resolve().parents[1] / 'repo_tvnn' / 'TVNN'
sys.path.insert(0, str(REPO_TVNN))
from read_dataset import data_from_name  # noqa: E402
from model import TVNN  # noqa: E402


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


def spectral_isolation_score(filter_fn, n=64, alpha=0.3):
    steps = np.arange(n)
    low = np.sin(2 * np.pi * steps / 32)[:, None]
    high = 0.3 * np.sin(2 * np.pi * steps / 4)[:, None]
    x = low + high
    high_part, low_part = filter_fn(x, alpha)
    def proj_energy(a, b):
        return float(np.dot(a[:, 0], b[:, 0]) ** 2 / (np.dot(b[:, 0], b[:, 0]) + 1e-12))
    return {
        'lowpart_on_low': proj_energy(low_part, low),
        'lowpart_on_high': proj_energy(low_part, high),
        'highpart_on_low': proj_energy(high_part, low),
        'highpart_on_high': proj_energy(high_part, high),
        'low_std': float(low_part.std()),
        'high_std': float(high_part.std()),
    }


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
    Xtest = np.array([Ztest[i * M_S : (i + 1) * M_S] for i in range(L // M_S)])
    Xtrain_g = torch.from_numpy(Xtrain_g).float().contiguous()
    Xtrain_l = torch.from_numpy(Xtrain_l).float().contiguous()
    train_data = torch.utils.data.TensorDataset(
        Xtrain_g[: S - 2], Xtrain_l[: S - 2], Xtrain_g[1 : S - 1], Xtrain_l[1 : S - 1], Xtrain_g[2:S], Xtrain_l[2:S]
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=8, shuffle=False)
    return Z, Xtrain_g, Xtrain_l, Xtest, train_loader


def train_and_eval(dataset: str, seed: int, filter_name: str, filter_fn, *, noise=0.1, M=512, L=64, M_S=8, alpha=0.3, epochs=20):
    set_seed(seed)
    Z, Xtrain_g, Xtrain_l, Xtest, train_loader = build_tensors(dataset, noise, M, L, M_S, alpha, filter_fn)
    D = Z.shape[1]
    model = TVNN(D, alpha)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, weight_decay=1e-5)
    criterion = nn.MSELoss()
    losses = []
    model.train()
    for epoch in range(epochs):
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
        losses.append(float(loss.item()))

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
            X_sec = X_sec_g + X_sec_l
            Xpred.append(X_sec.detach().numpy())
    Xpred = np.array(Xpred).reshape(-1, D)
    Ztest = Z[M : M + L]
    def rmse(k):
        return float(np.sqrt(np.sum(np.sum((Xpred[:k] - Ztest[:k]) ** 2, axis=1)) / k))
    return {
        'dataset': dataset,
        'seed': seed,
        'filter': filter_name,
        'epochs': epochs,
        'M': M,
        'L': L,
        'M_S': M_S,
        'alpha': alpha,
        'final_loss': losses[-1],
        'rmse16': rmse(16),
        'rmse32': rmse(32),
        'rmse48': rmse(48),
    }


def main():
    out_dir = Path(__file__).resolve().parents[1] / 'experiments' / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    filters = {
        'original': original_filter,
        'symmetric_lowpass': symmetric_lowpass_filter,
    }
    spectral = {name: spectral_isolation_score(fn) for name, fn in filters.items()}
    results = []
    for dataset in ['pendulum', 'lorenz']:
        for seed in [0, 1, 2]:
            for name, fn in filters.items():
                results.append(train_and_eval(dataset, seed, name, fn))
    payload = {'spectral_audit': spectral, 'runs': results}
    (out_dir / 'tvnn_fourier_audit.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()

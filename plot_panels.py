from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _coerce_curve(values: Any) -> list[float]:
    """Best-effort conversion of a curve-like object to a list of floats."""
    if values is None:
        return []

    torch = None
    try:
        import torch as _torch  # type: ignore

        torch = _torch
    except Exception:
        torch = None

    if torch is not None and isinstance(values, torch.Tensor):
        values = values.detach().cpu().flatten().tolist()
    elif isinstance(values, np.ndarray):
        values = values.flatten().tolist()
    elif not isinstance(values, (list, tuple)):
        try:
            return [float(values)]
        except Exception:
            return []

    out: list[float] = []
    for v in values:
        if v is None:
            out.append(float("nan"))
            continue
        try:
            out.append(float(v))
        except Exception:
            out.append(float("nan"))
    return out


def _finite_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    return xf if np.isfinite(xf) else None


@dataclass(frozen=True)
class SeriesSpec:
    title: str
    key: str
    ylabel: str
    start_at_zero: bool = False


def _plot_series(
    ax,
    y,
    *,
    x=None,
    title: str = "",
    xlabel: str = "Epoch",
    ylabel: str = "",
    color: str | None = None,
    linestyle: str = "-",
    baseline: float | None = None,
    baseline_color: str = "red",
    baseline_linestyle: str = ":",
    series_label: str | None = None,
    baseline_label: str | None = None,
):
    y_vals = _coerce_curve(y)
    if not y_vals:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        return

    if x is None:
        x_vals = list(range(len(y_vals)))
    else:
        x_vals = list(x)

    n = min(len(x_vals), len(y_vals))
    x_vals = x_vals[:n]
    y_vals = y_vals[:n]

    ax.plot(
        x_vals,
        y_vals,
        color=color or "C0",
        linewidth=1.6,
        linestyle=linestyle,
        label=series_label,
    )

    baseline_f = _finite_or_none(baseline)
    if baseline_f is not None:
        ax.axhline(
            y=baseline_f,
            color=baseline_color,
            linestyle=baseline_linestyle,
            linewidth=1.4,
            alpha=0.9,
            label=baseline_label,
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def save_loss_panel(
    out_path: str,
    *,
    train_curves: dict,
    val_curves: dict,
    eval_frequency: int,
    spokes_per_frame: int,
    suptitle: str | None = None,
):
    import matplotlib.pyplot as plt

    eval_frequency = max(int(eval_frequency or 1), 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    if suptitle is None:
        suptitle = "Loss Curves (train/val)"
    fig.suptitle(suptitle, fontsize=18)

    _plot_series(
        axes[0, 0],
        train_curves.get("train_adj_losses", []),
        title="Training Adjoint Loss",
        ylabel="Adjoint Loss",
    )
    _plot_series(
        axes[0, 1],
        train_curves.get("train_mc_losses", []),
        title="Training MC Loss",
        ylabel="MC Loss",
    )
    _plot_series(
        axes[0, 2],
        train_curves.get("train_ei_losses", []),
        title="Training EI Loss",
        ylabel="EI Loss",
    )

    def _val_epochs(n: int) -> list[int]:
        return list(range(0, int(n) * eval_frequency, eval_frequency))

    _plot_series(
        axes[1, 0],
        val_curves.get("val_adj_losses", []),
        x=_val_epochs(len(_coerce_curve(val_curves.get("val_adj_losses", [])))),
        title=f"Validation Adjoint Loss ({int(spokes_per_frame)} spokes/frame)",
        ylabel="Adjoint Loss",
        color="C1",
    )
    _plot_series(
        axes[1, 1],
        val_curves.get("val_mc_losses", []),
        x=_val_epochs(len(_coerce_curve(val_curves.get("val_mc_losses", [])))),
        title=f"Validation MC Loss ({int(spokes_per_frame)} spokes/frame)",
        ylabel="MC Loss",
        color="C1",
    )
    _plot_series(
        axes[1, 2],
        val_curves.get("val_ei_losses", []),
        x=_val_epochs(len(_coerce_curve(val_curves.get("val_ei_losses", [])))),
        title=f"Validation EI Loss ({int(spokes_per_frame)} spokes/frame)",
        ylabel="EI Loss",
        color="C1",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_eval_metrics_over_epochs(
    out_path: str,
    *,
    eval_curves: dict,
    eval_frequency: int,
    spokes_per_frame: int,
    baselines: dict | None = None,
    epochs: Any | None = None,
    suptitle: str | None = None,
):
    import matplotlib.pyplot as plt

    baselines = baselines or {}
    eval_frequency = max(int(eval_frequency or 1), 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    if suptitle is None:
        suptitle = f"Evaluation Metrics Over Epochs ({int(spokes_per_frame)} spokes/frame)"
    fig.suptitle(suptitle, fontsize=18)

    def _eval_epochs(n: int) -> list[int]:
        return list(range(0, int(n) * eval_frequency, eval_frequency))

    epoch_x = _coerce_curve(epochs) if epochs is not None else []

    specs = [
        SeriesSpec(title="DRO Evaluation SSIM", key="eval_ssims", ylabel="SSIM", start_at_zero=True),
        SeriesSpec(title="DRO Evaluation PSNR", key="eval_psnrs", ylabel="PSNR", start_at_zero=True),
        SeriesSpec(title="DRO Evaluation Image MSE", key="eval_mses", ylabel="MSE"),
        SeriesSpec(title="Evaluation LPIPS", key="eval_lpipses", ylabel="LPIPS"),
        SeriesSpec(title="Non-DRO Evaluation Raw k-space MAE", key="eval_raw_dc_maes", ylabel="MAE"),
        SeriesSpec(title="DRO Tumor Enhancement Curve Correlation", key="eval_curve_corrs", ylabel="Pearson Correlation Coefficient"),
    ]
    baseline_keys = ["ssim", "psnr", "mse", "lpips", "raw_dc_mae", "curve_corr"]
    for ax, spec, bkey in zip(axes.ravel(), specs, baseline_keys, strict=True):
        y = eval_curves.get(spec.key, [])
        x = epoch_x if epoch_x else _eval_epochs(len(_coerce_curve(y)))
        _plot_series(
            ax,
            y,
            x=x,
            title=spec.title,
            ylabel=spec.ylabel,
            baseline=baselines.get(bkey),
            series_label="BRISKNet",
            baseline_label="GRASP",
        )
        if spec.start_at_zero:
            ax.set_ylim(bottom=0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_inference_metrics_panel(
    out_path: str,
    *,
    results: list[dict],
    grasp_results: list[dict],
    raw_results: list[dict],
    zf_results: list[dict] | None,
    spokes_per_frame: int,
    num_frames: int,
    noise_level: float | None = None,
    suptitle: str | None = None,
):
    import matplotlib.pyplot as plt

    if not results:
        return

    zf_lookup = {row.get("sample"): row for row in (zf_results or []) if isinstance(row, dict)}

    def _vals(rows: list[dict], key: str) -> np.ndarray:
        out = []
        for row in rows:
            if not isinstance(row, dict):
                out.append(float("nan"))
                continue
            v = row.get(key)
            if v is None:
                out.append(float("nan"))
                continue
            try:
                out.append(float(v))
            except Exception:
                out.append(float("nan"))
        return np.asarray(out, dtype=np.float64)

    def _vals_zf(key: str) -> np.ndarray:
        out = []
        for r in results:
            sample = r.get("sample") if isinstance(r, dict) else None
            zf_row = zf_lookup.get(sample, {})
            v = zf_row.get(key) if isinstance(zf_row, dict) else None
            if v is None:
                out.append(float("nan"))
            else:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(float("nan"))
        return np.asarray(out, dtype=np.float64)

    def _plot_metric(ax, title: str, y_dl: np.ndarray, y_grasp: np.ndarray, ylabel: str, y_zf: np.ndarray | None = None):
        x = np.arange(len(y_dl), dtype=int)
        ax.plot(x, y_dl, "-o", label="BRISKNet", linewidth=1.6, markersize=4, color="C0")
        ax.plot(x, y_grasp, ":s", label="GRASP", linewidth=1.8, markersize=4, color="C1")
        if y_zf is not None and np.isfinite(y_zf).any():
            ax.plot(x, y_zf, "--^", label="ZF", linewidth=1.6, markersize=4, color="C2")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Sample")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    subtitle = f"{int(spokes_per_frame)} spokes/frame, {int(num_frames)} frames, N={len(results)}"
    if noise_level is not None:
        noise_f = _finite_or_none(noise_level)
        if noise_f is not None:
            subtitle += f", noise={noise_f:g}"
    if suptitle is None:
        suptitle = f"Inference Metrics ({subtitle})"
    fig.suptitle(suptitle, fontsize=18)

    _plot_metric(
        axes[0, 0],
        "DRO SSIM",
        _vals(results, "ssim"),
        _vals(grasp_results, "ssim"),
        "SSIM",
        y_zf=_vals_zf("ssim"),
    )
    axes[0, 0].set_ylim(bottom=0)

    _plot_metric(
        axes[0, 1],
        "DRO PSNR",
        _vals(results, "psnr"),
        _vals(grasp_results, "psnr"),
        "PSNR (dB)",
        y_zf=_vals_zf("psnr"),
    )
    axes[0, 1].set_ylim(bottom=0)

    _plot_metric(
        axes[0, 2],
        "DRO Image MSE",
        _vals(results, "mse"),
        _vals(grasp_results, "mse"),
        "MSE",
        y_zf=_vals_zf("mse"),
    )
    _plot_metric(
        axes[1, 0],
        "DRO LPIPS",
        _vals(results, "lpips"),
        _vals(grasp_results, "lpips"),
        "LPIPS",
        y_zf=_vals_zf("lpips"),
    )
    _plot_metric(
        axes[1, 1],
        "RAW k-space MAE",
        _vals(raw_results, "raw_dc_mae"),
        _vals(raw_results, "raw_grasp_dc_mae"),
        "MAE",
        y_zf=None,
    )
    _plot_metric(
        axes[1, 2],
        "DRO Enhancement Curve Correlation",
        _vals(results, "recon_corr"),
        _vals(results, "grasp_corr"),
        "Pearson r",
        y_zf=None,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)

    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

"""Regenerate LaTeX tables and inject them into the paper .tex file.

Runs make_inference_table.py / make_significance_table.py for each table,
then finds the matching \\begin{table}...\\end{table} block by \\label{} and
replaces it in-place.

Usage:
    python update_paper_tables.py                      # regenerate all tables
    python update_paper_tables.py --tables tab:acc_exp_spatial,tab:acc_exp_consistency
    python update_paper_tables.py --dry-run            # print diffs, don't write
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TEX_FILE = Path("/home/rachelgordon/mri_recon/brisknet-journal-paper/cas-dc-template.tex")

_MT = "inference/plot/make_inference_table.py"
_MS = "inference/plot/make_significance_table.py"
_LOG_REV = "inference/test_inference_logs_mri_journal_revised.json"
_LOG_ORIG = "inference/test_inference_logs_mri_journal.json"
_ACCEL_EXPS = ",".join([
    "ei_8spf_sampling_arrshift_fop", "ssdu_8spf_sampling",
    "ei_16spf_sampling_arrshift_fop", "ssdu_16spf_sampling",
    "ei_24spf_sampling_arrshift_fop", "ssdu_24spf_sampling",
    "ei_36spf_sampling_arrshift_fop", "ssdu_36spf_sampling",
])
_ULTRA_EXPS = "ei_2spf_sampling_no_rebin_fop,ei_4spf_sampling_no_rebin_fop"
_EI_EXPS = "ei_8spf_sampling_arrshift_fop,mc_8spf_slice_sampling"


# ---------------------------------------------------------------------------
# Table definitions
# Each value is either:
#   list[str]        — a single command; its stdout replaces the table block
#   list[list[str]]  — multiple commands; their stdout is joined and replaces
#                      the entire contiguous group of blocks sharing the label
# ---------------------------------------------------------------------------
TABLES: dict[str, list] = {

    # ---- Acceleration Sweep ------------------------------------------------

    "tab:acc_exp_spatial": [
        "python", _MT,
        "--log_file", _LOG_REV,
        "--exp_names", _ACCEL_EXPS,
        "--metric_type", "spatial_mc",
        "--decimals", "2",
        "--metrics", "ssim,psnr,lpips",
        "--include_metric_arrows",
        "--exclude_af", "--include_spf", "--include_temporal_resolution",
        "--caption", (
            "Spatial quality across accelerations. TR: temporal resolution (sec/frame). "
            "Values >1 SD above or below GRASP mean (same SPF) shown in red."
        ),
        "--label", "tab:acc_exp_spatial",
        "--format_mri_journal",
    ],

    "tab:acc_exp_consistency": [
        "python", _MT,
        "--log_file", _LOG_REV,
        "--exp_names", _ACCEL_EXPS,
        "--metric_type", "spatial_mc",
        "--decimals", "2",
        "--metrics", "dro_dc_mae,raw_ssdu_nmse",
        "--include_metric_arrows",
        "--exclude_af", "--include_spf", "--include_temporal_resolution",
        "--caption", (
            "Measurement consistency across accelerations. TR: temporal resolution (sec/frame). "
            "DC MAE is on DRO k-space; SSDU NMSE is on raw k-space. "
            "Values >1 SD below GRASP mean (same SPF) shown in red."
        ),
        "--label", "tab:acc_exp_consistency",
        "--format_mri_journal",
    ],

    "tab:acc_exp_temp_early": [
        "python", _MT,
        "--log_file", _LOG_ORIG,
        "--exp_names", _ACCEL_EXPS,
        "--metric_type", "temporal",
        "--temporal_subset", "all",
        "--temporal_region", "malignant",
        "--decimals", "2",
        "--metrics", "early curve correlation,early MAE,iAUC error",
        "--include_metric_arrows",
        "--exclude_af", "--include_spf", "--include_temporal_resolution",
        "--caption", (
            "Temporal fidelity in malignant ROIs: early enhancement metrics. "
            "TR: temporal resolution (sec/frame). "
            "Values >1 SD above or below GRASP mean (same SPF) shown in red."
        ),
        "--label", "tab:acc_exp_temp_early",
        "--format_mri_journal",
    ],

    "tab:acc_exp_temp_timing": [
        "python", _MT,
        "--log_file", _LOG_ORIG,
        "--exp_names", _ACCEL_EXPS,
        "--metric_type", "temporal",
        "--temporal_subset", "all",
        "--temporal_region", "malignant",
        "--decimals", "2",
        "--metrics", "arrival time error,wash in MAE",
        "--include_metric_arrows",
        "--exclude_af", "--include_spf", "--include_temporal_resolution",
        "--caption", (
            "Temporal fidelity in malignant ROIs: timing and wash-in metrics. "
            "TR: temporal resolution (sec/frame). "
            "Values >1 SD above GRASP mean (same SPF) shown in red."
        ),
        "--label", "tab:acc_exp_temp_timing",
        "--format_mri_journal",
    ],

    # ---- Ultra-High Acceleration -------------------------------------------

    "tab:ultra_acc_exp_spatial": [
        "python", _MT,
        "--log_file", _LOG_REV,
        "--exp_names", _ULTRA_EXPS,
        "--metric_type", "spatial_mc",
        "--decimals", "2",
        "--metrics", "ssim,psnr,lpips",
        "--include_metric_arrows",
        "--exclude_af", "--include_spf", "--include_temporal_resolution",
        "--caption", (
            "Spatial quality at ultra-high accelerations. TR: temporal resolution (sec/frame). "
            "Values >1 SD above or below GRASP mean (same SPF) shown in red."
        ),
        "--label", "tab:ultra_acc_exp_spatial",
        "--format_mri_journal",
    ],

    "tab:ultra_acc_exp_consistency": [
        "python", _MT,
        "--log_file", _LOG_REV,
        "--exp_names", _ULTRA_EXPS,
        "--metric_type", "spatial_mc",
        "--decimals", "2",
        "--metrics", "dro_dc_mae,raw_ssdu_nmse",
        "--include_metric_arrows",
        "--exclude_af", "--include_spf", "--include_temporal_resolution",
        "--caption", (
            "Measurement consistency at ultra-high accelerations. TR: temporal resolution (sec/frame). "
            "DC MAE is on DRO k-space; SSDU NMSE is on raw k-space. "
            "Values >1 SD below GRASP mean (same SPF) shown in red."
        ),
        "--label", "tab:ultra_acc_exp_consistency",
        "--format_mri_journal",
    ],

    "tab:ultra_acc_exp_temp_early": [
        "python", _MT,
        "--log_file", _LOG_ORIG,
        "--exp_names", _ULTRA_EXPS,
        "--metric_type", "temporal",
        "--temporal_subset", "all",
        "--temporal_region", "malignant",
        "--decimals", "2",
        "--metrics", "early curve correlation,early MAE,iAUC error",
        "--include_metric_arrows",
        "--exclude_af", "--include_spf", "--include_temporal_resolution",
        "--caption", (
            "Temporal fidelity at ultra-high accelerations in malignant ROIs: "
            "early enhancement metrics. TR: temporal resolution (sec/frame)."
        ),
        "--label", "tab:ultra_acc_exp_temp_early",
        "--format_mri_journal",
    ],

    "tab:ultra_acc_exp_temp_timing": [
        "python", _MT,
        "--log_file", _LOG_ORIG,
        "--exp_names", _ULTRA_EXPS,
        "--metric_type", "temporal",
        "--temporal_subset", "all",
        "--temporal_region", "malignant",
        "--decimals", "2",
        "--metrics", "arrival time error,wash in MAE",
        "--include_metric_arrows",
        "--exclude_af", "--include_spf", "--include_temporal_resolution",
        "--caption", (
            "Temporal fidelity at ultra-high accelerations in malignant ROIs: "
            "timing and wash-in metrics. TR: temporal resolution (sec/frame)."
        ),
        "--label", "tab:ultra_acc_exp_temp_timing",
        "--format_mri_journal",
    ],

    # ---- EI Ablation -------------------------------------------------------

    "tab:mc_ei_spatial": [
        "python", _MT,
        "--log_file", _LOG_ORIG,
        "--exp_names", _EI_EXPS,
        "--metric_type", "spatial_mc",
        "--decimals", "2",
        "--metrics", "ssim,psnr,lpips",
        "--include_metric_arrows",
        "--exclude_af", "--exclude_spf", "--exclude_temporal_resolution",
        "--caption", (
            "EI Ablation: Spatial quality at 8 SPF. "
            "Values >1 SD above GRASP mean are highlighted in red."
        ),
        "--label", "tab:mc_ei_spatial",
        "--format_mri_journal",
        "--config_keys", "EI Loss:model.losses.use_ei_loss",
    ],

    "tab:mc_ei_consistency": [
        "python", _MT,
        "--log_file", _LOG_REV,
        "--exp_names", _EI_EXPS,
        "--metric_type", "spatial_mc",
        "--decimals", "2",
        "--metrics", "dro_dc_mae,raw_ssdu_nmse",
        "--include_metric_arrows",
        "--exclude_af", "--exclude_spf", "--exclude_temporal_resolution",
        "--caption", (
            "EI Ablation: Measurement consistency at 8 SPF. "
            "Values >1 SD above GRASP mean are highlighted in red."
        ),
        "--label", "tab:mc_ei_consistency",
        "--format_mri_journal",
        "--config_keys", "EI Loss:model.losses.use_ei_loss",
    ],

    "tab:mc_ei_temp_early": [
        "python", _MT,
        "--log_file", _LOG_ORIG,
        "--exp_names", _EI_EXPS,
        "--metric_type", "temporal",
        "--temporal_subset", "all",
        "--temporal_region", "malignant",
        "--decimals", "2",
        "--metrics", "early curve correlation,early MAE,iAUC error",
        "--include_metric_arrows",
        "--exclude_af", "--exclude_spf", "--exclude_temporal_resolution",
        "--caption", (
            "EI Ablation: Early enhancement fidelity at 8 SPF. "
            "Values >1 SD above GRASP mean are highlighted in red."
        ),
        "--label", "tab:mc_ei_temp_early",
        "--format_mri_journal",
        "--config_keys", "EI Loss:model.losses.use_ei_loss",
    ],

    "tab:mc_ei_temp_timing": [
        "python", _MT,
        "--log_file", _LOG_ORIG,
        "--exp_names", _EI_EXPS,
        "--metric_type", "temporal",
        "--temporal_subset", "all",
        "--temporal_region", "malignant",
        "--decimals", "2",
        "--metrics", "arrival time error,wash in MAE",
        "--include_metric_arrows",
        "--exclude_af", "--exclude_spf", "--exclude_temporal_resolution",
        "--caption", (
            "EI Ablation: Timing and wash-in fidelity at 8 SPF. "
            "Values >1 SD above GRASP mean are highlighted in red."
        ),
        "--label", "tab:mc_ei_temp_timing",
        "--format_mri_journal",
        "--config_keys", "EI Loss:model.losses.use_ei_loss",
    ],

}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(args: list[str]) -> str:
    result = subprocess.run(
        args, capture_output=True, text=True, cwd=REPO_ROOT,
        env={**__import__("os").environ},
    )
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(args[:4])} ...")
    return result.stdout.strip()


def micromamba_run(args: list[str]) -> str:
    return run_cmd(["micromamba", "run", "-n", "recon_mri"] + args)


def find_table_block(lines: list[str], label: str) -> tuple[int, int] | None:
    """Return (begin_line, end_line) of the active table block containing label."""
    label_str = f"\\label{{{label}}}"
    label_line = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("%"):
            continue
        if label_str in line:
            label_line = i
            break
    if label_line is None:
        return None

    begin_line = None
    for i in range(label_line, -1, -1):
        if lines[i].lstrip().startswith("%"):
            continue
        if "\\begin{table}" in lines[i]:
            begin_line = i
            break

    end_line = None
    for i in range(label_line, len(lines)):
        if lines[i].lstrip().startswith("%"):
            continue
        if "\\end{table}" in lines[i]:
            end_line = i
            break

    if begin_line is None or end_line is None:
        return None
    return begin_line, end_line


def find_all_table_blocks(lines: list[str], label: str) -> list[tuple[int, int]]:
    """Return all contiguous table blocks that contain the given label."""
    label_str = f"\\label{{{label}}}"
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("%"):
            i += 1
            continue
        if label_str in lines[i]:
            # find enclosing begin/end
            begin_line = None
            for j in range(i, -1, -1):
                if lines[j].lstrip().startswith("%"):
                    continue
                if "\\begin{table}" in lines[j]:
                    begin_line = j
                    break
            end_line = None
            for j in range(i, len(lines)):
                if lines[j].lstrip().startswith("%"):
                    continue
                if "\\end{table}" in lines[j]:
                    end_line = j
                    break
            if begin_line is not None and end_line is not None:
                blocks.append((begin_line, end_line))
                i = end_line + 1
                continue
        i += 1
    return blocks


def replace_single(lines: list[str], label: str, new_content: str) -> list[str]:
    block = find_table_block(lines, label)
    if block is None:
        raise ValueError(f"Table block for label '{label}' not found in tex file.")
    begin, end = block
    return lines[:begin] + [new_content] + lines[end + 1:]


def replace_multi(lines: list[str], label: str, new_contents: list[str]) -> list[str]:
    """Replace the entire group of blocks sharing label with new_contents joined."""
    blocks = find_all_table_blocks(lines, label)
    if not blocks:
        raise ValueError(f"No table blocks found for label '{label}'.")
    first_begin = blocks[0][0]
    last_end = blocks[-1][1]
    joined = "\n\n".join(new_contents)
    return lines[:first_begin] + [joined] + lines[last_end + 1:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate and inject paper tables.")
    parser.add_argument(
        "--tables",
        default="",
        help="Comma-separated labels to regenerate (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated LaTeX but do not write the tex file.",
    )
    args = parser.parse_args()

    requested = {t.strip() for t in args.tables.split(",") if t.strip()}

    tex_content = TEX_FILE.read_text()
    lines = tex_content.split("\n")

    any_updated = False

    for key, cmd_or_cmds in TABLES.items():
        is_multi = key.startswith("multi:")
        label = key[len("multi:"):] if is_multi else key

        if requested and label not in requested:
            continue

        print(f"\n[{'MULTI' if is_multi else 'TABLE'}] {label}")

        try:
            if is_multi:
                outputs = []
                for cmd in cmd_or_cmds:
                    print(f"  running: {' '.join(cmd[:5])} ...")
                    outputs.append(micromamba_run(cmd))
                if args.dry_run:
                    print("\n\n".join(outputs))
                else:
                    lines = replace_multi(lines, label, outputs)
                    print(f"  -> replaced {len(cmd_or_cmds)} blocks in tex file")
            else:
                print(f"  running: {' '.join(cmd_or_cmds[:5])} ...")
                output = micromamba_run(cmd_or_cmds)
                if args.dry_run:
                    print(output)
                else:
                    lines = replace_single(lines, label, output)
                    print(f"  -> replaced block in tex file")
            any_updated = True
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    if not args.dry_run and any_updated:
        TEX_FILE.write_text("\n".join(lines))
        print(f"\nWrote {TEX_FILE}")


if __name__ == "__main__":
    main()

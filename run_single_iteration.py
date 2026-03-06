#!/usr/bin/env python3
"""
run_single_iteration.py — Run functional group optimization (FGO) only.

Calls loop_functional_group_optimization.py to perform FGO on a parents CSV and
writes iterations/iter_{iter:03d}/fgo_offspring.csv. No docking, PLIP, or selection.

Project layout (relative to project root):
  - loop_functional_group_optimization.py
  - data/pre_loop_df.csv (or --parents)
  - iterations/iter_{iter:03d}/  (outputs)

Usage:
  python run_single_iteration.py --iter 0 --parents data/pre_loop_df.csv
  python run_single_iteration.py --iter 0 --parents data/pre_loop_df.csv --smoke-test --smoke-size 10
  python run_single_iteration.py --iter 0 --parents data/pre_loop_df.csv --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Optional joblib for chunk-level parallelism
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# region agent log
DEBUG_LOG_PATH = "/Users/shreechatterjee/Downloads/jaks/.cursor/debug-97158e.log"
def _debug_log_run(location: str, data: Dict[str, Any]) -> None:
    payload = {"sessionId": "97158e", "runId": "pre-fix", "hypothesisId": "H1", "location": location, "message": "add_final_score columns", "data": data, "timestamp": int(time.time() * 1000)}
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# endregion

# ---------------------------------------------------------------------------
# Default analysis residue map (no LYS1154, GLY1132, SER1100, HIS1178 for heatmap default)
# ---------------------------------------------------------------------------
DEFAULT_HEATMAP_RESIDUES: Dict[str, List[str]] = {
    "JAK1": ["ARG879", "LEU891", "SER909", "SER961"],
    "JAK2": ["GLN853", "MET865", "LYS883", "TYR934"],
    "JAK3": ["SER826", "LEU838", "GLN856", "SER907"],
    "TYK2": ["ARG901", "LEU913", "ALA931", "LEU983"],
}

SMILES_COL = "Offspring Compounds"
PARENT_COL = "parent_SMILES"
COMPOUND_ID_COL = "compound_id"


# ---------------------------------------------------------------------------
# Paths and dirs
# ---------------------------------------------------------------------------

def get_project_root() -> str:
    """Project root = cwd when script is run (no hardcoded paths)."""
    return os.getcwd()


def iteration_dir(project_root: str, iter_num: int) -> str:
    return os.path.join(project_root, "iterations", f"iter_{iter_num:03d}")


def ensure_iteration_dirs(project_root: str, iter_num: int) -> Dict[str, str]:
    base = iteration_dir(project_root, iter_num)
    subdirs = ["chunks", "docking_chunk_outputs", "plip_finalists", "analysis", "logs", "complexes"]
    for d in subdirs:
        os.makedirs(os.path.join(base, d), exist_ok=True)
    return {
        "base": base,
        "chunks": os.path.join(base, "chunks"),
        "docking_chunk_outputs": os.path.join(base, "docking_chunk_outputs"),
        "plip_finalists": os.path.join(base, "plip_finalists"),
        "analysis": os.path.join(base, "analysis"),
        "logs": os.path.join(base, "logs"),
        "complexes": os.path.join(base, "complexes"),
    }


# ---------------------------------------------------------------------------
# Logging and manifest
# ---------------------------------------------------------------------------

def log_message(msg: str, log_path: Optional[str] = None) -> None:
    print(msg)
    if log_path:
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def append_error(errors_path: str, chunk_id: str, compound_id: str, stage: str, error_message: str) -> None:
    row = {"chunk_id": chunk_id, "compound_id": compound_id, "stage": stage, "error_message": error_message}
    file_exists = os.path.isfile(errors_path)
    pd.DataFrame([row]).to_csv(errors_path, mode="a", header=not file_exists, index=False)


def save_manifest(
    paths: Dict[str, str],
    args: argparse.Namespace,
    iter_num: int,
    git_commit: Optional[str] = None,
) -> None:
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit,
        "python_exe": sys.executable,
        "cli_args": vars(args),
        "iter": iter_num,
        "chunk_size": getattr(args, "chunk_size", None),
        "prefilter_fraction": getattr(args, "prefilter_fraction", None),
        "finalist_percent": getattr(args, "finalist_percent", None),
        "fgo_workers": getattr(args, "fgo_workers", None),
        "docking_workers": getattr(args, "docking_workers", None),
        "vina_cpu_per_job": getattr(args, "vina_cpu_per_job", None),
    }
    try:
        import git
        repo = git.Repo(get_project_root(), search_parent_directories=True)
        manifest["git_commit"] = repo.head.commit.hexsha
    except Exception:
        pass
    with open(os.path.join(paths["base"], "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Step 1 — Run FGO (import or subprocess)
# ---------------------------------------------------------------------------

def run_fgo(
    parents_path: str,
    output_path: str,
    project_root: str,
    max_compounds: Optional[int] = None,
    smoke_size: Optional[int] = None,
    plif_path: Optional[str] = None,
    plif_kinase: str = "JAK2",
    plif_residues: Optional[List[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run FGO; output must contain Curated_SMILES, parent_SMILES, pKi_JAK2, selectivity_score."""
    try:
        from loop_functional_group_optimization import run_functional_group_optimization
        n = max_compounds
        if smoke_size is not None:
            n = min(n or 999999, smoke_size)
        df = run_functional_group_optimization(
            input_path=parents_path,
            output_path=output_path,
            max_compounds=n,
            plif_path=plif_path,
            plif_kinase=plif_kinase,
            plif_target_residues=plif_residues,
            **kwargs,
        )
    except ImportError:
        cmd = [
            sys.executable,
            os.path.join(project_root, "loop_functional_group_optimization.py"),
            "--input", parents_path,
            "--output", output_path,
        ]
        if max_compounds is not None:
            cmd += ["--max-compounds", str(max_compounds)]
        if smoke_size is not None:
            cmd += ["--max-compounds", str(smoke_size)]
        if plif_path:
            cmd += ["--plif", plif_path, "--plif-kinase", plif_kinase]
        if plif_residues:
            cmd += ["--plif-residues", ",".join(plif_residues)]
        subprocess.run(cmd, check=True, cwd=project_root)
        df = pd.read_csv(output_path)
    required = [SMILES_COL, PARENT_COL, "pKi_JAK2", "selectivity_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"FGO output missing required columns: {missing}. Found: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Weighted QSAR; Step 3 — Per-parent cap; Step 4 — Prefilter
# ---------------------------------------------------------------------------

def compute_weighted_qsar(df: pd.DataFrame, alpha: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    out["weighted_qsar"] = out["pKi_JAK2"].fillna(0) + alpha * out["selectivity_score"].fillna(0)
    return out


def cap_per_parent(
    df: pd.DataFrame,
    parent_col: str = PARENT_COL,
    score_col: str = "weighted_qsar",
    n: int = 3,
) -> pd.DataFrame:
    if df.empty or parent_col not in df.columns or score_col not in df.columns:
        return df
    out = df.copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out = out.sort_values([parent_col, score_col], ascending=[True, False], na_position="last")
    return out.groupby(parent_col, as_index=False, group_keys=False).head(n).reset_index(drop=True)


def prefilter_bottom_fraction(df: pd.DataFrame, fraction: float, score_col: str = "weighted_qsar") -> pd.DataFrame:
    if fraction <= 0 or fraction >= 1 or df.empty:
        return df
    out = df.sort_values(score_col, ascending=False, na_position="last").reset_index(drop=True)
    keep = int(max(1, len(out) * (1 - fraction)))
    return out.head(keep)


# ---------------------------------------------------------------------------
# Step 5 — Chunking
# ---------------------------------------------------------------------------

def write_chunks(
    df: pd.DataFrame,
    chunks_dir: str,
    chunk_size: int,
    smiles_col: str = SMILES_COL,
) -> List[str]:
    """Write chunk CSVs; return list of chunk paths."""
    os.makedirs(chunks_dir, exist_ok=True)
    paths = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        path = os.path.join(chunks_dir, f"chunk_{i:04d}_{i + len(chunk):04d}.csv")
        chunk.to_csv(path, index=False)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Step 6 — Docking only (no PLIP); save complexes for later PLIP on finalists
# ---------------------------------------------------------------------------

DOCKING_OUTPUT_COLS = [COMPOUND_ID_COL, SMILES_COL, "isoform", "vina_score"]


def _run_docking_one_chunk(
    chunk_csv_path: str,
    chunk_id: str,
    complexes_dir: str,
    receptor_pdbqt_folder: str,
    receptor_pdb_folder: str,
    vina_bin: Optional[str],
    vina_timeout: int,
    done_marker_path: str,
    errors_path: str,
    sample_vina_log_path: Optional[str] = None,
    complexes_output_dir: Optional[str] = None,
    progress_log_path: Optional[str] = None,
    stream_first_vina: bool = False,
) -> List[Dict[str, Any]]:
    """Run Vina docking for one chunk; save complex PDBs (chunk dir + optional single folder); return rows. No PLIP."""
    try:
        from iteration_docking_plif import (
            prepare_ligand,
            prepare_receptor_cache,
            run_vina_docking,
            pdbqt_to_pdb_string_rdkit,
            get_vina_bin,
            DEFAULT_DOCKING_CFGS,
        )
    except ImportError:
        raise RuntimeError("iteration_docking_plif not importable; cannot run docking-only step.")
    df = pd.read_csv(chunk_csv_path)
    if SMILES_COL not in df.columns:
        df = df.rename(columns={c: SMILES_COL for c in df.columns if "smiles" in c.lower() and df[c].dtype == object})
    if SMILES_COL not in df.columns:
        raise ValueError(f"Chunk has no SMILES column: {list(df.columns)}")
    df[COMPOUND_ID_COL] = df[SMILES_COL].astype(str).apply(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    )
    receptor_map = prepare_receptor_cache(receptor_pdbqt_folder, receptor_pdb_folder)
    docking_cfgs = DEFAULT_DOCKING_CFGS
    vina_bin = vina_bin or get_vina_bin()
    results = []
    copied_sample_log = False
    first_dock_done = False
    chunk_complexes_dir = os.path.join(complexes_dir, chunk_id)
    os.makedirs(chunk_complexes_dir, exist_ok=True)
    if complexes_output_dir:
        os.makedirs(complexes_output_dir, exist_ok=True)
    for _, row in df.iterrows():
        cid = str(row[COMPOUND_ID_COL])
        smi = str(row[SMILES_COL]).strip()
        if not smi or smi.lower() in ("nan", "none"):
            continue
        try:
            pdbqt_str = prepare_ligand(smi)
        except Exception as e:
            append_error(errors_path, chunk_id, cid, "prepare_ligand", str(e))
            continue
        for iso_name, rec in receptor_map.items():
            cfg = docking_cfgs.get(iso_name)
            if not cfg:
                continue
            job_dir = os.path.join(chunk_complexes_dir, f"{cid[:8]}_{iso_name}")
            os.makedirs(job_dir, exist_ok=True)
            lig_in = os.path.join(job_dir, "lig.pdbqt")
            out_pdbqt = os.path.join(job_dir, "docked.pdbqt")
            log_path = os.path.join(job_dir, "vina.log")
            complex_pdb = os.path.join(chunk_complexes_dir, f"{cid[:8]}_{iso_name}_complex.pdb")
            try:
                with open(lig_in, "w") as f:
                    f.write(pdbqt_str)
                stream_this = stream_first_vina and not first_dock_done
                score = run_vina_docking(
                    lig_in, rec["pdbqt"], out_pdbqt, log_path,
                    cfg["center"], cfg["size"],
                    exhaustiveness=cfg.get("exhaustiveness", 8),
                    vina_bin=vina_bin,
                    timeout=vina_timeout,
                    use_api=True,
                    stream_output=stream_this,
                )
                first_dock_done = True
                if sample_vina_log_path and not copied_sample_log and os.path.isfile(log_path):
                    try:
                        shutil.copy2(log_path, sample_vina_log_path)
                        copied_sample_log = True
                    except Exception:
                        pass
                pdb_string = pdbqt_to_pdb_string_rdkit(out_pdbqt)
                with open(rec["pdb"], "r") as f:
                    protein_data = f.read()
                with open(complex_pdb, "w") as f:
                    f.write(protein_data)
                    f.write("\n")
                    f.write(pdb_string)
                if complexes_output_dir:
                    unified_name = f"{cid}_{iso_name}_complex.pdb"
                    unified_path = os.path.join(complexes_output_dir, unified_name)
                    try:
                        shutil.copy2(complex_pdb, unified_path)
                    except Exception:
                        pass
                if progress_log_path:
                    try:
                        with open(progress_log_path, "a") as pf:
                            pf.write(f"Docking {cid[:12]}... -> {iso_name} -> score {score}\n")
                    except Exception:
                        pass
                results.append({
                    COMPOUND_ID_COL: cid,
                    SMILES_COL: smi,
                    "isoform": iso_name,
                    "vina_score": score,
                })
            except Exception as e:
                append_error(errors_path, chunk_id, cid, "docking", str(e))
                results.append({COMPOUND_ID_COL: cid, SMILES_COL: smi, "isoform": iso_name, "vina_score": None})
            finally:
                if os.path.exists(job_dir):
                    shutil.rmtree(job_dir, ignore_errors=True)
    with open(done_marker_path, "w") as f:
        f.write("done")
    return results


def run_docking_only(
    chunk_paths: List[str],
    paths: Dict[str, str],
    receptor_pdbqt_folder: str,
    receptor_pdb_folder: str,
    vina_bin: Optional[str],
    vina_timeout: int,
    docking_workers: int,
    errors_path: str,
    sample_vina_log_path: Optional[str] = None,
    progress_log_path: Optional[str] = None,
    stream_first_vina: bool = False,
) -> pd.DataFrame:
    """Run docking (no PLIP) for all chunks; aggregate to docking_results.csv. Save complexes in chunk dirs + paths['complexes']."""
    complexes_output_dir = paths.get("complexes")
    all_rows = []
    if HAS_JOBLIB and docking_workers > 1:
        jobs = []
        for i, cp in enumerate(chunk_paths):
            chunk_id = os.path.splitext(os.path.basename(cp))[0]
            complexes_dir = os.path.join(paths["docking_chunk_outputs"], chunk_id)
            done = os.path.join(paths["docking_chunk_outputs"], f"{chunk_id}.done")
            if os.path.isfile(done):
                out_csv = os.path.join(paths["docking_chunk_outputs"], f"{chunk_id}_results.csv")
                if os.path.isfile(out_csv):
                    all_rows.append(pd.read_csv(out_csv))
                continue
            stream_this = stream_first_vina and (i == 0)
            jobs.append((cp, chunk_id, complexes_dir, done, errors_path, sample_vina_log_path, complexes_output_dir, progress_log_path, stream_this))
        if jobs:
            def run_one(job):
                cp, cid, cdir, done_path, err_path, sample_log, cplx_out, prog_log, stream_v = job
                return _run_docking_one_chunk(
                    cp, cid, cdir,
                    receptor_pdbqt_folder, receptor_pdb_folder,
                    vina_bin, vina_timeout, done_path, err_path,
                    sample_vina_log_path=sample_log,
                    complexes_output_dir=cplx_out,
                    progress_log_path=prog_log,
                    stream_first_vina=stream_v,
                )
            results_list = Parallel(n_jobs=docking_workers, backend="loky")(
                delayed(run_one)(j) for j in jobs
            )
            for rows in results_list:
                if rows:
                    all_rows.append(pd.DataFrame(rows))
            for (cp, cid, cdir, done_path, _err_path, _sample_log, _cplx, _prog, _), rows in zip(jobs, results_list):
                if rows:
                    out_csv = os.path.join(paths["docking_chunk_outputs"], f"{cid}_results.csv")
                    out_df = pd.DataFrame(rows)
                    out_df = out_df[[c for c in DOCKING_OUTPUT_COLS if c in out_df.columns]]
                    out_df.to_csv(out_csv, index=False)
    else:
        for i, cp in enumerate(chunk_paths):
            chunk_id = os.path.splitext(os.path.basename(cp))[0]
            complexes_dir = os.path.join(paths["docking_chunk_outputs"], chunk_id)
            done = os.path.join(paths["docking_chunk_outputs"], f"{chunk_id}.done")
            if os.path.isfile(done):
                out_csv = os.path.join(paths["docking_chunk_outputs"], f"{chunk_id}_results.csv")
                if os.path.isfile(out_csv):
                    all_rows.append(pd.read_csv(out_csv))
                continue
            stream_this = stream_first_vina and (i == 0)
            rows = _run_docking_one_chunk(
                cp, chunk_id, complexes_dir,
                receptor_pdbqt_folder, receptor_pdb_folder,
                vina_bin, vina_timeout, done, errors_path,
                sample_vina_log_path=sample_vina_log_path,
                complexes_output_dir=complexes_output_dir,
                progress_log_path=progress_log_path,
                stream_first_vina=stream_this,
            )
            if rows:
                out_df = pd.DataFrame(rows)
                all_rows.append(out_df)
                out_csv = os.path.join(paths["docking_chunk_outputs"], f"{chunk_id}_results.csv")
                out_df = out_df[[c for c in DOCKING_OUTPUT_COLS if c in out_df.columns]]
                out_df.to_csv(out_csv, index=False)
    if not all_rows:
        return pd.DataFrame(columns=DOCKING_OUTPUT_COLS)
    docking_long = pd.concat(all_rows, ignore_index=True)
    # Ensure vina_score and column order for downstream and CSV
    for c in DOCKING_OUTPUT_COLS:
        if c not in docking_long.columns:
            docking_long[c] = None if c == "vina_score" else ""
    docking_long = docking_long[DOCKING_OUTPUT_COLS]
    return docking_long


# ---------------------------------------------------------------------------
# Step 7 — Intermediate score (QSAR + docking, min-max normalized)
# ---------------------------------------------------------------------------

def add_intermediate_score(
    prefiltered_df: pd.DataFrame,
    docking_long: pd.DataFrame,
    qsar_weight: float = 0.7,
    dock_weight: float = 0.3,
) -> pd.DataFrame:
    """Merge prefiltered (weighted_qsar) with best vina per compound; compute intermediate_score."""
    # Best vina per compound (min = best)
    best_vina = docking_long.groupby(COMPOUND_ID_COL, as_index=False)["vina_score"].min()
    best_vina = best_vina.rename(columns={"vina_score": "best_vina_score"})
    # Ensure prefiltered has compound_id
    if COMPOUND_ID_COL not in prefiltered_df.columns:
        prefiltered_df = prefiltered_df.copy()
        prefiltered_df[COMPOUND_ID_COL] = prefiltered_df[SMILES_COL].astype(str).apply(
            lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
        )
    merged = prefiltered_df.merge(best_vina, on=COMPOUND_ID_COL, how="left")
    qsar = merged["weighted_qsar"].fillna(0)
    vina = merged["best_vina_score"]
    vina_inv = -vina  # higher = better
    vina_inv = vina_inv.fillna(vina_inv.min())
    qsar_norm = (qsar - qsar.min()) / (qsar.max() - qsar.min() + 1e-12)
    dock_norm = (vina_inv - vina_inv.min()) / (vina_inv.max() - vina_inv.min() + 1e-12)
    merged["qsar_norm"] = qsar_norm
    merged["dock_norm"] = dock_norm
    merged["intermediate_score"] = qsar_weight * qsar_norm + dock_weight * dock_norm
    merged["intermediate_score"] = pd.to_numeric(merged["intermediate_score"], errors="coerce").fillna(0)
    return merged


# ---------------------------------------------------------------------------
# Step 8 — Finalist selection (top finalist_percent %)
# ---------------------------------------------------------------------------

def select_finalists(df: pd.DataFrame, finalist_percent: float, score_col: str = "intermediate_score") -> pd.DataFrame:
    if df.empty or finalist_percent <= 0 or finalist_percent > 100:
        return df
    if score_col not in df.columns:
        return df
    # Ensure numeric so nlargest does not raise (e.g. when column came from CSV as object)
    score_vals = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    df = df.copy()
    df[score_col] = score_vals
    n = max(1, int(round(len(df) * finalist_percent / 100)))
    return df.nlargest(n, score_col, keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 9 — PLIP only for finalists
# ---------------------------------------------------------------------------

def run_plip_for_finalists(
    finalists_df: pd.DataFrame,
    paths: Dict[str, str],
    docking_chunk_outputs_dir: str,
    target_residues_map: Dict[str, List[str]],
    plip_timeout: int = 120,
) -> pd.DataFrame:
    """For each finalist, find saved complex PDBs (one per isoform), run PLIP, collect binary PLIF + plif_hits."""
    try:
        from iteration_docking_plif import (
            run_plip_subprocess_and_get_interactions,
            plip_interactions_to_binary_plif,
        )
    except ImportError:
        raise RuntimeError("iteration_docking_plif not importable; cannot run PLIP.")
    all_residues = set()
    for res_list in target_residues_map.values():
        all_residues.update(r.upper().replace(" ", "") for r in res_list)
    sorted_res = sorted(all_residues)
    compound_ids = finalists_df[COMPOUND_ID_COL].astype(str).unique().tolist()
    rows = []
    plip_dir = paths["plip_finalists"]
    os.makedirs(plip_dir, exist_ok=True)
    for cid in compound_ids:
        prefix = cid[:8]
        row = {COMPOUND_ID_COL: cid}
        plif_counts = []
        for iso_name, res_list in target_residues_map.items():
            complex_path = None
            for chunk_name in os.listdir(docking_chunk_outputs_dir):
                if not os.path.isdir(os.path.join(docking_chunk_outputs_dir, chunk_name)):
                    continue
                cand = os.path.join(docking_chunk_outputs_dir, chunk_name, f"{prefix}_{iso_name}_complex.pdb")
                if os.path.isfile(cand):
                    complex_path = cand
                    break
            if complex_path is None:
                for r in res_list:
                    row[r.upper().replace(" ", "")] = 0
                continue
            try:
                plip_raw = run_plip_subprocess_and_get_interactions(complex_path, timeout=plip_timeout)
                plif = plip_interactions_to_binary_plif(plip_raw, res_list)
                hits = sum(plif.values())
                plif_counts.append(hits)
                for k, v in plif.items():
                    row[k] = v
                with open(os.path.join(plip_dir, f"{prefix}_{iso_name}_plip.json"), "w") as f:
                    json.dump(plip_raw, f, indent=2)
            except Exception:
                for r in res_list:
                    row[r.upper().replace(" ", "")] = 0
        row["plif_hits"] = sum(plif_counts) if plif_counts else 0
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in sorted_res:
        if c not in out.columns:
            out[c] = 0
    return out


# ---------------------------------------------------------------------------
# Step 10 — Final score; Step 11 — Next parents
# ---------------------------------------------------------------------------

def add_final_score(
    finalists_with_plif: pd.DataFrame,
    intermediate_scored_df: pd.DataFrame,
    qsar_weight: float = 0.6,
    dock_weight: float = 0.3,
    plif_weight: float = 0.1,
) -> pd.DataFrame:
    """Merge finalists (with PLIF) with intermediate-scored df; normalize and compute final_score."""
    cols = [COMPOUND_ID_COL, "weighted_qsar", "best_vina_score"]
    available = [c for c in cols if c in intermediate_scored_df.columns]
    # region agent log
    _debug_log_run(
        "add_final_score",
        {"available": available, "has_weighted_qsar": "weighted_qsar" in intermediate_scored_df.columns, "inter_cols": list(intermediate_scored_df.columns)[:20]},
    )
    # endregion
    if not available or COMPOUND_ID_COL not in available:
        finalists_with_plif = finalists_with_plif.copy()
        finalists_with_plif["final_score"] = 0.0
        return finalists_with_plif
    merge_df = intermediate_scored_df[available].drop_duplicates(COMPOUND_ID_COL)
    merged = finalists_with_plif.merge(merge_df, on=COMPOUND_ID_COL, how="left")
    # Use only columns present in merged so we never KeyError (e.g. when intermediate_scored_df lacked weighted_qsar)
    qsar_n = merged["weighted_qsar"].fillna(0) if "weighted_qsar" in merged.columns else np.zeros(len(merged))
    qsar_norm = (qsar_n - qsar_n.min()) / (qsar_n.max() - qsar_n.min() + 1e-12)
    vina_inv = -merged["best_vina_score"].fillna(0) if "best_vina_score" in merged.columns else np.zeros(len(merged))
    dock_norm = (vina_inv - vina_inv.min()) / (vina_inv.max() - vina_inv.min() + 1e-12)
    plif = merged["plif_hits"].fillna(0) if "plif_hits" in merged.columns else np.zeros(len(merged))
    plif_norm = (plif - plif.min()) / (plif.max() - plif.min() + 1e-12)
    merged["final_score"] = qsar_weight * qsar_norm + dock_weight * dock_norm + plif_weight * plif_norm
    return merged


def select_next_parents(
    scored_df: pd.DataFrame,
    n: int,
    score_col: str = "final_score",
) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df
    return scored_df.nlargest(n, score_col, keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis section (4 figures + table)
# ---------------------------------------------------------------------------

def discover_iterations(project_root: str) -> List[Tuple[int, str]]:
    iters_dir = os.path.join(project_root, "iterations")
    if not os.path.isdir(iters_dir):
        return []
    out = []
    for name in sorted(os.listdir(iters_dir)):
        if name.startswith("iter_") and os.path.isdir(os.path.join(iters_dir, name)):
            try:
                num = int(name.split("_")[1])
                out.append((num, os.path.join(iters_dir, name)))
            except (IndexError, ValueError):
                continue
    return sorted(out, key=lambda x: x[0])


def run_analysis(
    project_root: str,
    current_iter: int,
    paths: Dict[str, str],
    top_k_plots: int = 20,
    panels_count: int = 4,
    heatmap_residues: Optional[Dict[str, List[str]]] = None,
    analysis_out_formats: str = "png,csv",
    top_final_table: int = 10,
) -> None:
    heatmap_residues = heatmap_residues or DEFAULT_HEATMAP_RESIDUES
    formats = [x.strip() for x in analysis_out_formats.split(",")]
    iterations = discover_iterations(project_root)
    if not iterations:
        return
    analysis_dir = paths["analysis"]
    os.makedirs(analysis_dir, exist_ok=True)
    results_analysis = os.path.join(project_root, "results", "analysis")
    os.makedirs(results_analysis, exist_ok=True)

    # Collect per-iteration data
    iter_data = []
    for num, it_path in iterations:
        fgo_path = os.path.join(it_path, "fgo_offspring.csv")
        prefilter_path = os.path.join(it_path, "prefiltered_selected.csv")
        dock_path = os.path.join(it_path, "docking_results.csv")
        finalists_path = os.path.join(it_path, "finalists.csv")
        plip_path = os.path.join(it_path, "plip_finalists", "plip_finalists.csv")
        if not os.path.isfile(fgo_path):
            continue
        df_fgo = pd.read_csv(fgo_path)
        score_col = "weighted_qsar" if "weighted_qsar" in df_fgo.columns else "selectivity_score"
        top = df_fgo.nlargest(top_k_plots, score_col, keep="first")
        median_sel = top["selectivity_score"].median() if "selectivity_score" in top.columns else np.nan
        best_idx = top["selectivity_score"].idxmax() if "selectivity_score" in top.columns else 0
        best_sel = top.loc[best_idx, "selectivity_score"] if best_idx in top.index else np.nan
        best_smi = top.loc[best_idx, SMILES_COL] if best_idx in top.index else ""
        iter_data.append({
            "iter": num,
            "path": it_path,
            "median_selectivity": median_sel,
            "best_selectivity": best_sel,
            "best_SMILES": best_smi,
            "df_fgo": df_fgo,
            "df_prefilter": pd.read_csv(prefilter_path) if os.path.isfile(prefilter_path) else None,
            "df_dock": pd.read_csv(dock_path) if os.path.isfile(dock_path) else None,
            "df_finalists": pd.read_csv(finalists_path) if os.path.isfile(finalists_path) else None,
            "plip_path": plip_path,
        })

    # Figure 1 — Iteration selectivity trend
    if iter_data:
        iters_n = [d["iter"] for d in iter_data]
        medians = [d["median_selectivity"] for d in iter_data]
        bests = [d["best_selectivity"] for d in iter_data]
        fig1_data = pd.DataFrame({"iteration": iters_n, "median_selectivity": medians, "best_selectivity": bests})
        fig1_data.to_csv(os.path.join(analysis_dir, "figure1_selectivity_trend.csv"), index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(iters_n, medians, "b-", label="Median (top-k)")
            ax.scatter(iters_n, bests, color="orange", s=50, label="Best")
            for i, (x, y) in enumerate(zip(iters_n, bests)):
                ax.annotate(f"{y:.2f}", (x, y), fontsize=8, alpha=0.8)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Selectivity")
            ax.legend()
            ax.set_title("Iteration Selectivity Trend")
            fig.savefig(os.path.join(analysis_dir, "figure1_selectivity_trend.png"), dpi=300, bbox_inches="tight")
            plt.close()
            if "png" in formats:
                shutil.copy(os.path.join(analysis_dir, "figure1_selectivity_trend.png"),
                            os.path.join(results_analysis, "figure1_selectivity_trend.png"))
        except Exception:
            pass
    # Figure 2 — JAK2 vs others panels
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panel_inds = np.linspace(0, len(iter_data) - 1, min(panels_count, len(iter_data)), dtype=int) if iter_data else []
        n_panels = len(panel_inds)
        if n_panels > 0:
            fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
            if n_panels == 1:
                axes = [axes]
            for pi, idx in enumerate(panel_inds):
                d = iter_data[idx]
                df = d["df_fgo"]
                if df is None or "pKi_JAK2" not in df.columns:
                    continue
                others = ["pKi_JAK1", "pKi_JAK3", "pKi_TYK2"]
                existing = [c for c in others if c in df.columns]
                if not existing:
                    continue
                mean_others = df[existing].mean(axis=1)
                axes[pi].scatter(mean_others, df["pKi_JAK2"], alpha=0.6)
                axes[pi].set_xlabel("Mean affinity (JAK1/3/TYK2)")
                axes[pi].set_ylabel("pKi_JAK2")
                axes[pi].set_title(f"Iter {d['iter']}")
            fig.suptitle("JAK2 vs Other Isoforms")
            fig.savefig(os.path.join(analysis_dir, "figure2_jak2_vs_others_panels.png"), dpi=300, bbox_inches="tight")
            plt.close()
            fig2_csv = []
            for idx in panel_inds:
                d = iter_data[idx]
                df = d["df_fgo"]
                if df is not None and "pKi_JAK2" in df.columns:
                    others = [c for c in ["pKi_JAK1", "pKi_JAK3", "pKi_TYK2"] if c in df.columns]
                    if others:
                        fig2_csv.append(df[["pKi_JAK2"] + others].assign(iteration=d["iter"]))
            if fig2_csv:
                pd.concat(fig2_csv, ignore_index=True).to_csv(
                    os.path.join(analysis_dir, "figure2_jak2_vs_others_panels.csv"), index=False)
    except Exception:
        pass
    # Figure 3 — Residue heatmap
    all_res = []
    for res_list in heatmap_residues.values():
        all_res.extend([r.upper().replace(" ", "") for r in res_list])
    residues = sorted(set(all_res))
    heatmap_rows = []
    for d in iter_data:
        row = {"iteration": d["iter"]}
        if os.path.isfile(d["plip_path"]):
            plip_df = pd.read_csv(d["plip_path"])
            top_cids = []
            if d["df_finalists"] is not None and not d["df_finalists"].empty:
                top_cids = d["df_finalists"][COMPOUND_ID_COL].head(top_k_plots).astype(str).tolist()
            if not top_cids and "compound_id" in plip_df.columns:
                top_cids = plip_df[COMPOUND_ID_COL].head(top_k_plots).astype(str).tolist()
            sub = plip_df[plip_df[COMPOUND_ID_COL].astype(str).isin(top_cids)] if top_cids else plip_df
            n = len(sub)
            for res in residues:
                if res in sub.columns:
                    row[res] = sub[res].sum() / (n or 1)
                else:
                    row[res] = 0.0
        else:
            for res in residues:
                row[res] = np.nan
        heatmap_rows.append(row)
    if heatmap_rows:
        heat_df = pd.DataFrame(heatmap_rows)
        heat_df.to_csv(os.path.join(analysis_dir, "figure3_residue_heatmap.csv"), index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(max(6, len(residues) * 0.5), max(4, len(heatmap_rows) * 0.4)))
            mat = heat_df[residues].values if all(c in heat_df.columns for c in residues) else np.zeros((len(heat_df), len(residues)))
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(residues)))
            ax.set_xticklabels(residues, rotation=45, ha="right")
            ax.set_yticks(range(len(heat_df)))
            ax.set_yticklabels(heat_df["iteration"].tolist())
            ax.set_ylabel("Iteration")
            plt.colorbar(im, ax=ax, label="Fraction interacting")
            fig.savefig(os.path.join(analysis_dir, "figure3_residue_heatmap.png"), dpi=300, bbox_inches="tight")
            plt.close()
        except Exception:
            pass
    # Figure 4 — Top N table (current iteration only)
    current_path = iteration_dir(project_root, current_iter)
    finalists_path = os.path.join(current_path, "finalists.csv")
    docking_plif_path = os.path.join(current_path, "docking_results_with_plif.csv")
    next_path = os.path.join(current_path, "next_parents.csv")
    parents_path = os.path.join(project_root, "data", "pre_loop_df.csv")
    for prev_iter in range(current_iter - 1, -1, -1):
        prev_next = os.path.join(iteration_dir(project_root, prev_iter), "next_parents.csv")
        if os.path.isfile(prev_next):
            parents_path = prev_next
            break
    if os.path.isfile(docking_plif_path):
        df_final = pd.read_csv(docking_plif_path)
        if "final_score" in df_final.columns:
            top_df = df_final.nlargest(top_final_table, "final_score", keep="first")
        else:
            top_df = df_final.head(top_final_table)
    elif os.path.isfile(next_path):
        top_df = pd.read_csv(next_path).head(top_final_table)
    else:
        top_df = pd.DataFrame()
    if not top_df.empty and PARENT_COL in top_df.columns:
        parents_df = pd.read_csv(parents_path) if os.path.isfile(parents_path) else pd.DataFrame()
        parent_lookup = parents_df.set_index(SMILES_COL) if SMILES_COL in parents_df.columns else pd.DataFrame()
        rows = []
        for _, r in top_df.iterrows():
            opt_id = r.get(COMPOUND_ID_COL, "")
            opt_smi = r.get(SMILES_COL, "")
            opt_pki = r.get("pKi_JAK2", np.nan)
            opt_sel = r.get("selectivity_score", np.nan)
            par_smi = r.get(PARENT_COL, "")
            par_pki = par_sel = np.nan
            if par_smi and not parent_lookup.empty and par_smi in parent_lookup.index:
                par_pki = parent_lookup.loc[par_smi, "pKi_JAK2"] if "pKi_JAK2" in parent_lookup.columns else np.nan
                par_sel = parent_lookup.loc[par_smi, "selectivity_score"] if "selectivity_score" in parent_lookup.columns else np.nan
            delta_pki = (opt_pki - par_pki) if np.isfinite(opt_pki) and np.isfinite(par_pki) else np.nan
            delta_sel = (opt_sel - par_sel) if np.isfinite(opt_sel) and np.isfinite(par_sel) else np.nan
            rows.append({
                "optimized_compound_id": opt_id,
                "optimized_SMILES": opt_smi,
                "optimized_pKi_JAK2": opt_pki,
                "optimized_selectivity_score": opt_sel,
                "parent_SMILES": par_smi,
                "parent_pKi_JAK2": par_pki,
                "parent_selectivity_score": par_sel,
                "delta_pKi_JAK2": delta_pki,
                "delta_selectivity": delta_sel,
            })
        table_df = pd.DataFrame(rows)
        table_df.to_csv(os.path.join(analysis_dir, "figure4_top10_table.csv"), index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.4)))
            ax.axis("off")
            tbl = ax.table(
                cellText=table_df.values,
                colLabels=table_df.columns,
                loc="center",
                cellLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            fig.savefig(os.path.join(analysis_dir, "figure4_top10_table.png"), dpi=300, bbox_inches="tight")
            plt.close()
        except Exception:
            pass
        try:
            table_df.to_html(os.path.join(analysis_dir, "figure4_top10_table.html"), index=False)
        except Exception:
            pass
    manifest_analysis = {
        "top_k_plots": top_k_plots,
        "panels_count": panels_count,
        "top_final_table": top_final_table,
        "heatmap_residues": heatmap_residues,
        "missing_plip_handled": "fraction computed over available PLIP data; missingness recorded",
    }
    with open(os.path.join(analysis_dir, "manifest_analysis.json"), "w") as f:
        json.dump(manifest_analysis, f, indent=2)
    for fname in os.listdir(analysis_dir):
        src = os.path.join(analysis_dir, fname)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(results_analysis, fname))


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_one_iteration(args: argparse.Namespace) -> None:
    """Run FGO only (no docking, PLIP, or selection). Calls loop_functional_group_optimization.py."""
    project_root = get_project_root()
    iter_num = args.iter
    paths = ensure_iteration_dirs(project_root, iter_num)
    log_path = os.path.join(paths["logs"], "run.log")

    parents_path = os.path.abspath(args.parents)
    if not os.path.isfile(parents_path):
        raise FileNotFoundError(f"Parents file not found: {parents_path}")

    save_manifest(paths, args, iter_num)

    # FGO only — call loop_functional_group_optimization
    fgo_out = os.path.join(paths["base"], "fgo_offspring.csv")
    if os.path.isfile(fgo_out) and not args.dry_run:
        log_message(f"Skip FGO (exists): {fgo_out}", log_path)
        print(f"Step 1: Skip (exists): {fgo_out}", file=sys.stderr)
        df_fgo = pd.read_csv(fgo_out)
    else:
        log_message("Step 1: Run FGO", log_path)
        print("Step 1: Run FGO", file=sys.stderr)
        df_fgo = run_fgo(
            parents_path,
            fgo_out,
            project_root,
            max_compounds=getattr(args, "max_compounds", None),
            smoke_size=getattr(args, "smoke_size", None),
            plif_path=getattr(args, "plif_path", None),
            plif_kinase=getattr(args, "plif_kinase", "JAK2"),
            plif_residues=getattr(args, "plif_residues", None),
        )
        df_fgo.to_csv(fgo_out, index=False)

    log_message(f"FGO complete. Saved {fgo_out} with {len(df_fgo)} compounds.", log_path)
    print(f"Saved: {fgo_out}", file=sys.stderr)


def main() -> int:
    root = get_project_root()
    parser = argparse.ArgumentParser(
        description="Run functional group optimization (FGO) only via loop_functional_group_optimization.py. No docking.",
    )
    parser.add_argument("--iter", type=int, required=True, help="Iteration number (e.g. 0).")
    parser.add_argument("--parents", required=True, help="Path to parents CSV (e.g. data/pre_loop_df.csv).")
    parser.add_argument("--max-compounds", type=int, default=None, dest="max_compounds", help="Max compounds for FGO (default: no limit).")
    parser.add_argument("--alpha-selectivity", type=float, default=1.0, dest="alpha_selectivity")
    parser.add_argument("--max-per-parent", type=int, default=3, dest="max_per_parent")
    parser.add_argument("--cap-after-prefilter", action="store_true", dest="cap_after_prefilter")
    parser.add_argument("--prefilter-fraction", type=float, default=0.5, dest="prefilter_fraction")
    parser.add_argument("--chunk-size", type=int, default=100, dest="chunk_size")
    parser.add_argument("--finalist-percent", type=float, default=10, dest="finalist_percent")
    parser.add_argument("--next-parent-count", type=int, default=50, dest="next_parent_count")
    parser.add_argument("--receptor-pdbqt", default="CHIMERA_PDBQT_FINAL_VALIDATED", dest="receptor_pdbqt")
    parser.add_argument("--receptor-pdb", default="PDB - Chimera", dest="receptor_pdb")
    parser.add_argument("--vina-bin", default=None, dest="vina_bin")
    parser.add_argument("--vina-timeout", type=int, default=300, dest="vina_timeout")
    parser.add_argument("--print-vina-log-once", action="store_true", dest="print_vina_log_once", help="Print Vina console log for one compound to stderr (to verify docking).")
    parser.add_argument("--plip-timeout", type=int, default=120, dest="plip_timeout")
    parser.add_argument("--fgo-workers", type=int, default=max(1, os.cpu_count() or 2 - 1), dest="fgo_workers")
    parser.add_argument("--docking-workers", type=int, default=2, dest="docking_workers")
    parser.add_argument("--vina-cpu-per-job", type=int, default=1, dest="vina_cpu_per_job")
    parser.add_argument("--qsar-weight-intermediate", type=float, default=0.7, dest="qsar_weight_intermediate")
    parser.add_argument("--dock-weight-intermediate", type=float, default=0.3, dest="dock_weight_intermediate")
    parser.add_argument("--qsar-weight", type=float, default=0.6, dest="qsar_weight")
    parser.add_argument("--dock-weight", type=float, default=0.3, dest="dock_weight")
    parser.add_argument("--plif-weight", type=float, default=0.1, dest="plif_weight")
    parser.add_argument("--plif-path", default=None, dest="plif_path")
    parser.add_argument("--plif-kinase", default="JAK2", dest="plif_kinase")
    parser.add_argument("--plif-residues", default=None, dest="plif_residues", help="Comma-separated residues")
    parser.add_argument("--analysis", action="store_true", default=True, dest="analysis")
    parser.add_argument("--no-analysis", action="store_false", dest="analysis")
    parser.add_argument("--top-k-plots", type=int, default=20, dest="top_k_plots")
    parser.add_argument("--panels-count", type=int, default=4, dest="panels_count")
    parser.add_argument("--top-final-table", type=int, default=10, dest="top_final_table")
    parser.add_argument("--heatmap-residues", default=None, dest="heatmap_residues", help="Path to JSON with kinase -> list of residues for heatmap; default uses built-in map.")
    parser.add_argument("--analysis-out-formats", default="png,csv", dest="analysis_out_formats")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--smoke-test", action="store_true", dest="smoke_test")
    parser.add_argument("--smoke-size", type=int, default=10, dest="smoke_size")
    args = parser.parse_args()

    if args.dock_weight_intermediate + args.qsar_weight_intermediate != 1.0:
        pass  # Allow any weights; user responsibility
    if args.docking_workers * args.vina_cpu_per_job > (os.cpu_count() or 4):
        print("Warning: docking_workers * vina_cpu_per_job may exceed CPU count.", file=sys.stderr)
    if args.plif_residues:
        args.plif_residues = [r.strip() for r in args.plif_residues.split(",") if r.strip()]

    if args.smoke_test:
        args.smoke_size = args.smoke_size or 10

    if args.dry_run:
        ensure_iteration_dirs(root, args.iter)
        print("Dry run: config valid; would run FGO only.")
        return 0

    run_one_iteration(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

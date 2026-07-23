"""Train Transolver on the SU2 hypersonic dataset.

One script covers both the fine-tune-from-AirfRANS-pretrain and from-scratch
baselines (toggle with ``--pretrain PATH`` or its absence). The slice-count
ablation is a single ``--slice-num`` flag; sweep it across {4, 8, 16, 32, 64}
by re-running with different values and different ``--out`` directories.

Splits
------
Train and val are drawn from the ``core`` group only, stratified by
``geom_id`` so val cases sit on geometries the train set has not seen. The
OOD slabs (``cone_high``, ``mach_high``, ``nose_large``, ``nose_small``) are
held out entirely as test sets; per-slab metrics are reported during eval.

Three evaluation tiers, in increasing difficulty:
``test_interp`` (interpolation: random cases held out from train geometries,
so unseen freestream conditions on seen shapes), ``test`` (family holdout:
whole geometry clusters unseen in training), and the OOD slabs
(extrapolation past the core parameter box).

Deep ensembles
--------------
``--init-seed`` decouples training stochasticity (model init, data order,
node subsampling) from the split seed. Ensemble members share ``--seed``
(identical splits and norm stats) and differ only in ``--init-seed``; a
run trained without ``--init-seed`` counts as the init-seed-0 member.

Eval-only mode
--------------
``--eval-only RUNDIR`` skips training, loads ``RUNDIR/best.pt`` and
``RUNDIR/norm_stats.pt``, rebuilds the splits from the current workdir and
evaluates all tiers. Use it to re-score existing checkpoints after a
dataset version update lands new OOD cases.

Metrics
-------
Per-channel relative L2 over the held-out set is the primary number, in
physical units after de-normalization. Stagnation heat flux from the
predicted T field (post-hoc via :func:`src.eval.sanity.compute_q_w_from_T`)
is the engineering-credibility figure; it is computed against the
SU2-postprocessed q_w stored in the ledger for each case.

Pretrain loading
----------------
The AirfRANS-pretrained Transolver was trained with ``space_dim=2,
fun_dim=5, out_dim=4`` and a different output convention
``(u, v, p/rho, nu_t)``. For SU2 we instantiate with ``space_dim=2,
fun_dim=8, out_dim=4`` and ``(rho, u, v, T)``. The input projection
(``preprocess``) and the final head (``blocks[-1].head``) cannot be
reused: their input/output dimensions or their target semantics differ.
We load all other parameters from the pretrain checkpoint and re-init
``preprocess`` and the final head from scratch. This is the standard
transfer move.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.su2 import (
    CASE_PARAM_ORDER,
    N_CASE_PARAMS,
    POS_DIM,
    SU2Dataset,
    SU2NormStats,
    TARGET_DIM,
    TARGET_ORDER,
    case_id_from_npz_path,
    compute_norm_stats,
    denormalize_targets,
    list_case_npzs,
)
from src.eval.sanity import (
    compute_q_w_from_T,
    identify_wall_nodes,
)
from src.models.transolver import Transolver


OOD_GROUPS = ("cone_high", "mach_high", "nose_large", "nose_small")


# ============================================================================================
#                                       split helpers
# ============================================================================================

def _connect_ro(ledger_path: Path) -> sqlite3.Connection:
    """Read-only immutable connection; works on read-only mounts (Kaggle input)."""
    return sqlite3.connect(f"file:{Path(ledger_path).as_posix()}?mode=ro&immutable=1", uri=True)


def load_case_groups(ledger_path: Path) -> dict[int, str]:
    """Return ``{case_id: group_name}`` for every row in the ledger."""
    con = _connect_ro(ledger_path)
    rows = con.execute("select case_id, group_name from cases").fetchall()
    con.close()
    return {int(cid): g for cid, g in rows}


def load_case_geom_ids(ledger_path: Path) -> dict[int, int]:
    con = _connect_ro(ledger_path)
    rows = con.execute("select case_id, geom_id from cases").fetchall()
    con.close()
    return {int(cid): int(g) for cid, g in rows}


def load_case_params_from_ledger(ledger_path: Path) -> dict[int, dict[str, float]]:
    """Return ``{case_id: {param: value}}`` for the 8 case parameters."""
    cols = ", ".join(CASE_PARAM_ORDER)
    con = _connect_ro(ledger_path)
    rows = con.execute(f"select case_id, {cols} from cases").fetchall()
    con.close()
    return {int(r[0]): dict(zip(CASE_PARAM_ORDER, map(float, r[1:]))) for r in rows}


def case_id_from_path(path: Path) -> int:
    """``case_0042/case.npz`` or ``case_0042.npz`` -> 42."""
    return case_id_from_npz_path(path)


def split_paths(
    paths: list[Path],
    groups: dict[int, str],
    geom_ids: dict[int, int],
    val_frac: float,
    test_frac: float,
    seed: int,
    interp_frac: float = 0.0,
) -> dict[str, list[Path]]:
    """Stratified train/val/test split by geometry cluster (core only).

    OOD groups go entirely to per-slab test sets. Within ``core``, full
    geometry clusters are assigned to train/val/test by a deterministic
    seeded shuffle of unique ``geom_id``s -- this prevents leakage between
    freestream-NN neighbours that share a mesh.

    With ``interp_frac > 0`` a random per-case fraction of the train
    geometries' cases is held out as ``test_interp``: unseen freestream
    conditions on shapes the model trained on. The geometry shuffle happens
    before the interp draw, so val/test geometry assignment for a given
    seed is unchanged by ``interp_frac``.

    Cases with ``group_name == 'loop'`` (active-learning acquired cases)
    are force-routed to ``train`` and never enter val/test/interp. They
    are also excluded from the geometry-shuffle input, so adding or
    removing loop cases leaves the core val/test/interp membership
    byte-identical for the same seed.
    """
    by_group: dict[str, list[Path]] = {"core": [], "loop": [], **{g: [] for g in OOD_GROUPS}}
    for p in paths:
        cid = case_id_from_path(p)
        if cid not in groups:
            continue
        by_group.setdefault(groups[cid], []).append(p)

    core_paths = by_group.pop("core")
    loop_paths = by_group.pop("loop", [])
    core_by_geom: dict[int, list[Path]] = {}
    for p in core_paths:
        gid = geom_ids[case_id_from_path(p)]
        core_by_geom.setdefault(gid, []).append(p)

    geom_ids_sorted = sorted(core_by_geom.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(geom_ids_sorted)
    n = len(geom_ids_sorted)
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    val_geoms = set(geom_ids_sorted[:n_val])
    test_geoms = set(geom_ids_sorted[n_val : n_val + n_test])
    train_geoms = set(geom_ids_sorted[n_val + n_test :])

    train_paths = [p for gid in sorted(train_geoms) for p in core_by_geom[gid]]
    interp_paths: list[Path] = []
    if interp_frac > 0:
        n_interp = int(round(len(train_paths) * interp_frac))
        pick = rng.permutation(len(train_paths))[:n_interp]
        interp_set = set(pick.tolist())
        interp_paths = [p for i, p in enumerate(train_paths) if i in interp_set]
        train_paths = [p for i, p in enumerate(train_paths) if i not in interp_set]

    # loop cases append after the core train, deterministic in case_id order
    train_paths = train_paths + sorted(loop_paths, key=case_id_from_path)

    splits = {
        "train": train_paths,
        "val":   [p for gid in sorted(val_geoms)   for p in core_by_geom[gid]],
        "test":  [p for gid in sorted(test_geoms)  for p in core_by_geom[gid]],
        "test_interp": interp_paths,
    }
    for g in OOD_GROUPS:
        splits[f"ood_{g}"] = by_group.get(g, [])
    return splits


# ============================================================================================
#                                       pretrain loading
# ============================================================================================

def load_pretrain_into(model: Transolver, pretrain_path: Path) -> dict[str, int]:
    """Load AirfRANS pretrain weights, skipping the input projection and head.

    The input projection (``preprocess``) is skipped because the input
    feature count differs (AirfRANS 7 channels vs SU2 10 channels). The
    final head (``blocks[-1].head``) is skipped because the 4 output
    channels carry different physical meanings (AirfRANS ``(u, v, p/rho,
    nu_t)`` vs SU2 ``(rho, u, v, T)``); reusing the AirfRANS head would
    carry irrelevant biases. The LayerNorms ``blocks[-1].ln_3`` operate on
    the hidden dim and are kept. Returns a summary dict for logging.
    """
    state = torch.load(str(pretrain_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_state = model.state_dict()
    n_blocks = len(model.blocks)
    skip_prefixes = ("preprocess.", f"blocks.{n_blocks - 1}.head.")
    loaded, skipped_shape, skipped_unknown, skipped_explicit = 0, [], [], []
    for k, v in state.items():
        if k not in model_state:
            skipped_unknown.append(k)
            continue
        if any(k.startswith(pfx) for pfx in skip_prefixes):
            skipped_explicit.append(k)
            continue
        if v.shape != model_state[k].shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        model_state[k] = v
        loaded += 1
    model.load_state_dict(model_state)
    print(f"[pretrain] loaded {loaded} tensors from {pretrain_path}")
    if skipped_explicit:
        print(f"[pretrain] skipped {len(skipped_explicit)} explicit (re-init): "
              f"{skipped_explicit}")
    if skipped_shape:
        print(f"[pretrain] skipped {len(skipped_shape)} shape-mismatched:")
        for k, src, dst in skipped_shape:
            print(f"  {k}: src{src} dst{dst}")
    if skipped_unknown:
        print(f"[pretrain] skipped {len(skipped_unknown)} unknown keys (first 5): "
              f"{skipped_unknown[:5]}")
    return {"loaded": loaded, "skipped_explicit": len(skipped_explicit),
            "skipped_shape": len(skipped_shape),
            "skipped_unknown": len(skipped_unknown)}


# ============================================================================================
#                                       collate + loop
# ============================================================================================

def collate_single(batch: list[dict]) -> dict:
    """Per-sample batches; meshes have variable N so no stacking."""
    if len(batch) != 1:
        raise NotImplementedError("batch_size > 1 not supported (variable N per mesh)")
    item = batch[0]
    return {
        "x": item["x"].unsqueeze(0),
        "y": item["y"].unsqueeze(0),
        "y_raw": item["y_raw"].unsqueeze(0),
        "pos": item["pos"].unsqueeze(0),
        "case_params": item["case_params"],
        "name": item["name"],
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    n, sum_loss = 0, 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pos = batch["pos"].to(device)
        optimizer.zero_grad()
        pred = model(x, pos=pos).float()
        loss = F.mse_loss(pred, y.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at step={n} sim={batch.get('name', '?')}; "
                f"x.range=[{float(x.min()):.3g}, {float(x.max()):.3g}], "
                f"pred.range=[{float(pred.min()):.3g}, {float(pred.max()):.3g}]"
            )
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        sum_loss += float(loss.detach())
        n += 1
    return {"loss": sum_loss / max(n, 1)}


# ============================================================================================
#                                       evaluation
# ============================================================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: SU2Dataset,
    stats: SU2NormStats,
    device: str,
    *,
    case_groups: dict[int, str] | None = None,
) -> tuple[dict, list[dict]]:
    """Per-channel rel-L2 in physical units, plus post-hoc stagnation q_w.

    For each case in the split, computes ||pred_c - true_c||_2 / ||true_c||_2
    per channel, plus q_w from the predicted T field via FD on the inward
    wall normal. The wall-node indices are identified from the ground-truth
    T field (not the predicted one) so the q_w comparison is well-defined
    even when the surrogate's near-wall T is noisy.

    Returns the aggregate summary dict and a per-case record list (name,
    case params, per-channel rL2, q_w pair) for envelope-distance analysis.
    """
    model.eval()
    saved_subsample = dataset.subsample
    dataset.subsample = None
    per_case: list[dict] = []
    try:
        for i in range(len(dataset)):
            item = dataset[i]
            x = item["x"].unsqueeze(0).to(device)
            pos = item["pos"].unsqueeze(0).to(device)
            y_true = item["y_raw"].to(device)
            pred_norm = model(x, pos=pos).squeeze(0)
            pred = denormalize_targets(pred_norm, stats.to(device))

            num = (pred - y_true).pow(2).sum(dim=0).sqrt()
            den = y_true.pow(2).sum(dim=0).sqrt().clamp_min(1e-12)
            rl2 = (num / den).cpu().numpy()
            rec = {
                "name": item["name"],
                "case_params": {k: float(item["case_params"][j])
                                for j, k in enumerate(CASE_PARAM_ORDER)},
                "rL2": {name: float(rl2[j]) for j, name in enumerate(TARGET_ORDER)},
                "qw_true": None,
                "qw_pred": None,
            }

            # post-hoc q_w on predicted T (Kelvin), wall ids from ground truth
            xy = item["pos"].cpu().numpy()
            T_true = y_true[:, TARGET_ORDER.index("T")].cpu().numpy()
            T_pred = pred[:, TARGET_ORDER.index("T")].cpu().numpy()
            T_w = float(item["case_params"][CASE_PARAM_ORDER.index("T_w")])
            R_n = float(item["case_params"][CASE_PARAM_ORDER.index("R_n")])
            try:
                wall = identify_wall_nodes(T_true, T_w)
                qw_true = float(compute_q_w_from_T(
                    x=xy[:, 0], r=xy[:, 1], T=T_true, T_w=T_w,
                    wall_indices=wall, y_axis_skip=0.05 * R_n, n_average=3,
                )["q_w"])
                qw_pred = float(compute_q_w_from_T(
                    x=xy[:, 0], r=xy[:, 1], T=T_pred, T_w=T_w,
                    wall_indices=wall, y_axis_skip=0.05 * R_n, n_average=3,
                )["q_w"])
                # assign only when both succeed; a lone qw_true poisons the
                # (true, pred) pair arithmetic downstream
                rec["qw_true"], rec["qw_pred"] = qw_true, qw_pred
            except ValueError:
                pass                       # mesh / wall-normal degeneracy; keep the rL2 record
            per_case.append(rec)
    finally:
        dataset.subsample = saved_subsample

    rl2 = np.stack([[r["rL2"][name] for name in TARGET_ORDER] for r in per_case]).mean(axis=0)
    out = {f"rL2_{name}": float(rl2[i]) for i, name in enumerate(TARGET_ORDER)}
    qw_pairs = [(r["qw_true"], r["qw_pred"]) for r in per_case if r["qw_true"] is not None]
    if qw_pairs:
        qw_arr = np.array(qw_pairs)
        rel = (qw_arr[:, 1] - qw_arr[:, 0]) / qw_arr[:, 0]
        out["qw_rel_err_mean"] = float(rel.mean())
        out["qw_rel_err_median"] = float(np.median(rel))
        out["qw_abs_rel_err_median"] = float(np.median(np.abs(rel)))
        out["n_qw_cases"] = int(len(qw_pairs))
    return out, per_case


# ============================================================================================
#                                       main
# ============================================================================================

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="train Transolver on SU2 hypersonic")
    p.add_argument("--workdir", default="data/raw/sweep",
                   help="dataset workdir containing case_*/case.npz and ledger.db")
    p.add_argument("--out", required=True, help="output directory for checkpoints + logs")
    p.add_argument("--pretrain", default=None,
                   help="path to AirfRANS pretrain .pt; omit for from-scratch baseline")
    p.add_argument("--slice-num", type=int, default=32)
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--subsample", type=int, default=8192,
                   help="random subsample of nodes per training step (full mesh at eval)")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--interp-frac", type=float, default=0.10,
                   help="fraction of train-geometry cases held out as the interpolation tier")
    p.add_argument("--eval-only", default=None, metavar="RUNDIR",
                   help="skip training; evaluate RUNDIR/best.pt with RUNDIR/norm_stats.pt "
                        "on all tiers and write results to --out")
    p.add_argument("--pinned-splits", default=None, metavar="JSON",
                   help="JSON of {split_name: [case_name, ...]} that overrides the "
                        "auto-computed splits for the tiers it lists; tiers absent from "
                        "the JSON keep their auto-computed membership. Use to reproduce "
                        "the exact core val/test/interp used by a prior run when "
                        "rescoring the same checkpoint against a refreshed dataset.")
    p.add_argument("--seed", type=int, default=0,
                   help="split seed; also seeds training RNGs unless --init-seed is given")
    p.add_argument("--init-seed", type=int, default=None,
                   help="training RNG seed (init, data order, subsampling); "
                        "splits stay on --seed. For deep-ensemble members.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=10)
    return p


def main() -> None:
    args = _parser().parse_args()
    workdir = Path(args.workdir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    init_seed = args.seed if args.init_seed is None else args.init_seed
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
    print(f"[main] split seed {args.seed}, init seed {init_seed}")

    ledger = workdir / "ledger.db"
    paths = list_case_npzs(workdir)
    if not paths:
        raise SystemExit(f"no case_*/case.npz under {workdir}")
    print(f"[main] found {len(paths)} case.npz files under {workdir}")

    groups = load_case_groups(ledger)
    geom_ids = load_case_geom_ids(ledger)
    splits = split_paths(paths, groups, geom_ids, args.val_frac, args.test_frac,
                         args.seed, args.interp_frac)
    if args.pinned_splits is not None:
        pinned = json.loads(Path(args.pinned_splits).read_text())
        def _cn(p: Path) -> str:
            return p.parent.name if p.name == "case.npz" else p.stem
        by_name = {_cn(p): p for p in paths}
        for split_name, names in pinned.items():
            resolved = [by_name[n] for n in names if n in by_name]
            missing = [n for n in names if n not in by_name]
            if missing:
                print(f"[pinned] {split_name}: {len(missing)} case(s) not present in "
                      f"workdir; dropping ({missing[:5]}{' ...' if len(missing) > 5 else ''})")
            splits[split_name] = resolved
        print(f"[pinned] applied {len(pinned)} pinned split(s) from {args.pinned_splits}")
    for name, ps in splits.items():
        print(f"[split] {name}: {len(ps)} cases")

    if not splits["train"]:
        raise SystemExit("empty train split; check ledger / workdir")

    # per-parameter training envelope, for envelope-distance analysis downstream
    case_params = load_case_params_from_ledger(ledger)
    train_ids = [case_id_from_path(p) for p in splits["train"]]
    train_envelope = {
        k: [min(case_params[c][k] for c in train_ids),
            max(case_params[c][k] for c in train_ids)]
        for k in CASE_PARAM_ORDER
    }

    if args.eval_only is not None:
        rundir = Path(args.eval_only).resolve()
        stats = SU2NormStats.load(rundir / "norm_stats.pt")
        # model shape must match the checkpoint; prefer the recorded run args
        run_args = dict(vars(args))
        prev_eval = rundir / "final_eval.json"
        if prev_eval.is_file():
            recorded = json.loads(prev_eval.read_text()).get("args", {})
            for k in ("slice_num", "n_hidden", "n_layers", "n_head"):
                if k in recorded:
                    run_args[k] = recorded[k]
        model = Transolver(
            space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=TARGET_DIM,
            n_hidden=run_args["n_hidden"], n_layers=run_args["n_layers"],
            n_head=run_args["n_head"], slice_num=run_args["slice_num"],
        ).to(args.device)
        model.load_state_dict(torch.load(rundir / "best.pt", map_location=args.device))
        print(f"[eval-only] loaded {rundir / 'best.pt'} "
              f"(slice_num={run_args['slice_num']})")
        pretrain_info, best_val = None, None
    else:
        stats = compute_norm_stats(splits["train"])
        stats.save(out / "norm_stats.pt")
        print(f"[main] norm stats fit on {len(splits['train'])} train cases, saved")

        train_ds = SU2Dataset(splits["train"], stats, subsample=args.subsample)
        val_ds = SU2Dataset(splits["val"], stats, subsample=None)
        loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single,
                            num_workers=0)

        model = Transolver(
            space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=TARGET_DIM,
            n_hidden=args.n_hidden, n_layers=args.n_layers, n_head=args.n_head,
            slice_num=args.slice_num,
        ).to(args.device)
        print(f"[main] model: {sum(p.numel() for p in model.parameters()):,} params, "
              f"slice_num={args.slice_num}")

        pretrain_info = None
        if args.pretrain is not None:
            pretrain_info = load_pretrain_into(model, Path(args.pretrain))

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )

        history: list[dict] = []
        best_val = float("inf")
        t_start = time.time()
        for epoch in range(1, args.epochs + 1):
            rec = train_one_epoch(model, loader, optimizer, args.device, args.grad_clip)
            if epoch % args.val_every == 0 or epoch == args.epochs:
                val, _ = evaluate(model, val_ds, stats, args.device)
                mean_rl2 = float(np.mean([val[k] for k in val if k.startswith("rL2_")]))
                rec.update({"epoch": epoch, "val": val, "mean_rL2": mean_rl2,
                            "wallclock_s": time.time() - t_start})
                print(f"[ep {epoch:4d}] loss={rec['loss']:.4f} "
                      f"val_mean_rL2={mean_rl2:.4f} val={val}")
                if mean_rl2 < best_val:
                    best_val = mean_rl2
                    torch.save(model.state_dict(), out / "best.pt")
            else:
                rec["epoch"] = epoch
                print(f"[ep {epoch:4d}] loss={rec['loss']:.4f}")
            history.append(rec)

        torch.save(model.state_dict(), out / "final.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        model.load_state_dict(torch.load(out / "best.pt", map_location=args.device))

    # final eval on all tiers from the best checkpoint
    final = {}
    per_case_all: dict[str, list[dict]] = {}
    for split_name in ["val", "test", "test_interp", *(f"ood_{g}" for g in OOD_GROUPS)]:
        if not splits[split_name]:
            continue
        ds = SU2Dataset(splits[split_name], stats, subsample=None)
        final[split_name], per_case_all[split_name] = evaluate(model, ds, stats, args.device)
        print(f"[final] {split_name}: {final[split_name]}")

    # in eval-only mode, args.slice_num / n_hidden / n_layers / n_head reflect the
    # CLI defaults, not the model that actually ran; overlay from run_args so the
    # provenance record matches the loaded checkpoint
    args_out = dict(vars(args))
    if args.eval_only is not None:
        for k in ("slice_num", "n_hidden", "n_layers", "n_head"):
            args_out[k] = run_args[k]
    (out / "final_eval.json").write_text(json.dumps({
        "splits": {k: len(v) for k, v in splits.items()},
        "args": args_out,
        "pretrain_info": pretrain_info,
        "best_val_mean_rL2": best_val,
        "train_envelope": train_envelope,
        "final": final,
    }, indent=2))
    (out / "per_case_eval.json").write_text(json.dumps(per_case_all, indent=2))


if __name__ == "__main__":
    main()

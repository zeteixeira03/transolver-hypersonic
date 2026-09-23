"""Train Transolver on the SU2 hypersonic dataset.

The slice-count ablation is the single flag ``--slice-num``. Sweep it by
running again with a new value and a new ``--out``.

Splits
------
The training and validation splits come from the ``core`` group alone,
stratified by ``geom_id``, so a validation case sits on a geometry the model
never trained on. The four extrapolation slabs, ``cone_high``, ``mach_high``,
``nose_large`` and ``nose_small``, are held out whole as test sets, and each
one reports its own metrics.

Three evaluation tiers rise in difficulty. ``test_interp`` is interpolation:
cases held back from the training geometries, so the shapes are seen and the
freestream conditions are not. ``test`` is the family holdout: whole geometry
clusters absent from training. The slabs are extrapolation past the core box.

Deep ensembles
--------------
``--init-seed`` separates the randomness of training, meaning the starting
weights, the data order and the node draw, from the randomness of the split.
Every member of an ensemble shares one ``--seed``, which gives identical splits
and identical normalization statistics, and differs only in ``--init-seed``. A
run with no ``--init-seed`` counts as the member at init-seed 0.

Evaluation without training
---------------------------
``--eval-only RUNDIR`` skips training. It loads ``RUNDIR/best.pt`` and
``RUNDIR/norm_stats.pt``, rebuilds the splits from the current dataset
directory, and scores every tier. Use it to re-score existing checkpoints once
a new dataset version brings new extrapolation cases.

Physics priors
--------------
The project leaves three priors on. Each one can be switched off, so the prior
ablation can price it against a baseline on a small training split.
``--predict-p`` drops the hard equation of state and emits pressure as a free
fifth channel instead of reconstructing it from density and temperature.
``--log-targets`` names the channels standardized in log10 space, and ``none``
applies plain standardization to every channel. ``--qw-head`` adds a case-level stagnation
heat-flux head, which predicts either log10 of q_w directly or the log of its
ratio to the Fay-Riddell value. The defaults hold the project convention, so an
existing command behaves as before.

Metrics
-------
The primary number is the relative L2 per channel over the held-out set, in
physical units after the normalization is undone.

Stagnation heat flux is what makes the result credible to an engineer, and two
different numbers carry that name here. The first runs the finite-difference
estimator :func:`src.eval.sanity.compute_q_w_from_T` over the predicted field
and over the true field and compares the two, which isolates the near-wall
gradient of the surrogate. It runs for every model, so the column stays
comparable across ablation cells. The second exists only where a heat-flux head
does, and that head trains and scores against the value SU2 itself reported in
the ledger.

The two disagree by a median factor of about 27 on this dataset, the
finite-difference value being the smaller, because a fit to the nearest fluid
neighbour cannot resolve the near-wall gradient at this mesh resolution. The two
columns are reported apart and must never be read as one quantity.

"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.analytical import fay_riddell_qw
from src.cfd.ledger import read_cases
from src.data.su2 import (
    CASE_PARAM_ORDER,
    DEFAULT_LOG_TARGETS,
    DEFAULT_LOG_TARGETS_WITH_P,
    N_CASE_PARAMS,
    OOD_GROUPS,
    POS_DIM,
    TARGET_ORDER,
    TARGET_ORDER_WITH_P,
    SU2Dataset,
    SU2NormStats,
    case_id_from_npz_path,
    case_name,
    compute_norm_stats,
    list_case_npzs,
    pin_splits,
    split_paths,
)
from src.models.transolver import Transolver
from src.training.loop import collate_single, evaluate, train_one_epoch

ROOT = Path(__file__).resolve().parents[1]


# ============================================================================================
#                                       physics priors
# ============================================================================================

# each of the three priors below can be switched off, so the ablation can price
# it. the defaults hold the project convention: pressure reconstructed and never
# predicted, log10 normalization on density and temperature, no heat-flux head.

def preflight_device(device: str) -> None:
    """Stop at once on an accelerator the installed torch has no kernels for.

    A rented session assigns a card rather than offering a choice. A card too old
    for the build otherwise fails as a CUDA error at the first forward pass,
    which arrives after dataset staging has already spent most of an hour of
    quota. This check turns that into an immediate and readable failure.
    """
    if not device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise SystemExit(f"--device {device} but torch reports no CUDA device")
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"[preflight] {name}, compute capability {major}.{minor}, "
          f"{torch.cuda.device_count()} device(s)")
    arch_list = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" not in arch_list:
        raise SystemExit(
            f"{name} is compute capability {major}.{minor} (sm_{major}{minor}), "
            f"which this torch build has no kernels for (it ships {arch_list}). "
            f"Training would die at the first CUDA call. Reassign the session "
            f"to a supported accelerator and rerun."
        )


def collect_env(device: str) -> dict:
    """Record the provenance of one run: code version, interpreter, accelerator.

    A run happens on a rented machine that does not persist, and pushing the
    notebook again destroys its output. Anything absent from the run directory
    costs another session of quota to recover. The commit and the device are what
    make a surprising number explicable months later.
    """
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        env["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (subprocess.CalledProcessError, OSError):
        env["git_commit"] = None          # no git available, or not a checkout
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        env["cuda_version"] = torch.version.cuda
        env["gpu_count"] = torch.cuda.device_count()
    return env


@dataclass
class QwSpec:
    """How the stagnation heat-flux head encodes its target.

    ``direct`` regresses the standardized ``log10 q_w``. ``residual`` regresses
    the standardized log of the ratio to the Fay-Riddell value, so the network
    learns only the departure from the correlation. Both work in log space,
    because q_w runs from 1.2e4 to 5.8e7 W/m^2 across the dataset.

    Attributes
    ----------
    mode : str
        ``direct`` or ``residual``.
    mean, std : float
        Standardization constants, fit on the train split only.
    """

    mode: str
    mean: float
    std: float

    def log_target(self, qw: float, q_fr: float) -> float:
        """Return the log-space target for one case, before standardization."""
        if self.mode == "residual":
            return float(np.log10(qw / q_fr))
        return float(np.log10(qw))

    def encode(self, qw: float, q_fr: float) -> float:
        return (self.log_target(qw, q_fr) - self.mean) / self.std

    def decode(self, z: float, q_fr: float) -> float:
        """Turn a standardized head output back into W/m^2."""
        log_val = z * self.std + self.mean
        if self.mode == "residual":
            return float(q_fr * 10.0 ** log_val)
        return float(10.0 ** log_val)


def fay_riddell_for_case(params: dict[str, float]) -> float:
    """Return the Fay-Riddell stagnation heat flux for one case-parameter dict."""
    return fay_riddell_qw(
        M_inf=params["mach"], T_inf=params["T_inf"], p_inf=params["p_inf"],
        R_n=params["R_n"], T_w=params["T_w"],
    )


def build_qw_maps(
    paths: list[Path],
    case_params: dict[int, dict[str, float]],
    qw_by_id: dict[int, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return, per case name, the measured and the Fay-Riddell q_w in W/m^2.

    A case with no q_w in the ledger is absent from both maps, so it falls out
    of the heat-flux loss rather than feeding it a NaN.
    """
    qw, q_fr = {}, {}
    for path in paths:
        cid = case_id_from_npz_path(path)
        if cid not in qw_by_id or cid not in case_params:
            continue
        name = case_name(path)
        qw[name] = qw_by_id[cid]
        q_fr[name] = fay_riddell_for_case(case_params[cid])
    return qw, q_fr


# ============================================================================================
#                                       main
# ============================================================================================

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="train Transolver on SU2 hypersonic")
    p.add_argument("--workdir", default="data/raw/sweep",
                   help="dataset workdir containing case_*/case.npz and ledger.db")
    p.add_argument("--out", required=True, help="output directory for checkpoints + logs")
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
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                   help="cosine anneals to zero over --epochs, stepped per epoch")
    p.add_argument("--predict-p", action="store_true",
                   help="emit pressure as a free fifth channel instead of "
                        "reconstructing it from density and temperature, which "
                        "drops the equation-of-state constraint")
    p.add_argument("--log-targets", default=None,
                   help="comma-separated target channels to standardize in log10 "
                        "space, or 'none' for plain standardization throughout. "
                        "Defaults to rho,T (plus p when --predict-p is set).")
    p.add_argument("--qw-head", choices=("none", "direct", "residual"), default="none",
                   help="auxiliary stagnation heat-flux head: predict log10 q_w "
                        "directly, or the log-ratio against Fay-Riddell")
    p.add_argument("--qw-loss-weight", type=float, default=0.1,
                   help="weight on the auxiliary q_w term; ignored without --qw-head")
    p.add_argument("--val-every", type=int, default=10)
    return p


def main() -> None:
    args = _parser().parse_args()
    workdir = Path(args.workdir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    preflight_device(args.device)
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

    rows = read_cases(ledger)
    qw_by_id = {cid: float(r["qw"]) for cid, r in rows.items() if r["qw"] is not None}
    groups = {cid: r["group_name"] for cid, r in rows.items()}
    geom_ids = {cid: int(r["geom_id"]) for cid, r in rows.items()}
    splits = split_paths(paths, groups, geom_ids, args.val_frac, args.test_frac,
                         args.seed, args.interp_frac)
    if args.pinned_splits is not None:
        pinned = json.loads(Path(args.pinned_splits).read_text())
        splits = pin_splits(splits, paths, pinned)
        print(f"[pinned] applied {len(pinned)} pinned split(s) from {args.pinned_splits}")
    for name, ps in splits.items():
        print(f"[split] {name}: {len(ps)} cases")

    if not splits["train"]:
        raise SystemExit("empty train split; check ledger / workdir")

    # the training range of each parameter, which the envelope-distance analysis
    # reads later
    case_params = {cid: {k: float(r[k]) for k in CASE_PARAM_ORDER} for cid, r in rows.items()}
    train_ids = [case_id_from_npz_path(p) for p in splits["train"]]
    train_envelope = {
        k: [min(case_params[c][k] for c in train_ids),
            max(case_params[c][k] for c in train_ids)]
        for k in CASE_PARAM_ORDER
    }

    # ---- prior configuration: target layout, log channels, heat-flux head ----
    target_order = TARGET_ORDER_WITH_P if args.predict_p else TARGET_ORDER
    if args.log_targets is None:
        log_targets = DEFAULT_LOG_TARGETS_WITH_P if args.predict_p else DEFAULT_LOG_TARGETS
    elif args.log_targets.strip().lower() == "none":
        log_targets = ()
    else:
        log_targets = tuple(s.strip() for s in args.log_targets.split(",") if s.strip())
        unknown = [s for s in log_targets if s not in target_order]
        if unknown:
            raise SystemExit(f"--log-targets names unknown channels {unknown}; "
                             f"available: {list(target_order)}")
    print(f"[priors] targets={list(target_order)} log={list(log_targets)} "
          f"qw_head={args.qw_head}")

    # built on every run, so that every per-case record carries both the ledger
    # and the analytical heat flux, and not only the runs that train a head
    qw_raw, fr_by_name = build_qw_maps(paths, case_params,
                                       qw_by_id)
    qw_spec: QwSpec | None = None
    qw_encoded: dict[str, float] | None = None
    if args.qw_head != "none":
        train_names = [case_name(p) for p in splits["train"]]
        logs = [np.log10(qw_raw[n] / (fr_by_name[n] if args.qw_head == "residual" else 1.0))
                for n in train_names if n in qw_raw]
        if not logs:
            raise SystemExit("--qw-head set but no train case has a ledger q_w")
        # the floor on the standard deviation guards against a degenerate
        # training split. a real spread is of order 0.1 to 1.
        qw_spec = QwSpec(args.qw_head, float(np.mean(logs)),
                         max(float(np.std(logs)), 1e-6))
        qw_encoded = {n: qw_spec.encode(qw_raw[n], fr_by_name[n]) for n in qw_raw}
        print(f"[priors] q_w head on {len(logs)}/{len(train_names)} train cases, "
              f"log-target mean={qw_spec.mean:.3f} std={qw_spec.std:.3f}")

    qw_weight = args.qw_loss_weight if qw_spec is not None else 0.0

    if args.eval_only is not None:
        rundir = Path(args.eval_only).resolve()
        stats = SU2NormStats.load(rundir / "norm_stats.pt")
        target_order = stats.targets            # the checkpoint sets the layout
        spec_path = rundir / "qw_spec.json"
        if spec_path.is_file():
            qw_spec = QwSpec(**json.loads(spec_path.read_text()))
            if fr_by_name is None:
                qw_raw, fr_by_name = build_qw_maps(
                    paths, case_params, qw_by_id)
        else:
            qw_spec = None
        # the shape of the model must match the checkpoint, so take it from the
        # arguments the run recorded
        run_args = dict(vars(args))
        prev_eval = rundir / "final_eval.json"
        if prev_eval.is_file():
            recorded = json.loads(prev_eval.read_text()).get("args", {})
            for k in ("slice_num", "n_hidden", "n_layers", "n_head"):
                if k in recorded:
                    run_args[k] = recorded[k]
        model = Transolver(
            space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=len(target_order),
            n_hidden=run_args["n_hidden"], n_layers=run_args["n_layers"],
            n_head=run_args["n_head"], slice_num=run_args["slice_num"],
            qw_head=qw_spec is not None,
        ).to(args.device)
        model.load_state_dict(torch.load(rundir / "best.pt", map_location=args.device))
        print(f"[eval-only] loaded {rundir / 'best.pt'} "
              f"(slice_num={run_args['slice_num']})")
        # the layout in the checkpoint outranks the command-line defaults here
        log_targets = stats.log_targets
        n_params = sum(p.numel() for p in model.parameters())
        best_val, best_epoch = None, None
    else:
        stats = compute_norm_stats(splits["train"], log_targets=log_targets,
                                   with_p=args.predict_p)
        stats.save(out / "norm_stats.pt")
        print(f"[main] norm stats fit on {len(splits['train'])} train cases, saved")
        if qw_spec is not None:
            (out / "qw_spec.json").write_text(json.dumps(vars(qw_spec), indent=2))

        train_ds = SU2Dataset(splits["train"], stats, subsample=args.subsample,
                              qw_by_name=qw_encoded)
        val_ds = SU2Dataset(splits["val"], stats, subsample=None)
        loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single,
                            num_workers=0)

        model = Transolver(
            space_dim=POS_DIM, fun_dim=N_CASE_PARAMS, out_dim=len(target_order),
            n_hidden=args.n_hidden, n_layers=args.n_layers, n_head=args.n_head,
            slice_num=args.slice_num, qw_head=qw_spec is not None,
        ).to(args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[main] model: {n_params:,} params, slice_num={args.slice_num}")


        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            if args.lr_schedule == "cosine" else None
        )

        history: list[dict] = []
        best_val, best_epoch = float("inf"), None
        t_start = time.time()
        for epoch in range(1, args.epochs + 1):
            rec = train_one_epoch(model, loader, optimizer, args.device, args.grad_clip,
                                  qw_weight=qw_weight)
            # record every cheap scalar on every epoch. the run directory is the
            # only lasting record, and recovering a curve later costs a whole
            # session of accelerator quota.
            rec["epoch"] = epoch
            rec["lr"] = float(optimizer.param_groups[0]["lr"])
            rec["wallclock_s"] = time.time() - t_start
            if torch.cuda.is_available():
                rec["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
            if scheduler is not None:
                scheduler.step()
            if epoch % args.val_every == 0 or epoch == args.epochs:
                val, _ = evaluate(model, val_ds, stats, args.device,
                                             qw_spec=qw_spec, fr_by_name=fr_by_name,
                                             qw_ledger_by_name=qw_raw)
                mean_rl2 = float(np.mean([val[k] for k in val if k.startswith("rL2_")]))
                rec.update({"val": val, "mean_rL2": mean_rl2})
                print(f"[ep {epoch:4d}] loss={rec['loss']:.4f} "
                      f"val_mean_rL2={mean_rl2:.4f} val={val}")
                if mean_rl2 < best_val:
                    best_val, best_epoch = mean_rl2, epoch
                    torch.save(model.state_dict(), out / "best.pt")
            else:
                print(f"[ep {epoch:4d}] loss={rec['loss']:.4f}")
            history.append(rec)
            # rewritten each epoch, so an interrupted session still holds its curve
            (out / "history.json").write_text(json.dumps(history, indent=2))

        torch.save(model.state_dict(), out / "final.pt")
        model.load_state_dict(torch.load(out / "best.pt", map_location=args.device))
        print(f"[main] best val mean_rL2 {best_val:.4f} at epoch {best_epoch} "
              f"of {args.epochs}")

    # final eval on all tiers from the best checkpoint
    final = {}
    per_case_all: dict[str, list[dict]] = {}
    for split_name in ["val", "test", "test_interp", *(f"ood_{g}" for g in OOD_GROUPS)]:
        if not splits[split_name]:
            continue
        ds = SU2Dataset(splits[split_name], stats, subsample=None)
        final[split_name], per_case_all[split_name] = evaluate(
            model, ds, stats, args.device, qw_spec=qw_spec, fr_by_name=fr_by_name,
            qw_ledger_by_name=qw_raw)
        print(f"[final] {split_name}: {final[split_name]}")

    # without training, args.slice_num, n_hidden, n_layers and n_head still hold
    # the command-line defaults rather than the shape of the model that ran.
    # overlay them from run_args, so the provenance record matches the checkpoint
    # that was loaded.
    args_out = dict(vars(args))
    if args.eval_only is not None:
        for k in ("slice_num", "n_hidden", "n_layers", "n_head"):
            args_out[k] = run_args[k]
    (out / "final_eval.json").write_text(json.dumps({
        "splits": {k: len(v) for k, v in splits.items()},
        # the names as well as the counts. without them a later run on a
        # refreshed dataset could not be lined up against this one case by case.
        "split_names": {k: [case_name(p) for p in v] for k, v in splits.items()},
        "args": args_out,
        "env": collect_env(args.device),
        "n_params": n_params,
        "best_val_mean_rL2": best_val,
        "best_epoch": best_epoch,
        "qw_spec": None if qw_spec is None else vars(qw_spec),
        "target_order": list(target_order),
        "log_targets": list(log_targets),
        "train_envelope": train_envelope,
        "final": final,
    }, indent=2))
    (out / "per_case_eval.json").write_text(json.dumps(per_case_all, indent=2))


if __name__ == "__main__":
    main()

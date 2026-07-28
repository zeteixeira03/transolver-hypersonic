# Deploying the dashboard to Hugging Face Spaces

The Space is a separate git repository from this project. It needs the app,
the `src/` package it imports, and the five ensemble checkpoints. Total payload
is about 10 MB (five `best.pt` at ~2 MB each), so plain git is fine, no LFS.

You run these steps: they involve your Hugging Face account, a write token, and
publishing a public page.

## 1. Create the Space

On https://huggingface.co/new-space: choose the **Streamlit** SDK, CPU basic
(free) hardware, and a name (e.g. `hypersonic-sphere-cone-surrogate`). This
creates an empty repo at `https://huggingface.co/spaces/<user>/<name>`.

## 2. Assemble the Space directory locally

From the project root, stage only what the app needs. The Space card
(`README.md` with the YAML front matter) and `requirements.txt` / `packages.txt`
must sit at the Space **root**; `app_file` in the card already points at
`app/app.py`.

```powershell
$dst = "..\hf_space"; New-Item -ItemType Directory -Force $dst; `
Copy-Item app\app.py, app\inference.py -Destination (New-Item -ItemType Directory -Force "$dst\app"); `
Copy-Item app\README.md, app\requirements.txt, app\packages.txt -Destination $dst; `
Copy-Item -Recurse src "$dst\src"; `
foreach ($s in 0..4) { $m = "data\processed\ensemble_v3\run_m32_v3_s$s"; `
  $d = New-Item -ItemType Directory -Force "$dst\$m"; `
  Copy-Item "$m\best.pt","$m\norm_stats.pt","$m\final_eval.json" -Destination $d }
```

The checkpoint layout under `$dst\data\processed\...` must match, because
`app/inference.py` builds `ENSEMBLE_DIRS` from `ROOT / "data/processed/..."`.

## 3. Push

```powershell
cd ..\hf_space; git init; git add .; git commit -m "Hypersonic sphere-cone surrogate dashboard"; `
git remote add origin https://huggingface.co/spaces/<user>/<name>; git push -u origin main
```

Authenticate with a write token when prompted (create one at
https://huggingface.co/settings/tokens). Do not commit any token.

## 4. Verify

The Space builds (installs `packages.txt` via apt, then `requirements.txt`),
then launches. First build takes a few minutes. Once live, set a flight
condition and press **Run prediction**; a prediction takes tens of seconds on
the free CPU tier (meshing plus five-member inference), which is expected.

## Notes

- CPU-only torch is pinned in `requirements.txt` via the PyTorch CPU index, so
  the image stays small.
- `packages.txt` carries the apt libraries the gmsh wheel links against.
- If the free tier feels slow, HF's paid CPU-upgrade (8 vCPU) drops a prediction
  to roughly ten seconds; the free tier is the default target.

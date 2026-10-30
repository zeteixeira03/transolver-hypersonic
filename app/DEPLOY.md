# Deploying the dashboard to Streamlit Community Cloud

Streamlit Community Cloud hosts the dashboard for free. It builds the app from
a public GitHub repository, which is kept separate from this project so the
checkpoints never enter it. The deploy repository holds the app, the `src`
package it imports, and the five ensemble checkpoints. That comes to about
10 MB, so plain git is enough.

You run these steps yourself, because they use your GitHub account and
publish a public page.

## 1. Create the repository

On https://github.com/new, create an empty public repository, for example
`hypersonic-sphere-cone-demo`. Add no README, license or ignore file, so the
first push lands on an empty history.

## 2. Assemble the deploy directory

From the project root, copy only what the app needs:

```powershell
$dst = "..\dashboard_deploy"; New-Item -ItemType Directory -Force $dst; `
Copy-Item app\app.py, app\inference.py -Destination (New-Item -ItemType Directory -Force "$dst\app"); `
Copy-Item app\README.md, app\requirements.txt, app\packages.txt, pyproject.toml -Destination $dst; `
robocopy src "$dst\src" /E /XD __pycache__ | Out-Null; `
foreach ($s in 0..4) { $m = "data\processed\ensemble_v3\run_m32_v3_s$s"; `
  $d = New-Item -ItemType Directory -Force "$dst\$m"; `
  Copy-Item "$m\best.pt","$m\norm_stats.pt","$m\final_eval.json" -Destination $d }; `
Copy-Item data\processed\ensemble_v3\thresholds.json -Destination "$dst\data\processed\ensemble_v3"
```

Each file has a fixed place:

- `requirements.txt` sits at the root. The host looks for a dependency file
  in the entry point's directory first and then at the root, so `app/` must
  hold none. A root `requirements.txt` takes priority over the root
  `pyproject.toml`.
- `pyproject.toml` sits at the root because the last line of
  `requirements.txt`, a single `.`, installs the `src` package from it.
  The host runs `streamlit run app/app.py`, which puts only `app/` on the
  import path, so `from src...` works only once the package is installed.
- `packages.txt` sits at the root, the only place the host reads it. It lists
  the system libraries the gmsh wheel links against.
- The checkpoints and `thresholds.json` sit under
  `data/processed/ensemble_v3`, because `app/inference.py` reads them from
  there.

## 3. Push

```powershell
cd ..\dashboard_deploy; git init -b main; git add .; git commit -m "Hypersonic sphere-cone surrogate dashboard"; `
git remote add origin https://github.com/<user>/<name>.git; git push -u origin main
```

## 4. Deploy

On https://share.streamlit.io, sign in with GitHub and create an app from the
repository: branch `main`, main file path `app/app.py`. Under Advanced
settings, choose the Python version this project runs locally (3.13, within
the `requires-python` range of `pyproject.toml`). Then deploy.

## 5. Verify

The first build installs `packages.txt` and then `requirements.txt`, and takes
a few minutes. Once the app is live, set a flight condition and press
**Run prediction**. The first prediction at a cone half-angle meshes the body;
later ones at the same angle reuse that mesh, scaled to the nose radius, and
only run the five members.

## Limits

- The host gives an app between 690 MB and 2.7 GB of memory and between 0.078
  and 2 processors, by its own published figures, which it may change without
  notice. The app peaks at about 720 MB through a prediction with all five
  members loaded (measured on Python 3.13 with the processor-only torch
  build), just above the guaranteed floor. If the host reports that the app
  went over its resource limits, copy fewer `run_*` member folders in step 2.
  The app loads every member folder it finds.
- The first prediction at a new cone half-angle took 90 s on a desktop
  processor, most of it meshing. On a small share of the host's processors it
  can take several minutes. Later predictions at the same angle skip the
  meshing.
- The first build log should show no missing shared library. A line such as
  `libGLU.so.1: cannot open shared object file` names a library that
  `packages.txt` lacks. The host tries uv first and falls back to pip, so a
  uv error followed by a pip install is harmless.
- An app with no visitors for 12 hours goes to sleep. The next visitor wakes
  it with one click and waits about a minute while it starts.

# External scenes in RoboGen

The adapter imports local scene bundles into PyBullet without asset retrieval,
normalization, layout repair, size clamping or an extra table. Both modes retain
the author's suction constraints, action primitives, OMPL algorithms, SAC
configuration, observation format and sequential executor. The external mode
preserves the source Franka meshes, inertias, joint frames and limits. Its open
fingers are held symmetrically and the original suction constraint attaches to
the left fingertip. This is a robot-model comparison, **not physical two-finger
grasping**. A physical gripper comparison would require a different grasp policy.

The original mobile Panda is placed at the source robot's XY location and heading,
with a 0.28 m base-frame offset matching the author's floor-mounted model. The
external model uses the source base transform. These offsets and grasp semantics
are recorded in `experiment.json`. All other scene entities keep their original
poses, scales and fixed flags. Normal physical settling is allowed, without pose
correction; its drift is reported.

## Environment

The uv environment can be recreated from this repository's root with:

```bash
bash scripts/setup_uv.sh
CUDA_VISIBLE_DEVICES=0 uv run python run_external_scene.py --help
```

`pyproject.toml` declares the Python 3.9 experiment dependencies.
`uv.lock` is generated locally and is not tracked in Git.
The configuration uses the Linux x86_64 wheels' dependency metadata for
PyTorch 2.0.0 and Open3D 0.14.1, whose older releases have different metadata
across platforms. This ensures CUDA 11.7 libraries and Open3D's notebook
dependencies are installed. The OMPL build disables its optional registration
webpage so unattended installation does not block on opening a browser.
The setup script downloads and verifies Gym 0.21's source archive under
`output/vendor`, then fixes only its invalid `opencv-python>=3.` packaging
metadata to `>=3.0`. Gym's runtime code is unchanged. Run the setup script
before `uv sync` on a new checkout so this local dependency exists.
Use `bash scripts/setup_uv.sh --skip-ompl` to install only the Python packages.
The full setup builds OMPL 1.5.2 with the existing isolated build script;
its tools and native libraries remain under `output/`. No system packages
are installed. `source prepare.sh` now activates `.venv`.
This environment covers the external scene generation/execution workflow.
The original Objaverse retrieval pipeline additionally needs LAVIS and
SentenceTransformer models and separately downloaded datasets.

Use Python 3.9 and the original Gym 0.21 / Ray 1.13 interfaces. For Gym's old
packaging metadata, install pip 23.0.1, setuptools 65.5.0 and wheel 0.38.4 first,
then `python -m pip install --no-build-isolation gym==0.21.0`. Install
`external_scene_requirements.txt`, PyTorch 2.0.0 and OMPL 1.5.2 in this environment.
The repository README's OMPL prerelease wheel URL may no longer be available.
MuJoCo is used only to compile the source MJCF into URDF; it is not the simulator.
MoviePy 1.0.3 still calls `Image.ANTIALIAS`, removed in Pillow 10. The shared
utility restores that name as an alias of `Image.Resampling.LANCZOS` on newer
Pillow versions, retaining the same filter and original GIF export function.

For the existing installation, OMPL 1.5.2 (the author's source-install version)
and its native libraries reside in `output/ompl_env`. The conda package contains
native libraries only; Python bindings must be generated and compiled from the
same source version. Recreate the isolated installation using:

```bash
bash scripts/build_ompl_bindings.sh
```

The build script adapts three Python binding type-name selectors for current
CastXML/Py++ (string aliases, default template arguments and QRRT's alias).
It does not modify any OMPL planner implementation. Runtime package versions
and the scene hash are recorded with each experiment.

## macOS Apple Silicon environment with uv

From the repository root, run:

```bash
bash scripts/setup_uv.sh
```

This creates a uv-managed Python 3.9 `.venv`, installs the versions declared in
`pyproject.toml` / `uv.lock`, and builds the original OMPL 1.5.2 Python bindings.
The project uses the Tsinghua PyPI mirror; this does not change global uv settings.
Gym 0.21 is bootstrapped with pip 23.0.1, setuptools 65.5.0 and wheel 0.38.4.
The script corrects only its installed optional OpenCV dependency metadata
(`>=3.` to `>=3`), because uv rejects that malformed version. Gym runtime code
is unchanged. gRPC 1.43 is built against macOS's existing system zlib to avoid
its bundled zlib conflicting with current SDK headers. PyBullet 3.2.7 is built
with `-include stdio.h` so system declarations are parsed before its old zlib
defines an incompatible `fdopen` macro. The PyTorch 2.0 ARM wheel incorrectly
labels its installed WHEEL metadata as x86_64; the setup corrects that tag
without changing its ARM binaries. Run the setup script
rather than plain `uv sync` on a fresh checkout.

OMPL native dependencies and CastXML are isolated under `output/ompl_env` and
`output/castxml_env`, using a project-local micromamba. No Homebrew or system
packages are installed. Building requires the existing Xcode Command Line Tools.
Bindings and generated C++ files stay under `output/tools/ompl-1.5.2`; a `.pth`
file in `.venv` makes them importable. Keep these output directories for runtime.
The `ompl-build` dependency group is needed only when regenerating bindings.

Check the command-line entry point without running a simulation:

```bash
uv run python run_external_scene.py --help
```

For the commands below, replace `python` with `uv run python`. Training uses the
macOS PyTorch build; the original Linux CUDA environment is not reproduced.

## Usage and scene-local outputs

Run commands from the repository root. `--scene-yaml` accepts a bundle directory
or its `scene.yaml`. Every invocation creates a fresh run unless `--run-dir`
explicitly selects an existing run. Historical `output/external_scene` files
are left untouched; existing solution directories can still be supplied with
`--solution`.

```text
<scene directory>/robogen/
├── _assets/<scene-and-path-hash>/  # Shared converted assets and manifest.json
└── 20261009_143000_s000/           # Same-name runs get -01, -02, ...
    ├── run_meta.json
    ├── result.json
    ├── run.log
    ├── llm/
    │   ├── index.json
    │   ├── turn_0001.json
    │   └── reward.json            # Original generator response format
    ├── code/
    │   ├── task.yaml
    │   ├── substeps.txt
    │   ├── substep_types.txt
    │   ├── action_spaces.txt
    │   └── <substep>.py
    ├── trajectory/<execution timestamp>/<primitive>/
    │   ├── execute.gif
    │   └── state_*.pkl
    ├── training/<execution timestamp>/<reward step>/
    │   ├── ray_results/
    │   ├── checkpoint_*/
    │   ├── best_model/
    │   ├── best_state/
    │   ├── eval_*/
    │   └── execute.gif
    └── artifacts/
        ├── experiment.json
        ├── steps.json
        ├── experiment/all_last_state_files_<timestamp>.txt
        ├── generated_solution_<timestamp>/
        ├── check/                # scene.png, scene_check.json, initial_state.pkl
        ├── evaluation/           # scene.png, scene_check.json, final_state.pkl
        └── all-<timestamp>.gif
```

The run ID defaults to local time plus the scene seed (default 0).
`--run-id NAME` sets its name; `--output-dir ROOT` overrides the root containing
run directories and `_assets`. `--run-dir PATH` continues a run, inherits its
scene seed and robot mode, and rejects a different scene, changed scene YAML,
or explicitly mismatched seed/mode. It appends Python console output to
`run.log` and records each command as a session. Native PyBullet/C++ output and
Ray worker output may remain in the terminal or Ray logs rather than `run.log`.
Continuation means running another pipeline stage with the existing code; it
is not policy-checkpoint resumption and does not skip completed substeps.

Ray uses a short, stable `/tmp/rg-<hash>` runtime directory for each run. This
keeps its Unix socket paths within the platform limit even when the scene or
output path is long. Training and replay of the same run reuse the directory;
separate run IDs, including different seeds, use separate directories.
The ManiGen batch runner copies Ray session diagnostics into the task's
`logs/ray/` after the job stops; sockets and object-store runtime files remain
in `/tmp`. Batch-stage console logs are also retained under the task's `logs/`.

`run_meta.json` records the scene and manifest hashes, source commit/dirty state,
robot mode, scene seed, training seed, package versions, solution provenance and
non-secret command arguments. `result.json` records `status`, `native_success`,
`success`, failure stage/reason, elapsed time, LLM call count and substep outcomes.
`status=completed` means the requested stage finished, not that manipulation
succeeded. `native_success` is populated only by the supported final evaluator;
`success` stays null until a common cross-method evaluator is connected.
Checks and generation leave both success fields null. Continued stages preserve
previous result snapshots in `run_meta.json.sessions`.

Failed pipeline stages leave a result and traceback when the run directory has
been initialized. SIGKILL or power loss can leave status `running`.
`llm/turn_*.json` records the actual request parameters/messages, response,
provider usage, duration and errors without authentication headers or keys.
`llm/reward.json` retains the original generator log. `steps.json` records primitive
success when returned; reward-step success is not inferred from training return.
State PKLs retain the original simulation snapshot format, not a standardized
observation/action/reward dataset.

Configure the LLM in the repository-root `.env` (ignored by Git):

```dotenv
LLM_BASE_URL='https://api.deepseek.com'
LLM_MODEL='deepseek-flash'
LLM_API_KEY=''
```

Copy `.env.example` on a fresh checkout and fill in the key. Existing shell
variables override `.env`; explicit `--base-url`, `--model` and `--api-key-env`
override defaults. Legacy OpenAI/DeepSeek environment variables remain supported.
The base URL is the API root, not `/chat/completions`. Choose a model supported
by that endpoint. `run` and `generate` require a nonempty key; checks do not.

```bash
SCENE=/path/to/002_living_room/scene.yaml

# Scene import check only; no LLM calls or RL training.
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --action check

# Full workflow: prepare, check, generate if needed, execute, evaluate.
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --seed 0

# Separate generation and execution within the SAME experiment.
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" \
  --action generate --run-id experiment_01
RUN=/path/to/002_living_room/robogen/experiment_01
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" \
  --action execute --run-dir "$RUN"

# Same generated code on the source Franka, in a new experiment.
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" \
  --robot-mode external --solution "$RUN/code"

# Check a final saved state in the same experiment.
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" \
  --action evaluate --run-dir "$RUN" --state /path/to/final_state.pkl
```

`--action prepare` refreshes the shared conversion cache and preserves existing
code when continuing a run. It creates the bundled manual remote-to-tray reference
only for that supported task. Arbitrary tasks use `generate` or `--solution`.
Source and converted-asset hashes are checked before reuse; changed assets require
`prepare`. Keep `_assets` for state restoration. No source assets are edited.

Absolute solution files are loaded directly, including inside Ray workers;
legacy repository-relative module imports remain supported. Generated code is
adapted only by replacing its `SimpleEnv` import, with the author's generated
source archived under `artifacts`. The existing solution validation remains.

Original primitives, suction, OMPL planners and SAC hyperparameters are retained.
Each reward stage still trains for one million steps. `--seed` controls the
scene/NumPy seed; the original SAC seed remains 0 and is separately recorded as
`training_seed`. The old `execute.py` entry point keeps its default output paths.

## Replay and export a final state

After generation and training complete, replay the saved primitive code and load
the saved RL policy checkpoints without calling the generator or training a new
policy:

```bash
.venv/bin/python replay_save_state.py "$RUN" --seed 0 \
  --settle-seconds 0.5
```

Pass `--training-algo ppo` only when the run used PPO; current external-scene
execution defaults to SAC.

Each invocation allocates the next unused `state/replay_NNN/` directory. It
contains `manifest.json`, `final_state.json`, the native `final_state.pkl`, a
a step result file. The JSON state
uses canonical source-scene entity IDs, world positions, `wxyz` quaternions, and
revolute/prismatic joint values. RoboGen's robot entity is listed under
`skipped_entities`, because its robot topology is not converted into Genesis.
The manifest stores relative paths to the source scene, generated code, prepared
RoboGen scene manifest, and selected best checkpoints.
Physics settling runs once at the end and is included in the recorded
`settle_seconds` value.
Policy replay writes RLlib logs to each reward stage's `ray_results/` inside
the new replay directory, rather than the shared `data/local/ray_results` default.

## Verification and evaluation

`--action check` checks pre-settling positions/orientations, fixed-object drift,
observation/action shapes, and saves `scene.png`, `scene_check.json` and
`initial_state.pkl` under `artifacts/check/`. It does not require a generated or manual task solution and does not evaluate
manipulation success (`evaluation` is null in the check report). The camera uses
`source_support_id`, `surface` or `target_id` to focus on the task area.
Final tray evaluation requires `task.goal_id`. Goal-point tasks save their final
state/render but report `evaluation_status=unsupported_task` and null success;
they require a separate task-specific evaluator. Source hashes
are verified before each run. Restore files from a different manifest or robot
mode are rejected before any state is changed.
Final evaluation from `run` or `evaluate` writes `artifacts/evaluation/scene.png`,
`artifacts/evaluation/scene_check.json` and `artifacts/evaluation/final_state.pkl`, preserving the
initial check artifacts. `--state` supplies an initial simulation state; it
does not resume a policy checkpoint or skip completed substeps.
The external robot interface clips only limit roundoff within 1e-7 radians when
reading joints, so a nanoradian Bullet constraint error cannot invalidate OMPL's
start state. Physical scene poses and planner bounds are left intact. Bullet's
visual-record fields and empty fixed frames are handled by the camera helper;
source robot colors are baked into a single textured visual per link so temporary
hiding/restoration preserves them.

```bash
ROBOGEN_EXTERNAL_MANIFEST=/path/to/robogen/_assets/HASH/manifest.json \
  .venv/bin/python -m unittest discover -s tests -v

.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --robot-mode robogen \
  --action evaluate --run-dir "$RUN" --state PATH_TO_FINAL_STATE.pkl
```

Independent evaluation checks full-object XY containment with a 5 mm margin,
lengthwise alignment within 10 degrees, contact with the tray without more than
2 mm penetration, low linear/angular velocity and released suction. This is a
bounding-box approximation of the tray interior and does not change the training
reward. Inspect the final render and collision model alongside these metrics.

Non-triangle geometries (such as GLB line paths) are excluded from visual and
collision export, with a console notice. Assets containing no triangle meshes
are rejected. Conversion preserves GLB scene graph transforms and matches Genesis's automatic
Y-up to Z-up conversion before applying scale. Static meshes use concave
triangle collisions; dynamic meshes use convex collisions or PyBullet VHACD's
default settings, with individual hulls retained. Dynamic mass defaults to 0.1 kg
and friction to 1.0, following the author's mesh helper conventions; these do not
reproduce Genesis density-based mass or contact dynamics. Diffuse material support
depends on PyBullet; normal/roughness/specular PBR maps are not reproduced.

The current external robot converter explicitly supports the sample's unscaled,
fixed-base Franka Panda MJCF. Unsupported articulated assets, free bases, joint
layouts and joint anchors fail instead of being silently approximated. No claim
of paper reproduction or task success follows from scene import alone.

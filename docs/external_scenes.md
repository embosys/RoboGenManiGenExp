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

## Usage

Run from the repository root. `--scene-yaml` accepts the original bundle file.
The output directory must be inside this repository for the original solution
module imports. Outputs for each robot and seed are separate; converted assets
are shared. Choose a different output directory for each source scene.

The default action is `run`: prepare missing assets, check the scene, generate
the task through the original LLM pipeline if no formal solution exists, execute
the original primitives and SAC training, then evaluate the executor's final
saved state. Existing generated or explicitly supplied solutions are reused.
An automatic manual reference is replaced through the original generator.
Any failed stage stops the pipeline; generation requires valid API credentials.
The original executor and all training/planning settings remain unchanged.
The adapter removes Markdown heading prefixes from `Substep N:` lines before
passing the response to the unchanged author parser. It preserves fenced Python
and reward code and the raw `gpt_response/reward.json` response. Generated and
supplied solutions must have nonempty, equally sized step/type/action lists,
supported action spaces and compilable modules with the required task methods;
invalid solutions stop before execution instead of failing on an empty GIF.

Configure the LLM in the repository-root `.env` (ignored by Git). Copy
`.env.example` on a fresh checkout and fill in your key:

```dotenv
LLM_BASE_URL='https://api.deepseek.com'
LLM_MODEL='deepseek-flash'
LLM_API_KEY=''
```

The model must be supported by your endpoint. Existing shell variables override
`.env`; explicit `--base-url`, `--model` and `--api-key-env` override their defaults.
Legacy `OPENAI_API_KEY`, `YUFEI_OPENAI_API_KEY`, `DEEPSEEK_API_KEY`,
`ROBOGEN_API_KEY_ENV`, `OPENAI_BASE_URL` and `OPENAI_API_BASE` remain supported.
`--api-key-env` selects the variable containing the key, not the key itself.
`run` and `generate` require a nonempty key; scene checks do not call the LLM
and allow an empty key. The base URL must be the API root, not `/chat/completions`.

```bash
SCENE=/home/zhy/SceneSynthesis/output/living_room_26_items_20261006_1791270090985_attempt02/scene.yaml
# Optional custom key variable; MY_API_KEY must already be exported.
# .venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --api-key-env MY_API_KEY
# Complete workflow; omitting --action is equivalent to --action run.
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --robot-mode robogen
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --robot-mode external \
  --solution output/external_scene/robogen/seed_0/task_solution

# Scene checks only, without generation or training.
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --robot-mode robogen --action check
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --robot-mode external --action check

# Original task decomposition / reward generator; requires the repo's OpenAI setup.
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --robot-mode robogen --action generate

# Compare exactly the same generated solution on the other robot.
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --robot-mode external --action prepare \
  --solution output/external_scene/robogen/seed_0/task_solution

# Calls the unmodified execute.py, including its original one-million-step SAC stage.
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --robot-mode robogen --action execute
.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --robot-mode external --action execute
```

`--action prepare` rebuilds asset conversion and preserves existing task code.
`--seed N` selects a separate
experiment directory and sets the external loader / NumPy seeds; the original
Ray learning seed is left as authored (0), so this flag alone does not create
multiple SAC training seeds. No planner limits, retry budgets or training
hyperparameters are changed.

The bundled remote-to-tray solution is a **manual reference task**, adapted from
the author's example reward pattern. It calls the original grasp/release
primitives and uses their bounding-box reward pattern. It is not represented as
LLM-generated. Use `--action generate` for the original LLM generation pipeline,
or `--solution PATH` for existing generated solutions. Both are adapted only by
changing their `SimpleEnv` import in a copied directory. Original solution files
remain untouched. Arbitrary tasks require a supplied or generated solution.
`--action execute` refuses the automatic manual reference until a generated or
explicitly supplied solution is selected for the research run.

## Verification and evaluation

`--action check` checks pre-settling positions/orientations, fixed-object drift,
observation/action shapes, and saves `scene.png`, `scene_check.json` and
`initial_state.pkl`. It does not require a generated or manual task solution and does not evaluate
manipulation success (`evaluation` is null in the check report). The camera uses
`source_support_id`, `surface` or `target_id` to focus on the task area.
Final tray evaluation requires `task.goal_id`; goal-point tasks require a
separate task-specific evaluator. Source hashes
are verified before each run. Restore files from a different manifest or robot
mode are rejected before any state is changed.
Final evaluation from `run` or `evaluate` writes `evaluation/scene.png`,
`evaluation/scene_check.json` and `evaluation/final_state.pkl`, preserving the
initial check artifacts. `--state` supplies an initial simulation state; it
does not resume a policy checkpoint or skip completed substeps.
The external robot interface clips only limit roundoff within 1e-7 radians when
reading joints, so a nanoradian Bullet constraint error cannot invalidate OMPL's
start state. Physical scene poses and planner bounds are left intact. Bullet's
visual-record fields and empty fixed frames are handled by the camera helper;
source robot colors are baked into a single textured visual per link so temporary
hiding/restoration preserves them.

```bash
ROBOGEN_EXTERNAL_MANIFEST=output/external_scene/manifest.json \
  .venv/bin/python -m unittest discover -s tests -v

.venv/bin/python run_external_scene.py --scene-yaml "$SCENE" --robot-mode robogen \
  --action evaluate --state PATH_TO_FINAL_STATE.pkl
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

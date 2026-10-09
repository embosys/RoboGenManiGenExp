"""Import fixed external scenes; execute them through the original RoboGen pipeline."""
import argparse
import ast
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
from importlib.metadata import version
import shutil
from urllib.parse import urlparse
from unittest.mock import patch

import numpy as np
import yaml
from dotenv import load_dotenv

from manipulation.external_scene_assets import prepare_scene
from run_output import RunOutput, capture_console, write_json
import traceback

ROOT = Path(__file__).resolve().parent


def reference_solution(directory, task):
    if task.get("name") != "move_remote_to_tray":
        raise ValueError("Supply --solution or use --action generate for other tasks")
    header = """from manipulation.external_scene import ExternalSceneEnv as SimpleEnv
import numpy as np
from manipulation.gpt_reward_api import *
from manipulation.gpt_primitive_api import *

class {name}(SimpleEnv):
    def __init__(self, task_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name
"""
    # Reference task only: use the original grasp/release primitives and example reward pattern.
    sources = {
        "grasp_remote": '''    def execute(self):
        rgbs, states = grasp_object(self, "remote")
        return rgbs, states, check_grasped(self, "remote")
''',
        "move_remote_into_tray": '''    def _compute_reward(self):
        position = get_position(self, "remote")
        low, high = get_bounding_box(self, "tray")
        center = (low + high) / 2
        success = in_bbox(self, position, low, high)
        reward = 5 * int(success) - np.linalg.norm(position - center)
        reward -= np.linalg.norm(get_eef_pos(self)[0] - position)
        return reward, success
''',
        "release_remote": '''    def execute(self):
        rgbs, states = release_grasp(self)
        return rgbs, states, not self.activated
''',
    }
    for name, body in sources.items():
        (directory / (name+".py")).write_text(header.format(name=name)+body)
    (directory / "substeps.txt").write_text("grasp remote\nmove remote into tray\nrelease remote\n")
    (directory / "substep_types.txt").write_text("primitive\nreward\nprimitive\n")
    (directory / "action_spaces.txt").write_text("None\nnormalized-direct-translation\nNone\n")


def prepare_task(manifest_path, mode, seed, output, solution=None, create_reference=True):
    manifest = json.loads(manifest_path.read_text())
    directory = output / "code"
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory.parent / "artifacts" / "experiment.json"
    previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if solution:
        source = Path(solution).resolve()
        if source != directory:
            shutil.copytree(source, directory, dirs_exist_ok=True)
        for path in directory.glob("*.py"):
            content = path.read_text()
            path.write_text(content.replace("from manipulation.sim import SimpleEnv",
                                            "from manipulation.external_scene import ExternalSceneEnv as SimpleEnv"))
        provenance = "provided_solution"
    elif (directory / "substeps.txt").exists():
        provenance = previous.get("solution_provenance", "provided_solution")
    elif create_reference:
        reference_solution(directory, manifest["task"])
        provenance = "manual_reference_adapted_from_author_example"
    else:
        provenance = "pending_original_generator"
    relative = str(directory.resolve())
    config = [{"solution_path": relative}, {"external_scene": {
        "manifest": str(manifest_path), "robot_mode": mode, "seed": seed,
        "solution_path": relative}}, {"task_name": manifest["task"]["name"],
                                      "task_description": manifest["task"]["description"]}]
    config_path = directory / "task.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {**previous, "robot_mode": mode, "seed": seed, "source_scene": manifest["source_scene"],
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "solution_provenance": provenance, "grasp_model": "original_robogen_suction",
                "robot_base_offset": [0, 0, 0.28] if mode == "robogen" else [0, 0, 0],
                "python": platform.python_version(),
                "packages": {name: version(name) for name in ("numpy", "gym", "ray", "pybullet", "torch", "mujoco", "trimesh")},
                "solution_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(directory.iterdir()) if path.suffix in (".py", ".txt")},
                "core_logic": "original task execution, primitives, reward generation, SAC and OMPL algorithms"}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return config_path, directory


def verify_sources(manifest):
    for name, expected in {**manifest["source_hashes"], **manifest.get("converted_hashes", {})}.items():
        path = Path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Source changed; rerun --action prepare: %s" % name)


def normalize_solution_response(response):
    """Adapt Markdown headings without touching generated Python or rewards."""
    lines, fenced = [], False
    for line in response.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            heading = re.sub(r"^\s*(?:#{1,6}\s+)?", "", line)
            if re.match(r"substep\s+\d+\s*:", heading, re.IGNORECASE):
                line = heading
        lines.append(line)
    return "\n".join(lines)


def validate_solution(directory):
    """Reject empty or incompatible solutions before the original executor."""
    try:
        steps, types, actions = [(directory / name).read_text().splitlines() for name in
                                 ("substeps.txt", "substep_types.txt", "action_spaces.txt")]
    except FileNotFoundError as error:
        raise ValueError("Incomplete task solution: "+str(directory)) from error
    if not steps or len(steps) != len(types) or len(steps) != len(actions):
        raise ValueError("Empty or mismatched task lists in %s; inspect gpt_response/reward.json and regenerate the solution" % directory)
    for step, kind, action in zip(steps, types, actions):
        name, kind, action = step.strip().replace(" ", "_"), kind.strip(), action.strip()
        if not name.isidentifier() or kind not in ("primitive", "reward"):
            raise ValueError("Invalid substep or type: %r / %r" % (step, kind))
        if action not in (("None",) if kind == "primitive" else ("delta-translation", "normalized-direct-translation")):
            raise ValueError("Invalid action space for %s: %r" % (step, action))
        path = directory / (name+".py")
        if not path.is_file():
            raise ValueError("Missing substep module: "+str(path))
        source = path.read_text()
        try:
            compile(source, str(path), "exec")
        except SyntaxError as error:
            raise ValueError("Invalid generated Python in "+str(path)) from error
        classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == name]
        method = "execute" if kind == "primitive" else "_compute_reward"
        if not classes or not any(isinstance(node, ast.FunctionDef) and node.name == method for node in classes[0].body):
            raise ValueError("Missing %s.%s in %s" % (name, method, path))


def final_task_check(env):
    """Independent evaluation; never changes the author's training reward."""
    if not env.manifest["task"].get("goal_id"):
        raise ValueError("Final tray evaluation requires task.goal_id; goal-point tasks need their own evaluator")
    # ponytail: bounding-box interior; use authored regions for irregular containers.
    import pybullet as p
    from scipy.spatial.transform import Rotation
    entities = {e["id"]: e for e in env.manifest["entities"]}
    target, goal = env.manifest["task"]["target_id"], env.manifest["task"]["goal_id"]
    target_id, goal_id = env.external_entity_ids[target], env.external_entity_ids[goal]
    low, high = np.array(entities[target]["mesh_asset"]["bounds_local"])
    corners = np.array(np.meshgrid(*zip(low, high))).reshape(3, -1).T
    pos, quat = p.getBasePositionAndOrientation(target_id, physicsClientId=env.id)
    goal_pos, goal_quat = p.getBasePositionAndOrientation(goal_id, physicsClientId=env.id)
    rotation = Rotation.from_quat(quat).as_matrix()
    goal_rotation = Rotation.from_quat(goal_quat).as_matrix()
    local = ((corners @ rotation.T + pos) - goal_pos) @ goal_rotation
    goal_low, goal_high = np.array(entities[goal]["mesh_asset"]["bounds_local"])
    contained = bool(np.all(local[:, :2] >= goal_low[:2]+0.005)
                     and np.all(local[:, :2] <= goal_high[:2]-0.005))
    axis = rotation[:, np.argmax(high-low)]
    goal_axis = goal_rotation[:, np.argmax((goal_high-goal_low)[:2])]
    angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(axis, goal_axis)), 0, 1))))
    contacts = p.getContactPoints(target_id, goal_id, physicsClientId=env.id)
    supported = bool(contacts) and all(c[8] >= -0.002 for c in contacts)
    linear, angular = p.getBaseVelocity(target_id, physicsClientId=env.id)
    stable = np.linalg.norm(linear) < 0.01 and np.linalg.norm(angular) < 0.1
    released = not env.activated
    return {"success": bool(contained and angle < 10 and supported and stable and released),
            "contained_xy_with_5mm_margin": contained, "alignment_error_deg": angle,
            "supported_by_tray": supported, "stable": bool(stable), "released": released}


def check_scene(config_path, directory, gui=False, state=None, evaluation=False):
    import pybullet as p
    from PIL import Image
    from manipulation.external_scene import ExternalSceneEnv
    from manipulation.utils import save_env
    manifest = json.loads(Path(yaml.safe_load(config_path.read_text())[1]["external_scene"]["manifest"]).read_text())
    translation_mode = "delta-translation"
    if (directory / "substep_types.txt").exists() and (directory / "action_spaces.txt").exists():
        types = (directory / "substep_types.txt").read_text().splitlines()
        actions = (directory / "action_spaces.txt").read_text().splitlines()
        index = types.index("reward") if "reward" in types else 0
        if index < len(actions):
            translation_mode = actions[index]
    report_dir = directory.parent / "artifacts" / ("evaluation" if evaluation else "check")
    report_dir.mkdir(parents=True, exist_ok=True)
    env = ExternalSceneEnv(config_path=str(config_path), restore_state_file=state,
                           translation_mode=translation_mode, gui=gui)
    try:
        initial, settled, max_drift = {}, {}, 0.0
        for entity in manifest["entities"]:
            name = entity["id"]
            initial[name] = env.imported_poses[name]
            settled[name] = p.getBasePositionAndOrientation(env.external_entity_ids[name], physicsClientId=env.id)
            if not entity.get("robot"):
                if not np.allclose(initial[name][0], entity.get("position", [0, 0, 0]), atol=1e-9):
                    raise AssertionError("Imported position differs: "+name)
                if not np.isclose(abs(np.dot(initial[name][1], entity["quaternion_xyzw"])), 1, atol=1e-9):
                    raise AssertionError("Imported orientation differs: "+name)
                drift = np.linalg.norm(np.array(settled[name][0])-initial[name][0])
                max_drift = max(max_drift, drift)
                if entity.get("fixed") and drift > 1e-9:
                    raise AssertionError("Fixed object moved: "+name)
        task = manifest["task"]
        focus_id = task.get("source_support_id") or task.get("surface") or task.get("target_id")
        focus = next((entity for entity in manifest["entities"] if entity["id"] == focus_id), None)
        center = np.array(focus.get("position", [0, 0, 0]) if focus else [0, 0, 0])
        env.setup_camera(camera_eye=[2.7, -2.7, 2.5], camera_target=center)
        Image.fromarray(env.render()[0]).save(report_dir / "scene.png")
        obs = env._get_obs() if state else env.reset()
        if obs.shape != env.observation_space.shape or not np.isfinite(obs).all():
            raise AssertionError("Observation is incompatible with the original policy")
        report = {"robot_mode": env.robot_mode, "scene_entity_count": len(env.external_entity_ids),
                  "initial_poses_match": True, "max_nonrobot_settle_drift_m": max_drift,
                  "settling_steps": 500, "initial_poses": initial, "settled_poses": settled,
                  "observation_shape": list(obs.shape), "action_shape": list(env.action_space.shape),
                  "arm_joint_indices": env.robot.right_arm_joint_indices,
                  "evaluation": final_task_check(env) if evaluation and manifest["task"].get("goal_id") else None,
                  "evaluation_status": "supported" if evaluation and manifest["task"].get("goal_id") else "unsupported_task" if evaluation else "not_requested"}
        (report_dir / "scene_check.json").write_text(json.dumps(report, indent=2))
        save_env(env, report_dir / ("final_state.pkl" if evaluation else "initial_state.pkl"))
        print(json.dumps({k: v for k, v in report.items() if k not in ("initial_poses", "settled_poses")}, indent=2))
        return report
    finally:
        env.close()


def generate_solution(manifest_path, config_path, directory, model, cached_response=None):
    key_env = os.environ.get("ROBOGEN_API_KEY_ENV")
    key = os.environ.get(key_env) if key_env else os.environ.get("OPENAI_API_KEY") or os.environ.get("YUFEI_OPENAI_API_KEY")
    if cached_response is None and not key:
        raise ValueError("Original RoboGen generation requires "+(key_env or "OPENAI_API_KEY or YUFEI_OPENAI_API_KEY"))
    from gpt_4.prompts import prompt_manipulation_reward_primitive as generator
    manifest = json.loads(manifest_path.read_text())
    objects = [{"name": e["id"], "type": "mesh", "center": str(tuple(e.get("position", [0, 0, 0]))),
                "lang": e["id"], "size": float(np.linalg.norm(np.diff(e["mesh_asset"]["bounds_local"], axis=0))),
                "on_table": False} for e in manifest["entities"] if "mesh_asset" in e]
    log = directory.parent / "llm" / "reward.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    task = manifest["task"]
    original_query = generator.query

    def compatible_query(*args, **kwargs):
        response = cached_response if cached_response is not None else original_query(*args, **kwargs)
        return normalize_solution_response(response)

    with patch.object(generator, "query", compatible_query):
        generated = generator.decompose_and_generate_reward_or_primitive(
            task["name"], task["description"], yaml.safe_dump(objects),
            "All scene objects are rigid meshes with a single base link and no joints.",
            "\n".join(e["id"]+": rigid "+("fixed" if e.get("fixed") else "movable")+" object"
                      for e in manifest["entities"] if not e.get("robot")),
            "base", "None", None, str(config_path), str(log), model=model)
    validate_solution(Path(generated))
    prepare_task(manifest_path, yaml.safe_load(config_path.read_text())[1]["external_scene"]["robot_mode"],
                 yaml.safe_load(config_path.read_text())[1]["external_scene"]["seed"],
                 directory.parent, solution=generated)
    archive = directory.parent / "artifacts" / ("generated_solution_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    shutil.move(generated, archive)
    metadata_path = directory.parent / "artifacts" / "experiment.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["solution_provenance"] = "original_robogen_llm_generator"
    metadata["model"] = model
    metadata["response_format_adapter"] = "markdown_substep_headings"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-yaml", required=True, type=Path)
    parser.add_argument("--robot-mode", choices=("robogen", "external"), default=None)
    parser.add_argument("--action", choices=("run", "prepare", "check", "generate", "execute", "evaluate"),
                        default="run", help="Default: run the complete preparation, check, generation, execution and evaluation pipeline")
    parser.add_argument("--output-dir", type=Path, help="Run root; default: <scene>/robogen")
    run_options = parser.add_mutually_exclusive_group()
    run_options.add_argument("--run-id", help="Name for a new run (duplicates receive a suffix)")
    run_options.add_argument("--run-dir", type=Path, help="Continue an existing run, including its generated code")
    parser.add_argument("--solution", type=Path, help="Existing RoboGen solution directory (copied, never edited in place)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL") or "deepseek-flash")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "https://api.deepseek.com",
                        help="OpenAI-compatible API root, including /v1 when required; overrides OPENAI_BASE_URL / OPENAI_API_BASE")
    parser.add_argument("--api-key-env", default=os.environ.get("ROBOGEN_API_KEY_ENV") or ("LLM_API_KEY" if "LLM_API_KEY" in os.environ else "YUFEI_OPENAI_API_KEY" if "YUFEI_OPENAI_API_KEY" in os.environ else "OPENAI_API_KEY" if "OPENAI_API_KEY" in os.environ else "DEEPSEEK_API_KEY"),
                        help="Environment variable containing the API key; overrides ROBOGEN_API_KEY_ENV and legacy key selection")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--state", help="Saved state for final evaluation or continuing execution")
    args = parser.parse_args()
    scene = args.scene_yaml.expanduser().resolve()
    if scene.is_dir():
        scene = scene / "scene.yaml"
    run = RunOutput(args, scene)
    print("Run directory:", run.path)
    previous_run = os.environ.get("ROBOGEN_RUN_DIR")
    os.environ["ROBOGEN_RUN_DIR"] = str(run.path)
    error = None
    with capture_console(run.path / "run.log"):
        try:
            run.phase = "configuration"
            if args.action in ("run", "generate") and not os.environ.get(args.api_key_env):
                raise ValueError("API key environment variable is missing or empty: " + args.api_key_env)
            os.environ["ROBOGEN_API_KEY_ENV"] = args.api_key_env
            os.environ["LLM_MODEL"] = args.model
            if args.base_url:
                endpoint = urlparse(args.base_url)
                if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
                    raise ValueError("--base-url must be an absolute HTTP(S) URL")
                os.environ["OPENAI_BASE_URL"] = args.base_url.rstrip("/")
            os.chdir(ROOT)
            run.phase = "prepare"
            cache_key = hashlib.sha256((str(scene) + run.meta["scene_sha256"]).encode()).hexdigest()
            cache = run.path.parent / "_assets" / cache_key
            manifest_path = cache / "manifest.json"
            if args.action == "prepare" or not manifest_path.exists():
                manifest_path = prepare_scene(scene, cache)
            manifest = json.loads(manifest_path.read_text())
            if manifest["source_scene"] != str(scene):
                raise ValueError("Asset cache belongs to another scene")
            verify_sources(manifest)
            run.meta["manifest"] = str(manifest_path)
            run.meta["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            config_path, directory = prepare_task(manifest_path, args.robot_mode, args.seed, run.path, args.solution,
                create_reference=args.action == "prepare" and manifest["task"].get("name") == "move_remote_to_tray")
            run.persist()
            np.random.seed(args.seed)
            action = args.action
            if action == "run":
                run.phase = "check"
                check_scene(config_path, directory, args.gui, args.state)
                metadata = json.loads((run.path / "artifacts/experiment.json").read_text())
                if metadata["solution_provenance"] in ("manual_reference_adapted_from_author_example", "pending_original_generator"):
                    run.phase = "generate"
                    generate_solution(manifest_path, config_path, directory, args.model)
                action = "execute"
            run.phase = action
            if action == "generate":
                generate_solution(manifest_path, config_path, directory, args.model)
            elif action in ("check", "evaluate"):
                if action == "evaluate" and not args.state:
                    raise ValueError("--action evaluate requires --state")
                report = check_scene(config_path, directory, args.gui, args.state, evaluation=action == "evaluate")
                if action == "evaluate":
                    run.result["evaluation"] = report["evaluation"]
                    run.result["evaluation_status"] = report["evaluation_status"]
                    if report["evaluation"] is not None:
                        run.result["native_success"] = report["evaluation"]["success"]
            elif action == "execute":
                metadata = json.loads((run.path / "artifacts/experiment.json").read_text())
                if metadata["solution_provenance"] == "manual_reference_adapted_from_author_example":
                    raise ValueError("Use --action generate or --solution before research execution")
                validate_solution(directory)
                from execute import execute
                run.result["attempts"] = execute(str(config_path), gui=args.gui, randomize=False,
                    last_restore_state_file=args.state, output_dir=str(run.path))
                if args.action == "run":
                    run.phase = "evaluate"
                    final_state = run.result["attempts"][-1]["final_state"]
                    report = check_scene(config_path, directory, args.gui, final_state, evaluation=True)
                    run.result["evaluation"] = report["evaluation"]
                    run.result["evaluation_status"] = report["evaluation_status"]
                    if report["evaluation"] is not None:
                        run.result["native_success"] = report["evaluation"]["success"]
            print("Task config:", config_path)
        except BaseException as caught:
            error = caught
            traceback.print_exc()
            raise
        finally:
            steps = run.path / "artifacts/steps.json"
            if run.phase in ("execute", "evaluate") and steps.exists():
                run.result["attempts"] = json.loads(steps.read_text())
                if error is not None and run.result["attempts"] and run.result["attempts"][-1]["status"] == "running":
                    run.result["attempts"][-1].update(status="error", failure_reason=str(error))
                    write_json(steps, run.result["attempts"])
            metadata = run.path / "artifacts/experiment.json"
            if metadata.exists():
                run.meta["experiment"] = json.loads(metadata.read_text())
            run.finish(error)
            if previous_run is None:
                os.environ.pop("ROBOGEN_RUN_DIR", None)
            else:
                os.environ["ROBOGEN_RUN_DIR"] = previous_run


if __name__ == "__main__":
    main()

"""Replay a completed RoboGen run and export its final scene state."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_json(path):
    return json.loads(Path(path).read_text())


def _relative(path, base):
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def _new_replay_dir(run_dir):
    state_root = run_dir / "state"
    state_root.mkdir(exist_ok=True)
    for index in range(1, 1000):
        path = state_root / ("replay_%03d" % index)
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Too many replay directories under %s" % state_root)


def _checkpoint_for(step, steps_file):
    records = _read_json(steps_file)
    record = next((item for item in records
                   if item.get("step") == step["index"] and item.get("name") == step["name"]), None)
    if record is None:
        raise FileNotFoundError("No execution record for reward step %r in %s" % (step["name"], steps_file))
    artifact = Path(record.get("artifacts") or "")
    if not artifact.is_absolute():
        artifact = (steps_file.parent.parent / artifact).resolve()
    best_model = artifact / "best_model"
    checkpoints = []
    for candidate in best_model.rglob("checkpoint-*") if best_model.is_dir() else ():
        match = re.fullmatch(r"checkpoint-(\d+)", candidate.name)
        if match and candidate.is_file():
            checkpoints.append((int(match.group(1)), candidate))
    if not checkpoints:
        raise FileNotFoundError(
            "Missing trained RL checkpoint for reward step %r. Expected a saved policy under %s; "
            "this replay command never trains a policy." % (step["name"], best_model)
        )
    return max(checkpoints, key=lambda item: item[0])[1].resolve()


def _read_steps(solution_dir):
    filenames = ("substeps.txt", "substep_types.txt", "action_spaces.txt")
    values = [(solution_dir / name).read_text().splitlines() for name in filenames]
    if not values[0] or len({len(items) for items in values}) != 1:
        raise ValueError("RoboGen substep, type, and action-space files are empty or have mismatched lengths")
    steps = []
    for index, (name, kind, action_space) in enumerate(zip(*values)):
        name, kind, action_space = name.strip(), kind.strip(), action_space.strip()
        if not name or kind not in ("primitive", "reward"):
            raise ValueError("Invalid RoboGen step at index %d: %r / %r" % (index, name, kind))
        code = solution_dir / (name.replace(" ", "_") + ".py")
        if not code.is_file():
            raise FileNotFoundError("Missing generated step code: %s" % code)
        steps.append({"index": index, "name": name, "kind": kind,
                      "action_space": action_space, "code": code})
    return steps


def _resolve_run(run_dir):
    import yaml

    run_meta_path = run_dir / "run_meta.json"
    config_path = run_dir / "code" / "task.yaml"
    steps_path = run_dir / "artifacts" / "steps.json"
    for path in (run_meta_path, config_path, steps_path):
        if not path.is_file():
            raise FileNotFoundError("Incomplete RoboGen run; missing %s" % path)

    run_meta = _read_json(run_meta_path)
    scene_path = Path(run_meta["scene"]).expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError("Source scene recorded by this run is missing: %s" % scene_path)
    task_config = yaml.safe_load(config_path.read_text())
    solution_dir = Path(next(item["solution_path"] for item in task_config if "solution_path" in item))
    if not solution_dir.is_absolute():
        solution_dir = config_path.parent / solution_dir
    solution_dir = solution_dir.resolve()
    scene_config = next(item["external_scene"] for item in task_config if "external_scene" in item)
    scene_manifest = Path(scene_config["manifest"]).expanduser()
    if not scene_manifest.is_absolute():
        scene_manifest = config_path.parent / scene_manifest
    scene_manifest = scene_manifest.resolve()
    if not scene_manifest.is_file():
        raise FileNotFoundError("Prepared RoboGen scene manifest is missing: %s" % scene_manifest)
    steps = _read_steps(solution_dir)
    return run_meta, config_path, scene_path, scene_manifest, solution_dir, steps, steps_path


def _recorded_initial_state(run_meta):
    for session in reversed(run_meta.get("sessions", [])):
        if session.get("action") not in ("run", "execute"):
            continue
        value = (session.get("args") or {}).get("state")
        if not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError("Initial state recorded by this run is missing: %s" % path)
        return path
    return None


def _run_primitive(task_config, solution_dir, step, previous_state, stage_dir, gui, obj_id):
    from execute import execute_primitive

    outcome = {"native_success": None}
    _, states = execute_primitive(
        str(task_config), str(solution_dir), step["name"],
        str(previous_state) if previous_state else None, str(stage_dir),
        gui=gui, randomize=False, obj_id=obj_id, outcome=outcome,
    )
    if not states:
        raise RuntimeError("Primitive %r produced no replay state" % step["name"])
    return Path(states[-1]).resolve(), outcome.get("native_success")


def _run_reward(task_config, solution_dir, step, previous_state, checkpoint, output_state,
                gui, seed, training_algo, options):
    import pybullet as p
    import ray
    from ray import tune
    from RL.ray_learn import load_policy, make_env

    task_name = step["name"].replace(" ", "_")
    env_config = {
        "task_config_path": str(task_config),
        "solution_path": str(solution_dir),
        "task_name": task_name,
        "last_restore_state_file": str(previous_state) if previous_state else None,
        "action_space": step["action_space"],
        "randomize": False,
        "use_bard": options.get("using_bard", True),
        "obj_id": options.get("obj_id", 0),
        "use_gpt_size": options.get("using_gpt_size", True),
        "use_gpt_joint_angle": options.get("using_gpt_joint_angle", True),
        "use_gpt_spatial_relationship": options.get("using_gpt_spatial_relationship", True),
        "use_distractor": options.get("use_distractor", False),
    }
    tune.register_env(task_name, lambda config: make_env(config))
    env = make_env(env_config, render=gui)
    agent = None
    try:
        agent, _ = load_policy(training_algo, task_name, str(checkpoint), seed=seed,
                               env_config=env_config, eval=True)
        observation = env.reset()
        done = False
        total_reward = 0.0
        max_steps = int(getattr(env, "horizon", 120))
        for _ in range(max_steps):
            action = agent.compute_action(observation, explore=False)
            observation, reward, done, _ = env.step(action)
            total_reward += float(reward)
            if done:
                break
        if not done:
            raise RuntimeError("Reward policy %r exceeded the environment horizon (%d steps)" % (step["name"], max_steps))
        from manipulation.utils import save_env
        save_env(env, str(output_state))
        return output_state.resolve(), total_reward
    finally:
        if agent is not None:
            agent.stop()
        p.disconnect(env.id)


def _settle(env, seconds):
    import pybullet as p

    parameters = p.getPhysicsEngineParameters(physicsClientId=env.id)
    dt = float(parameters.get("fixedTimeStep", 1.0 / 240.0))
    for _ in range(math.ceil(seconds / dt)):
        p.stepSimulation(physicsClientId=env.id)


def _scene_world_pose(p, body, client_id):
    # Bullet stores a base's inertial/COM frame; convert it back to the model root.
    com_position, com_quaternion = p.getBasePositionAndOrientation(body, physicsClientId=client_id)
    dynamics = p.getDynamicsInfo(body, -1, physicsClientId=client_id)
    local_inertial = (dynamics[3], dynamics[4])
    inverse_inertial = p.invertTransform(*local_inertial)
    root_position, root_quaternion = p.multiplyTransforms(
        com_position, com_quaternion, *inverse_inertial
    )
    return [float(value) for value in root_position], [float(value) for value in root_quaternion]


def _export_scene_state(env, state_path, manifest_path):
    import pybullet as p

    source_manifest = _read_json(manifest_path)
    entities, skipped = {}, []
    for entity in source_manifest.get("entities", []):
        entity_id = entity["id"]
        if entity.get("robot"):
            skipped.append({"id": entity_id, "reason": "robot topology is not transferred by the scene-state converter"})
            continue
        if entity_id not in env.external_entity_ids:
            raise KeyError("Scene entity %r has no Bullet body mapping" % entity_id)
        body = env.external_entity_ids[entity_id]
        position, quaternion_xyzw = _scene_world_pose(p, body, env.id)
        joints = {}
        for index in range(p.getNumJoints(body, physicsClientId=env.id)):
            info = p.getJointInfo(body, index, physicsClientId=env.id)
            if info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                continue
            name = info[1].decode("utf-8")
            joints[name] = float(p.getJointState(body, index, physicsClientId=env.id)[0])
        entities[entity_id] = {
            "position": position,
            "quaternion": [quaternion_xyzw[3], *quaternion_xyzw[:3]],
            "joints": joints,
        }
    state = {
        "schema_version": 1,
        "coordinate_system": "scene_world",
        "quaternion_order": "wxyz",
        "entities": entities,
        "skipped_entities": skipped,
    }
    state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    return skipped


def replay(run_dir, seed=None, settle_seconds=0.5, gui=False,
           training_algo="sac"):
    import numpy as np
    import yaml

    run_dir = Path(run_dir).expanduser().resolve()
    if not math.isfinite(settle_seconds) or settle_seconds < 0:
        raise ValueError("--settle-seconds must be a finite, nonnegative number")
    (run_meta, task_config, scene_path, scene_manifest, solution_dir,
     steps, steps_path) = _resolve_run(run_dir)
    initial_state = _recorded_initial_state(run_meta)
    replay_seed = run_meta.get("seed", 0) if seed is None else seed
    if not 0 <= replay_seed <= 2**32 - 1:
        raise ValueError("--seed must be between 0 and 4294967295")
    random.seed(replay_seed)
    np.random.seed(replay_seed)

    checkpoints = {}
    for step in steps:
        if step["kind"] == "reward":
            checkpoints[step["name"]] = _checkpoint_for(step, steps_path)

    # Keep the final state snapshot independent of existing run artifacts.
    replay_dir = _new_replay_dir(run_dir)

    metadata_options = {}
    option_files = sorted((run_dir / "artifacts" / "experiment").glob("meta_info_*.json"))
    if option_files:
        metadata_options = _read_json(option_files[-1])

    replay_task_config = yaml.safe_load(task_config.read_text())
    external_config = next(item["external_scene"] for item in replay_task_config if "external_scene" in item)
    external_config["seed"] = replay_seed
    replay_task_path = replay_dir / "task.yaml"
    replay_task_path.write_text(yaml.safe_dump(replay_task_config, sort_keys=False))

    current_state = initial_state
    initial_states = []
    reward_totals = {}
    ray_started_here = False
    final_env = None
    try:
        if checkpoints:
            import ray
            if not ray.is_initialized():
                from run_output import ray_temp_dir
                ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False,
                         _temp_dir=ray_temp_dir(run_dir))
                ray_started_here = True

        for step in steps:
            step_dir = replay_dir / "stages" / ("%03d_%s" % (step["index"], step["name"].replace(" ", "_")))
            step_dir.mkdir(parents=True)
            if step["kind"] == "primitive":
                current_state, native_success = _run_primitive(
                    replay_task_path, solution_dir, step, current_state, step_dir,
                    gui, metadata_options.get("obj_id", 0),
                )
                initial_states.append({"step": step["name"], "kind": "primitive",
                                      "native_success": native_success})
            else:
                state_output = step_dir / "final_state.pkl"
                current_state, total_reward = _run_reward(
                    replay_task_path, solution_dir, step, current_state,
                    checkpoints[step["name"]], state_output, gui, replay_seed,
                    training_algo, metadata_options,
                )
                reward_totals[step["name"]] = total_reward
                initial_states.append({"step": step["name"], "kind": "reward",
                                      "policy_checkpoint": _relative(checkpoints[step["name"]], replay_dir),
                                      "total_reward": total_reward})

        if current_state is None:
            raise RuntimeError("The RoboGen task did not produce a final simulator state")

        # Rebuild the final state once so settling, native pickle and JSON all capture the same frame.
        from manipulation.utils import build_up_env, save_env
        last_step = steps[-1]
        final_env, _ = build_up_env(
            str(replay_task_path), str(solution_dir), last_step["name"].replace(" ", "_"),
            str(current_state), action_space=last_step["action_space"],
            render=gui, randomize=False, obj_id=metadata_options.get("obj_id", 0),
        )
        final_env.reset()
        _settle(final_env, settle_seconds)
        native_state_path = replay_dir / "final_state.pkl"
        save_env(final_env, str(native_state_path))
        state_path = replay_dir / "final_state.json"
        skipped = _export_scene_state(final_env, state_path, scene_manifest)
        source_files = [step["code"] for step in steps]
        policies = {name: _relative(path, replay_dir) for name, path in checkpoints.items()}
        import pybullet as p
        fixed_time_step = p.getPhysicsEngineParameters(physicsClientId=final_env.id).get(
            "fixedTimeStep", 1.0 / 240.0
        )
        manifest = {
            "schema_version": 1,
            "method": "robogen",
            "engine": "pybullet",
            "scene": _relative(scene_path, replay_dir),
            "scene_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            "scene_manifest": _relative(scene_manifest, replay_dir),
            "state_file": "final_state.json",
            "native_state_file": "final_state.pkl",
            "seed": replay_seed,
            **{key: run_meta[key] for key in ("batch_hash", "batch_run_id", "batch_name") if key in run_meta},
            "settle_seconds": settle_seconds,
            "source": {
                "code": {
                    "directory": _relative(solution_dir, replay_dir),
                    "files": [_relative(path, replay_dir) for path in source_files],
                },
                "policies": policies,
                "training_algo": training_algo,
                "task_config": "task.yaml",
                **({"initial_state": _relative(initial_state, replay_dir)} if initial_state else {}),
            },
            "native_env_config": {
                "physics_engine": "pybullet",
                "robot_mode": run_meta.get("robot_mode"),
                "fixed_time_step": float(fixed_time_step),
            },
            "env_config": {
                "auto_add_ground": True,
                "genesis_precision": "32",
                "dt": 0.01,
            },
            "skipped_entities": skipped,
        }
        (replay_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        result = {
            "schema_version": 1,
            "method": "robogen",
            "seed": replay_seed,
            "steps": initial_states,
            "reward_totals": reward_totals,
            "settle_seconds": settle_seconds,
            "native_state_file": "final_state.pkl",
            "state_file": "final_state.json",
        }
        (replay_dir / "replay_result.json").write_text(json.dumps(result, indent=2) + "\n")
        return replay_dir
    except Exception as error:
        (replay_dir / "replay_error.json").write_text(json.dumps({"error": str(error)}, indent=2) + "\n")
        raise
    finally:
        if final_env is not None:
            import pybullet as p
            p.disconnect(final_env.id)
        if ray_started_here:
            import ray
            ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed RoboGen run directory")
    parser.add_argument("--seed", type=int, default=None, help="Replay seed; defaults to the recorded scene seed")
    parser.add_argument("--settle-seconds", type=float, default=0.5,
                        help="Physics settling time after replay and before export (default: 0.5 seconds)")
    parser.add_argument("--gui", action="store_true", help="Replay with the PyBullet GUI")
    parser.add_argument("--training-algo", choices=("sac", "ppo"), default="sac",
                        help="Algorithm used by the saved reward-stage policies (default: sac)")
    args = parser.parse_args()
    try:
        run_dir = args.run_dir.expanduser().resolve()
        os.chdir(ROOT)
        output = replay(run_dir, seed=args.seed, settle_seconds=args.settle_seconds,
                        gui=args.gui,
                        training_algo=args.training_algo)
    except Exception as error:
        print("Replay failed: %s" % error, file=sys.stderr)
        raise SystemExit(1) from error
    print("Saved RoboGen replay state:", output)


if __name__ == "__main__":
    main()

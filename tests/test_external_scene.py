"""Run with .venv/bin/python -m unittest discover -s tests -v.

Set ROBOGEN_EXTERNAL_MANIFEST to include the converted real-scene checks.
"""
import json
import os
import runpy
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from manipulation.external_scene_assets import convert_mesh, convert_mjcf, xyzw


class ExternalSceneChecks(unittest.TestCase):
    def test_gif_export_with_current_pillow_and_original_moviepy(self):
        from PIL import Image
        from manipulation.utils import save_numpy_as_gif
        frames = np.zeros((3, 12, 16, 3), dtype=np.uint8)
        for index in range(3):
            frames[index, :, :, index] = 255
        with tempfile.TemporaryDirectory() as temporary:
            for scale in (1.0, 0.5, 2.0):
                with self.subTest(scale=scale):
                    path = Path(temporary) / ("scale_%s.gif" % scale)
                    clip = save_numpy_as_gif(frames, str(path), fps=5, scale=scale)
                    try:
                        with Image.open(path) as gif:
                            self.assertEqual(gif.size, (int(16*scale), int(12*scale)))
                            self.assertEqual(gif.n_frames, 3)
                            for index in range(3):
                                gif.seek(index)
                                self.assertEqual(gif.convert("RGB").getpixel((0, 0)),
                                                 tuple(frames[index, 0, 0]))
                    finally:
                        clip.close()

    def test_default_run_pipeline_reuses_solutions_and_stops_on_generation_failure(self):
        import run_external_scene as runner
        root = Path(__file__).resolve().parents[1]
        for origin in ("fresh", "manual_reference_adapted_from_author_example", "provided_solution", "failed", "empty"):
            with self.subTest(origin=origin), tempfile.TemporaryDirectory(dir=root) as temporary:
                output = Path(temporary) / "output"
                scene = Path(temporary) / "scene.yaml"
                scene.write_text("fixture")
                events = []

                def prepare(path, destination):
                    events.append("prepare")
                    destination.mkdir(parents=True, exist_ok=True)
                    manifest = destination / "manifest.json"
                    manifest.write_text(json.dumps({"source_scene": str(path), "source_hashes": {},
                                                   "task": {"name": "move_remote_to_tray", "description": "Move remote"}}))
                    return manifest

                if origin in ("manual_reference_adapted_from_author_example", "provided_solution", "empty"):
                    manifest = prepare(scene, output)
                    config, directory = runner.prepare_task(manifest, "robogen", 0, output)
                    metadata = json.loads((directory.parent / "experiment.json").read_text())
                    metadata["solution_provenance"] = "provided_solution" if origin == "empty" else origin
                    (directory.parent / "experiment.json").write_text(json.dumps(metadata))
                    if origin == "empty":
                        for name in ("substeps.txt", "substep_types.txt", "action_spaces.txt"):
                            (directory / name).write_text("")
                    events.clear()

                def check(config, directory, gui=False, state=None, evaluation=False):
                    events.append("evaluate" if evaluation else "check")
                    if evaluation:
                        self.assertEqual(state, str(Path(temporary) / "final.pkl"))

                def generate(manifest, config, directory, model):
                    events.append("generate")
                    self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://cli.example/v1")
                    self.assertEqual(os.environ["ROBOGEN_API_KEY_ENV"], "TEST_PROVIDER_KEY")
                    if origin == "failed":
                        raise ValueError("generation failed")
                    runner.reference_solution(directory, {"name": "move_remote_to_tray"})
                    metadata = json.loads((directory.parent / "experiment.json").read_text())
                    metadata["solution_provenance"] = "original_robogen_llm_generator"
                    (directory.parent / "experiment.json").write_text(json.dumps(metadata))

                def execute(config, **kwargs):
                    events.append("execute")
                    self.assertFalse(kwargs["randomize"])
                    directory = Path(config).parent / "task_solution" / "experiment"
                    directory.mkdir(parents=True)
                    (directory / ("all_last_state_files_%s.txt" % kwargs["time_string"])).write_text(
                        str(Path(temporary) / "final.pkl")+"\n")

                with patch("sys.argv", ["run_external_scene.py", "--scene-yaml", str(scene), "--output-dir", str(output),
                                        "--base-url", "https://cli.example/v1/", "--api-key-env", "TEST_PROVIDER_KEY"]), \
                        patch.dict(os.environ, {"OPENAI_BASE_URL": "https://environment.example/v1",
                                                "ROBOGEN_API_KEY_ENV": "OTHER_PROVIDER_KEY", "TEST_PROVIDER_KEY": "test-custom-key"}), \
                        patch.object(runner, "prepare_scene", prepare), patch.object(runner, "check_scene", check), \
                        patch.object(runner, "generate_solution", generate), patch("builtins.print"), \
                        patch.dict("sys.modules", {"execute": types.SimpleNamespace(execute=execute)}):
                    if origin in ("failed", "empty"):
                        with self.assertRaisesRegex(ValueError, "generation failed" if origin == "failed" else "Empty"):
                            runner.main()
                    else:
                        runner.main()
                expected = [] if origin in ("manual_reference_adapted_from_author_example", "provided_solution", "empty") else ["prepare"]
                expected.append("check")
                if origin not in ("provided_solution", "empty"):
                    expected.append("generate")
                if origin not in ("failed", "empty"):
                    expected.extend(["execute", "evaluate"])
                self.assertEqual(events, expected)

    def test_markdown_response_uses_original_generator_and_rejects_empty_solutions(self):
        import re
        import run_external_scene as runner
        from gpt_4.prompts import prompt_manipulation_reward_primitive as generator
        response = '''### Substep 1: Grasp the remote
```primitive
rgbs, final_state = grasp_object(self, "remote")
success = check_grasped(self, "remote")
```
### Substep 2: Move the remote into the tray
```reward
def _compute_reward(self):
    reward = -np.linalg.norm(get_position(self, "remote") - get_position(self, "tray"))
    return reward, False
```
```action space
normalized-direct-translation
```
### Substep 3: Release the remote
```primitive
rgbs, final_state = release_grasp(self)
success = not check_grasped(self, "remote")
```
'''
        normalized = runner.normalize_solution_response(response)
        self.assertEqual(re.findall(r"```.*?```", normalized, re.DOTALL),
                         re.findall(r"```.*?```", response, re.DOTALL))
        self.assertIn("Substep 1: Grasp the remote", normalized.splitlines())
        for cached in (False, True):
            with self.subTest(cached=cached), tempfile.TemporaryDirectory(dir=runner.ROOT) as temporary:
                output = Path(temporary)
                manifest = output / "manifest.json"
                manifest.write_text(json.dumps({"source_scene": "synthetic", "entities": [], "task": {
                    "name": "move_remote_to_tray", "description": "Move remote into tray"}}))
                config, directory = runner.prepare_task(manifest, "robogen", 0, output, create_reference=False)
                log = directory.parent / "gpt_response/reward.json"
                log.parent.mkdir()
                log.write_text(json.dumps({"res": response}))
                original_log = log.read_bytes()
                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
                        patch.object(generator, "query", return_value=response) as query, patch("builtins.print"):
                    runner.generate_solution(manifest, config, directory, "test-model",
                                             cached_response=response if cached else None)
                self.assertEqual(query.call_count, 0 if cached else 1)
                self.assertEqual(log.read_bytes(), original_log)
                self.assertEqual((directory / "substep_types.txt").read_text().splitlines(), ["primitive", "reward", "primitive"])
                runner.validate_solution(directory)
                self.assertEqual(json.loads((directory.parent / "experiment.json").read_text())["solution_provenance"],
                                 "original_robogen_llm_generator")
                (directory / "action_spaces.txt").write_text("")
                with self.assertRaisesRegex(ValueError, "mismatched"):
                    runner.validate_solution(directory)
                for name in ("substeps.txt", "substep_types.txt"):
                    (directory / name).write_text("")
                with self.assertRaisesRegex(ValueError, "Empty"):
                    runner.validate_solution(directory)
                with patch.dict(os.environ, {}, clear=True), patch("builtins.print"):
                    with self.assertRaisesRegex(ValueError, "Empty"):
                        runner.generate_solution(manifest, config, directory, "test-model", cached_response="No task code")

    def test_api_base_routes_original_query_without_changing_its_logic(self):
        import ast
        import openai
        import requests
        import run_external_scene as runner
        path = runner.ROOT / "gpt_4/query.py"
        original = subprocess.check_output(["git", "show", "HEAD:gpt_4/query.py"], cwd=runner.ROOT)
        function = lambda source: ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)))
        self.assertEqual(function(original), function(path.read_text()))
        official = "https://api.openai.com/v1"
        for settings, expected, key in (({}, official, "test-key"),
                                        ({"OPENAI_API_BASE": "https://legacy.example/v1/"}, "https://legacy.example/v1", "test-key"),
                                        ({"OPENAI_BASE_URL": "https://custom.example/v1/",
                                          "OPENAI_API_BASE": "https://legacy.example/v1"}, "https://custom.example/v1", "test-key"),
                                        ({"YUFEI_OPENAI_API_KEY": "test-legacy-key"}, official, "test-legacy-key"),
                                        ({"ROBOGEN_API_KEY_ENV": "TEST_PROVIDER_KEY", "TEST_PROVIDER_KEY": "test-custom-key",
                                          "YUFEI_OPENAI_API_KEY": "test-legacy-key", "OPENAI_BASE_URL": "https://custom.example/v1"},
                                         "https://custom.example/v1", "test-custom-key")):
            with self.subTest(settings=settings), patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", **settings}, clear=True), \
                    patch.object(openai, "api_base", official), patch.object(openai, "api_key", None), \
                    patch("builtins.print"):
                query = runpy.run_path(str(path))["query"]
                response = requests.Response()
                response.status_code = 200
                response.headers["Content-Type"] = "application/json"
                response._content = b'{"choices": [{"message": {"content": "ok"}}]}'
                with patch("requests.sessions.Session.request", return_value=response) as request:
                    self.assertEqual(query("system", ["task"], [], model="gpt-4"), "ok")
                self.assertEqual(request.call_args.args[:2], ("post", expected+"/chat/completions"))
                self.assertEqual(json.loads(request.call_args.kwargs["data"])["model"], "gpt-4")
                self.assertEqual(request.call_args.kwargs["headers"]["Authorization"], "Bearer "+key)
        for value in (None, ""):
            settings = {"ROBOGEN_API_KEY_ENV": "TEST_PROVIDER_KEY", "OPENAI_API_KEY": "fallback-key"}
            if value is not None:
                settings["TEST_PROVIDER_KEY"] = value
            with self.subTest(value=value), patch.dict(os.environ, settings, clear=True):
                with self.assertRaisesRegex(ValueError, "TEST_PROVIDER_KEY"):
                    runpy.run_path(str(path))
                with self.assertRaisesRegex(ValueError, "TEST_PROVIDER_KEY"):
                    runner.generate_solution(None, None, None, "gpt-4")
                with patch("sys.argv", ["run_external_scene.py", "--scene-yaml", "unused.yaml",
                                        "--api-key-env", "TEST_PROVIDER_KEY"]), patch("sys.stderr"):
                    with self.assertRaises(SystemExit):
                        runner.main()
        with patch("sys.argv", ["run_external_scene.py", "--scene-yaml", "unused.yaml", "--base-url", "invalid"]), \
                patch.object(runner, "prepare_scene") as prepare, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                runner.main()
            prepare.assert_not_called()

    def test_original_core_is_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("execute.py", "execute_long_horizon.py", "RL/ray_learn.py",
                     "manipulation/sim.py", "manipulation/panda.py", "manipulation/agent.py",
                     "manipulation/gpt_primitive_api.py", "manipulation/gpt_reward_api.py",
                     "gpt_4/prompts/prompt_manipulation_reward_primitive.py"):
            original = subprocess.check_output(["git", "show", "HEAD:"+name], cwd=root)
            self.assertEqual(original, (root / name).read_bytes(), name)

    def test_quaternion_order_and_invalid_input(self):
        self.assertEqual(xyzw([1, 0, 0, 0]), [0, 0, 0, 1])
        self.assertEqual(xyzw([0, 0, 0, 1]), [0, 0, 1, 0])
        with self.assertRaises(ValueError):
            xyzw([2, 0, 0, 0])

    def test_prepare_preserves_an_existing_solution(self):
        from run_external_scene import prepare_task
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            directory = Path(temporary)
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"source_scene": "synthetic", "task": {
                "name": "move_remote_to_tray", "description": "Move the remote"}}))
            _, solution = prepare_task(manifest, "robogen", 0, directory)
            path = solution / "grasp_remote.py"
            path.write_text("existing research solution\n")
            prepare_task(manifest, "robogen", 0, directory)
            self.assertEqual(path.read_text(), "existing research solution\n")

    def test_mesh_graph_transform_and_scale_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            scene = trimesh.Scene()
            transform = trimesh.transformations.translation_matrix([2, -3, 4])
            scene.add_geometry(trimesh.creation.box(extents=[1, 2, 3]), transform=transform)
            path = directory / "source.glb"
            scene.export(path)
            asset = convert_mesh(path, 0.25, True, directory)
            expected = np.array([[1.5, -5.5, -4], [2.5, -2.5, -2]])*0.25
            np.testing.assert_allclose(asset["bounds_local"], expected)
            exported = trimesh.load(asset["visual"], force="scene").dump(concatenate=True)
            np.testing.assert_allclose(exported.bounds, expected)

    @unittest.skipUnless(os.environ.get("ROBOGEN_EXTERNAL_MANIFEST"), "real scene manifest not specified")
    def test_pointcloud_camera_preserves_scene_and_handles_empty_frames(self):
        import pybullet as p
        from manipulation.external_scene import ExternalSceneEnv
        from manipulation.utils import take_round_images_around_object
        from run_external_scene import prepare_task
        manifest = Path(os.environ["ROBOGEN_EXTERNAL_MANIFEST"]).resolve()
        for mode in ("robogen", "external"):
            config, _ = prepare_task(manifest, mode, 9123, manifest.parent)
            env = ExternalSceneEnv(config_path=str(config))
            try:
                env.setup_camera([2.7, -2.7, 2.5], [0, 0, 0.35], camera_width=160, camera_height=120)
                rgb = env.render()[0]
                poses = {n: p.getBasePositionAndOrientation(b, physicsClientId=env.id)
                         for n, b in env.urdf_ids.items()}
                take_round_images_around_object(env, "remote", only_object=True,
                                                azimuth_interval=360, camera_width=160, camera_height=120)
                np.testing.assert_array_equal(env.render()[0], rgb)
                self.assertEqual(poses, {n: p.getBasePositionAndOrientation(b, physicsClientId=env.id)
                                        for n, b in env.urdf_ids.items()})
            finally:
                env.close()

    @unittest.skipUnless(os.environ.get("ROBOGEN_EXTERNAL_MANIFEST"), "real scene manifest not specified")
    def test_all_collision_hulls_are_loaded(self):
        import pybullet as p
        manifest = json.loads(Path(os.environ["ROBOGEN_EXTERNAL_MANIFEST"]).read_text())
        client = p.connect(p.DIRECT)
        try:
            for entity in manifest["entities"]:
                if "mesh_asset" not in entity or entity.get("fixed"):
                    continue
                asset = entity["mesh_asset"]
                body = p.loadURDF(asset["urdf"], physicsClientId=client)
                self.assertEqual(len(p.getCollisionShapeData(body, -1, physicsClientId=client)),
                                 len(asset["collisions"]), entity["id"])
        finally:
            p.disconnect(client)

    @unittest.skipUnless(os.environ.get("ROBOGEN_EXTERNAL_MANIFEST"), "real scene manifest not specified")
    def test_mjcf_forward_kinematics(self):
        import mujoco
        import pybullet as p
        from scipy.spatial.transform import Rotation
        manifest = json.loads(Path(os.environ["ROBOGEN_EXTERNAL_MANIFEST"]).read_text())
        entity = next(e for e in manifest["entities"] if e.get("robot"))
        model = mujoco.MjModel.from_xml_path(entity["robot_asset"]["source_mjcf"])
        data = mujoco.MjData(model)
        client = p.connect(p.DIRECT)
        try:
            body = p.loadURDF(entity["robot_asset"]["urdf"], useFixedBase=True, physicsClientId=client)
            links = {p.getJointInfo(body, i, physicsClientId=client)[12].decode(): i
                     for i in range(p.getNumJoints(body, physicsClientId=client))}
            joints = {p.getJointInfo(body, i, physicsClientId=client)[1].decode(): i
                      for i in range(p.getNumJoints(body, physicsClientId=client))}
            for values in (model.qpos0, model.key_qpos[0]):
                data.qpos[:] = values
                mujoco.mj_forward(model, data)
                for name, value in zip(entity["robot_asset"]["joint_names"], values):
                    p.resetJointState(body, joints[name], value, physicsClientId=client)
                for name, link in links.items():
                    if name == "robogen_tcp":
                        continue
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                    state = p.getLinkState(body, link, computeForwardKinematics=True, physicsClientId=client)
                    np.testing.assert_allclose(state[4], data.xpos[bid], atol=2e-6)
                    np.testing.assert_allclose(Rotation.from_quat(state[5]).as_matrix(),
                                               data.xmat[bid].reshape(3, 3), atol=2e-6)
        finally:
            p.disconnect(client)

    @unittest.skipUnless(os.environ.get("ROBOGEN_EXTERNAL_MANIFEST"), "real scene manifest not specified")
    def test_both_robot_modes_step_restore_and_plan(self):
        from manipulation.utils import build_up_env, save_env, load_env
        from manipulation.motion_planning_utils import motion_planning
        from run_external_scene import prepare_task
        manifest_path = Path(os.environ["ROBOGEN_EXTERNAL_MANIFEST"]).resolve()
        states = {}
        for mode in ("robogen", "external"):
            config, directory = prepare_task(manifest_path, mode, 9123, manifest_path.parent)
            env, _ = build_up_env(str(config), str(directory.relative_to(Path.cwd())),
                                  "move_remote_into_tray", None)
            try:
                obs = env.reset()
                self.assertEqual(obs.shape, env.observation_space.shape)
                state = save_env(env)
                observation, reward, done, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1]))
                self.assertTrue(np.isfinite(observation).all())
                self.assertTrue(np.isfinite(reward))
                load_env(env, state=state)
                np.testing.assert_allclose(env._get_obs(), obs, atol=1e-5)
                states[mode] = state
                position, quaternion = env.robot.get_pos_orient(env.robot.right_end_effector)
                success, path = motion_planning(env, position, quaternion, obstacles=[])
                self.assertTrue(success)
                self.assertEqual(len(path[0]), len(env.robot.right_arm_joint_indices))
                if mode == "external":
                    before = env.robot.get_joint_angles(env.robot.right_gripper_indices)
                    import pybullet_ompl.pb_ompl as pb_ompl
                    robot = pb_ompl.PbOMPLRobot(env.robot.body, env.robot.planning_joint_indices)
                    robot.set_state(env.robot.get_joint_angles(env.robot.right_arm_joint_indices))
                    np.testing.assert_allclose(before, env.robot.get_joint_angles(env.robot.right_gripper_indices))
                    with self.assertRaises(ValueError):
                        load_env(env, state=states["robogen"])
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()

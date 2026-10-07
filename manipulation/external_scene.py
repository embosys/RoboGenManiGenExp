"""Scene/robot compatibility only; task actions, planning and learning stay in RoboGen."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pybullet as p
import yaml

from manipulation.robot import Robot
from manipulation.panda import Panda
from manipulation.sim import SimpleEnv
from manipulation.utils import load_env


class ExternalPanda(Robot):
    def __init__(self, entity):
        self.entity = entity
        super().__init__("right", [], -1, [])

    def get_joint_angles(self, indices=None):
        values = super().get_joint_angles(indices)
        if self.lower_limits is not None:
            for i, joint in enumerate(self.all_joint_indices if indices is None else indices):
                clipped = np.clip(values[i], self.lower_limits[joint], self.upper_limits[joint])
                # Bullet can settle a joint nanoradians outside its hard limit.
                if abs(values[i] - clipped) <= 1e-7:
                    values[i] = clipped
        return values

    def init(self, directory, id, np_random, fixed_base=True, use_suction=True):
        body = p.loadURDF(self.entity["robot_asset"]["urdf"],
                          basePosition=self.entity.get("position", [0, 0, 0]),
                          baseOrientation=self.entity["quaternion_xyzw"],
                          useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_INERTIA_FROM_FILE,
                          physicsClientId=id)
        joints = {p.getJointInfo(body, i, physicsClientId=id)[1].decode(): i
                  for i in range(p.getNumJoints(body, physicsClientId=id))}
        links = {p.getJointInfo(body, i, physicsClientId=id)[12].decode(): i
                 for i in range(p.getNumJoints(body, physicsClientId=id))}
        self.right_arm_joint_indices = [joints["joint%d" % i] for i in range(1, 8)]
        self.controllable_joint_indices = self.right_arm_joint_indices
        self.right_gripper_indices = [joints["finger_joint1"], joints["finger_joint2"]]
        self.right_end_effector = links["robogen_tcp"]
        self.suction_link = links["left_finger"]
        # Map the original planner/IK interface to the seven arm DOFs only.
        self.planning_joint_indices = self.right_arm_joint_indices
        super().init(body, id, np_random)
        motor_joints = self.right_arm_joint_indices + self.right_gripper_indices
        self.set_joint_angles(motor_joints, self.entity["robot_asset"]["initial_qpos"])
        self.set_gripper_open_position(self.right_gripper_indices, [0.04, 0.04])


class ExternalSceneEnv(SimpleEnv):
    def __init__(self, *args, config_path=None, **kwargs):
        config = yaml.safe_load(Path(config_path).read_text())
        self.external_config = next(item["external_scene"] for item in config if "external_scene" in item)
        self.manifest_path = Path(self.external_config["manifest"]).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.manifest_hash = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.robot_mode = self.external_config["robot_mode"]
        if self.robot_mode not in ("robogen", "external"):
            raise ValueError("robot_mode must be robogen or external")
        if kwargs.get("randomize", False):
            raise ValueError("External scene layout is fixed; randomize must be disabled")
        if kwargs.get("use_suction", True) is not True:
            raise ValueError("Both robot modes retain the original RoboGen suction logic")
        super().__init__(*args, config_path=config_path, **kwargs)

    def seed(self, seed=None):
        from gym.utils import seeding
        self.np_random, actual_seed = seeding.np_random(self.external_config.get("seed", seed))
        return [actual_seed]

    def restore_metadata(self):
        return {"manifest_sha256": self.manifest_hash, "robot_mode": self.robot_mode,
                "grasp_model": "original_robogen_suction"}

    def validate_restore(self, state):
        if state.get("external_scene") != self.restore_metadata():
            raise ValueError("Restore state belongs to a different scene or robot mode")

    def capture_velocities(self):
        return {name: {"base": p.getBaseVelocity(body, physicsClientId=self.id),
                       "joints": [p.getJointState(body, i, physicsClientId=self.id)[1]
                                  for i in range(p.getNumJoints(body, physicsClientId=self.id))]}
                for name, body in self.urdf_ids.items()}

    def restore_velocities(self, state):
        for name, values in state.get("external_velocities", {}).items():
            body = self.urdf_ids[name]
            p.resetBaseVelocity(body, *values["base"], physicsClientId=self.id)
            for index, velocity in enumerate(values["joints"]):
                angle = p.getJointState(body, index, physicsClientId=self.id)[0]
                p.resetJointState(body, index, angle, targetVelocity=velocity, physicsClientId=self.id)

    def set_scene(self):
        p.resetSimulation(physicsClientId=self.id)
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)
        self.use_table = False  # Original coffee table is an ordinary scene entity.
        self.solution_path = self.external_config["solution_path"]
        restore_state = None
        if self.restore_state_file is not None:
            with open(self.restore_state_file, "rb") as file:
                restore_state = pickle.load(file)
            self.validate_restore(restore_state)
        robot_entity = next(e for e in self.manifest["entities"] if e.get("robot"))
        self.robot_name = "panda"  # Preserve original Panda primitive orientation conventions.
        if self.robot_mode == "robogen":
            self.robot = Panda()
            self.robot.init(self.asset_dir, self.id, self.np_random, fixed_base=True, use_suction=True)
            # Original mobile Panda's base frame is 0.28 m above its floor support.
            position = np.array(robot_entity.get("position", [0, 0, 0]), dtype=float)
            position[2] += 0.28
            self.robot.set_base_pos_orient(position, robot_entity["quaternion_xyzw"])
            self.robot.set_joint_angles(self.robot.right_arm_joint_indices, self.get_robot_init_joint_angles())
            self.suction_id = self.robot.right_gripper_indices[0]
        else:
            self.robot = ExternalPanda(robot_entity)
            self.robot.init(self.asset_dir, self.id, self.np_random)
            self.suction_id = self.robot.suction_link
        self.robot_class = type(self.robot)
        self.robot_base_orient = robot_entity["quaternion_xyzw"]
        self.agents = [self.robot]
        self.robot.motor_gains = 0.05
        self.robot.motor_forces = 100.0
        self.urdf_ids = {"robot": self.robot.body}
        self.urdf_paths, self.urdf_types = {}, {}
        self.init_positions, self.on_tables, self.simulator_sizes = {}, {}, {}
        self.is_distractor = {"robot": 0, "plane": 0}
        self.external_entity_ids = {robot_entity["id"]: self.robot.body}
        task = self.manifest["task"]
        active = {task.get(k) for k in ("target_id", "source_support_id", "goal_id")}
        for entity in self.manifest["entities"]:
            if entity.get("robot"):
                continue
            body = self.load_external_object(entity)
            name = "plane" if entity["id"] == "room_envelope_floor" else entity["id"]
            self.external_entity_ids[entity["id"]] = body
            self.urdf_ids[name] = body
            self.urdf_paths[name] = entity.get("file", "")
            self.urdf_types[name] = "mesh"
            self.init_positions[name] = np.array(entity.get("position", [0, 0, 0]))
            self.on_tables[name] = False
            self.simulator_sizes[name] = entity.get("scale", 1)
            self.is_distractor[name] = int(entity["id"] not in active)
        if "plane" not in self.urdf_ids:
            self.urdf_ids["plane"] = p.loadURDF(str(Path(self.asset_dir) / "plane/plane.urdf"), physicsClientId=self.id)
        self.is_distractor["plane"] = 0
        self.imported_poses = {name: p.getBasePositionAndOrientation(body, physicsClientId=self.id)
                               for name, body in self.external_entity_ids.items()}
        # Match the author's settling period; never correct or reposition imported objects.
        self.record_initial_joint_and_pose()
        for _ in range(500):
            p.stepSimulation(physicsClientId=self.id)
        if restore_state is not None:
            load_env(self, state=restore_state)
        self.init_state = p.saveState(physicsClientId=self.id)

    def load_external_object(self, entity):
        fixed = entity.get("fixed", False)
        if entity["type"] == "mesh" and not fixed:
            body = p.loadURDF(entity["mesh_asset"]["urdf"],
                              basePosition=entity.get("position", [0, 0, 0]),
                              baseOrientation=entity["quaternion_xyzw"],
                              flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL, physicsClientId=self.id)
            p.changeDynamics(body, -1, mass=entity.get("mass", self.manifest["physics"]["dynamic_mass"]),
                             lateralFriction=self.manifest["physics"]["lateral_friction"], physicsClientId=self.id)
            return body
        if entity["type"] == "box":
            half = np.asarray(entity["size"], dtype=float) / 2
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.id)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.id)
        else:
            asset = entity["mesh_asset"]
            collision = p.createCollisionShape(p.GEOM_MESH, fileName=asset["collisions"][0],
                                               flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.id)
            visual = p.createVisualShape(p.GEOM_MESH, fileName=asset["visual"],
                                        flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL, physicsClientId=self.id)
        body = p.createMultiBody(baseMass=0 if fixed else entity.get("mass", self.manifest["physics"]["dynamic_mass"]),
                                 baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                                 basePosition=entity.get("position", [0, 0, 0]),
                                 baseOrientation=entity["quaternion_xyzw"], physicsClientId=self.id)
        p.changeDynamics(body, -1, lateralFriction=self.manifest["physics"]["lateral_friction"], physicsClientId=self.id)
        if "surface_texture" in entity:
            texture = p.loadTexture(entity["surface_texture"], physicsClientId=self.id)
            p.changeVisualShape(body, -1, textureUniqueId=texture, physicsClientId=self.id)
        return body

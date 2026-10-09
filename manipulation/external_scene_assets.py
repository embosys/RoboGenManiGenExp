"""Convert a scene bundle without recentering, normalizing, or replacing assets."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml


def xyzw(rotation):
    q = np.asarray(rotation, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1):
        raise ValueError("rotation must be a unit quaternion in wxyz order")
    return q[[1, 2, 3, 0]].tolist()


def numbers(values):
    return " ".join(format(float(v), ".17g") for v in values)


def origin(parent, position, quaternion):
    ET.SubElement(parent, "origin", xyz=numbers(position),
                  rpy=numbers(Rotation.from_quat(xyzw(quaternion)).as_euler("xyz")))


def convert_mjcf(source, directory):
    """Use MuJoCo's compiler to resolve defaults, mesh frames and inertias."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(source))
    root = ET.Element("robot", name="external_franka")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(model.nbody)]
    if model.nbody < 2 or sum(model.body_parentid == 0) != 2:
        raise ValueError("MJCF must contain one fixed robot root body")
    meshes = {}
    for i in range(model.nmesh):
        va, vn = model.mesh_vertadr[i], model.mesh_vertnum[i]
        fa, fn = model.mesh_faceadr[i], model.mesh_facenum[i]
        file = directory / ("mesh_%d.obj" % i)
        trimesh.Trimesh(model.mesh_vert[va:va+vn], model.mesh_face[fa:fa+fn],
                        process=False).export(file)
        meshes[i] = str(file)
    joint_names = []
    for i in range(1, model.nbody):
        link = ET.SubElement(root, "link", name=names[i])
        visual_scene = trimesh.Scene()
        inertia = ET.SubElement(link, "inertial")
        origin(inertia, model.body_ipos[i], model.body_iquat[i])
        ET.SubElement(inertia, "mass", value=str(model.body_mass[i]))
        ix, iy, iz = model.body_inertia[i]
        ET.SubElement(inertia, "inertia", ixx=str(ix), iyy=str(iy), izz=str(iz),
                      ixy="0", ixz="0", iyz="0")
        for g in range(model.body_geomadr[i], model.body_geomadr[i]+model.body_geomnum[i]):
            visual = model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0
            element = ET.SubElement(link, "visual" if visual else "collision")
            origin(element, model.geom_pos[g], model.geom_quat[g])
            geometry = ET.SubElement(element, "geometry")
            kind = model.geom_type[g]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                ET.SubElement(geometry, "mesh", filename=meshes[model.geom_dataid[g]])
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                ET.SubElement(geometry, "box", size=numbers(model.geom_size[g]*2))
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                ET.SubElement(geometry, "sphere", radius=str(model.geom_size[g, 0]))
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                ET.SubElement(geometry, "cylinder", radius=str(model.geom_size[g, 0]),
                              length=str(model.geom_size[g, 1]*2))
            else:
                raise ValueError("Unsupported MJCF geom type: %s" % kind)
            if visual:
                rgba = model.geom_rgba[g]
                if model.geom_matid[g] >= 0:
                    rgba = model.mat_rgba[model.geom_matid[g]]
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = trimesh.load(meshes[model.geom_dataid[g]], process=False)
                elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                    mesh = trimesh.creation.box(model.geom_size[g]*2)
                elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    mesh = trimesh.creation.icosphere(radius=model.geom_size[g, 0])
                else:
                    mesh = trimesh.creation.cylinder(radius=model.geom_size[g, 0], height=model.geom_size[g, 1]*2)
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_quat(xyzw(model.geom_quat[g])).as_matrix()
                transform[:3, 3] = model.geom_pos[g]
                mesh.apply_transform(transform)
                colors = np.tile(np.rint(rgba*255).astype(np.uint8), (len(mesh.vertices), 1))
                mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors).to_texture()
                mesh.visual.material.diffuse = [255, 255, 255, 255]
                mesh.visual.material.name = "geom_%d" % g
                visual_scene.add_geometry(mesh)
                link.remove(element)
        if visual_scene.geometry:
            # Bullet changes RGBA for an entire link, not individual visual geoms.
            # One textured visual keeps all source colors through hide/restore.
            folder = directory / names[i]
            folder.mkdir(exist_ok=True)
            visual_file = folder / "visual.obj"
            visual_scene.export(visual_file)
            geometry = ET.SubElement(ET.SubElement(link, "visual"), "geometry")
            ET.SubElement(geometry, "mesh", filename=str(visual_file))
        if model.body_parentid[i] == 0:
            if model.body_jntnum[i]:
                raise ValueError("Free robot bases are not supported")
            continue
        if model.body_jntnum[i] > 1:
            raise ValueError("Only one joint per MJCF body is supported")
        j = model.body_jntadr[i]
        moving = model.body_jntnum[i] == 1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) if moving else names[i]+"_fixed"
        kind = "fixed"
        if moving:
            if not np.allclose(model.jnt_pos[j], 0):
                raise ValueError("Nonzero MJCF joint anchors require a separate converter")
            if model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError("Unsupported MJCF joint type")
            kind = "revolute" if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else "prismatic"
            joint_names.append(name)
        joint = ET.SubElement(root, "joint", name=name, type=kind)
        ET.SubElement(joint, "parent", link=names[model.body_parentid[i]])
        ET.SubElement(joint, "child", link=names[i])
        origin(joint, model.body_pos[i], model.body_quat[i])
        if moving:
            ET.SubElement(joint, "axis", xyz=numbers(model.jnt_axis[j]))
            ET.SubElement(joint, "limit", lower=str(model.jnt_range[j, 0]),
                          upper=str(model.jnt_range[j, 1]), effort="100", velocity="2.175")
            ET.SubElement(joint, "dynamics", damping=str(model.dof_damping[model.jnt_dofadr[j]]))
    expected = ["joint%d" % i for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
    if joint_names != expected:
        raise ValueError("External MJCF adapter supports the Franka Panda joint layout only")
    # A frame at the open left fingertip, independent of the held finger DOFs.
    tcp = ET.SubElement(root, "link", name="robogen_tcp")
    inertial = ET.SubElement(tcp, "inertial")
    ET.SubElement(inertial, "mass", value="0")
    ET.SubElement(inertial, "inertia", ixx="0", iyy="0", izz="0", ixy="0", ixz="0", iyz="0")
    joint = ET.SubElement(root, "joint", name="robogen_tcp_fixed", type="fixed")
    ET.SubElement(joint, "parent", link="hand")
    ET.SubElement(joint, "child", link="robogen_tcp")
    ET.SubElement(joint, "origin", xyz="0 0.04 0.1029", rpy="0 0 0")
    path = directory / "robot.urdf"
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    qpos = np.zeros(model.nq)
    qpos[-2:] = 0.04
    return {"urdf": str(path), "joint_names": expected,
            "initial_qpos": qpos.tolist(), "source_mjcf": str(source),
            "conversion_notes": ["MJCF actuators are replaced by RoboGen motor control",
                                 "Finger equality/tendon uses symmetric position holding",
                                 "Original suction constraint acts on the left fingertip"]}


def convert_mesh(source, scale, fixed, directory, file_meshes_are_zup=None):
    import pybullet as p
    scene = trimesh.load(source, force="scene", process=False)
    non_mesh = [name for name, geometry in scene.geometry.items()
                if not isinstance(geometry, trimesh.Trimesh)]
    if non_mesh:
        scene.delete_geometry(non_mesh)
        print("Skipping %d non-triangle geometries in %s" % (len(non_mesh), source), flush=True)
    if not scene.geometry:
        raise ValueError("Asset contains no triangle meshes: %s" % source)
    convert_up = (Path(source).suffix.lower() in (".glb", ".gltf") and file_meshes_are_zup is not True) or file_meshes_are_zup is False
    if convert_up:
        # Match Genesis GLTF loading: Y-up file coordinates become Z-up local coordinates.
        scene.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                         [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float))
    # Scene.export applies the GLB scene graph transforms before exporting OBJ.
    scene.apply_scale(scale)
    for mesh in scene.geometry.values():
        if isinstance(mesh.visual, trimesh.visual.ColorVisuals) and mesh.visual.defined:
            mesh.visual = mesh.visual.to_texture()
    visual = directory / "visual.obj"
    scene.export(visual)
    geometries = scene.dump(concatenate=True)
    plain = directory / "collision_source.obj"
    previous_hash = hashlib.sha256(plain.read_bytes()).hexdigest() if plain.exists() else None
    trimesh.Trimesh(geometries.vertices, geometries.faces, process=False).export(plain)
    if fixed:
        collisions = [str(plain)]
    elif geometries.is_convex:
        collisions = [str(plain)]
    else:
        vhacd = directory / "vhacd.obj"
        if not vhacd.exists() or previous_hash != hashlib.sha256(plain.read_bytes()).hexdigest():
            p.vhacd(str(plain), str(vhacd), str(directory / "vhacd.log"))
        hulls = trimesh.load(vhacd, force="scene", process=False, split_object=True, group_material=False)
        collisions = []
        for i, mesh in enumerate(hulls.geometry.values()):
            path = directory / ("hull_%d.obj" % i)
            mesh.export(path)
            collisions.append(str(path))
        if not collisions:
            raise ValueError("VHACD produced no collision hulls for %s" % source)
    # Bullet's shape-array API silently truncates at 16. URDF accepts all hulls.
    root = ET.Element("robot", name="external_object")
    link = ET.SubElement(root, "link", name="base")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="0.1")
    ET.SubElement(inertial, "inertia", ixx="1", iyy="1", izz="1", ixy="0", ixz="0", iyz="0")
    for tag, files in (("visual", [str(visual)]), ("collision", collisions)):
        for file in files:
            geometry = ET.SubElement(ET.SubElement(link, tag), "geometry")
            ET.SubElement(geometry, "mesh", filename=file)
    ET.indent(root, space="  ")
    urdf = directory / "object.urdf"
    ET.ElementTree(root).write(urdf, encoding="utf-8", xml_declaration=True)
    return {"urdf": str(urdf), "visual": str(visual), "collisions": collisions,
            "file_coordinate_conversion": "y_up_to_z_up" if convert_up else "none",
            "bounds_local": geometries.bounds.tolist(),
            "collision_method": "static_concave" if fixed else "convex" if geometries.is_convex else "vhacd"}


def prepare_scene(scene_path, output):
    scene_path, output = Path(scene_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    document = yaml.safe_load(scene_path.read_text())
    entities = document["scene"]["entities"]
    ids = [e["id"] for e in entities]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", i) for i in ids):
        raise ValueError("Entity IDs must be unique lowercase Python-style names")
    robots = [e for e in entities if e.get("robot")]
    if len(robots) != 1:
        raise ValueError("Exactly one robot entity is required")
    converted = []
    hashes = {str(scene_path): hashlib.sha256(scene_path.read_bytes()).hexdigest()}
    for entity in entities:
        item = dict(entity)
        item["quaternion_xyzw"] = xyzw(item.get("rotation", [1, 0, 0, 0]))
        position = np.asarray(item.get("position", [0, 0, 0]))
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Entity position must be a finite three-vector")
        directory = output / "assets" / item["id"]
        directory.mkdir(parents=True, exist_ok=True)
        kind = item["type"]
        if kind in ("mesh", "mjcf", "urdf"):
            source = (scene_path.parent / item["file"]).resolve()
            hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
            scale = item.get("scale", 1.0)
            if np.asarray(scale).shape not in ((), (3,)) or not np.isfinite(scale).all() or np.any(np.asarray(scale) <= 0):
                raise ValueError("Scale must be a positive scalar or three-vector")
            if item.get("robot"):
                if kind != "mjcf" or scale != 1:
                    raise ValueError("External robot conversion currently requires unscaled Franka MJCF")
                item["robot_asset"] = convert_mjcf(source, directory)
            elif kind == "mesh":
                item["mesh_asset"] = convert_mesh(source, scale, item.get("fixed", False), directory,
                                                   item.get("file_meshes_are_zup"))
            else:
                raise ValueError("Non-robot articulated assets need explicit joint-state adaptation")
        elif kind != "box":
            raise ValueError("Unsupported external entity type: %s" % kind)
        elif np.asarray(item.get("size", [])).shape != (3,) or not np.isfinite(item["size"]).all() or np.any(np.asarray(item["size"]) <= 0):
            raise ValueError("Box size must be a finite positive three-vector")
        if "surface_texture" in item:
            texture = (scene_path.parent / item["surface_texture"]).resolve()
            item["surface_texture"] = str(texture)
            hashes[str(texture)] = hashlib.sha256(texture.read_bytes()).hexdigest()
        converted.append(item)
        print("Converted", item["id"], flush=True)
    # Hash all robot mesh files as well as the MJCF document.
    robot_source = Path(converted[ids.index(robots[0]["id"])]["robot_asset"]["source_mjcf"])
    for path in sorted((robot_source.parent / "assets").glob("*")):
        if path.is_file():
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    converted_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((output / "assets").rglob("*"))
                        if path.is_file() and path.suffix in (".obj", ".urdf", ".png", ".mtl")}
    manifest = {"source_scene": str(scene_path), "source_hashes": hashes,
                "converted_hashes": converted_hashes,
                "entities": converted, "task": document.get("task", {}),
                "physics": {"dynamic_mass": 0.1, "lateral_friction": 1.0},
                "notes": ["No asset recentering, normalization or placement correction",
                          "Dynamic mass follows RoboGen obj_to_urdf default, not Genesis density",
                          "PyBullet rendering does not reproduce Genesis PBR materials"]}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path

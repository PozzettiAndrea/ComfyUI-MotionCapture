"""
GVHMR → Maya Export Script
===========================
Exports the current Blender scene to Alembic (.abc) and/or FBX (.fbx) for use in Maya.

This version is specifically designed to improve Maya FBX compatibility by:
- duplicating the export objects into a temporary collection,
- baking evaluated source armature pose onto the duplicate armature,
- exporting selection only,
- exporting only one baked action per armature.

Usage
-----
1. First run your Blender import script so the scene contains:
   - animated armature
   - skinned mesh
   - animated camera
2. Edit the CONFIGURE section below.
3. Run this script from Blender's Scripting workspace.
"""

import bpy
from pathlib import Path


# ── CONFIGURE ────────────────────────────────────────────────────────────────

OUTPUT_DIR = "/home/innovation/dev-lbringer/ComfyUI_nightly/custom_nodes/ComfyUI-MotionCapture/maya_files"
EXPORT_NAME = "karoshi_mocap"

EXPORT_ALEMBIC = True
EXPORT_FBX = True

# Set to 0 to keep current Blender scene render resolution
RESOLUTION_X = 0
RESOLUTION_Y = 0

TEMP_EXPORT_COLLECTION = "_TEMP_MAYA_EXPORT"

# ── END CONFIGURE ────────────────────────────────────────────────────────────


def print_header(msg):
    print(f"\n=== {msg} ===")


def remove_collection_and_contents(col):
    if col is None:
        return

    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.data.collections.remove(col)


def duplicate_for_export(objects, collection_name):
    old = bpy.data.collections.get(collection_name)
    if old is not None:
        remove_collection_and_contents(old)

    temp_col = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(temp_col)

    obj_map = {}

    # Duplicate objects and their data
    for obj in objects:
        obj_copy = obj.copy()
        if obj.data is not None:
            obj_copy.data = obj.data.copy()
        temp_col.objects.link(obj_copy)
        obj_map[obj] = obj_copy

    # Rebuild parenting if parent is also in the duplicated set
    for src, dst in obj_map.items():
        if src.parent in obj_map:
            dst.parent = obj_map[src.parent]
            dst.parent_type = src.parent_type
            dst.parent_bone = src.parent_bone
            dst.matrix_parent_inverse = src.matrix_parent_inverse.copy()

    # IMPORTANT: retarget mesh armature modifiers to duplicated armatures
    for src, dst in obj_map.items():
        if dst.type != 'MESH':
            continue

        for mod in dst.modifiers:
            if mod.type == 'ARMATURE' and getattr(mod, "object", None) is not None:
                src_target = mod.object
                if src_target in obj_map and obj_map[src_target].type == 'ARMATURE':
                    mod.object = obj_map[src_target]

        # Also fix direct armature parenting if needed
        if dst.parent and dst.parent.type == 'ARMATURE':
            # already remapped above if parent was in obj_map
            pass

    return temp_col, obj_map


def clear_animation_data_recursive(obj):
    if obj.animation_data:
        obj.animation_data_clear()

    if obj.data and hasattr(obj.data, "animation_data") and obj.data.animation_data:
        obj.data.animation_data_clear()


def find_export_objects():
    cameras = [o for o in bpy.data.objects if o.type == 'CAMERA' and o.visible_get()]
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE' and o.visible_get()]
    meshes = [o for o in bpy.data.objects if o.type == 'MESH' and o.visible_get()]
    return cameras, armatures, meshes


def ensure_animation_data(obj):
    if obj.animation_data is None:
        obj.animation_data_create()


def bake_armature_evaluated_pose(scene, src_arm_obj, dst_arm_obj, frame_start, frame_end):
    """
    Sample evaluated pose from the SOURCE armature, but write keys onto the
    DUPLICATE armature.
    """
    ensure_animation_data(dst_arm_obj)
    dst_arm_obj.animation_data.action = None

    baked_action = bpy.data.actions.new(name=f"{dst_arm_obj.name}_MAYA_BAKE")
    dst_arm_obj.animation_data.action = baked_action

    depsgraph = bpy.context.evaluated_depsgraph_get()
    baked = {}

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        src_eval = src_arm_obj.evaluated_get(depsgraph)
        baked[frame] = {}

        for src_pb in src_eval.pose.bones:
            mat_pose = src_pb.matrix.copy()

            parent = src_pb.parent
            if parent:
                mat_local_pose = parent.matrix.inverted() @ mat_pose
            else:
                mat_local_pose = mat_pose.copy()

            rest_local = src_pb.bone.matrix_local.copy()
            if parent:
                rest_parent = parent.bone.matrix_local.copy()
                rest_rel = rest_parent.inverted() @ rest_local
            else:
                rest_rel = rest_local

            mat_basis = rest_rel.inverted() @ mat_local_pose
            loc, rot_q, scale = mat_basis.decompose()
            baked[frame][src_pb.name] = (loc.copy(), rot_q.copy(), scale.copy())

    for frame, bone_data in baked.items():
        for bone_name, (loc, rot_q, scale) in bone_data.items():
            dst_pb = dst_arm_obj.pose.bones.get(bone_name)
            if dst_pb is None:
                continue

            dst_pb.rotation_mode = 'XYZ'
            dst_pb.location = loc
            dst_pb.rotation_euler = rot_q.to_euler('XYZ')
            dst_pb.scale = scale

            dst_pb.keyframe_insert(data_path='location', frame=frame)
            dst_pb.keyframe_insert(data_path='rotation_euler', frame=frame)
            dst_pb.keyframe_insert(data_path='scale', frame=frame)

    print(f"  Baked action '{baked_action.name}' with {len(baked_action.fcurves)} fcurves.")
    return baked_action


def copy_camera_animation(scene, src_cam, dst_cam, frame_start, frame_end):
    clear_animation_data_recursive(dst_cam)
    ensure_animation_data(dst_cam)

    cam_action = bpy.data.actions.new(name=f"{dst_cam.name}_MAYA_BAKE")
    dst_cam.animation_data.action = cam_action

    if dst_cam.data and hasattr(dst_cam.data, "animation_data_create"):
        dst_cam.data.animation_data_create()
        lens_action = bpy.data.actions.new(name=f"{dst_cam.data.name}_MAYA_BAKE")
        dst_cam.data.animation_data.action = lens_action

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        dst_cam.matrix_world = src_cam.matrix_world.copy()
        dst_cam.keyframe_insert(data_path='location', frame=frame)
        dst_cam.keyframe_insert(data_path='rotation_euler', frame=frame)
        dst_cam.keyframe_insert(data_path='scale', frame=frame)

        if src_cam.data and dst_cam.data:
            if hasattr(src_cam.data, "lens") and hasattr(dst_cam.data, "lens"):
                dst_cam.data.lens = src_cam.data.lens
                dst_cam.data.keyframe_insert(data_path='lens', frame=frame)

    print(f"  Baked duplicate camera '{dst_cam.name}'.")


def select_only(objects):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)


def main():
    print_header("GVHMR → Maya Export")

    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end = scene.frame_end

    print(f"  Scene frame range: {frame_start}–{frame_end}")

    if not EXPORT_ALEMBIC and not EXPORT_FBX:
        print("  Nothing to export. Set EXPORT_ALEMBIC and/or EXPORT_FBX to True.")
        return

    maya_dir = Path(OUTPUT_DIR)
    maya_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder: {maya_dir}")

    if RESOLUTION_X and RESOLUTION_Y:
        scene.render.resolution_x = RESOLUTION_X
        scene.render.resolution_y = RESOLUTION_Y
        print(f"  Resolution forced to: {RESOLUTION_X} × {RESOLUTION_Y}")

    cameras, armatures, meshes = find_export_objects()
    print(f"  Found {len(cameras)} camera(s), {len(armatures)} armature(s), {len(meshes)} mesh(es).")

    if not cameras and not armatures and not meshes:
        print("  No visible exportable objects found.")
        return

    export_objects = cameras + armatures + meshes

    print("\n  Building clean duplicate export set ...")
    temp_col, obj_map = duplicate_for_export(export_objects, TEMP_EXPORT_COLLECTION)

    dup_cameras = [obj_map[o] for o in cameras if o in obj_map]
    dup_armatures = [obj_map[o] for o in armatures if o in obj_map]
    dup_meshes = [obj_map[o] for o in meshes if o in obj_map]

    for obj in temp_col.objects:
        clear_animation_data_recursive(obj)

    baked_actions = []
    if dup_armatures:
        print("\n  Baking duplicate armature animation ...")
        for src_arm in armatures:
            dst_arm = obj_map[src_arm]
            print(f"  Armature: '{src_arm.name}' → '{dst_arm.name}'")
            baked_action = bake_armature_evaluated_pose(
                scene,
                src_arm,
                dst_arm,
                frame_start,
                frame_end,
            )
            baked_actions.append(baked_action)

    if cameras:
        print("\n  Baking duplicate camera animation ...")
        for src_cam in cameras:
            dst_cam = obj_map[src_cam]
            copy_camera_animation(scene, src_cam, dst_cam, frame_start, frame_end)

    scene.frame_set(frame_start)
    bpy.context.view_layer.update()

    if EXPORT_ALEMBIC:
        abc_path = maya_dir / f"{EXPORT_NAME}.abc"
        print(f"\n  Exporting Alembic → {abc_path}")

        select_only(list(temp_col.objects))
        if dup_armatures:
            bpy.context.view_layer.objects.active = dup_armatures[0]
        elif dup_cameras:
            bpy.context.view_layer.objects.active = dup_cameras[0]
        elif dup_meshes:
            bpy.context.view_layer.objects.active = dup_meshes[0]

        bpy.ops.wm.alembic_export(
            filepath=str(abc_path),
            start=frame_start,
            end=frame_end,
            xsamples=1,
            gsamples=1,
            sh_open=0.0,
            sh_close=1.0,
            selected=True,
            visible_objects_only=False,
            flatten=False,
            uvs=True,
            normals=True,
            vcolors=False,
            orcos=False,
            face_sets=False,
            subdiv_schema=False,
            apply_subdiv=False,
            curves_as_mesh=False,
            use_instancing=True,
            global_scale=1.0,
            triangulate=False,
            quad_method='SHORTEST_DIAGONAL',
            ngon_method='BEAUTY',
            export_hair=False,
            export_particles=False,
            export_custom_properties=True,
            as_background_job=False,
            init_scene_frame_range=False,
        )
        print(f"  Alembic written: {abc_path}")

    if EXPORT_FBX:
        fbx_path = maya_dir / f"{EXPORT_NAME}.fbx"
        print(f"\n  Exporting FBX → {fbx_path}")

        export_selection = list(temp_col.objects)
        select_only(export_selection)

        if dup_armatures:
            bpy.context.view_layer.objects.active = dup_armatures[0]
        elif dup_cameras:
            bpy.context.view_layer.objects.active = dup_cameras[0]
        elif dup_meshes:
            bpy.context.view_layer.objects.active = dup_meshes[0]

        bpy.ops.export_scene.fbx(
            filepath=str(fbx_path),
            use_selection=True,
            use_visible=False,

            global_scale=1.0,
            apply_unit_scale=True,
            apply_scale_options='FBX_SCALE_NONE',

            use_space_transform=True,
            bake_space_transform=False,

            object_types={'ARMATURE', 'CAMERA', 'MESH'},
            use_mesh_modifiers=True,
            use_armature_deform_only=True,
            add_leaf_bones=False,

            primary_bone_axis='Y',
            secondary_bone_axis='X',
            armature_nodetype='ROOT',

            bake_anim=True,
            bake_anim_use_all_bones=True,
            bake_anim_use_nla_strips=False,
            bake_anim_use_all_actions=False,
            bake_anim_force_startend_keying=True,
            bake_anim_step=1.0,
            bake_anim_simplify_factor=0.0,

            path_mode='AUTO',
            embed_textures=False,
            batch_mode='OFF',
            axis_forward='-Z',
            axis_up='Y',
        )

        print(f"  FBX written: {fbx_path}")

    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    fps = scene.render.fps / scene.render.fps_base

    print("\nDone!")
    print(f"  Render resolution : {res_x} × {res_y}")
    print(f"  Frame range       : {frame_start} – {frame_end} @ {fps:.6g} fps")
    print("  Maya import tips:")
    print("    • Use a new empty scene")
    print("    • File > Import > choose FBX")
    print("    • Preset: Autodesk Media and Entertainment")
    print("    • Make sure Animation / Deformed Models / Skins are enabled")
    print("    • Check whether joints have keys in Maya if animation still appears static")


if __name__ == "__main__":
    main()
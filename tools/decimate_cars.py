# Blender (bpy) script: lowers the triangle count of the hero car .glb files. Node names, mesh names and material names are KEPT
# (the game code finds doors / hoods / glass / paint by those names).  Usage: python decimate_cars.py <in.glb> <out.glb> [target_tris]
import bpy, sys, bmesh
src, dst = sys.argv[1], sys.argv[2]
target = int(sys.argv[3]) if len(sys.argv) > 3 else 45000
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
def tris(o): return sum(len(p.vertices) - 2 for p in o.data.polygons)
total = sum(tris(o) for o in meshes)
# small / glass-like meshes are left alone: only meshes above 1500 tris are reduced, by the ratio that reaches the target
big = sum(tris(o) for o in meshes if tris(o) > 500)
small = total - big
ratio = min(1.0, max(0.12, (target - small) / max(big, 1)))
print("total", total, "big", big, "ratio", round(ratio, 3))
for o in meshes:
    if tris(o) <= 500 or "glass" in o.name.lower() or "window" in o.name.lower():
        continue
    bpy.context.view_layer.objects.active = o
    for s in bpy.context.selected_objects: s.select_set(False)
    o.select_set(True)
    if o.data.users > 1: o.data = o.data.copy()
    m = o.modifiers.new("dec", 'DECIMATE')
    m.decimate_type = 'COLLAPSE'
    m.ratio = ratio
    m.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=m.name)
after = sum(tris(o) for o in meshes)
print("after", after)
bpy.ops.export_scene.gltf(filepath=dst, export_format='GLB', export_apply=False, export_yup=True,
                          export_image_format='AUTO', export_materials='EXPORT', export_cameras=False, export_lights=False)

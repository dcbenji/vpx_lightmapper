#    Copyright (C) 2022  Vincent Bousquet
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>

"""Rules for the joined bake of unwrapped objects (scene['vlm_joined_bake']).

A part may only share the joined bake object with other parts when its bake stays equivalent to its own per object bake:
- Object dependent shading is rewritten into per part constants in the part's material copies: Texture Coordinate Object
  (no object set) / Generated / UV / Normal, implicit UVs and implicit Generated coordinates, the default (radial) BSDF
  tangent and Geometry Tangent, the default color attribute. The maths follow Cycles, so the rewritten material renders
  the same on the part alone and inside a joined object.
- Shading that reads other geometry of the same object (local AO, bevel, subsurface, island random / pointiness), or that
  cannot be expressed as a constant (Object Info Random: the baked duplicate and the original seen as a neighbour have
  different names), makes the part bake alone, as do IES textures, object space displacement, tangent space vector
  displacement, the default color attribute of a mesh without any, shape keys, object linked or empty material slots,
  the part's own light linking, and Generated coordinates (or radial tangents) on an object with render modifiers.
- Other object level reads (Object Info, object attributes, box projection, object space normal maps, object vector
  transforms) are kept as they are: the part then leads its join group (it is the active object of the join, whose
  transform and properties the joined object keeps). A group has at most one such leader.
- Parts only join parts with the same object level Cycles settings (AO distance and shadow caustics included) and the same
  handedness (join does not flip the winding of a part mirrored relative to the active object).
- A transparent part never shares a join with a part whose padded object mask overlaps it: camera rays would see the other
  part through it (in a per object bake the other objects are only in the indirect-only collection).

Rules ported from dcbenji's batch-bake (see vbousquet/vpx_lightmapper#54 discussion). They are an allowlist: any other
object dependent or cross object shading is not detected.
"""

import numpy as np

# Rewrite results that make a part bake alone
BAKE_ALONE = {'objinfo-Random', 'ao-only-local', 'bevel', 'subsurface', 'geometry-island-pointiness', 'ies',
              'displacement-object', 'vector-displacement-tangent', 'color-attribute-default', 'shape-keys', 'object-linked-material',
              'no-material', 'own-light-linking', 'generated-with-modifiers'}
# BSDFs whose unconnected Tangent input defaults to the object radial tangent
TANGENT_BSDFS = {'BSDF_PRINCIPLED', 'BSDF_GLOSSY', 'BSDF_ANISOTROPIC', 'BSDF_METALLIC'}
# Texture nodes whose unconnected Vector input is the object Generated coordinates
GENERATED_TEXTURES = {'TEX_NOISE', 'TEX_VORONOI', 'TEX_WAVE', 'TEX_MUSGRAVE', 'TEX_GRADIENT', 'TEX_MAGIC', 'TEX_CHECKER', 'TEX_BRICK', 'TEX_GABOR'}
# Object level Cycles settings: a joined object has its active object's, so joined parts must share them (name, default)
OBJECT_SETTINGS = (('visible_camera', True), ('visible_diffuse', True), ('visible_glossy', True), ('visible_transmission', True),
                   ('visible_volume_scatter', True), ('visible_shadow', True), ('is_holdout', False), ('is_shadow_catcher', False),
                   ('shadow_terminator_shading_offset', 0.0), ('shadow_terminator_geometry_offset', 0.1))
CYCLES_OBJECT_SETTINGS = (('ao_distance', 0.0), ('is_caustics_caster', False), ('is_caustics_receiver', False)) # on obj.cycles (Cycles add-on properties)


def join_key(obj):
    """Parts only join parts with the same key: object level Cycles settings and handedness of the transform"""
    settings = tuple(getattr(obj, attr, default) for attr, default in OBJECT_SETTINGS)
    cycles = getattr(obj, 'cycles', None)
    settings += tuple(getattr(cycles, attr, default) for attr, default in CYCLES_OBJECT_SETTINGS)
    mirrored = bool(np.linalg.det(np.array(obj.matrix_world, dtype=np.float64)[:3, :3]) < 0)
    return settings, mirrored


def is_transparent(obj, dup, bake_col):
    """Non opaque bake collection (of the original), or a see-through material (of the modifier applied duplicate)"""
    for col in bake_col.children:
        if obj.name in col.all_objects and not col.vlmSettings.is_opaque:
            return True

    def tree_transparent(tree, seen):
        if tree is None or tree.name in seen:
            return False
        seen.add(tree.name)
        for n in tree.nodes:
            if n.type in ('BSDF_TRANSPARENT', 'HOLDOUT', 'BSDF_GLASS', 'BSDF_REFRACTION', 'BSDF_RAY_PORTAL'):
                return True
            if n.type == 'BSDF_PRINCIPLED':
                alpha = n.inputs.get('Alpha')
                if alpha is not None and (alpha.is_linked or alpha.default_value < 1.0):
                    return True
                transmission = n.inputs.get('Transmission Weight') or n.inputs.get('Transmission')
                if transmission is not None and (transmission.is_linked or transmission.default_value > 0.0):
                    return True
            if n.type == 'GROUP' and tree_transparent(n.node_tree, seen):
                return True
        return False

    def volume_only(tree): # active output with a volume and no surface: rays pass through
        return any(n.type == 'OUTPUT_MATERIAL' and n.is_active_output and n.inputs['Volume'].is_linked and not n.inputs['Surface'].is_linked
                   for n in tree.nodes)
    return any(slot.material is not None and slot.material.use_nodes and slot.material.node_tree is not None
               and (tree_transparent(slot.material.node_tree, set()) or volume_only(slot.material.node_tree)) for slot in dup.material_slots)


def part_alone_reason(obj, dup):
    """Reasons that make a part bake alone before looking at its materials (obj: the original, dup: its modifier applied duplicate)"""
    if not dup.data.materials or any(m is None for m in dup.data.materials): # nothing to rewrite, or a slot without material copy
        return 'no-material'
    if dup.data.shape_keys: # the rewrite constants use the mesh without its shape keys, the bake uses the deformed mesh
        return 'shape-keys'
    if any(slot.link == 'OBJECT' for slot in dup.material_slots): # object linked materials are not the part's material copies
        return 'object-linked-material'
    light_linking = getattr(obj, 'light_linking', None) # the part's own light linking would apply to the whole joined object
    if light_linking is not None and (light_linking.receiver_collection is not None or light_linking.blocker_collection is not None):
        return 'own-light-linking'
    return None


def generated_reason(obj, ctx):
    """After the rewrite: Generated coordinates (and radial tangents) of the part are taken from its modifier applied duplicate,
    while the original, seen as a neighbour of the other bakes, keeps its own; the part then bakes alone"""
    if ctx.get('generated') and any(m.show_render for m in obj.modifiers):
        return 'generated-with-modifiers'
    return None


def mask_silhouette(mask):
    """Padded object mask (w, h, alpha bytes) as a boolean array"""
    w, h, alpha = mask
    return np.frombuffer(alpha, np.uint8).reshape(h, w) > 0


def join_groups(parts):
    """Split joinable parts into join groups. Each part is a dict with 'key', 'leader' (keeps object dependent shading, must be
    the active object of its join), 'transparent' and 'silhouette'. Greedy: leaders first, each one opens its own group, then
    every other part goes to the first group with the same key and no transparency conflict. The leader of a group is its
    first part. Returns lists of parts (some may be single)."""
    def conflict(a, b):
        if not (a['transparent'] or b['transparent']):
            return False
        sa, sb = a['silhouette'], b['silhouette']
        return sa is None or sb is None or sa.shape != sb.shape or bool((sa & sb).any())
    groups = []
    for p in [p for p in parts if p['leader']] + [p for p in parts if not p['leader']]:
        for g in groups:
            if g[0]['key'] == p['key'] and not (p['leader'] and g[0]['leader']) and not any(conflict(p, q) for q in g):
                g.append(p)
                break
        else:
            groups.append([p])
    return groups


# ---- Material rewrite: object dependent inputs become per part constants ----

def texspace(mesh):
    """(location, size) of the mesh texture space as Cycles reads it"""
    if not getattr(mesh, 'use_auto_texspace', True):
        return np.array(mesh.texspace_location[:]), np.array(mesh.texspace_size[:])
    co = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype(np.float64)
    if not len(co):
        return np.zeros(3), np.ones(3)
    lo, hi = co.min(0), co.max(0)
    loc, size = (lo + hi) / 2, (hi - lo) / 2
    size = np.where(size == 0, 1.0, np.where(size < 1e-5, 1e-5, size))
    return loc, size


def part_context(dup):
    """Per part constants, from the modifier applied duplicate"""
    mw = np.array(dup.matrix_world, dtype=np.float64)
    inv = np.linalg.inv(mw)
    loc, size = texspace(dup.data)
    k = 1.0 / (2.0 * size)
    uv = next((l.name for l in dup.data.uv_layers if l.active_render), 'UVMap')
    colors = dup.data.color_attributes
    color = colors[colors.render_color_index].name if len(colors) and 0 <= colors.render_color_index < len(colors) else ''
    return dict(obj_rows=inv[:3, :3], obj_off=inv[:3, 3], gen_rows=inv[:3, :3] * k[:, None], gen_off=(inv[:3, 3] - (loc - size)) * k,
                uv=uv, normal_mat=inv[:3, :3].T, tfm_t=mw[:3, :3].T, color=color)


def _linear_out(nt, rows, off, source='Position'):
    """Node chain computing rows (3 x 3) @ world Position (or Normal) + off, returns its output socket"""
    geometry = nt.nodes.new('ShaderNodeNewGeometry')
    combine = nt.nodes.new('ShaderNodeCombineXYZ')
    for i in range(3):
        dot = nt.nodes.new('ShaderNodeVectorMath')
        dot.operation = 'DOT_PRODUCT'
        dot.inputs[1].default_value = [float(x) for x in rows[i]]
        nt.links.new(geometry.outputs[source], dot.inputs[0])
        nt.links.new(dot.outputs['Value'], combine.inputs[i])
    add = nt.nodes.new('ShaderNodeVectorMath')
    add.operation = 'ADD'
    add.inputs[1].default_value = [float(x) for x in off]
    nt.links.new(combine.outputs[0], add.inputs[0])
    return add.outputs['Vector']


def _object_normal(nt, ctx):
    """Cycles Texture Coordinate Normal: normalize(transposed object transform @ N)"""
    normalize = nt.nodes.new('ShaderNodeVectorMath')
    normalize.operation = 'NORMALIZE'
    nt.links.new(_linear_out(nt, ctx['tfm_t'], np.zeros(3), 'Normal'), normalize.inputs[0])
    return normalize.outputs['Vector']


def _radial_tangent(nt, ctx, axis='Z'):
    """Cycles radial tangent (Tangent node RADIAL, also the default of every BSDF Tangent input) of THIS part, in world space:
    T = cross(N, normalize(cross(Nt @ d(generated), N))), Nt being the object normal matrix"""
    ctx['generated'] = True
    G, g0 = ctx['gen_rows'], ctx['gen_off']
    z = np.zeros(3)
    if axis == 'X':
        A, b = np.array([z, -G[2], G[1]]), np.array([0.0, 0.5 - g0[2], g0[1] - 0.5])
    elif axis == 'Y':
        A, b = np.array([-G[2], z, G[0]]), np.array([0.5 - g0[2], 0.0, g0[0] - 0.5])
    else:
        A, b = np.array([-G[1], G[0], z]), np.array([0.5 - g0[1], g0[0] - 0.5, 0.0])
    Nt = ctx['normal_mat']
    d = _linear_out(nt, Nt @ A, Nt @ b)
    geometry = nt.nodes.new('ShaderNodeNewGeometry')
    c1 = nt.nodes.new('ShaderNodeVectorMath')
    c1.operation = 'CROSS_PRODUCT'
    nt.links.new(d, c1.inputs[0])
    nt.links.new(geometry.outputs['Normal'], c1.inputs[1])
    normalize = nt.nodes.new('ShaderNodeVectorMath')
    normalize.operation = 'NORMALIZE'
    nt.links.new(c1.outputs['Vector'], normalize.inputs[0])
    c2 = nt.nodes.new('ShaderNodeVectorMath')
    c2.operation = 'CROSS_PRODUCT'
    nt.links.new(geometry.outputs['Normal'], c2.inputs[0])
    nt.links.new(normalize.outputs['Vector'], c2.inputs[1])
    return c2.outputs['Vector']


def _relink(nt, socket, new_src):
    for link in list(socket.links):
        nt.links.new(new_src, link.to_socket)


def rewrite(nt, ctx, made_groups, memo):
    """In a part's material (or node group) copy, make object dependent inputs explicit constants of this part.
    Node groups are copied once per part (memo, new groups appended to made_groups). Returns the object dependent reads
    that were left as they are: any of BAKE_ALONE makes the part bake alone, anything else makes it lead its join group."""
    left = set()
    for n in list(nt.nodes):
        if n.mute:
            continue
        t = n.type
        tangent = n.inputs.get('Tangent') if t in TANGENT_BSDFS else None
        anisotropy = (n.inputs.get('Anisotropy') or n.inputs.get('Anisotropic')) if tangent is not None else None
        if anisotropy is not None and not anisotropy.is_linked and anisotropy.default_value == 0.0: # isotropic: the tangent is not read
            tangent = None
        if tangent is not None and not tangent.is_linked: # default tangent = radial Z of the object generated coordinates
            nt.links.new(_radial_tangent(nt, ctx), tangent)
        if t == 'GROUP' and n.node_tree is not None:
            key = n.node_tree.name
            if key not in memo:
                group = n.node_tree.copy()
                made_groups.append(group)
                memo[key] = group
                left |= rewrite(group, ctx, made_groups, memo)
            n.node_tree = memo[key]
        elif t == 'TEX_COORD':
            if n.from_instancer:
                left.add('texco-instancer')
                continue
            if n.object is None and n.outputs['Object'].is_linked: # with an object set, Object output does not depend on the part
                _relink(nt, n.outputs['Object'], _linear_out(nt, ctx['obj_rows'], ctx['obj_off']))
            if n.outputs['Generated'].is_linked:
                ctx['generated'] = True
                _relink(nt, n.outputs['Generated'], _linear_out(nt, ctx['gen_rows'], ctx['gen_off']))
            if n.outputs['UV'].is_linked:
                uv = nt.nodes.new('ShaderNodeUVMap')
                uv.uv_map = ctx['uv']
                _relink(nt, n.outputs['UV'], uv.outputs['UV'])
            if n.outputs['Normal'].is_linked:
                _relink(nt, n.outputs['Normal'], _object_normal(nt, ctx))
        elif t == 'TEX_IMAGE':
            if n.projection == 'BOX':
                left.add('image-box')
            if not n.inputs['Vector'].is_linked:
                uv = nt.nodes.new('ShaderNodeUVMap')
                uv.uv_map = ctx['uv']
                nt.links.new(uv.outputs['UV'], n.inputs['Vector'])
        elif t == 'UVMAP' and not n.uv_map:
            n.uv_map = ctx['uv']
        elif t == 'NORMAL_MAP':
            if n.space == 'TANGENT' and not n.uv_map:
                n.uv_map = ctx['uv']
            elif n.space in ('OBJECT', 'BLENDER_OBJECT'):
                left.add('normalmap-object')
        elif t == 'TANGENT' and n.direction_type == 'UV_MAP' and not n.uv_map:
            n.uv_map = ctx['uv']
        elif t == 'TANGENT' and n.direction_type == 'RADIAL' and n.outputs[0].is_linked:
            _relink(nt, n.outputs[0], _radial_tangent(nt, ctx, n.axis))
        elif t == 'VECT_TRANSFORM' and 'OBJECT' in (n.convert_from, n.convert_to):
            left.add('vector-transform-object')
        elif t == 'ATTRIBUTE' and n.attribute_type in ('OBJECT', 'INSTANCER'):
            left.add('attribute-object')
        elif t == 'OBJECT_INFO':
            for output in n.outputs:
                if output.is_linked:
                    left.add('objinfo-' + output.name)
        elif t in GENERATED_TEXTURES:
            vector = n.inputs.get('Vector')
            if vector is not None and not vector.is_linked: # implicit Generated coordinates
                ctx['generated'] = True
                nt.links.new(_linear_out(nt, ctx['gen_rows'], ctx['gen_off']), vector)
        elif t == 'AMBIENT_OCCLUSION' and n.only_local:
            left.add('ao-only-local')
        elif t == 'BEVEL':
            left.add('bevel')
        elif t == 'SUBSURFACE_SCATTERING':
            left.add('subsurface')
        elif t == 'BSDF_PRINCIPLED':
            weight = n.inputs.get('Subsurface Weight') or n.inputs.get('Subsurface')
            if weight is not None and (weight.is_linked or weight.default_value > 0.0):
                left.add('subsurface')
        elif t == 'NEW_GEOMETRY':
            if n.outputs['Tangent'].is_linked: # radial Z tangent of the object, as the default BSDF tangent
                _relink(nt, n.outputs['Tangent'], _radial_tangent(nt, ctx))
            if n.outputs['Random Per Island'].is_linked or n.outputs['Pointiness'].is_linked:
                left.add('geometry-island-pointiness')
        elif t == 'TEX_IES': # reads the normal in object space
            left.add('ies')
        elif t in ('DISPLACEMENT', 'VECTOR_DISPLACEMENT') and n.space == 'OBJECT':
            left.add('displacement-object')
        elif t == 'VECTOR_DISPLACEMENT' and n.space == 'TANGENT': # tangent from the joined mesh default UV map
            left.add('vector-displacement-tangent')
        elif t == 'VERTEX_COLOR' and not n.layer_name: # default color attribute of the joined mesh may not be this part's
            if ctx['color']:
                n.layer_name = ctx['color']
            else:
                left.add('color-attribute-default')
    return left

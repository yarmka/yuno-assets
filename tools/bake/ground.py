"""Shared ground masks for the bake scripts (copy of the setup in gen_buildings.py). Run scripts from the project root."""
import re, base64, math
import numpy as np
OX, OZ, C = -308.8, -179.6, 0.5
def read(p): return open(p, encoding="utf-8").read()
sm = read("world/SurfaceMapData.gd")
W = int(re.search(r"const W: int = (\d+)", sm).group(1)); H = int(re.search(r"const H: int = (\d+)", sm).group(1))
b64 = re.search(r'RLE_B64: String = "([^"]+)"', sm).group(1)
raw = base64.b64decode(b64 + "=" * (-len(b64) % 4))
cells = []
for i in range(0, len(raw) - 2, 3): cells += [raw[i]] * (raw[i + 1] | (raw[i + 2] << 8))
G = np.array(cells[: W * H], dtype=np.uint8).reshape(H, W)
def arr(src, name):
    m = re.search(r"const " + name + r": Array = \[(.*?)\]", src, re.S)
    return [float(x) for x in m.group(1).split(",") if x.strip()]
lay = read("world/CityLayout.gd")
TYPES = eval(re.search(r"const TYPES: Array = (\[\[.*?\]\])\n", lay, re.S).group(1).replace('"', "'"))
LAMPS = arr(lay, "LAMPS") + arr(lay, "EXTRA_LAMPS")
PALMS = arr(read("world/PalmLayout.gd"), "PALMS")
npc = read("world/npc/NpcPathData.gd")
i0 = npc.index("const ROUTES: Array = [") + len("const ROUTES: Array = ")
depth, j = 0, i0
while True:
    depth += npc[j] == "["; depth -= npc[j] == "]"; j += 1
    if depth == 0: break
ROUTES = eval(npc[i0:j].replace("\t", " "))
def cell(x, z): return int((x - OX) / C), int((z - OZ) / C)
def dilate(mask, r):
    out = mask.copy()
    for ax in (0, 1):
        acc = out.copy()
        for s in range(1, r + 1):
            sh = np.zeros_like(out)
            if ax == 0:
                sh[s:, :] = out[:-s, :]; acc |= sh; sh = np.zeros_like(out); sh[:-s, :] = out[s:, :]; acc |= sh
            else:
                sh[:, s:] = out[:, :-s]; acc |= sh; sh = np.zeros_like(out); sh[:, :-s] = out[:, s:]; acc |= sh
        out = acc
    return out
route_m = np.zeros((H, W), bool)
for r in ROUTES:
    pts = [(r[k], r[k + 1]) for k in range(1, len(r) - 1, 2)]
    if r[0] == 1: pts.append(pts[0])
    for (ax, az), (bx, bz) in zip(pts, pts[1:]):
        n = max(2, int(math.hypot(bx - ax, bz - az) / 0.25))
        for t in range(n + 1):
            cx, cz = cell(ax + (bx - ax) * t / n, az + (bz - az) * t / n)
            if 0 <= cx < W and 0 <= cz < H: route_m[cz, cx] = True
pole_m = np.zeros((H, W), bool)
for i in range(0, len(LAMPS), 4):
    cx, cz = cell(LAMPS[i + 1], LAMPS[i + 2]); pole_m[cz, cx] = True
for i in range(0, len(PALMS), 5):
    cx, cz = cell(PALMS[i + 1], PALMS[i + 2]); pole_m[cz, cx] = True
def rect_cells(cx, cz, w, d, yaw, pad):
    a = math.radians(yaw); c, s = math.cos(a), math.sin(a)
    ext = max(w, d) / 2 + pad + 1
    c0, r0 = cell(cx - ext, cz - ext); c1, r1 = cell(cx + ext, cz + ext)
    rr, cc = np.mgrid[max(r0, 0):min(r1 + 1, H), max(c0, 0):min(c1 + 1, W)]
    px = OX + (cc + .5) * C - cx; pz = OZ + (rr + .5) * C - cz
    m = (np.abs(px * c - pz * s) <= w / 2 + pad) & (np.abs(px * s + pz * c) <= d / 2 + pad)
    return rr[m], cc[m]
def paint(mask, cx, cz, w, d, yaw, pad=0.0):
    r, c = rect_cells(cx, cz, w, d, yaw, pad); mask[r, c] = True
def fits(mask, cx, cz, w, d, yaw):
    r, c = rect_cells(cx, cz, w, d, yaw, 0.0)
    return len(r) > 0 and bool(mask[r, c].all()) and r.min() > 0 and c.min() > 0 and r.max() < H - 1 and c.max() < W - 1
def yaw_facing(x, z, tx, tz): return math.degrees(math.atan2(-(tx - x), -(tz - z)))

import json
def landmark_rects():
    """[(x, z, yaw, w, d)] of the landmark buildings (world/LandmarkLayout.gd), empty before gen_landmarks.py ran."""
    try:
        lay = read("world/LandmarkLayout.gd")
    except FileNotFoundError:
        return []
    names = json.loads(re.search(r"const NAMES: Array = (\[.*?\])", lay).group(1))
    B = arr(lay, "BUILDINGS"); out = []
    for i in range(0, len(B), 5):
        dim = json.load(open("tools/bake/lm_dim_%s.json" % names[int(B[i])])); k = B[i + 4]
        out.append((B[i + 1], B[i + 2], B[i + 3], dim[0] * k, dim[1] * k))
    return out

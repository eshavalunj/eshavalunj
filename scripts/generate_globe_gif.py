#!/usr/bin/env python3
"""
ESHA_OS — Spinning Contribution Globe Generator
Custom numpy ray-caster renders a photorealistic lit sphere
with teal grid lines and monthly contribution spikes.
Outputs an animated GIF. No matplotlib needed.
"""

import os, json, math, urllib.request
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.environ["GH_TOKEN"]
GITHUB_USER  = os.environ.get("GH_USER", "YOUR_GITHUB_USERNAME")
OUTPUT_PATH  = os.environ.get("OUTPUT_PATH", "assets/contrib-globe.gif")

FRAMES      = 60       # rotation frames (full 360°)
FPS         = 20       # playback speed
SIZE        = 520      # canvas px (square)
R_PX        = 196      # globe radius in pixels
CX          = SIZE // 2
CY          = SIZE // 2 + 8
SPIKE_MAX   = 78       # max spike length px
TILT_DEG    = -18      # Earth-like axial tilt
LOOP        = 0        # 0 = loop forever

# ESHA_OS palette
C_BG           = (4,   15,  15)
C_GLOBE_DARK   = (8,   35,  32)
C_GLOBE_MID    = (12,  55,  50)
C_GLOBE_LIGHT  = (18,  82,  72)
C_GRID         = (0,   140, 115)
C_SPIKE_BRIGHT = (77,  255, 212)   # #4dffd4
C_SPIKE_MID    = (0,   229, 204)   # #00e5cc
C_SPIKE_DIM    = (0,   155, 130)   # dimmer teal
C_TEXT         = (179, 255, 232)
C_DIM          = (60,  120, 110)
C_ACCENT       = (0,   229, 204)

MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN",
          "JUL","AUG","SEP","OCT","NOV","DEC"]

# ── GitHub GraphQL ─────────────────────────────────────────────────────────────

def fetch_contributions():
    now   = datetime.now(timezone.utc)
    start = now.replace(year=now.year-1, month=now.month, day=1,
                        hour=0, minute=0, second=0, microsecond=0)
    query = """
    query($login:String!,$from:DateTime!,$to:DateTime!){
      user(login:$login){
        contributionsCollection(from:$from,to:$to){
          contributionCalendar{
            totalContributions
            weeks{ contributionDays{ date contributionCount } }
          }
        }
      }
    }"""
    payload = json.dumps({"query": query, "variables": {
        "login": GITHUB_USER, "from": start.isoformat(), "to": now.isoformat()
    }}).encode()
    req = urllib.request.Request(
        "https://api.github.com/graphql", data=payload,
        headers={"Authorization": f"bearer {GITHUB_TOKEN}",
                 "Content-Type": "application/json",
                 "User-Agent": "ESHA_OS-globe"},
        method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw  = resp.read()
        data = json.loads(raw)

    print("[ESHA_OS] GitHub API response (first 800 chars):")
    print(json.dumps(data, indent=2)[:800])

    if "errors" in data:
        for e in data["errors"]:
            print(f"[ESHA_OS] GraphQL error: {e.get('message')}")
        raise RuntimeError("GraphQL errors — check token scopes")

    if data.get("data") is None or data["data"].get("user") is None:
        print(f"[ESHA_OS] user is null. GH_USER='{GITHUB_USER}'")
        raise RuntimeError(
            f"GitHub returned null for user '{GITHUB_USER}'. "
            "Ensure GH_TOKEN secret has scopes: read:user + repo"
        )

    cal   = data["data"]["user"]["contributionsCollection"]["contributionCalendar"]
    total = cal["totalContributions"]
    by_month = defaultdict(int)
    for week in cal["weeks"]:
        for day in week["contributionDays"]:
            dt = datetime.strptime(day["date"], "%Y-%m-%d")
            by_month[(dt.year, dt.month)] += day["contributionCount"]
    return by_month, total


# ── Rotation matrices ─────────────────────────────────────────────────────────

def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def sphere_pt(lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(lat)*math.cos(lon),
                     math.sin(lat),
                     math.cos(lat)*math.sin(lon)], dtype=np.float64)

def project(p3, r=R_PX):
    """Orthographic projection."""
    return int(CX + p3[0]*r), int(CY - p3[1]*r)


# ── Ray-cast sphere ────────────────────────────────────────────────────────────

def render_sphere(Rtotal):
    """
    Returns an RGB numpy array (SIZE×SIZE×3) with the shaded sphere.
    Rtotal: combined rotation matrix (3×3) mapping object→camera space.
    """
    arr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    arr[:, :] = C_BG

    # Camera-space light direction (upper-left-front)
    light = np.array([0.40, 0.55, 0.75], dtype=np.float64)
    light /= np.linalg.norm(light)

    xs = (np.arange(SIZE) - CX) / R_PX          # shape (W,)
    ys = (CY - np.arange(SIZE)) / R_PX           # shape (H,)
    XX, YY = np.meshgrid(xs, ys)                 # (H, W)
    D2 = XX**2 + YY**2
    mask = D2 <= 1.0                             # pixels on sphere

    ZZ = np.where(mask, np.sqrt(np.clip(1.0 - D2, 0, 1)), 0)

    # Camera-space normals
    Ncam = np.stack([XX, YY, ZZ], axis=-1)       # (H, W, 3)

    # Object-space normals (for lat-based tinting)
    Nobj = Ncam @ Rtotal.T                       # Rtotal.T = inverse rotation

    lat_abs = np.abs(np.arcsin(np.clip(Nobj[..., 1], -1, 1))) / (math.pi/2)

    # Lighting
    diffuse = np.clip(np.einsum('hwc,c->hw', Ncam, light), 0, 1)
    spec    = np.clip(np.einsum('hwc,c->hw', Ncam, light), 0, 1) ** 36 * 0.40
    ambient = 0.20
    intensity = (ambient + diffuse * 0.72 + spec)[..., np.newaxis]  # (H,W,1)

    # Base globe color from latitude
    dark  = np.array(C_GLOBE_DARK,  dtype=np.float64)
    light_ = np.array(C_GLOBE_LIGHT, dtype=np.float64)
    base  = dark + (light_ - dark) * lat_abs[..., np.newaxis]

    lit   = np.clip(base * intensity, 0, 255).astype(np.uint8)

    arr[mask] = lit[mask]
    return arr


# ── Grid lines ────────────────────────────────────────────────────────────────

def draw_grid(draw, Rtotal):
    n = 140
    def line3d(pts3, alpha, lw=1):
        pts2 = []
        vis  = []
        for p in pts3:
            pr = Rtotal @ p
            pts2.append(project(pr))
            vis.append(pr[2] > -0.12)
        col = tuple(int(c * alpha) for c in C_GRID)
        for i in range(len(pts2)-1):
            if vis[i] and vis[i+1]:
                draw.line([pts2[i], pts2[i+1]], fill=col, width=lw)

    for lat in [-60, -30, 0, 30, 60]:
        pts = [sphere_pt(lat, lon) for lon in np.linspace(0, 360, n)]
        lw  = 2 if lat == 0 else 1
        alp = 0.80 if lat == 0 else 0.50
        line3d(pts, alp, lw)

    for lon in range(0, 360, 30):
        pts = [sphere_pt(lat, lon) for lat in np.linspace(-82, 82, n)]
        line3d(pts, 0.38, 1)


# ── Spikes ────────────────────────────────────────────────────────────────────

def draw_spikes(draw, Rtotal, layout, peak):
    """
    layout: list of (lat, lon, count, label)
    Draws glowing contribution spikes sorted back-to-front.
    """
    # Compute depth of each spike base for painter's sort
    decorated = []
    for lat, lon, count, label in layout:
        base3 = sphere_pt(lat, lon)
        br    = Rtotal @ base3
        norm  = count / peak if peak > 0 else 0
        h_frac = 0.05 + norm * (SPIKE_MAX / R_PX)
        tip3  = sphere_pt(lat, lon) * (1.0 + h_frac)
        tr    = Rtotal @ tip3
        decorated.append((br[2], br, tr, norm, count, label))

    decorated.sort(key=lambda x: x[0])   # paint far spikes first

    for depth, br, tr, norm, count, label in decorated:
        bp = project(br)
        tp = project(tr)

        # Skip fully hidden spikes
        if br[2] < -0.35:
            continue

        facing = br[2]
        if facing < 0:
            alpha_scale = max(0, 1.0 + facing * 2)   # fade as it goes around edge
        else:
            alpha_scale = 1.0

        # Choose spike color by intensity
        if norm > 0.72:
            col = C_SPIKE_BRIGHT
        elif norm > 0.38:
            col = C_SPIKE_MID
        else:
            col = C_SPIKE_DIM

        # Glow halo (wide transparent line)
        g_alpha = int(55 * alpha_scale + 90 * norm * alpha_scale)
        draw.line([bp, tp], fill=(*col, g_alpha), width=7)

        # Core spike
        draw.line([bp, tp], fill=(*col, int(220 * alpha_scale)), width=2)

        # Tip dot
        dr = int(3 + 6 * norm)
        draw.ellipse([tp[0]-dr, tp[1]-dr, tp[0]+dr, tp[1]+dr],
                     fill=(*col, int(245 * alpha_scale)))
        # Outer glow on tip
        gr = int(dr * 2.8)
        draw.ellipse([tp[0]-gr, tp[1]-gr, tp[0]+gr, tp[1]+gr],
                     fill=(*col, int(38 * alpha_scale)))

        # Label — only for front-facing spikes
        if facing > 0.08:
            lx = tp[0] + (10 if tp[0] > CX else -10)
            ly = tp[1] - 14
            a  = int(200 * alpha_scale)
            # Drop shadow
            draw.text((lx+1, ly+1), f"{label}\n{count}",
                      fill=(0, 0, 0, int(a*0.6)), anchor="mm")
            draw.text((lx, ly), f"{label}\n{count}",
                      fill=(*C_TEXT, a), anchor="mm")


# ── HUD ───────────────────────────────────────────────────────────────────────

def draw_hud(draw, total, now, frame_idx):
    # Top title
    draw.text((SIZE//2, 20), "ESHA_OS  —  CONTRIBUTION GLOBE",
              fill=(*C_TEXT, 200), anchor="mm")

    # Bottom info
    draw.text((SIZE//2, SIZE-16),
              f"@{GITHUB_USER}  ·  {total} commits  ·  {now.strftime('%Y-%m-%d')}",
              fill=(*C_DIM, 170), anchor="mm")

    # Progress dots
    dot_total = min(FRAMES, 30)
    dot_idx   = int(frame_idx / FRAMES * dot_total)
    dot_start = SIZE//2 - dot_total*5//2
    for i in range(dot_total):
        dx = dot_start + i * 5
        dy = SIZE - 30
        if i == dot_idx:
            draw.ellipse([dx-2, dy-2, dx+2, dy+2], fill=(*C_ACCENT, 230))
        else:
            draw.ellipse([dx-1, dy-1, dx+1, dy+1], fill=(*C_DIM, 80))


# ── Build spike layout ────────────────────────────────────────────────────────

def build_layout(by_month, now):
    months = []
    for i in range(11, -1, -1):
        m = now.month - i; y = now.year
        while m <= 0: m += 12; y -= 1
        months.append((y, m))

    # Stagger latitudes so they feel natural, not a flat ring
    lats = [20, -18, 32, -28, 12, -10, 26, -22, 38, -14, 16, -32]
    layout = []
    for idx, (yr, mo) in enumerate(months):
        count = by_month.get((yr, mo), 0)
        lon   = (idx * 30 + 15) % 360
        lat   = lats[idx]
        layout.append((lat, lon, count, MONTHS[mo-1]))
    return layout


# ── Render one frame ──────────────────────────────────────────────────────────

def make_frame(layout, peak, total, azim_rad, frame_idx, now):
    Ry     = rot_y(azim_rad)
    Rx     = rot_x(math.radians(TILT_DEG))
    Rtotal = Rx @ Ry

    # Sphere via numpy ray-cast
    sphere_arr = render_sphere(Rtotal)
    img  = Image.fromarray(sphere_arr, "RGB").convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    # Grid + spikes + HUD
    draw_grid(draw, Rtotal)
    draw_spikes(draw, Rtotal, layout, peak)
    draw_hud(draw, total, now, frame_idx)

    return img.convert("RGB")


# ── GIF assembly ──────────────────────────────────────────────────────────────

def render_globe_gif(by_month, total):
    now    = datetime.now(timezone.utc)
    layout = build_layout(by_month, now)
    peak   = max(c for _, _, c, _ in layout) or 1

    frames_out = []
    for f in range(FRAMES):
        azim = (f / FRAMES) * 2 * math.pi
        frame = make_frame(layout, peak, total, azim, f, now)
        frames_out.append(
            frame.convert("P", palette=Image.ADAPTIVE, colors=240)
        )
        if (f+1) % 10 == 0:
            print(f"  [GLOBE] {f+1}/{FRAMES} frames ...")

    return frames_out


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[ESHA_OS] Fetching contributions for @{GITHUB_USER} ...")
    by_month, total = fetch_contributions()
    print(f"[ESHA_OS] {total} total commits — rendering {FRAMES} frames ...")

    frames = render_globe_gif(by_month, total)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    delay = 1000 // FPS
    frames[0].save(
        OUTPUT_PATH, format="GIF", save_all=True,
        append_images=frames[1:],
        duration=delay, loop=LOOP, optimize=True,
    )
    print(f"[ESHA_OS] Saved → {OUTPUT_PATH}  ({os.path.getsize(OUTPUT_PATH)//1024} KB)")

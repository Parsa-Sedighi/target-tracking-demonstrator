"""Closed-loop simulator: a virtual lab room seen through a camera on a pan/tilt gimbal.

The rendered image depends on the *true* gimbal angles at render time, so the
whole loop (detect -> track -> predict -> command -> gimbal moves -> image moves)
is exercised exactly as it will be on the real hardware. Target set and motions
follow the SDR (Table 1 / ES-02.x / ES-03.x):

* T1 small quadcopter drone (hover then lateral flight up to 3 m/s)
* T2 printed fiducial / index card carried at <= 1 m/s (has text for OCR)
* T3 falling tennis ball dropped from rest over 1.5 m
* distractor "hand" (for the drone-vs-hand discrimination test, ES-01.4)
* optional pillar occluder (reacquisition test, ES-03.1)

World frame: x forward from the gimbal, y right, z up, metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from ..perception.geometry import Intrinsics, azel_to_dir, world_to_pixel

FT = 0.3048
G = 9.81


# --------------------------------------------------------------------------- sprites
def _sprite_drone(px: int = 240) -> np.ndarray:
    """RGBA quadcopter seen roughly from the front/below."""
    s = np.zeros((px, px, 4), np.uint8)
    c = px // 2
    arm = int(px * 0.36)
    rotor_r = int(px * 0.13)
    body = (45, 45, 50, 255)
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        x2, y2 = c + dx * arm, c + int(dy * arm * 0.45)
        cv2.line(s, (c, c), (x2, y2), body, max(3, px // 22), cv2.LINE_AA)
        cv2.ellipse(s, (x2, y2 - rotor_r // 3), (rotor_r, rotor_r // 3), 0, 0, 360, (150, 150, 155, 230), -1, cv2.LINE_AA)
        cv2.circle(s, (x2, y2), max(3, px // 30), (30, 30, 30, 255), -1, cv2.LINE_AA)
    cv2.rectangle(s, (c - px // 7, c - px // 12), (c + px // 7, c + px // 12), body, -1, cv2.LINE_AA)
    cv2.rectangle(s, (c - px // 10, c + px // 12), (c + px // 10, c + px // 7), (60, 60, 70, 255), -1, cv2.LINE_AA)
    cv2.circle(s, (c, c + px // 9), px // 28, (20, 20, 20, 255), -1, cv2.LINE_AA)  # camera
    cv2.circle(s, (c - px // 9, c), px // 40, (40, 40, 240, 255), -1, cv2.LINE_AA)  # red LED
    cv2.circle(s, (c + px // 9, c), px // 40, (60, 230, 60, 255), -1, cv2.LINE_AA)  # green LED
    return s


def _sprite_ball(px: int = 160) -> np.ndarray:
    s = np.zeros((px, px, 4), np.uint8)
    c, r = px // 2, int(px * 0.47)
    cv2.circle(s, (c, c), r, (45, 225, 205, 255), -1, cv2.LINE_AA)  # yellow-green felt (BGR)
    # simple shading
    shade = np.zeros((px, px), np.float32)
    yy, xx = np.mgrid[0:px, 0:px]
    shade = 1.0 - 0.35 * np.clip(((xx - c * 0.8) ** 2 + (yy - c * 0.8) ** 2) / (r * r * 1.6), 0, 1)
    s[..., :3] = (s[..., :3] * shade[..., None]).astype(np.uint8)
    cv2.ellipse(s, (c - r, c), (int(r * 0.75), int(r * 0.95)), 0, -70, 70, (235, 245, 245, 255), max(2, px // 26), cv2.LINE_AA)
    cv2.ellipse(s, (c + r, c), (int(r * 0.75), int(r * 0.95)), 0, 110, 250, (235, 245, 245, 255), max(2, px // 26), cv2.LINE_AA)
    return s


def _sprite_card(text: str = "AIM 23", px: int = 360) -> np.ndarray:
    """6x6 in white index card: thick border, fiducial blocks, large OCR text."""
    s = np.full((px, px, 4), 255, np.uint8)
    b = px // 22
    cv2.rectangle(s, (0, 0), (px - 1, px - 1), (0, 0, 0, 255), b)
    q = px // 6
    for (x, y) in ((b * 2, b * 2), (px - b * 2 - q, b * 2)):
        cv2.rectangle(s, (x, y), (x + q, y + q), (0, 0, 0, 255), -1)
        cv2.rectangle(s, (x + q // 4, y + q // 4), (x + 3 * q // 4, y + 3 * q // 4), (255, 255, 255, 255), -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    lines = text.split("\n") if "\n" in text else [text]
    scale = 1.0
    for _ in range(40):
        widths = [cv2.getTextSize(ln, font, scale, 2)[0][0] for ln in lines]
        if max(widths) > px * 0.82:
            break
        scale += 0.1
    th = max(2, int(scale * 2.2))
    y = int(px * 0.62) if len(lines) == 1 else int(px * 0.52)
    for ln in lines:
        (w, h), _ = cv2.getTextSize(ln, font, scale, th)
        cv2.putText(s, ln, ((px - w) // 2, y + h // 2), font, scale, (0, 0, 0, 255), th, cv2.LINE_AA)
        y += int(h * 1.6)
    return s


def _sprite_hand(px: int = 240) -> np.ndarray:
    s = np.zeros((px, px, 4), np.uint8)
    skin = (120, 160, 215, 255)
    c = px // 2
    cv2.ellipse(s, (c, int(px * 0.62)), (int(px * 0.24), int(px * 0.27)), 0, 0, 360, skin, -1, cv2.LINE_AA)
    for i, ang in enumerate((-28, -10, 6, 22)):
        L = int(px * (0.30 if i in (1, 2) else 0.26))
        a = math.radians(ang - 90)
        x0, y0 = c + int((i - 1.5) * px * 0.11), int(px * 0.45)
        x1, y1 = int(x0 + L * math.cos(a)), int(y0 + L * math.sin(a))
        cv2.line(s, (x0, y0), (x1, y1), skin, int(px * 0.085), cv2.LINE_AA)
    cv2.line(s, (c - int(px * 0.2), int(px * 0.62)), (c - int(px * 0.40), int(px * 0.42)), skin, int(px * 0.09), cv2.LINE_AA)
    cv2.rectangle(s, (c - int(px * 0.15), int(px * 0.80)), (c + int(px * 0.15), px - 1), (70, 60, 140, 255), -1)  # sleeve
    return s


# --------------------------------------------------------------------------- targets
@dataclass
class SimTarget:
    tid: int
    cls: str                       # class reported by the sim "truth" detector
    size_m: float                  # characteristic size (span) used for drawing & range
    path: Callable[[float], Optional[np.ndarray]]
    sprite: np.ndarray
    aspect: float = 1.0            # bbox height / width
    label: str = ""

    def position(self, t: float) -> Optional[np.ndarray]:
        return self.path(t)


@dataclass
class Occluder:
    """Vertical pillar at (x, y) with radius r from z0 to z1 (metres)."""
    x: float
    y: float
    r: float
    z0: float = -1.0
    z1: float = 1.6
    color: tuple = (70, 85, 105)


def _drone_lateral(dist: float, amp: float, peak_speed: float, hover: float, height: float = 0.15):
    w = peak_speed / max(amp, 1e-6)

    def p(t: float):
        if t < hover:
            return np.array([dist, 0.0, height + 0.02 * math.sin(4 * t)])
        tt = t - hover
        return np.array([dist + 0.1 * math.sin(0.4 * tt), amp * math.sin(w * tt), height + 0.05 * math.sin(1.3 * tt)])

    return p


def _drone_lissajous(dist: float, speed: float = 1.0):
    def p(t: float):
        k = speed
        return np.array([dist + 0.35 * math.sin(0.45 * k * t),
                         1.0 * math.sin(0.85 * k * t),
                         0.25 + 0.30 * math.sin(1.55 * k * t + 0.5)])

    return p


def _ball_drop(dist: float, drop_m: float, hold_s: float, period_s: float, y: float = 0.0, t0: float = 0.5):
    z_top = drop_m / 2.0
    t_fall = math.sqrt(2 * drop_m / G)

    def p(t: float):
        if t < t0:
            return None
        tau = (t - t0) % period_s
        if tau < hold_s:
            return np.array([dist, y, z_top])
        tau -= hold_s
        if tau < t_fall:
            return np.array([dist, y, z_top - 0.5 * G * tau * tau])
        return None  # caught / out of the scene until the next drop

    return p


def _card_walk(dist: float, speed: float):
    amp = 1.0
    w = speed / amp

    def p(t: float):
        return np.array([dist, amp * math.sin(w * t), 0.15 * math.sin(0.5 * w * t)])

    return p


def _hand_path(dist: float, phase: float = 0.0, amp: float = 0.6, mirror: float = 1.0, speed: float = 1.0):
    def p(t: float):
        k = speed
        return np.array([dist + 0.15 * math.sin(0.7 * k * t + phase),
                         mirror * amp * math.sin(0.8 * k * t + phase),
                         0.05 + 0.18 * math.sin(1.1 * k * t + 2 * phase)])

    return p


def build_scenario(name: str, card_text: str = "AIM 23"):
    drone, ball, card, hand = _sprite_drone(), _sprite_ball(), _sprite_card(card_text), _sprite_hand()
    occ: list[Occluder] = []
    if name == "drone_lateral":      # ES-02.2 design case: lateral up to 3 m/s at >= 5 ft
        tg = [SimTarget(1, "drone", 0.30, _drone_lateral(5.5 * FT, 1.0, 3.0, 2.0), drone, 0.62, "T1 drone")]
    elif name == "ball_drop":        # ES-02.2: free fall 1.5 m at >= 10 ft
        tg = [SimTarget(3, "sports ball", 0.067, _ball_drop(10 * FT, 1.5, 1.0, 8.0), ball, 1.0, "T3 ball")]
    elif name == "card_walk":        # ES-02.1: T2 at <= 1 m/s, 10 ft
        tg = [SimTarget(2, "card", 0.152, _card_walk(10 * FT, 1.0), card, 1.0, "T2 card")]
    elif name == "crossing":         # ES-01.4: two targets cross, keep the selected one
        tg = [SimTarget(1, "drone", 0.30, _hand_path(8 * FT, 0.0, 1.0, 1.0, 1.0), drone, 0.62, "T1 drone"),
              SimTarget(4, "hand", 0.20, _hand_path(6 * FT, 0.0, 1.0, -1.0, 1.0), hand, 1.0, "hand")]
    elif name == "occlusion":        # ES-03.1: target blocked 1-3 s mid-track
        tg = [SimTarget(1, "drone", 0.30, _drone_lateral(9 * FT, 1.3, 0.9, 1.5), drone, 0.62, "T1 drone")]
        occ = [Occluder(6 * FT, 0.0, 0.17)]
    else:                            # "mixed": everything at once (default demo)
        tg = [SimTarget(1, "drone", 0.30, _drone_lissajous(9 * FT, 0.8), drone, 0.62, "T1 drone"),
              SimTarget(2, "card", 0.152, _card_walk(6 * FT, 0.6), card, 1.0, "T2 card"),
              SimTarget(3, "sports ball", 0.067, _ball_drop(10 * FT, 1.5, 1.0, 7.0, y=-0.9), ball, 1.0, "T3 ball"),
              SimTarget(4, "hand", 0.20, _hand_path(5 * FT, 1.0, 0.7, 1.0, 0.9), hand, 1.0, "hand")]
        occ = [Occluder(7 * FT, 1.25, 0.15)]
    return tg, occ


SCENARIOS = ["mixed", "drone_lateral", "ball_drop", "card_walk", "crossing", "occlusion"]


# --------------------------------------------------------------------------- world
class SimWorld:
    """Renders camera frames for given gimbal angles and answers LiDAR range queries."""

    PANO_PPD = 8.0          # background panorama pixels per degree
    EL_MIN, EL_MAX = -80.0, 115.0

    def __init__(self, intr: Intrinsics, scenario: str = "mixed", seed: int = 7, noise_sigma: float = 3.0,
                 card_text: str = "AIM 23", room: Optional[dict] = None):
        self.intr = intr
        self.rng = np.random.default_rng(seed)
        self.noise_sigma = noise_sigma
        self.card_text = card_text
        r = room or {}
        # Box room around the gimbal (20 x 20 ft lab, gimbal near the back wall).
        self.room = {"front": r.get("front", 20 * FT), "back": r.get("back", -1.0),
                     "left": r.get("left", -10 * FT), "right": r.get("right", 10 * FT),
                     "floor": r.get("floor", -1.0), "ceiling": r.get("ceiling", 1.7)}
        self._pano = self._make_panorama(seed)
        self._noise = self._make_noise_bank()
        self._noise_i = 0
        self.set_scenario(scenario)

    # ---- scenario
    def set_scenario(self, name: str) -> None:
        self.scenario = name if name in SCENARIOS else "mixed"
        self.targets, self.occluders = build_scenario(self.scenario, self.card_text)
        self.t_origin: Optional[float] = None

    def scene_time(self, t: float) -> float:
        if self.t_origin is None:
            self.t_origin = t
        return t - self.t_origin

    # ---- background
    def _make_panorama(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        ppd = self.PANO_PPD
        W = int(360 * ppd)
        H = int((self.EL_MAX - self.EL_MIN) * ppd)
        pano = np.zeros((H, W, 3), np.uint8)

        def row(el):
            return int((self.EL_MAX - el) * ppd)

        horizon_floor = row(-14)
        ceiling_row = row(32)
        # walls: soft colour drift with azimuth
        az = np.linspace(0, 2 * np.pi, W, endpoint=False)
        base = np.stack([150 + 18 * np.sin(az * 2 + 1), 160 + 14 * np.sin(az * 3), 168 + 10 * np.cos(az)], -1)
        pano[:, :, :] = base[None, :, :].astype(np.uint8)
        pano[:ceiling_row] = (205, 208, 210)
        for x in range(0, W, int(12 * ppd)):  # ceiling tiles / lights
            cv2.line(pano, (x, 0), (x, ceiling_row), (185, 188, 190), 2)
            cv2.rectangle(pano, (x + 20, ceiling_row - 90), (x + 70, ceiling_row - 40), (245, 248, 250), -1)
        pano[horizon_floor:] = (88, 98, 110)
        for x in range(0, W, int(6 * ppd)):   # floor tile seams
            cv2.line(pano, (x, horizon_floor), (x + rng.integers(-40, 40), H), (75, 84, 95), 2)
        # clutter: shelves, posters, doors, monitors, cables, boxes
        for _ in range(140):
            x = int(rng.integers(0, W))
            kind = rng.integers(0, 6)
            if kind == 0:   # shelf with boxes
                w, h = int(rng.integers(60, 200)), int(rng.integers(80, 220))
                y = horizon_floor - h
                cv2.rectangle(pano, (x, y), (x + w, horizon_floor), (60, 75, 95), 3)
                for k in range(4):
                    yy = y + k * h // 4
                    cv2.line(pano, (x, yy), (x + w, yy), (60, 75, 95), 3)
                    bx = x + 4
                    while bx < x + w - 10:
                        bw = int(rng.integers(10, 40))
                        col = tuple(int(c) for c in rng.integers(40, 220, 3))
                        cv2.rectangle(pano, (bx, yy + 4), (min(bx + bw, x + w - 4), yy + h // 4 - 3), col, -1)
                        bx += bw + 3
            elif kind == 1:  # poster
                w, h = int(rng.integers(30, 90)), int(rng.integers(40, 110))
                y = int(rng.integers(row(28), row(0)))
                col = tuple(int(c) for c in rng.integers(30, 250, 3))
                cv2.rectangle(pano, (x, y), (x + w, y + h), col, -1)
                for k in range(3):
                    cv2.line(pano, (x + 5, y + 10 + k * 12), (x + w - 5, y + 10 + k * 12), (20, 20, 20), 2)
            elif kind == 2:  # door
                w, h = 70, int(30 * ppd)
                cv2.rectangle(pano, (x, horizon_floor - h), (x + w, horizon_floor), (70, 95, 130), -1)
                cv2.circle(pano, (x + w - 12, horizon_floor - h // 2), 5, (60, 110, 150), -1)  # brass handle
            elif kind == 3:  # monitor / whiteboard
                w, h = int(rng.integers(40, 160)), int(rng.integers(30, 90))
                y = int(rng.integers(row(20), row(-6)))
                cv2.rectangle(pano, (x, y), (x + w, y + h), (35, 35, 35), -1)
                cv2.rectangle(pano, (x + 4, y + 4), (x + w - 4, y + h - 4), tuple(int(c) for c in rng.integers(60, 255, 3)), -1)
            elif kind == 4:  # cable
                y0 = int(rng.integers(row(30), horizon_floor))
                pts = np.array([[x + i * 15, y0 + int(20 * math.sin(i * 0.7))] for i in range(12)], np.int32)
                cv2.polylines(pano, [pts], False, (25, 25, 25), 2, cv2.LINE_AA)
            else:            # bench / table
                w = int(rng.integers(100, 260))
                y = horizon_floor - int(rng.integers(40, 70))
                cv2.rectangle(pano, (x, y), (x + w, y + 10), (40, 60, 80), -1)
                cv2.line(pano, (x + 8, y), (x + 8, horizon_floor + 15), (40, 50, 60), 4)
                cv2.line(pano, (x + w - 8, y), (x + w - 8, horizon_floor + 15), (40, 50, 60), 4)
        noise = rng.normal(0, 4, pano.shape).astype(np.int16)
        pano = np.clip(pano.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        pano = cv2.GaussianBlur(pano, (3, 3), 0)
        pad = int(90 * ppd)
        self._pano_pad = pad
        return np.ascontiguousarray(np.hstack([pano[:, -pad:], pano, pano[:, :pad]]))

    def _make_noise_bank(self, n: int = 6):
        if self.noise_sigma <= 0:
            return []
        h, w = self.intr.height, self.intr.width
        bank = []
        for _ in range(n):
            nz = self.rng.normal(0, self.noise_sigma, (h, w, 3))
            bank.append((np.clip(nz, 0, 255).astype(np.uint8), np.clip(-nz, 0, 255).astype(np.uint8)))
        return bank

    def _render_background(self, pan: float, tilt: float) -> np.ndarray:
        intr = self.intr
        ppd = self.PANO_PPD
        pan_deg = (math.degrees(pan) + 180.0) % 360.0 - 180.0
        Xc = (pan_deg + 180.0) * ppd + self._pano_pad
        Yc = (self.EL_MAX - math.degrees(tilt)) * ppd
        k = intr.px_per_deg / ppd   # camera px per panorama px
        M = np.array([[1.0 / k, 0.0, Xc - intr.cx / k], [0.0, 1.0 / k, Yc - intr.cy / k]])
        return cv2.warpAffine(self._pano, M, (intr.width, intr.height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    # ---- drawing helpers
    @staticmethod
    def _blit(img: np.ndarray, sprite: np.ndarray, u: float, v: float, w_px: float, h_px: float) -> None:
        """Alpha-blend sprite scaled to (w_px, h_px) centred at sub-pixel (u, v)."""
        if w_px < 1.0 or h_px < 1.0:
            return
        H, W = img.shape[:2]
        x0, y0 = int(math.floor(u - w_px / 2)) - 1, int(math.floor(v - h_px / 2)) - 1
        x1, y1 = int(math.ceil(u + w_px / 2)) + 1, int(math.ceil(v + h_px / 2)) + 1
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sh, sw = sprite.shape[:2]
        sx, sy = w_px / sw, h_px / sh
        # dst(x,y) = src((x - tx)/sx, ...) ; tx chosen so sprite centre lands on (u - cx0, v - cy0)
        tx = (u - cx0) - sx * sw / 2.0
        ty = (v - cy0) - sy * sh / 2.0
        M = np.array([[sx, 0.0, tx], [0.0, sy, ty]])
        interp = cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR
        patch = cv2.warpAffine(sprite, M, (cx1 - cx0, cy1 - cy0), flags=interp, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0, 0))
        a = patch[..., 3:4].astype(np.float32) * (1.0 / 255.0)
        roi = img[cy0:cy1, cx0:cx1]
        roi[:] = (roi.astype(np.float32) * (1 - a) + patch[..., :3].astype(np.float32) * a).astype(np.uint8)

    def _occluder_polys(self, pan: float, tilt: float):
        polys = []
        for oc in self.occluders:
            top = world_to_pixel(self.intr, np.array([oc.x, oc.y, oc.z1]), pan, tilt)
            bot = world_to_pixel(self.intr, np.array([oc.x, oc.y, oc.z0]), pan, tilt)
            if top is None or bot is None:
                continue
            depth = 0.5 * (top[2] + bot[2])
            half = oc.r * self.intr.fx / max(depth, 1e-3)
            poly = np.array([[top[0] - half, top[1]], [top[0] + half, top[1]],
                             [bot[0] + half, bot[1]], [bot[0] - half, bot[1]]], np.float32)
            polys.append((poly, depth, oc))
        return polys

    # ---- main API
    def render(self, t: float, pan: float, tilt: float):
        """Return (image BGR, truth list) for host time t and true gimbal angles."""
        ts = self.scene_time(t)
        img = self._render_background(pan, tilt)
        intr = self.intr
        occ = self._occluder_polys(pan, tilt)
        items = []
        for tg in self.targets:
            p = tg.position(ts)
            if p is None:
                continue
            proj = world_to_pixel(intr, p, pan, tilt)
            if proj is None:
                continue
            u, v, depth = proj
            w_px = tg.size_m * intr.fx / depth
            h_px = w_px * tg.aspect
            items.append((depth, tg, p, u, v, w_px, h_px))
        # far to near (painter's algorithm), occluders are drawn at their depth
        drawables = [(d, "t", it) for it in items for d in (it[0],)] + [(o[1], "o", o) for o in occ]
        drawables.sort(key=lambda x: -x[0])
        for _, kind, obj in drawables:
            if kind == "t":
                _, tg, _, u, v, w_px, h_px = obj
                self._blit(img, tg.sprite, u, v, w_px, h_px * (1.0 if tg.aspect == 1.0 else 1.0))
            else:
                poly = obj[0]
                cv2.fillConvexPoly(img, (poly * 16).astype(np.int32), obj[2].color, cv2.LINE_AA, shift=4)
                cv2.polylines(img, [(poly * 16).astype(np.int32)], True, (50, 60, 75), 2, cv2.LINE_AA, shift=4)
        if self._noise:
            pos, neg = self._noise[self._noise_i % len(self._noise)]
            self._noise_i += 1
            cv2.add(img, pos, dst=img)
            cv2.subtract(img, neg, dst=img)
        truth = []
        H, W = intr.height, intr.width
        for depth, tg, p, u, v, w_px, h_px in items:
            x1, y1, x2, y2 = u - w_px / 2, v - h_px / 2, u + w_px / 2, v + h_px / 2
            bx1, by1, bx2, by2 = max(0.0, x1), max(0.0, y1), min(W - 1.0, x2), min(H - 1.0, y2)
            if bx2 <= bx1 or by2 <= by1:
                continue
            area = (x2 - x1) * (y2 - y1)
            vis_area = (bx2 - bx1) * (by2 - by1)
            # subtract occluder overlap (only occluders nearer than the target)
            for poly, odepth, _ in occ:
                if odepth >= depth:
                    continue
                ox1, oy1 = poly[:, 0].min(), poly[:, 1].min()
                ox2, oy2 = poly[:, 0].max(), poly[:, 1].max()
                iw = max(0.0, min(bx2, ox2) - max(bx1, ox1))
                ih = max(0.0, min(by2, oy2) - max(by1, oy1))
                vis_area -= iw * ih
            frac = max(0.0, vis_area) / area if area > 0 else 0.0
            truth.append({"id": tg.tid, "cls": tg.cls, "bbox": (bx1, by1, bx2, by2), "center": (u, v),
                          "full_bbox": (x1, y1, x2, y2), "visible": frac, "depth": depth,
                          "pos": (float(p[0]), float(p[1]), float(p[2])), "label": tg.label})
        return img, truth

    def target_truth(self, t: float) -> list[dict]:
        ts = self.scene_time(t)
        out = []
        for tg in self.targets:
            p = tg.position(ts)
            if p is not None:
                out.append({"id": tg.tid, "cls": tg.cls, "pos": p})
        return out

    def range_along(self, t: float, pan: float, tilt: float, beam_half_deg: float = 1.5) -> float:
        """Distance the LiDAR (on the boresight, 3 deg beam) would report, metres."""
        d = azel_to_dir(pan, tilt)
        best = self._room_distance(d)
        ts = self.scene_time(t)
        for tg in self.targets:
            p = tg.position(ts)
            if p is None:
                continue
            dist = float(np.linalg.norm(p))
            ang = math.acos(float(np.clip(np.dot(d, p / dist), -1, 1)))
            ang_radius = math.atan2(tg.size_m / 2.0, dist)
            if ang <= math.radians(beam_half_deg) + ang_radius * 0.8:
                best = min(best, dist - tg.size_m / 2.0)
        for oc in self.occluders:
            # ray vs vertical cylinder (2-D in xy)
            dx, dy = d[0], d[1]
            a = dx * dx + dy * dy
            if a < 1e-9:
                continue
            b = -2 * (dx * oc.x + dy * oc.y)
            c = oc.x ** 2 + oc.y ** 2 - oc.r ** 2
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            s = (-b - math.sqrt(disc)) / (2 * a)
            if s > 0 and oc.z0 <= s * d[2] <= oc.z1:
                best = min(best, s)
        return best

    def _room_distance(self, d: np.ndarray) -> float:
        r = self.room
        ts = []
        if d[0] > 1e-9:
            ts.append(r["front"] / d[0])
        if d[0] < -1e-9:
            ts.append(r["back"] / d[0])
        if d[1] > 1e-9:
            ts.append(r["right"] / d[1])
        if d[1] < -1e-9:
            ts.append(r["left"] / d[1])
        if d[2] > 1e-9:
            ts.append(r["ceiling"] / d[2])
        if d[2] < -1e-9:
            ts.append(r["floor"] / d[2])
        return float(min(t for t in ts if t > 0)) if ts else 40.0

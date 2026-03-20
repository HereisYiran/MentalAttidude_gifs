from __future__ import annotations

import math
from pathlib import Path
import json
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gridworld.helper.objects import Bush, EmptyBush, RedBerryBush, BlueBerryBush, OrangeBerryBush
    from gridworld.helper.walls import OuterWall, InnerWall
else:
    from .objects import Bush, EmptyBush, RedBerryBush, BlueBerryBush, OrangeBerryBush
    from .walls import OuterWall, InnerWall

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Floor
from minigrid.utils.rendering import fill_coords, point_in_rect

class WarmGrassFloor(Floor):

    BASE = (126, 200, 80)   # #7ec850
    DARK = (109, 184, 68)   # #6db844
    LIGHT = (142, 212, 90)  # #8ed45a
    GRID = (95, 165, 55)

    def __init__(self):
        super().__init__("green")

    def render(self, img):

        # base color
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.BASE)

        # darker texture rectangles
        patches_dark = [
            (0.12,0.18,0.22,0.30),
            (0.70,0.76,0.14,0.20),
            (0.45,0.52,0.55,0.63),
            (0.30,0.35,0.70,0.78)
        ]

        for x1,x2,y1,y2 in patches_dark:
            fill_coords(img, point_in_rect(x1,x2,y1,y2), self.DARK)

        # lighter texture rectangles
        patches_light = [
            (0.60,0.66,0.22,0.30),
            (0.18,0.25,0.60,0.68),
            (0.74,0.80,0.60,0.68)
        ]

        for x1,x2,y1,y2 in patches_light:
            fill_coords(img, point_in_rect(x1,x2,y1,y2), self.LIGHT)

        # grid lines
        fill_coords(img, point_in_rect(0.0,1.0,0.0,0.02), self.GRID)
        fill_coords(img, point_in_rect(0.0,0.02,0.0,1.0), self.GRID)

_CARDINAL_TO_DIR = {
    "R": 0,
    "D": 1,
    "L": 2,
    "U": 3,
}

_BUSH_MAP = {
    "empty": EmptyBush,
    "red": RedBerryBush,
    "blue": BlueBerryBush,
    "orange": OrangeBerryBush,
}


class JsonScenarioEnv(MiniGridEnv):
    def __init__(self, scenario: dict[str, Any], **kwargs: Any):
        grid_cfg = scenario["grid"]
        mission = scenario.get("name", "gridworld scenario")
        mission_space = MissionSpace(mission_func=lambda: mission)
        super().__init__(
            width=grid_cfg["width"],
            height=grid_cfg["height"],
            max_steps=grid_cfg.get("max_steps", 100),
            see_through_walls=False,
            agent_view_size=grid_cfg.get("agent_view_size", 3),
            mission_space=mission_space,
            **kwargs,
        )
        self._scenario = scenario
        self.agent_start_pos = tuple(grid_cfg["agent_start_pos"])
        self.agent_start_dir = int(grid_cfg["agent_start_dir"])

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)

        walls_cfg = self._scenario.get("walls", {})
        if walls_cfg.get("outer", True):
            for x in range(width):
                self.grid.set(x, 0, OuterWall())
                self.grid.set(x, height - 1, OuterWall())
            for y in range(height):
                self.grid.set(0, y, OuterWall())
                self.grid.set(width - 1, y, OuterWall())

        for seg in walls_cfg.get("segments", []):
            row = int(seg["row"])
            for col in seg["cols"]:
                self.grid.set(int(col), row, InnerWall())

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if self.grid.get(x, y) is None:
                    self.grid.set(x, y, WarmGrassFloor())

        for bush in self._scenario.get("bushes", []):
            bush_type = bush["type"]
            bush_cls = _BUSH_MAP[bush_type]
            x, y = bush["pos"]
            self.grid.set(int(x), int(y), bush_cls())

        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        self.mission = self._scenario.get("name", "gridworld scenario")


def load_scenario(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_action(env: MiniGridEnv, action: str):
    if action in _CARDINAL_TO_DIR:
        env.agent_dir = _CARDINAL_TO_DIR[action]
        env.step(env.actions.forward)
        return

    if action == "F":
        env.step(env.actions.forward)
        return

    if action in {"TL", "TURN_LEFT"}:
        env.step(env.actions.left)
        return

    if action in {"TR", "TURN_RIGHT"}:
        env.step(env.actions.right)
        return

    raise ValueError(f"Unsupported action token: {action}")


def _get_highlighted_cells(env: MiniGridEnv) -> set[tuple[int, int]]:
    _, vis_mask = env.gen_obs_grid()
    f_vec = env.dir_vec
    r_vec = env.right_vec
    top_left = (
        env.agent_pos
        + f_vec * (env.agent_view_size - 1)
        - r_vec * (env.agent_view_size // 2)
    )
    cells: set[tuple[int, int]] = set()

    for vis_j in range(1, env.agent_view_size):
        for vis_i in range(env.agent_view_size):
            if not vis_mask[vis_i, vis_j]:
                continue
            abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
            ai, aj = int(abs_i), int(abs_j)
            if 0 <= ai < env.width and 0 <= aj < env.height:
                cells.add((ai, aj))
    return cells


def _check_berry_discovery(env: MiniGridEnv):
    for (x, y) in _get_highlighted_cells(env):
        obj = env.grid.get(x, y)
        if isinstance(obj, Bush) and obj.berry_color is not None and not obj.discovered:
            obj.discovered = True


# Legacy 8-cell orbit (kept for reference)
def _clockwise_orbit(center: tuple[int, int]) -> list[tuple[int, int]]:
    cx, cy = center
    return [
        (cx, cy - 1),
        (cx + 1, cy - 1),
        (cx + 1, cy),
        (cx + 1, cy + 1),
        (cx, cy + 1),
        (cx - 1, cy + 1),
        (cx - 1, cy),
        (cx - 1, cy - 1),
    ]


ORBIT_POSITIONS = 8  # number of discrete positions in the original orbit


def _smooth_bug_pixel_pos(
    center: tuple[int, int],
    orbit_frac: float,
    tile_size: int,
    orbit_radius: float = 1.0,
) -> tuple[int, int]:
    """Return (px, py) pixel coords for a bug on a smooth circular orbit."""
    cx, cy = center
    angle = 2.0 * math.pi * orbit_frac
    px = cx * tile_size + tile_size / 2 + orbit_radius * tile_size * math.sin(angle)
    py = cy * tile_size + tile_size / 2 - orbit_radius * tile_size * math.cos(angle)
    return int(round(px)), int(round(py))


def _draw_dot(img: np.ndarray, cx: int, cy: int, radius: int, color: tuple[int, int, int]):
    h, w, _ = img.shape
    x0 = max(0, cx - radius)
    x1 = min(w - 1, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h - 1, cy + radius)
    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[y0:y1 + 1, x0:x1 + 1][mask] = color

def _draw_triangle_alpha(img, cx, cy, size, direction, alpha):

    RED = np.array([255, 0, 0], dtype=np.float32)

    if direction == 0:   # right
        pts = [(cx+size,cy),(cx-size,cy-size),(cx-size,cy+size)]
    elif direction == 2: # left
        pts = [(cx-size,cy),(cx+size,cy-size),(cx+size,cy+size)]
    elif direction == 3: # up
        pts = [(cx,cy-size),(cx-size,cy+size),(cx+size,cy+size)]
    else:                # down
        pts = [(cx,cy+size),(cx-size,cy-size),(cx+size,cy-size)]

    pts = np.array(pts)

    minx = int(pts[:,0].min())
    maxx = int(pts[:,0].max())
    miny = int(pts[:,1].min())
    maxy = int(pts[:,1].max())

    for y in range(miny,maxy+1):
        for x in range(minx,maxx+1):

            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:

                v0 = pts[2]-pts[0]
                v1 = pts[1]-pts[0]
                v2 = np.array([x,y])-pts[0]

                dot00=np.dot(v0,v0)
                dot01=np.dot(v0,v1)
                dot02=np.dot(v0,v2)
                dot11=np.dot(v1,v1)
                dot12=np.dot(v1,v2)

                inv=1/(dot00*dot11-dot01*dot01+1e-6)

                u=(dot11*dot02-dot01*dot12)*inv
                v=(dot00*dot12-dot01*dot02)*inv

                if u>=0 and v>=0 and u+v<1:

                    img[y,x] = (
                        img[y,x]*(1-alpha) + RED*alpha
                    ).astype(np.uint8)

def _overlay_bug(img: np.ndarray, cell: tuple[int, int], tile_size: int):
    x, y = cell
    cx = x * tile_size + tile_size // 2
    cy = y * tile_size + tile_size // 2
    _overlay_bug_at_pixel(img, cx, cy)


def _overlay_bug_at_pixel(img: np.ndarray, px: int, py: int):
    """Draw a more intuitive bug sprite."""

    BODY = (40, 40, 40)
    SHELL = (200, 50, 50)
    HEAD = (25, 25, 25)
    LEG = (30, 30, 30)
    HIGHLIGHT = (240, 120, 120)

    # body
    _draw_dot(img, px, py, 5, BODY)

    # shell
    _draw_dot(img, px, py, 4, SHELL)

    # head
    _draw_dot(img, px, py - 6, 2, HEAD)

    # legs
    _draw_dot(img, px - 6, py + 1, 1, LEG)
    _draw_dot(img, px + 6, py + 1, 1, LEG)
    _draw_dot(img, px - 6, py - 2, 1, LEG)
    _draw_dot(img, px + 6, py - 2, 1, LEG)

    # highlight
    _draw_dot(img, px - 2, py - 1, 1, HIGHLIGHT)


_LABEL_FONT = None


def _get_label_font():
    global _LABEL_FONT
    if _LABEL_FONT is not None:
        return _LABEL_FONT

    try:
        _LABEL_FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 19)
    except OSError:
        _LABEL_FONT = ImageFont.load_default()
    return _LABEL_FONT


def _overlay_bush_labels(
    img: np.ndarray,
    bushes: list[tuple[int, int]],
    tile_size: int,
    label_position: str = "above",
):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    font = _get_label_font()

    for index, (x, y) in enumerate(bushes, start=1):
        label = str(index)
        center_x = x * tile_size + tile_size // 2
        top_y = y * tile_size
        bottom_y = (y + 1) * tile_size
        right_x = x * tile_size + tile_size
        center_y = y * tile_size + tile_size // 2
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        if label_position == "right":
            label_x = right_x + 2
            label_y = center_y - text_h // 2
        elif label_position == "below":
            label_x = center_x - text_w // 2
            label_y = bottom_y - text_h - 2
        elif label_position == "bottom_right":
            label_x = right_x - text_w - 2
            label_y = bottom_y - text_h - 2
        else:  # "above" (default)
            label_x = center_x - text_w // 2
            label_y = max(1, top_y + 1 - text_h)

        draw.text(
            (label_x, label_y),
            label,
            font=font,
            fill=(250, 250, 250),
            stroke_width=2,
            stroke_fill=(20, 20, 20),
        )

    img[:, :, :] = np.array(pil_img)


def _consume_on_step(env: MiniGridEnv, bush_types: list[str]):
    x, y = int(env.agent_pos[0]), int(env.agent_pos[1])
    obj = env.grid.get(x, y)
    berry_type = getattr(obj, "berry_color", None)
    if isinstance(obj, Bush) and berry_type in bush_types:
        env.grid.set(x, y, EmptyBush())


def _dim_outside_view(
    frame: np.ndarray,
    visible_cells: set[tuple[int, int]],
    tile_size: int,
    brightness: float,
):
    if brightness >= 1.0:
        return

    brightness = max(0.0, min(1.0, brightness))
    h, w, _ = frame.shape
    visible_mask = np.zeros((h, w), dtype=bool)

    for x, y in visible_cells:
        x0 = max(0, x * tile_size)
        x1 = min(w, (x + 1) * tile_size)
        y0 = max(0, y * tile_size)
        y1 = min(h, (y + 1) * tile_size)
        visible_mask[y0:y1, x0:x1] = True

    original = frame.copy()
    frame[:, :, :] = (frame.astype(np.float32) * brightness).astype(np.uint8)
    boosted = np.clip(original.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
    frame[visible_mask] = boosted[visible_mask]


def _eat_berry(env: MiniGridEnv, berry_type: str | None = None):
    def _consume_if_match(x: int, y: int) -> bool:
        obj = env.grid.get(x, y)
        obj_berry_type = getattr(obj, "berry_color", None)
        if not isinstance(obj, Bush) or obj_berry_type is None:
            return False
        if berry_type is not None and obj_berry_type != berry_type:
            return False
        env.grid.set(x, y, EmptyBush())
        return True

    x, y = int(env.agent_pos[0]), int(env.agent_pos[1])
    if _consume_if_match(x, y):
        return

    fx, fy = int(env.front_pos[0]), int(env.front_pos[1])
    if 0 <= fx < env.width and 0 <= fy < env.height:
        if _consume_if_match(fx, fy):
            return

    visible_cells = sorted(
        _get_highlighted_cells(env),
        key=lambda cell: (abs(cell[0] - x) + abs(cell[1] - y), cell[1], cell[0]),
    )
    for vx, vy in visible_cells:
        if _consume_if_match(vx, vy):
            return

def _eat_bug(env: MiniGridEnv, active_bugs: list[tuple[int,int]]):

    x, y = int(env.agent_pos[0]), int(env.agent_pos[1])

    # eat bug in same cell
    if (x,y) in active_bugs:
        active_bugs.remove((x,y))
        return

    # eat bug in front
    fx, fy = int(env.front_pos[0]), int(env.front_pos[1])
    if (fx,fy) in active_bugs:
        active_bugs.remove((fx,fy))

# ---------------------------------------------------------------------------
# Scripted bug: follows a waypoint path, triggered by game events
# ---------------------------------------------------------------------------

class ScriptedBug:
    """A bug that lurks at a hidden position, then walks a path on trigger.

    JSON config keys (inside render.scripted_bugs list):
      start           [x, y]        initial (lurking) tile position
      path            [[x,y], ...]  waypoints to walk after trigger fires
      trigger         str           "after_eat_orange" | "after_eat_red" |
                                    "after_eat" | "immediate" | "after_step_N"
      eat_at_end      bool          consume a berry after the path finishes
      eat_target      [x, y]        which cell to eat from (defaults to path[-1]);
                                    use this to eat an adjacent bush without
                                    stepping onto it
      eat_pause_frames int          sub-frames to wait at final waypoint before
                                    eating (fps * bug_sub_frames ≈ 1 second)
      steps_per_cell  int           sub-frames spent moving between waypoints
    """

    def __init__(self, cfg: dict):
        self.start: tuple[int, int] = tuple(cfg["start"])
        self.path: list[tuple[int, int]] = [tuple(p) for p in cfg.get("path", [])]
        self.trigger: str = cfg.get("trigger", "after_eat_orange")
        self.trigger_cell = tuple(cfg["trigger_cell"]) if "trigger_cell" in cfg else None
        self.eat_at_end: bool = bool(cfg.get("eat_at_end", True))
        

        # eat_target: explicit cell to consume; falls back to path[-1] if omitted
        raw_target = cfg.get("eat_target", None)
        self.eat_target: tuple[int, int] | None = tuple(raw_target) if raw_target is not None else None
        self.eat_pause_frames: int = max(0, int(cfg.get("eat_pause_frames", 0)))
        self.steps_per_cell: int = max(1, int(cfg.get("steps_per_cell", 4)))

        # runtime state
        self.triggered: bool = False
        self.finished: bool = False
        self._path_idx: int = 0
        self._step_in_cell: int = 0
        self._pos: tuple[float, float] = (float(self.start[0]), float(self.start[1]))
        self._berry_eaten: bool = False
        self._eat_pause_elapsed: int = 0   # sub-frames spent in pre-eat pause

    def notify_event(self, event: str):
        if self.triggered:
            return
        if self.trigger == "immediate" or self.trigger == event:
            self.triggered = True

    def notify_step(self, step_count: int):
        if self.triggered:
            return
        if self.trigger.startswith("after_step_"):
            try:
                if step_count >= int(self.trigger.split("_")[-1]):
                    self.triggered = True
            except ValueError:
                pass

    def advance(self, env: MiniGridEnv):
        """Move one sub-frame along the path."""
        if not self.triggered or self.finished:
            return

        if self._path_idx >= len(self.path):
            # Arrived — pause, then eat, then finish
            if self.eat_at_end and not self._berry_eaten:
                if self._eat_pause_elapsed < self.eat_pause_frames:
                    self._eat_pause_elapsed += 1
                    return
                # Determine which cell holds the berry
                if self.eat_target is not None:
                    cx, cy = self.eat_target
                elif self.path:
                    cx, cy = self.path[-1]
                else:
                    cx, cy = self.start
                obj = env.grid.get(int(cx), int(cy))
                if isinstance(obj, Bush) and getattr(obj, "berry_color", None) is not None:
                    env.grid.set(int(cx), int(cy), EmptyBush())
                self._berry_eaten = True
            self.finished = True
            return

        target = self.path[self._path_idx]
        tx, ty = float(target[0]), float(target[1])

        if self._path_idx == 0:
            sx, sy = float(self.start[0]), float(self.start[1])
        else:
            prev = self.path[self._path_idx - 1]
            sx, sy = float(prev[0]), float(prev[1])

        self._step_in_cell += 1
        frac = min(self._step_in_cell / self.steps_per_cell, 1.0)
        self._pos = (sx + (tx - sx) * frac, sy + (ty - sy) * frac)

        if frac >= 1.0:
            self._path_idx += 1
            self._step_in_cell = 0

    def pixel_pos(self, tile_size: int) -> tuple[int, int]:
        px = self._pos[0] * tile_size + tile_size / 2
        py = self._pos[1] * tile_size + tile_size / 2
        return int(round(px)), int(round(py))


def render_scenario(scenario_path: Path, output_root: Path) -> Path:
    scenario = load_scenario(scenario_path)
    render_cfg = scenario.get("render", {})
    tile_size = int(render_cfg.get("tile_size", 48))
    fps = float(render_cfg.get("fps", 1.3))
    enable_discovery = bool(render_cfg.get("enable_discovery", True))
    show_bush_labels = bool(render_cfg.get("show_bush_labels", True))
    label_position = str(render_cfg.get("label_position", "above")).lower()
    consume_types = [str(t) for t in render_cfg.get("consume_on_step", [])]
    dim_outside_view = bool(render_cfg.get("dim_outside_view", False))
    outside_view_brightness = float(render_cfg.get("outside_view_brightness", 0.28))
    eat_bugs_on_bump = bool(render_cfg.get("eat_bugs_on_bump", False))
    has_outer_wall = bool(scenario.get("walls", {}).get("outer", True))

    category = scenario["category"]
    output_dir = output_root / category
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scenario['name']}.gif"

    env = JsonScenarioEnv(scenario, render_mode="rgb_array", tile_size=tile_size)

    if dim_outside_view:
        env.highlight = False

    # --- orbital bugs (legacy) ---
    bug_cfg = render_cfg.get("bugs", [])
    bug_phase = [int(b.get("phase", 0)) for b in bug_cfg]
    bug_centers = [(int(b["center"][0]), int(b["center"][1])) for b in bug_cfg]
    active_bugs = bug_centers.copy()
    bug_orbit_radius = float(render_cfg.get("bug_orbit_radius", 1.2))

    # --- scripted bugs ---
    scripted_bugs: list[ScriptedBug] = [
        ScriptedBug(cfg) for cfg in render_cfg.get("scripted_bugs", [])
    ]

    has_any_bugs = bool(bug_cfg) or bool(scripted_bugs)
    bug_sub_frames = int(render_cfg.get("bug_sub_frames", 4)) if has_any_bugs else 1

    bush_positions = [
        (int(bush["pos"][0]), int(bush["pos"][1]))
        for bush in scenario.get("bushes", [])
    ]

    env.reset()
    frames = []
    bug_time = 0.0
    bug_dt = 1.0 / bug_sub_frames
    agent_step_count = 0

    trajectory: list[dict] = []
    trajectory.append({"pos": tuple(env.agent_pos), "age": 0, "dir": env.agent_dir})

    def _render_frame(bt: float) -> np.ndarray:
        frame = env.render()

        for t in trajectory:
            t["age"] += 1 / bug_sub_frames

        MAX_AGE = 15
        trajectory[:] = [t for t in trajectory if t["age"] < MAX_AGE]


        # draw trajectory triangles
        for t in trajectory:

            x, y = t["pos"]
            direction = t["dir"]

            cx = x * tile_size + tile_size // 2
            cy = y * tile_size + tile_size // 2

            fade = 1 - t["age"] / MAX_AGE
            fade = max(0, fade)

            _draw_triangle_alpha(
                frame,
                cx,
                cy,
                4,
                direction,
                fade
            )

        # Orbital bugs

        agent_px = env.agent_pos[0] * tile_size + tile_size // 2
        agent_py = env.agent_pos[1] * tile_size + tile_size // 2

        remaining_bugs = []
        remaining_phases = []

        for idx, center in enumerate(active_bugs):
            phase_offset = bug_phase[idx] / ORBIT_POSITIONS
            orbit_frac = (bt / ORBIT_POSITIONS + phase_offset) % 1.0

            px, py = _smooth_bug_pixel_pos(center, orbit_frac, tile_size, bug_orbit_radius)

            # bug gets eaten when it bumps into the agent
            dist = math.hypot(px - agent_px, py - agent_py)

            if eat_bugs_on_bump and dist < tile_size * 0.35:
                continue

            remaining_bugs.append(center)
            remaining_phases.append(bug_phase[idx])
            _overlay_bug_at_pixel(frame, px, py)

        active_bugs[:] = remaining_bugs
        bug_phase[:] = remaining_phases

        # Scripted bugs
        for sb in scripted_bugs:
            px, py = sb.pixel_pos(tile_size)
            _overlay_bug_at_pixel(frame, px, py)
        if dim_outside_view:
            visible_cells = _get_highlighted_cells(env)
            visible_cells.add((int(env.agent_pos[0]), int(env.agent_pos[1])))
            _dim_outside_view(frame, visible_cells, tile_size, outside_view_brightness)
        if show_bush_labels:
            _overlay_bush_labels(frame, bush_positions, tile_size, label_position)
        if has_outer_wall:
            frame = frame[tile_size:-tile_size, tile_size:-tile_size]
            b = 2
            border_color = np.array([55, 95, 32], dtype=np.uint8)
            h, w = frame.shape[:2]
            padded = np.full((h + 2 * b, w + 2 * b, 3), border_color, dtype=np.uint8)
            padded[b:b + h, b:b + w] = frame
            frame = padded
        return frame

    def _advance_scripted_bugs():
        for sb in scripted_bugs:
            sb.advance(env)

    def _notify_event(event: str):
        for sb in scripted_bugs:
            sb.notify_event(event)

    def _notify_step():
        for sb in scripted_bugs:
            sb.notify_step(agent_step_count)

    if enable_discovery:
        _check_berry_discovery(env)
    frames.append(_render_frame(bug_time))

    for symbol, count in scenario.get("actions", []):
        action = str(symbol).upper()
        amount = int(count)

        if action in {"WAIT", "PAUSE"}:
            hold_count = max(0, int(round(amount * fps)))
            for _ in range(hold_count * bug_sub_frames):
                bug_time += bug_dt
                _advance_scripted_bugs()
                frames.append(_render_frame(bug_time))
            continue

        if action in {"EAT_BERRY", "CONSUME_BERRY", "EAT", "DISAPPEAR_BERRY"}:
            _eat_berry(env)
            if enable_discovery:
                _check_berry_discovery(env)
            _notify_event("after_eat")
            for _ in range(bug_sub_frames):
                bug_time += bug_dt
                _advance_scripted_bugs()
                frames.append(_render_frame(bug_time))
            continue

        if action in {"EAT_ORANGE", "CONSUME_ORANGE"}:
            _eat_berry(env, "orange")
            if enable_discovery:
                _check_berry_discovery(env)
            _notify_event("after_eat_orange")
            for _ in range(bug_sub_frames):
                bug_time += bug_dt
                _advance_scripted_bugs()
                frames.append(_render_frame(bug_time))
            continue

        if action in {"EAT_RED", "CONSUME_RED"}:
            _eat_berry(env, "red")
            if enable_discovery:
                _check_berry_discovery(env)
            _notify_event("after_eat_red")
            for _ in range(bug_sub_frames):
                bug_time += bug_dt
                _advance_scripted_bugs()
                frames.append(_render_frame(bug_time))
            continue

        for _ in range(max(0, amount)):
            _apply_action(env, action)

            # trigger bug when agent reaches specific bush
            for sb in scripted_bugs:
                if not sb.triggered and sb.trigger == "reach_cell":
                    if tuple(env.agent_pos) == sb.trigger_cell:
                        sb.triggered = True
            trajectory.append({
                "pos": tuple(env.agent_pos),
                "age": 0,
                "dir": env.agent_dir
            })
            if consume_types:
                _consume_on_step(env, consume_types)
            if enable_discovery:
                _check_berry_discovery(env)
            agent_step_count += 1
            _notify_step()
            for _ in range(bug_sub_frames):
                bug_time += bug_dt
                _advance_scripted_bugs()
                frames.append(_render_frame(bug_time))

    env.close()
    imageio.mimsave(output_path, frames, fps=fps * bug_sub_frames, loop=0)
    return output_path
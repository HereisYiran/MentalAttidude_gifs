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
                    self.grid.set(x, y, Floor("green"))

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
    """Return (px, py) pixel coords for a bug on a smooth circular orbit.

    *orbit_frac* goes from 0 → 1 for one full revolution (clockwise, starting
    from directly above *center*).
    """
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


def _overlay_bug(img: np.ndarray, cell: tuple[int, int], tile_size: int):
    x, y = cell
    cx = x * tile_size + tile_size // 2
    cy = y * tile_size + tile_size // 2
    _overlay_bug_at_pixel(img, cx, cy)


def _overlay_bug_at_pixel(img: np.ndarray, px: int, py: int):
    """Draw a bug sprite at arbitrary pixel coordinates."""
    _draw_dot(img, px, py, 5, (25, 25, 25))
    _draw_dot(img, px - 5, py - 4, 2, (210, 210, 210))
    _draw_dot(img, px + 5, py - 4, 2, (210, 210, 210))
    _draw_dot(img, px, py + 1, 2, (255, 120, 0))


_LABEL_FONT = None


def _get_label_font():
    global _LABEL_FONT
    if _LABEL_FONT is not None:
        return _LABEL_FONT

    try:
        _LABEL_FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 21)
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
        right_x = x * tile_size + tile_size
        center_y = y * tile_size + tile_size // 2
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        if label_position == "right":
            label_x = right_x + 2
            label_y = center_y - text_h // 2
        else:
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

    category = scenario["category"]
    output_dir = output_root / category
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scenario['name']}.gif"

    env = JsonScenarioEnv(scenario, render_mode="rgb_array", tile_size=tile_size)

    if dim_outside_view:
        env.highlight = False

    bug_cfg = render_cfg.get("bugs", [])
    bug_phase = [int(b.get("phase", 0)) for b in bug_cfg]
    bug_centers = [(int(b["center"][0]), int(b["center"][1])) for b in bug_cfg]
    bug_orbit_radius = float(render_cfg.get("bug_orbit_radius", 1.2))
    # Sub-frames between each agent step for smooth bug flight.
    # Only used when there are bugs; otherwise keep 1 to avoid bloating frames.
    bug_sub_frames = int(render_cfg.get("bug_sub_frames", 4)) if bug_cfg else 1
    bush_positions = [
        (int(bush["pos"][0]), int(bush["pos"][1]))
        for bush in scenario.get("bushes", [])
    ]

    env.reset()
    frames = []
    bug_time = 0.0          # continuous bug tick (advances by 1.0 per agent step)
    bug_dt = 1.0 / bug_sub_frames  # how much bug_time advances per sub-frame

    def _render_frame(bt: float) -> np.ndarray:
        frame = env.render()
        for idx, center in enumerate(bug_centers):
            phase_offset = bug_phase[idx] / ORBIT_POSITIONS
            orbit_frac = (bt / ORBIT_POSITIONS + phase_offset) % 1.0
            px, py = _smooth_bug_pixel_pos(center, orbit_frac, tile_size, bug_orbit_radius)
            _overlay_bug_at_pixel(frame, px, py)
        if dim_outside_view:
            visible_cells = _get_highlighted_cells(env)
            visible_cells.add((int(env.agent_pos[0]), int(env.agent_pos[1])))
            _dim_outside_view(frame, visible_cells, tile_size, outside_view_brightness)
        if show_bush_labels:
            _overlay_bush_labels(frame, bush_positions, tile_size, label_position)
        return frame

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
                frames.append(_render_frame(bug_time))
            continue

        if action in {"EAT_BERRY", "CONSUME_BERRY", "EAT", "DISAPPEAR_BERRY"}:
            _eat_berry(env)
            if enable_discovery:
                _check_berry_discovery(env)
            for _ in range(bug_sub_frames):
                bug_time += bug_dt
                frames.append(_render_frame(bug_time))
            continue

        if action in {"EAT_ORANGE", "CONSUME_ORANGE"}:
            _eat_berry(env, "orange")
            if enable_discovery:
                _check_berry_discovery(env)
            for _ in range(bug_sub_frames):
                bug_time += bug_dt
                frames.append(_render_frame(bug_time))
            continue

        for _ in range(max(0, amount)):
            _apply_action(env, action)
            if consume_types:
                _consume_on_step(env, consume_types)
            if enable_discovery:
                _check_berry_discovery(env)
            for _ in range(bug_sub_frames):
                bug_time += bug_dt
                frames.append(_render_frame(bug_time))

    env.close()
    # Scale fps by sub-frames so overall animation speed is preserved.
    imageio.mimsave(output_path, frames, fps=fps * bug_sub_frames, loop=0)
    return output_path

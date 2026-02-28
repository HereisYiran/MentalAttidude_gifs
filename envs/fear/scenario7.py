# Fear_Scenario 7

# file path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import imageio
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor
from minigrid.minigrid_env import MiniGridEnv

from render.objects import Bush, RedBerryBush, BlueBerryBush
from render.walls import OuterWall, InnerWall


LAYOUT = [
    "WWWWWWWWWWWWW",
    "W...........W",
    "W.........B.W",
    "W...........W",
    "W...WWWWWWWWW",
    "W....SSS....W",
    "W....SBS....W",
    "W....SSS....W",
    "W...........W",
    "W...........W",
    "WWWWWW......W",
    "W........SSSW",
    "WG.......SBSW",
    "W........SSSW",
    "WWWWWWWWWWWWW",
]


def _direct_backoff(env):
    back = env.agent_pos - env.dir_vec
    bx, by = int(back[0]), int(back[1])
    if not (0 <= bx < env.width and 0 <= by < env.height):
        return
    obj = env.grid.get(bx, by)
    if obj is None or obj.can_overlap():
        env.agent_pos = (bx, by)


def _move_to(x, y):
    def _inner(env):
        obj = env.grid.get(x, y)
        if obj is None or obj.can_overlap():
            old_x, old_y = int(env.agent_pos[0]), int(env.agent_pos[1])
            env.agent_pos = (x, y)
            dx, dy = x - old_x, y - old_y
            if abs(dx) > abs(dy):
                env.agent_dir = 0 if dx > 0 else 2
            elif abs(dy) > 0:
                env.agent_dir = 1 if dy > 0 else 3
    return _inner


def _wait(seconds):
    return ("wait", float(seconds))


def _get_highlighted_cells(env):
    _, vis_mask = env.gen_obs_grid()
    f_vec = env.dir_vec
    r_vec = env.right_vec
    top_left = (
        env.agent_pos
        + f_vec * (env.agent_view_size - 1)
        - r_vec * (env.agent_view_size // 2)
    )
    cells = set()
    for vis_j in range(1, env.agent_view_size):
        for vis_i in range(env.agent_view_size):
            if not vis_mask[vis_i, vis_j]:
                continue
            abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
            ai, aj = int(abs_i), int(abs_j)
            if 0 <= ai < env.width and 0 <= aj < env.height:
                cells.add((ai, aj))
    return cells


def _check_berry_discovery(env):
    for (x, y) in _get_highlighted_cells(env):
        obj = env.grid.get(x, y)
        if isinstance(obj, Bush) and obj.berry_color is not None and not obj.discovered:
            obj.discovered = True


def _clockwise_orbit(center):
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


def _draw_dot(img, cx, cy, radius, color):
    h, w, _ = img.shape
    x0 = max(0, cx - radius)
    x1 = min(w - 1, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h - 1, cy + radius)
    yy, xx = np.ogrid[y0:y1 + 1, x0:x1 + 1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[y0:y1 + 1, x0:x1 + 1][mask] = color


def _overlay_bug(img, cell, tile_size):
    x, y = cell
    cx = x * tile_size + tile_size // 2
    cy = y * tile_size + tile_size // 2
    _draw_dot(img, cx, cy, 4, (25, 25, 25))
    _draw_dot(img, cx - 4, cy - 3, 2, (210, 210, 210))
    _draw_dot(img, cx + 4, cy - 3, 2, (210, 210, 210))
    _draw_dot(img, cx, cy + 1, 1, (255, 120, 0))


def make_scenario7_gif(env, actions, output_path="scenario7_fear.gif", fps=2.0, tile_size=48):
    env.reset()
    frames = []
    tick = 0

    bug_orbits = [
        _clockwise_orbit((6, 6)),
        _clockwise_orbit((10, 12)),
    ]
    bug_phase = [0, 3]

    def _render_with_bugs():
        frame = env.render()
        for idx, orbit in enumerate(bug_orbits):
            cell = orbit[(tick + bug_phase[idx]) % len(orbit)]
            _overlay_bug(frame, cell, tile_size)
        return frame

    def _append_frame():
        nonlocal tick
        _check_berry_discovery(env)
        frames.append(_render_with_bugs())
        tick += 1

    _append_frame()

    for action in actions:
        if isinstance(action, tuple) and len(action) == 2 and action[0] == "wait":
            hold_frames = max(1, int(round(action[1] * fps)))
            for _ in range(hold_frames):
                _append_frame()
            continue

        if callable(action):
            action(env)
        else:
            env.step(action)

        _append_frame()

    env.close()
    imageio.mimsave(output_path, frames, fps=fps, loop=0)


class Scenario7Env(MiniGridEnv):

    def __init__(self, max_steps=160, **kwargs):
        width = len(LAYOUT[0])
        height = len(LAYOUT)
        mission_space = MissionSpace(mission_func=lambda: "fear scenario 7")
        super().__init__(
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=3,
            mission_space=mission_space,
            **kwargs,
        )
        self.agent_start_pos = (1, 12)
        self.agent_start_dir = 0

    def get_full_render(self, highlight, tile_size):
        _, vis_mask = self.gen_obs_grid()
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        highlight_mask = np.zeros((self.width, self.height), dtype=bool)
        if highlight:
            for vis_j in range(1, self.agent_view_size):
                for vis_i in range(self.agent_view_size):
                    if not vis_mask[vis_i, vis_j]:
                        continue
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
                    if 0 <= abs_i < self.width and 0 <= abs_j < self.height:
                        highlight_mask[int(abs_i), int(abs_j)] = True

        agent_pos = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        return self.grid.render(
            tile_size,
            agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        for y, row in enumerate(LAYOUT):
            for x, ch in enumerate(row):
                if ch == "W":
                    self.grid.set(x, y, OuterWall())
                elif ch in {".", "S", "G"}:
                    self.grid.set(x, y, Floor("green"))
                elif ch == "B":
                    if (x, y) == (10, 12):
                        self.grid.set(x, y, RedBerryBush())
                    else:
                        self.grid.set(x, y, BlueBerryBush())

        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        self.mission = "fear scenario 7"


if __name__ == "__main__":
    env = Scenario7Env(render_mode="rgb_array", tile_size=48)

    L = env.actions.left
    R = env.actions.right
    F = env.actions.forward

    orbit_second_bush = [
        _move_to(9, 11),
        _wait(0.35),
        _move_to(10, 11),
        _wait(0.35),
        _move_to(11, 11),
        _wait(0.35),
        _move_to(11, 12),
        _wait(0.35),
        _move_to(11, 13),
        _wait(0.35),
        _move_to(10, 13),
        _wait(0.35),
        _move_to(9, 13),
        _wait(0.35),
        _move_to(9, 12),
        _wait(0.35),
    ]

    actions = [
        *([F] * 5),
        L,
        *([F] * 6),
        _direct_backoff,
        R,
        R,
        *([F] * 5),
        L,
        *([F] * 4),
        _wait(1.0),
        *orbit_second_bush,
    ]

    make_scenario7_gif(env, actions, output_path="scenario7_fear.gif", fps=2.0, tile_size=48)
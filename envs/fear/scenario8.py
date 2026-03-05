# Fear_Scenario 8

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

from render.objects import EmptyBush, OrangeBerryBush
from render.walls import OuterWall, InnerWall


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
    _draw_dot(img, cx, cy, 5, (25, 25, 25))
    _draw_dot(img, cx - 5, cy - 4, 2, (210, 210, 210))
    _draw_dot(img, cx + 5, cy - 4, 2, (210, 210, 210))
    _draw_dot(img, cx, cy + 1, 2, (255, 120, 0))


def make_scenario8_gif(
    env,
    actions,
    output_path="output/scenario8_fear.gif",
    fps=2.0,
    tile_size=48,
):
    env.reset()
    frames = []
    tick = 0

    bug_orbits = [
        _clockwise_orbit((2, 2)),
    ]
    bug_phase = [0]

    def _render_with_bugs():
        frame = env.render()
        for idx, orbit in enumerate(bug_orbits):
            cell = orbit[(tick + bug_phase[idx]) % len(orbit)]
            _overlay_bug(frame, cell, tile_size)
        return frame

    def _append_frame():
        frames.append(_render_with_bugs())
        nonlocal tick
        tick += 1

    _append_frame()

    for action in actions:
        if callable(action):
            action(env)
        else:
            env.step(action)

        _append_frame()

    env.close()
    imageio.mimsave(output_path, frames, fps=fps, loop=0)


class Scenario8Env(MiniGridEnv):

    def __init__(self, max_steps=100, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "fear scenario 8")
        super().__init__(
            width=16,
            height=15,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=3,
            mission_space=mission_space,
            **kwargs,
        )
        self.agent_start_pos = (14, 2)
        self.agent_start_dir = 2

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

        # Outer walls
        for x in range(width):
            self.grid.set(x, 0, OuterWall())
            self.grid.set(x, height - 1, OuterWall())
        for y in range(height):
            self.grid.set(0, y, OuterWall())
            self.grid.set(width - 1, y, OuterWall())

        # Inner wall: 
        for col in range(6, 15):
            self.grid.set(col, 4, InnerWall())

        # Inner wall: 
        for col in list(range(1, 7)) + list(range(12, 15)):
            self.grid.set(col, 8, InnerWall())

        # Inner wall: 
        for col in range(12, 15):
            self.grid.set(col, 9, InnerWall())
        
        # Inner wall: 
        for col in [4]:
            self.grid.set(col, 12, InnerWall())

        # Inner wall: 
        for col in [4]:
            self.grid.set(col, 13, InnerWall())

        # Forest floor
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if self.grid.get(x, y) is None:
                    self.grid.set(x, y, Floor("green"))

        # Bushes
        self.grid.set(2, 2, EmptyBush())
        self.grid.set(8, 9, OrangeBerryBush())
        self.grid.set(1, 10, EmptyBush())

        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        self.mission = "find the preferred berry bush"

# GIF
if __name__ == "__main__":
    env = Scenario8Env(render_mode="rgb_array", tile_size=48)

    L = env.actions.left
    R = env.actions.right
    F = env.actions.forward

    actions = [
        *[F]* 10,
        L, *[F]* 4,
        L, *[F]* 7,
        R, *[F]* 6,
        R, *[F]* 6,
        R, *[F]* 2,
    ]

    make_scenario8_gif(env, actions, output_path="output/scenario8_fear.gif", fps=1.3, tile_size=48)
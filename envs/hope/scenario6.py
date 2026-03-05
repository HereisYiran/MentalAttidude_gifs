# Hope_Scenario 6

# file path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import imageio
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor
from minigrid.minigrid_env import MiniGridEnv

from render.objects import Bush, OrangeBerryBush, RedBerryBush, BlueBerryBush
from render.walls import OuterWall, InnerWall


def _get_highlighted_cells_s5(env):
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


def _check_berry_discovery_s5(env):
    for (x, y) in _get_highlighted_cells_s5(env):
        obj = env.grid.get(x, y)
        if isinstance(obj, Bush) and obj.berry_color is not None and not obj.discovered:
            obj.discovered = True


def make_scenario5_gif(env, actions, output_path="output/scenario5_hope.gif", fps=1.3):
    env.reset()
    frames = []

    _check_berry_discovery_s5(env)
    frames.append(env.render())

    for action in actions:
        if callable(action):
            action(env)
        else:
            env.step(action)

        _check_berry_discovery_s5(env)
        frames.append(env.render())

    env.close()
    imageio.mimsave(output_path, frames, fps=fps, loop=0)

class Scenario6Env(MiniGridEnv):

    def __init__(self, max_steps=100, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "hope scenario 6")
        super().__init__(
            width=13,
            height=13,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=3,
            mission_space=mission_space,
            **kwargs,
        )
        self.agent_start_pos = (1, 2)
        self.agent_start_dir = 0  # facing right

    def get_full_render(self, highlight, tile_size):
        # 2×3 wall-blocked view: use MiniGrid's built-in visibility
        # (respects walls) but clip to 2 rows deep instead of 3.
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

        img = self.grid.render(
            tile_size,
            agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
        )
        return img

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Outer walls (row 0, row 10, col 0, col 9)
        for x in range(width):
            self.grid.set(x, 0, OuterWall())
            self.grid.set(x, height - 1, OuterWall())
        for y in range(height):
            self.grid.set(0, y, OuterWall())
            self.grid.set(width - 1, y, OuterWall())

        # Inner wall
        for col in [1, 2, 3, 4, 5, 6]:
            self.grid.set(col, 4, InnerWall())

        # Inner wall
        for col in [4]:
            self.grid.set(col, 5, InnerWall())

        # Inner wall
        for col in [4]:
            self.grid.set(col, 6, InnerWall())

        # Inner wall
        for col in [4]:
            self.grid.set(col, 10, InnerWall())

        # Inner wall
        for col in range (4, 12):
            self.grid.set(col, 11, InnerWall())

        # Forest floor
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if self.grid.get(x, y) is None:
                    self.grid.set(x, y, Floor("green"))

        # Bushes
        self.grid.set(10, 2, RedBerryBush())       
        self.grid.set(2, 6, OrangeBerryBush())       
        self.grid.set(2, 10, RedBerryBush())         

        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        self.mission = "hope scenario 6"

# GIF

if __name__ == "__main__":
    env = Scenario6Env(render_mode="rgb_array", tile_size=48)

    L = env.actions.left
    R = env.actions.right
    F = env.actions.forward

    actions = [
        *[F]*8,
        R, *[F]*6,        
        R, *[F]*3,
    ]
    make_scenario5_gif(env, actions, output_path="output/scenario6_hope.gif", fps=1.3)
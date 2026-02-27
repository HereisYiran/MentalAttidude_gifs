# Desire_Scenario 3

# file path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor
from minigrid.minigrid_env import MiniGridEnv

from render.objects import EmptyBush, RedBerryBush, BlueBerryBush, OrangeBerryBush
from render.walls import OuterWall, InnerWall
from render.render_gif import make_gif

class Scenario3Env(MiniGridEnv):

    def __init__(self, max_steps=100, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "desire scenario 3")
        super().__init__(
            width=10,
            height=11,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=3,
            mission_space=mission_space,
            **kwargs,
        )
        self.agent_start_pos = (8, 8)
        self.agent_start_dir = 2  # facing left

    def get_full_render(self, highlight, tile_size):
        # 2×3 wall-blocked view: use MiniGrid's built-in visibility
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

        # Outer walls
        for x in range(width):
            self.grid.set(x, 0, OuterWall())
            self.grid.set(x, height - 1, OuterWall())
        for y in range(height):
            self.grid.set(0, y, OuterWall())
            self.grid.set(width - 1, y, OuterWall())

        # Inner wall: row 3, x=1,2 and x=7,8
        for col in [1, 2, 7, 8]:
            self.grid.set(col, 3, InnerWall())

        # Inner wall: row 6, x=5,6,7,8
        for col in [5, 6, 7, 8]:
            self.grid.set(col, 6, InnerWall())

        # Forest floor
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if self.grid.get(x, y) is None:
                    self.grid.set(x, y, Floor("green"))

        # Bushes
        self.grid.set(5, 1, OrangeBerryBush())  # Top bush
        self.grid.set(4, 1, BlueBerryBush())    # Left bush
        self.grid.set(6, 1, RedBerryBush())     # Right bush

        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        self.mission = "find the preferred berry bush"

# GIF

if __name__ == "__main__":
    env = Scenario3Env(render_mode="rgb_array", tile_size=48)

    L = env.actions.left
    R = env.actions.right
    F = env.actions.forward

    actions = [
        *[F]*6,           # left along row 8 → (1,8)
        R, *[F]*3,        # turn up, through gap in wall y=6 → (1,5)
        R, *[F]*3,        # turn right → (5,5)
        L, *[F]*3,        # turn up, through wall gap y=3 → (5,2)
        L,                # turn left
        *[F]*1,           # step onto (4,2) blue berry bush
        R, *[F]*1,        # turn right, step onto (5,1) orange berry bush
    ]
    make_gif(env, actions, output_path="scenario3_desire.gif", fps=1.2, tile_size=48)
    env.close()

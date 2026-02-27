import imageio
import numpy as np
from minigrid.core.grid import Grid
from render.objects import Bush
from typing import Union


def _get_highlighted_cells(env):
    """Return the set of (x, y) world cells currently in the agent's scope view.
    Matches the 2-deep clipped view used by get_full_render (skips farthest row)."""
    _, vis_mask = env.gen_obs_grid()
    f_vec = env.dir_vec
    r_vec = env.right_vec
    top_left = (
        env.agent_pos
        + f_vec * (env.agent_view_size - 1)
        - r_vec * (env.agent_view_size // 2)
    )
    cells = set()
    # Start from vis_j=1 to skip the farthest row (matching 2-deep scope)
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
    """Check if the agent's current view contains an undiscovered berry bush.
    If so, mark it discovered and return the list of newly discovered bushes."""
    newly_discovered = []
    for (x, y) in _get_highlighted_cells(env):
        obj = env.grid.get(x, y)
        if isinstance(obj, Bush) and obj.berry_color is not None and not obj.discovered:
            obj.discovered = True
            newly_discovered.append(obj)
    return newly_discovered


def _animate_discovery(env, bushes, frames, grow_steps=5):
    """Animate berry enlargement over several frames by stepping discovery_scale
    from 0.0 to 1.0. Berries stay at full size afterwards."""
    for step in range(1, grow_steps + 1):
        scale = step / grow_steps
        for bush in bushes:
            bush.discovery_scale = scale
        Grid.tile_cache.clear()  
        frames.append(env.render())


def make_gif(env, actions, output_path="scenario.gif", fps: Union[int, float] = 4, tile_size=48):
    env.reset()

    frames = []
    newly = _check_berry_discovery(env)
    frames.append(env.render())
    if newly:
        _animate_discovery(env, newly, frames)

    for action in actions:
        env.step(action)
        newly = _check_berry_discovery(env)
        frame = env.render()
        frames.append(frame)
        if newly:
            _animate_discovery(env, newly, frames)

    env.close()
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
import imageio
import numpy as np
from render.objects import Bush
from typing import Union


def _hold_frame(frames, frame, fps: Union[int, float], seconds: float = 1.0):
    extra_frames = max(0, int(round(fps * seconds)) - 1)
    for _ in range(extra_frames):
        frames.append(frame.copy())


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
    discovered = []
    for (x, y) in _get_highlighted_cells(env):
        obj = env.grid.get(x, y)
        if isinstance(obj, Bush) and obj.berry_color is not None and not obj.discovered:
            obj.discovered = True
            discovered.append(obj)
    return discovered


def make_gif(
    env,
    actions,
    output_path="scenario.gif",
    fps: Union[int, float] = 4,
    tile_size=48,
):
    env.reset()

    frames = []
    discovered = _check_berry_discovery(env)
    frame = env.render()
    frames.append(frame)
    if discovered:
        _hold_frame(frames, frame, fps, seconds=1.0)

    for action in actions:
        env.step(action)

        discovered = _check_berry_discovery(env)
        frame = env.render()
        frames.append(frame)

        if discovered:
            _hold_frame(frames, frame, fps, seconds=1.2)

    env.close()
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
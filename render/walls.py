import numpy as np
from minigrid.core.constants import COLORS, COLOR_TO_IDX
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_rect

# Register custom colors for walls
def _register_color(name, rgb):
    if name not in COLORS:
        COLORS[name] = np.array(rgb)
        COLOR_TO_IDX[name] = max(COLOR_TO_IDX.values()) + 1

_register_color("dark_brown", (80, 50, 30))
_register_color("warm_brown", (140, 100, 60))

class OuterWall(WorldObj):
    def __init__(self):
        super().__init__("wall", "dark_brown")

    def see_behind(self):
        return False

    def render(self, r: np.ndarray) -> np.ndarray:
        # Dark brown stone with mortar lines
        fill_coords(r, point_in_rect(0, 1, 0, 1), (65, 42, 24))
        fill_coords(r, point_in_rect(0.04, 0.96, 0.04, 0.96), (80, 50, 30))
        # Subtle mortar lines
        fill_coords(r, point_in_rect(0, 1, 0.48, 0.52), (55, 38, 20))
        fill_coords(r, point_in_rect(0.48, 0.52, 0, 1), (55, 38, 20))
        return r


class InnerWall(WorldObj):
    def __init__(self):
        super().__init__("wall", "warm_brown")

    def see_behind(self):
        return False

    def render(self, r: np.ndarray) -> np.ndarray:
        # Warm wooden plank look
        fill_coords(r, point_in_rect(0, 1, 0, 1), (140, 100, 60))
        fill_coords(r, point_in_rect(0.06, 0.94, 0.06, 0.94), (160, 115, 70))
        # Horizontal wood grain lines
        fill_coords(r, point_in_rect(0.06, 0.94, 0.30, 0.33), (130, 90, 50))
        fill_coords(r, point_in_rect(0.06, 0.94, 0.63, 0.66), (130, 90, 50))
        return r
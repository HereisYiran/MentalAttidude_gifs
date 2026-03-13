import numpy as np
from minigrid.core.constants import COLORS, COLOR_TO_IDX
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle


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

        fill_coords(r, point_in_rect(0, 1, 0, 1), (95, 62, 35))
        fill_coords(r, point_in_rect(0.04, 0.96, 0.04, 0.96), (115, 75, 45))
        fill_coords(r, point_in_rect(0, 1, 0.48, 0.52), (80, 55, 30))
        fill_coords(r, point_in_rect(0.48, 0.52, 0, 1), (80, 55, 30))

        return r


class InnerWall(WorldObj):

    def __init__(self):
        super().__init__("wall", "warm_brown")

    def see_behind(self):
        return False

    def render(self, r: np.ndarray) -> np.ndarray:

        # ---------- grass tile ----------
        BASE  = (126, 200, 80)
        DARK  = (109, 184, 68)
        LIGHT = (142, 212, 90)
        GRID  = (95, 165, 55)

        fill_coords(r, point_in_rect(0, 1, 0, 1), BASE)
        fill_coords(r, point_in_rect(0.12, 0.18, 0.25, 0.30), DARK)
        fill_coords(r, point_in_rect(0.65, 0.70, 0.55, 0.60), DARK)
        fill_coords(r, point_in_rect(0.22, 0.27, 0.45, 0.50), LIGHT)
        fill_coords(r, point_in_rect(0.70, 0.75, 0.30, 0.35), LIGHT)
        fill_coords(r, point_in_rect(0, 1, 0, 0.02), GRID)
        fill_coords(r, point_in_rect(0, 0.02, 0, 1), GRID)

        # ---------- rock pile (back-to-front painter's order) ----------
        ROCK_BASE  = (176, 184, 154)  # main stone fill
        ROCK_MID   = (148, 156, 128)  # mid stones
        ROCK_DARK  = (112, 120, 96)   # darkest / back stones
        ROCK_HL    = (210, 218, 190)  # highlight faces
        ROCK_SHAD  = (80,  88,  64)   # shadow faces
        ROCK_OUT   = (90,  98,  74)   # outline/stroke (approximated via border rects)

        # ground shadow
        fill_coords(r, point_in_rect(0.15, 0.85, 0.74, 0.82), (60, 70, 45))

        # --- back-left stone ---
        fill_coords(r, point_in_rect(0.10, 0.42, 0.44, 0.65), ROCK_DARK)   # body
        fill_coords(r, point_in_rect(0.12, 0.38, 0.44, 0.52), ROCK_MID)    # top face
        fill_coords(r, point_in_rect(0.13, 0.28, 0.45, 0.51), ROCK_HL)     # highlight

        # --- back-right stone ---
        fill_coords(r, point_in_rect(0.58, 0.90, 0.42, 0.63), ROCK_DARK)
        fill_coords(r, point_in_rect(0.60, 0.88, 0.42, 0.50), ROCK_MID)
        fill_coords(r, point_in_rect(0.61, 0.74, 0.43, 0.49), ROCK_HL)

        # --- back-center stone (top of pile) ---
        fill_coords(r, point_in_rect(0.32, 0.68, 0.30, 0.50), ROCK_MID)
        fill_coords(r, point_in_rect(0.34, 0.66, 0.30, 0.38), ROCK_BASE)   # top face
        fill_coords(r, point_in_rect(0.35, 0.48, 0.31, 0.37), ROCK_HL)     # highlight

        # --- front-center large boulder (main) ---
        fill_coords(r, point_in_rect(0.20, 0.80, 0.52, 0.78), ROCK_DARK)   # shadow base
        fill_coords(r, point_in_rect(0.22, 0.78, 0.48, 0.74), ROCK_BASE)   # body
        fill_coords(r, point_in_rect(0.24, 0.76, 0.48, 0.57), ROCK_MID)    # top face
        fill_coords(r, point_in_rect(0.25, 0.44, 0.49, 0.56), ROCK_HL)     # highlight left
        fill_coords(r, point_in_rect(0.58, 0.74, 0.63, 0.72), ROCK_SHAD)   # shadow right
        # beveled corners (clip the rectangle to look more polygonal)
        fill_coords(r, point_in_rect(0.20, 0.26, 0.48, 0.54), ROCK_DARK)   # cut top-left corner
        fill_coords(r, point_in_rect(0.74, 0.80, 0.48, 0.54), ROCK_DARK)   # cut top-right corner
        fill_coords(r, point_in_rect(0.20, 0.26, 0.70, 0.78), ROCK_DARK)   # cut bot-left corner
        fill_coords(r, point_in_rect(0.74, 0.80, 0.70, 0.78), ROCK_DARK)   # cut bot-right corner
        # crack
        fill_coords(r, point_in_rect(0.36, 0.64, 0.595, 0.610), ROCK_OUT)
        fill_coords(r, point_in_rect(0.49, 0.505, 0.560, 0.610), ROCK_OUT)

        # --- front-left small stone ---
        fill_coords(r, point_in_rect(0.10, 0.35, 0.63, 0.78), ROCK_DARK)
        fill_coords(r, point_in_rect(0.11, 0.34, 0.63, 0.70), ROCK_MID)
        fill_coords(r, point_in_rect(0.12, 0.22, 0.64, 0.69), ROCK_HL)

        # --- front-right pebble ---
        fill_coords(r, point_in_rect(0.66, 0.88, 0.65, 0.77), ROCK_DARK)
        fill_coords(r, point_in_rect(0.67, 0.87, 0.65, 0.72), ROCK_MID)
        fill_coords(r, point_in_rect(0.68, 0.76, 0.66, 0.71), ROCK_HL)
        return r
from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import fill_coords, point_in_circle, point_in_rect
from typing import Optional
import numpy as np

# Register Objects
def register_object(name):
    if name not in OBJECT_TO_IDX:
        OBJECT_TO_IDX[name] = max(OBJECT_TO_IDX.values()) + 1
        IDX_TO_OBJECT[OBJECT_TO_IDX[name]] = name

register_object("bush")
register_object("red_berry_bush")
register_object("blue_berry_bush")
register_object("orange_berry_bush")

def draw_grass_tile(r):
    BASE  = (126,200,80)   # #7ec850
    DARK  = (109,184,68)   # #6db844
    LIGHT = (142,212,90)   # #8ed45a
    GRID  = (95,165,55)

    # base
    fill_coords(r, point_in_rect(0,1,0,1), BASE)

    # darker patches
    fill_coords(r, point_in_rect(0.12,0.18,0.25,0.30), DARK)
    fill_coords(r, point_in_rect(0.65,0.70,0.55,0.60), DARK)
    fill_coords(r, point_in_rect(0.35,0.40,0.65,0.70), DARK)

    # lighter patches
    fill_coords(r, point_in_rect(0.22,0.27,0.45,0.50), LIGHT)
    fill_coords(r, point_in_rect(0.70,0.75,0.30,0.35), LIGHT)

    # grid lines
    fill_coords(r, point_in_rect(0,1,0,0.02), GRID)
    fill_coords(r, point_in_rect(0,0.02,0,1), GRID)

# Bushes
class Bush(WorldObj):
    BERRY_COLORS = {
        "red":    (220, 30, 30),
        "blue":   (30, 80, 220),
        "orange": (255, 140, 0),
    }
    def __init__(self, berry_color: Optional[str] = None):
        super().__init__("bush", "green")
        self.berry_color = berry_color
        self.discovered = False
        self.discovery_scale = 0  # 0 = undiscovered, 1 = fully discovered

    def can_overlap(self):
        return True

    def render(self, r: np.ndarray) -> np.ndarray:
        draw_grass_tile(r)


        fill_coords(r, point_in_rect(0.44, 0.56, 0.58, 0.82), (90, 55, 20))
        DARK_GREEN  = (30, 110, 30)
        MID_GREEN   = (50, 160, 50)
        LIGHT_GREEN = (80, 200, 80)
        fill_coords(r, point_in_circle(0.50, 0.44, 0.32), DARK_GREEN)
        fill_coords(r, point_in_circle(0.28, 0.50, 0.24), DARK_GREEN)
        fill_coords(r, point_in_circle(0.72, 0.50, 0.24), DARK_GREEN)
        fill_coords(r, point_in_circle(0.50, 0.38, 0.26), MID_GREEN)
        fill_coords(r, point_in_circle(0.31, 0.44, 0.18), MID_GREEN)
        fill_coords(r, point_in_circle(0.69, 0.44, 0.18), MID_GREEN)
        fill_coords(r, point_in_circle(0.50, 0.28, 0.16), LIGHT_GREEN)

        if self.berry_color is not None:
            bc = self.BERRY_COLORS.get(self.berry_color, (200, 0, 0))
            hi = tuple(min(255, c + 90) for c in bc)

            for (cx, cy) in [(0.37, 0.47), (0.57, 0.41), (0.48, 0.56)]:
                fill_coords(r, point_in_circle(cx, cy, 0.075), bc)
                fill_coords(r, point_in_circle(cx - 0.02, cy - 0.025, 0.026), hi)
        return r

class EmptyBush(Bush):
    def __init__(self): 
        super().__init__(berry_color=None)

# Berries
class RedBerryBush(Bush):
    def __init__(self): 
        super().__init__(berry_color="red")
        self.type = "red_berry_bush"

    def render(self, r: np.ndarray) -> np.ndarray:
        super().render(r)
        RED, RED_HI, RED_DARK = (220, 30, 30), (255, 160, 160), (140, 0, 0)
        s = self.discovery_scale
        radius = 0.13 + s * 0.12
        ri = radius - 0.02
        rh = 0.04 + s * 0.04  
        for (cx, cy) in [(0.30, 0.52), (0.55, 0.45), (0.50, 0.62)]:
            fill_coords(r, point_in_circle(cx, cy, radius), RED_DARK)
            fill_coords(r, point_in_circle(cx, cy, ri), RED)
            fill_coords(r, point_in_circle(cx - 0.03, cy - 0.03, rh), RED_HI)
        return r

class BlueBerryBush(Bush):
    def __init__(self):
        super().__init__(berry_color="blue")
        self.type = "blue_berry_bush"

    def render(self, r: np.ndarray) -> np.ndarray:
        super().render(r)
        BLUE, BLUE_HI, BLUE_DARK = (30, 80, 220), (160, 180, 255), (0, 30, 140)
        s = self.discovery_scale
        radius = 0.13 + s * 0.12
        ri = radius - 0.02
        rh = 0.04 + s * 0.04
        for cx, cy in [(0.30, 0.52), (0.55, 0.45), (0.50, 0.62)]:
            fill_coords(r, point_in_circle(cx, cy, radius), BLUE_DARK)
            fill_coords(r, point_in_circle(cx, cy, ri), BLUE)
            fill_coords(r, point_in_circle(cx - 0.03, cy - 0.03, rh), BLUE_HI)
        return r


class OrangeBerryBush(Bush):
    def __init__(self):
        super().__init__(berry_color="orange")
        self.type = "orange_berry_bush"

    def render(self, r: np.ndarray) -> np.ndarray:
        super().render(r)
        ORANGE, ORANGE_HI, ORANGE_DARK = (255, 140, 0), (255, 210, 140), (180, 80, 0)
        s = self.discovery_scale
        radius = 0.13 + s * 0.12
        ri = radius - 0.02
        rh = 0.04 + s * 0.04
        for cx, cy in [(0.30, 0.52), (0.55, 0.45), (0.50, 0.62)]:
            fill_coords(r, point_in_circle(cx, cy, radius), ORANGE_DARK)
            fill_coords(r, point_in_circle(cx, cy, ri), ORANGE)
            fill_coords(r, point_in_circle(cx - 0.03, cy - 0.03, rh), ORANGE_HI)
        return r

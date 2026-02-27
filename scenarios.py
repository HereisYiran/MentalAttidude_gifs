## BELIEF
# scenario_1

SCENARIOS = {
    "belief_1": {
        "grid": [
            "WWWWWWWWWW",
            "W.......BW",  # B at (9,2)
            "W........W",
            "W..WWWWWWW",
            "W........W",
            "WB.......W",  # B at (2,6)
            "WWWWWWW..W",
            "W........W",
            "WG......BW",  # G at (2,9), B at (9,9)
            "WWWWWWWWWW",
        ],
        "path": [(2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9)],
    },
    "belief_2": {
        "grid": [
            "WWWWWWWWWWW",
            "W........WW",
            "W.......GWW",  # G at (8,2)
            "W........WW",
            "W..WWWWWWWW",
            "W........WW",
            "WB.......WW",  # B at (1,6)
            "WWWWWWW..WW",
            "W........WW",
            "WB......BWW",  # B at (1,9), B at (8,9)
            "WWWWWWWWWWW",
        ],
        "path": [(8, 2), (7, 2), (6, 2), (5, 2), (4, 2), (3, 2), (2, 2), (1, 2)],
    },
    "desire_3": {
        "grid": [
            "WWWWWWW",
            "W.BBBWW",  # B at (2,2),(3,2),(4,2)
            "W....WW",
            "WWW.WWW",
            "W....WW",
            "W....WW",
            "W.WWWWW",
            "W....WW",
            "W...GWW",  # G at (4,9)
            "WWWWWWW",
        ],
        "path": [(4, 9), (4, 8), (4, 7), (4, 6), (4, 5), (4, 4), (4, 3), (3, 2)],
    },
    "desire_4": {
        "grid": [
            "WWWWWWW",
            "WGXXXXW",  # G at (1,2) — col labels start at 1
            "W....WW",
            "WWWWW.W",
            "W....WW",
            "W....WW",
            "W.WWWWW",
            "W....BWW",  # B at (5,8)
            "W....BWW",  # B at (5,9)
            "W....BWW",  # B at (5,10)
            "WWWWBWW",   # B at (4,11)
            "WWWWWWW",
        ],
        "path": [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10)],
    },
    "hope_5": {
        "grid": [
            "WWWWWWW",
            "WGXXXXW",  # G at (1,2)
            "W....WW",
            "WWWWW.W",
            "WB....W",  # B at (1,5)
            "W....WW",
            "W.WWWWW",
            "W....BWW",  # B at (5,8)
            "W....WW",
            "W....BWW",  # B at (5,10)
            "WWWWBWW",   # B at (4,11)
            "WWWWWWW",
        ],
        "path": [(1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (5, 6), (5, 7), (5, 8)],
    },
    "hope_6": {
        "grid": [
            "WWWWWWWWW",
            "WGXXXXXXW",  # G at (1,2)
            "WWWWWW.BW",  # B at (7,3)
            "WWWW....W",
            "WWWW...WW",
            "W......WW",
            "W......WW",
            "W.WW.WWWW",
            "WBWWBWWWW",  # B at (1,9), B at (4,9)
            "WWWWWWWWW",
        ],
        "path": [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (5, 7), (4, 8), (3, 9), (2, 9), (1, 9)],
    },
    "fear_7": {
        "grid": [
            "WWWWWWWWWWW",
            "W........WW",
            "W.......BWW",  # B at (8,2)
            "W........WW",
            "W..WWWWWWWW",
            "W.B......WW",  # B at (2,5)
            "W........WW",
            "WWWWWWW..WW",
            "W........WW",
            "WG.....B.WW",  # G at (1,9), B at (7,9)
            "WWWWWWWWWWW",
        ],
        "path": [(1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9)],
    },
    "fear_8": {
        # Grid not provided in PDF — placeholder
        "grid": [
            "WWWWWWWWWWW",
            "W........WW",
            "W...B....WW",  # B at (4,2)
            "W........WW",
            "W........WW",
            "W........WW",
            "WB.......WW",  # B at (1,6) — actually (2,6) per description
            "W........WW",
            "W....B...WW",  # B at (5,8)
            "W........WW",
            "WWWWWWWWWWW",
        ],
        "path": [],  # path not specified
    },
}
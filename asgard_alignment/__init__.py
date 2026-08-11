BALDR_ALLOWED_PHASEMASK_POSITIONS = ["J1", "J2", "J3", "J4", "H2", "H3", "H4", "H5", 
                                    #  "J5", "H1",
                                     "LL", "CLEAR"
                                     ]

"""Phasemask positions to use if the incoming JSON does not contain them.

This is structured like BALDR_PHASEMASK_INITIAL_POSITIONS[mask_no][posn]
"""
BALDR_PHASEMASK_INITIAL_POSITIONS = {
    1: {
        "LL": [7640, 4000],
        "CLEAR": [4640, 1000],
    },
    2: {
        "LL": [9171, 3090],
        "CLEAR": [6171, 90],
    },
    3: {
        "LL": [7630, 3745],
        "CLEAR": [4630, 745],
    },
    4: {
        "LL": [1720, 4360],
        "CLEAR": [4720, 7360],
    },
}

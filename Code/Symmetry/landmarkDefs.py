
SD_MAX_NORM_VALUE = 30000
SD_MIN_NORM_VALUE = 0

MOUTH_SIZE_MAX_NORM_VALUE = 26000
MOUTH_SIZE_MIN_NORM_VALUE = 25000

SD_FILTER_CONST = 0.5
LANDMARK_FILTER_CONST = 0.5

#up-down landmarks sets of 2 points, to pronounce the vertical symmetry of a face.
LIPS_VERTICAL_LANDMARK_SYMMETRY = [

                            #Center Line - 3 (Left)
                            (40,91),
                            (74,90),
                            (42,89),
                            #(80,88),

                            #Center Line - 2 (Left)
                            (39,181),
                            (73,180),
                            (41,179),
                            #(81,178),

                            #Center Line - 1 (Left)
                            (37,84),
                            (72,85),
                            (38,86),
                            #(82,87),

                            #Center Line
                            (0,17),
                            (11,16),
                            (12,15),
                            #(13,14),

                            #center line + 1 (Right)
                            (267,314),
                            (302,315),
                            (268, 316),
                            #(312,317),

                            #center line + 2 (Right)
                            (269,405),
                            (303,404),
                            (271,403),
                            #(311,402),

                            #center line + 3 (Right)
                            (270,321),
                            (304,320),
                            (272,319),
                            #(310,318)
                        ]

#right-left landmarks sets of 2 points, to pronounce the Horizontal symmetry of a face.
LIPS_HORIZONTAL_LANDMARK_SYMMETRY = [
                            #most external segment, big numbers on the right (upper point 0, lower point 17)
                            (267, 37),
                            (269, 39),
                            (270, 40),
                            (409, 185), 
                            #(287, 57),
                            (375, 146),
                            (321, 91),
                            (405, 181),
                            (314, 84),

                            # #2nd external (upper point 11, lower point 16)
                            # (302, 72),
                            # (303, 73),
                            # (304, 74),
                            # (408, 184),
                            # (307, 77),
                            # (320, 90),
                            # (404, 180),
                            # (315, 85),

                            # #3rd external (upper point 12, lower point 15 )
                            # (268, 38),
                            # (271, 41),
                            # (272, 42),
                            # (407, 183),
                            # (325, 96),
                            # (319, 89),
                            # (403, 179),
                            # (316, 86),

                            #4th external (upper point 13, lower point 14 )
                            (312, 82),
                            (311, 81),
                            (310, 80),
                            (415, 191),
                            (317, 87),
                            (402, 178),
                            (318, 88),
                            (324, 95),
                            (308, 78) ]

#TODO: not in use, remove later
LIPS_LANDMARKS = [ 61,
                  146,
                  91,
                  181,
                  84,
                  17,
                  314,
                  405,
                  321,
                  375,
                  291,
                  185,
                  40,
                  39,
                  37,
                  0,
                  267,
                  269,
                  270,
                  409,
                  78,
                  95,
                  88,
                  178,
                  87,
                  14,
                  317,
                  402,
                  318,
                  324,
                  308,
                  191,
                  80,
                  81,
                  82,
                  13,
                  312,
                  311,
                  310,
                  415]

UP_MARKER = 9
DOWN_MARKER = 94
LEFT_MARKER = 78 # left lips point
RIGHT_MARKER = 308 # right lips point
MOUTH_UPPER_LIP_MIN_HEIGHT = 13
MOUTH_LOWER_LIP_MAX_HEIGHT = 14

FACE_GUIDE = [ MOUTH_UPPER_LIP_MIN_HEIGHT, MOUTH_LOWER_LIP_MAX_HEIGHT, LEFT_MARKER, RIGHT_MARKER, UP_MARKER, DOWN_MARKER ] #Middle of forhead to endn of nose ref. line + lips up and down

MY_FACE_CONNECTIONS = frozenset([
    # Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308)
])

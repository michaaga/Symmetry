#Alg Settings
useNormalizedLandmarks = True  # Alg: work in normalized landmarks domain, or image landmark domain.
filterLandmarks        = True  # Alg: filter landmark points position between frames based on previous position
ignoreSmallMouthSize   = True  # Alg: ignore small mouth size calculation to remove jitter.

#Display Settings
normalizeOutputSD  = True        # Display: normalize SD output values for display
filterAngleOutputs = True        # Display: filter SD Angles output display
filterSDOutputs    = False       # Display: filter SD & MS for output display
createVideoOutput  = False       # Display: join all frames to output video

MIN_MOUTH_SIZE_FOR_SD_CALC = 0.2

# Image Const Values
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1920
IMAGE_LOAD_SKIP_CNT = 1
IMAGE_WRITE_SKIP_CNT = 1

# Normalization Factors
SD_MAX_NORM_VALUE = 100
SD_MIN_NORM_VALUE = 0
MOUTH_SIZE_MAX_NORM_VALUE = 100
MOUTH_SIZE_MIN_NORM_VALUE = 0
NORM_VAR = 10

#Filter Const
SD_FILTER_CONST = 0.5
LANDMARK_FILTER_CONST = 0.5

## Landmark Markers and Tuples
LEFT_LIPS_MARKER = 78 # left lips point
RIGHT_LIPS_MARKER = 308 # right lips point

UPPER_LIP_MIN = 13
UPPER_LIP_MAX = 0

LOWER_LIP_MIN = 17
LOWER_LIP_MAX = 14

LEFT_LIPS_REF_1 = 80
LEFT_LIPS_REF_2 = 88
RIGHT_LIPS_REF_1 = 310
RIGHT_LIPS_REF_2 = 318

LIPS_GUIDE_SYMMETRY_POINTS = [
    (LEFT_LIPS_MARKER, RIGHT_LIPS_MARKER),
    (UPPER_LIP_MIN, LOWER_LIP_MAX),
    (LOWER_LIP_MIN, UPPER_LIP_MAX),
    (LEFT_LIPS_REF_1, LEFT_LIPS_REF_2),
    (RIGHT_LIPS_REF_1, RIGHT_LIPS_REF_2)
]

#up-down landmarks sets of 2 points, to pronounce the vertical symmetry of a face.
LIPS_VERTICAL_LANDMARK_SYMMETRY = [
    (37,84),
    (267,314),
    (39,181),
    (269,405),
    (40,91),
    (270,321), 

    (82, 87), 
    (81, 178),
    (80, 88),
    (191, 95), 
    (312, 317), 
    (311, 402), 
    (310, 318),
    (324, 415)
]

#right-left landmarks sets of 2 points, to pronounce the Horizontal symmetry of a face.
LIPS_HORIZONTAL_LANDMARK_SYMMETRY = [
    (84, 314),
    (37, 267),
    (181,405),
    (39,269),
    (40,270),
    (91,321),

    (82, 312),
    (81, 311),
    (80, 310),
    (191, 415),
    (87, 317), 
    (178, 402), 
    (88, 318),
    (95, 324)
 ]

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

#FACE_GUIDE = [ MOUTH_UPPER_LIP_MIN_HEIGHT, MOUTH_LOWER_LIP_MAX_HEIGHT, LEFT_MARKER, RIGHT_MARKER, UP_MARKER, DOWN_MARKER ] #Middle of forehead to end of nose ref. line + lips up and down

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

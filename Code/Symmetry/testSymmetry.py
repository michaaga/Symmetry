import Symmetry
import utils
import random
import cv2
import math
import landmarkDefs

#return random values in H/W ranges
def randX(): return random.randint(5,1080 - 5)
def randY(): return random.randint(5,Symmetry.HEIGHT - 5)

#test Symmetry distance of points
testImgPath = 'C:\\GIT\\Symmetry\\TestImages\\test_image.jpg'

def testReflectPoint():
    global img 
    img = cv2.imread(testImgPath)
    
    random.seed()

    #first point is the center of image, the second is random.
    p1 = {'X': Symmetry.WIDTH / 2, 'Y': Symmetry.HEIGHT / 2}
    p2 = {'X': randX(), 'Y': randY()}

    #draw the symmetry line testpoints.
    img = cv2.drawMarker(img, ((int)(p1['X']), (int)(p1['Y'])) , (0, 255, 0), 0, 30)
    img = cv2.drawMarker(img, ((int)(p2['X']), (int)(p2['Y'])) , (0, 255, 0), 0, 30)

    #draw the actual symmetry line on the image.
    img = cv2.line(img, ((int)(p1['X']), (int)(p1['Y'])), ((int)(p2['X']), (int)(p2['Y'])), (0, 0, 0), 5)

    #test for 50 points
    for i in range(1, 1000):
        testPt = {'X': randX(), 'Y': randY()}
        reflectedPoint =  Symmetry.reflectPoint(p1, p2, testPt)
        refBackPoint = Symmetry.reflectPoint(p1, p2, reflectedPoint)
        diff = math.sqrt(Symmetry.pointsPairSqrDistance(testPt, refBackPoint))

        if(diff > 0.01):
            return False
        
        draw = False
        if draw:
            #only draw points that have both original and reflection on the image.
            if(Symmetry.inRange(testPt) and Symmetry.inRange(reflectedPoint)):
                img = cv2.drawMarker(img, ((int)(testPt['X']), (int)(testPt['Y'])) , (255, 0, 0), 0, 30)
                img = cv2.drawMarker(img, ((int)(reflectedPoint['X']), (int)(reflectedPoint['Y'])) , (0, 0, 0), 0, 30)

                cv2.imshow('Image', img)
                utils.resize_and_show(img, True)
                cv2.waitKey()

    return True

def testCheckSymmetryOfLine():
    global img 
    random.seed(10)

    img = cv2.imread(testImgPath)
    center = {'X': Symmetry.WIDTH / 2, 'Y': Symmetry.HEIGHT / 2}
    utils.annotatePoint(img, center, str('C'), (0, 255, 0))


    points = {}

    #create test points across the horizonal line
    for pair in landmarkDefs.LIPS_LANDMARK_SYMMTERY:
        pt1 = pair[0]
        pt2 = pair[1]

        deltaX =  random.randint(0, Symmetry.WIDTH /2)
        deltaY =  random.randint(0, Symmetry.HEIGHT /2)
        dstX = center['X'] + deltaX
        dstY = center['Y'] + deltaY
        points[pt1] = {'X': dstX, 'Y': dstY}

        dstX = center['X'] - deltaX
        points[pt2] = {'X': dstX, 'Y': dstY}

    #calculate Dst point to angle
    angle = 90
    lineLength = 500

    dstX = center['X'] + lineLength * math.cos(math.radians(angle))
    dstY = center['Y'] + lineLength * math.sin(math.radians(angle))
    dst = {'X': dstX, 'Y': dstY}

    #draw Symmetry line
    img = cv2.line(img, ((int)(center['X']), (int)(center['Y'])), ((int)(dst['X']), (int)(dst['Y'])), (0, 0, 0), 5)

    #calculate Symmetry line SD
    val = Symmetry.checkSymmetryOfLine(img, center, dst, points)
    print("value is: %d", val)

    printing = False
    if(printing):
        utils.resize_and_show(img, True)

    if(val > 0.001):
        return False

    return True

def testNormalizeLandmarks():
    global img 
    random.seed(10)

    points = {}
    VAR = 100

    #create test points
    for i in range(1,1000):
        pt1 = {'X': randX(), 'Y': randY()}
        pt2 = {'X': randX(), 'Y': randY()}
        points[i] = pt1
        points[i] = pt2

    pointsCenter = Symmetry.centerMass(points)

    list, scale = Symmetry.normalizeLandmarks(points, VAR)
    NormMean = Symmetry.centerMass(list)

    #calculate the avg sqr distance to the center.
    totalDistance = 0
    for p in list.items():
        totalDistance += Symmetry.pointsPairSqrDistance(NormMean, p[1])

    #avg the sqr distance
    stdSqr = totalDistance / len(list)
    std = math.sqrt(stdSqr)

    if(VAR - std > 0.001):
        return False

    return True

def runAllTests():

    result = True
    result &= testReflectPoint()
    result &= testCheckSymmetryOfLine()
    result &= testNormalizeLandmarks()

    if(result == False):
        print('Tests Failed!')
    else:
        print('Tests Passed!')

    return

#degugging code
#*******************************

#images = {}
#utils.extractImagesFromVideo('C:\\GIT\\Symmetry\\TestVideos', images, True)

#run All Tests Manually
#runAllTests()
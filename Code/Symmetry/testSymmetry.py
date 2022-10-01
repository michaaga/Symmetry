from statistics import NormalDist
from sqlalchemy import false
import Symmetry
import utils
import random
import cv2
import math
import landmarkDefs
import matplotlib.pylab as plt


#return random values in H/W ranges
def randX(): return random.randint(5,Symmetry.WIDTH - 5)
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

def testVerticalSymmetryOfLine():
    global img 
    random.seed(10)

    img = cv2.imread(testImgPath)
    center = {'X': Symmetry.WIDTH / 2, 'Y': Symmetry.HEIGHT / 2}
    utils.annotatePoint(img, center, str('C'), (0, 255, 0))

    srcLinePoint = {'X': center['X'] - 400, 'Y': center['Y']}
    dstLinePoint = {'X': center['X'] + 400, 'Y': center['Y']}
    
    #draw Symmetry line
    utils.drawLineOnImage(img, srcLinePoint, dstLinePoint)

    #create test points across the horizonal line
    points = {}
    for pair in landmarkDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY:
        pt1 = pair[0]
        pt2 = pair[1]

        PointX =  random.randint(0, Symmetry.WIDTH)
        deltaY =  random.randint(0, Symmetry.HEIGHT /2)
        points[pt1] = {'X': PointX, 'Y': center['Y'] + deltaY}
        points[pt2] = {'X': PointX, 'Y': center['Y'] - deltaY}
        utils.annotatePoint(img, points[pt1])
        utils.annotatePoint(img, points[pt2])
        utils.drawLineOnImage(img, points[pt1], points[pt2])

    #calculate Symmetry line SD
    val = Symmetry.checkSymmetryOfLine(img, srcLinePoint, dstLinePoint, points, landmarkDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY )

    #if __debug__:
    #    utils.resize_and_show(img, True)

    if(val > 0.001):
        return False

    return True

def testHorizonatalSymmetryOfLine():
    global img 
    random.seed(10)

    img = cv2.imread(testImgPath)
    center = {'X': Symmetry.WIDTH / 2, 'Y': Symmetry.HEIGHT / 2}
    utils.annotatePoint(img, center, str('C'), (0, 255, 0))

    srcLinePoint = {'X': center['X'], 'Y': center['Y']  - 400}
    dstLinePoint = {'X': center['X'], 'Y': center['Y']  + 400}
    
    #draw Symmetry line
    utils.drawLineOnImage(img, srcLinePoint, dstLinePoint)

    #create test points across the horizonal line
    points = {}
    for pair in landmarkDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY:
        pt1 = pair[0]
        pt2 = pair[1]

        PointY =  random.randint(0, Symmetry.HEIGHT)
        deltaX =  random.randint(0, Symmetry.WIDTH /2)
        points[pt1] = {'X': center['X'] + deltaX , 'Y': PointY}
        points[pt2] = {'X': center['X'] - deltaX , 'Y': PointY}
        utils.annotatePoint(img, points[pt1])
        utils.annotatePoint(img, points[pt2])
        utils.drawLineOnImage(img, points[pt1], points[pt2])

    #calculate Symmetry line SD
    val = Symmetry.checkSymmetryOfLine(img, srcLinePoint, dstLinePoint, points, landmarkDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY )

    #if __debug__:
    #    utils.resize_and_show(img, True)

    if(val > 0.001):
        return False

    return True

def testNormalizeLandmarks():
    global img 
    random.seed(10)

    points = {}
    VAR = 300
    testListX = []
    testListY = []

    #create & plot test points
    for i in range(1,1000):
        pt = {'X': randX(), 'Y': randY()}
        points[i] = pt
        testListX.append(pt['X'])
        testListY.append(pt['Y'])

#   plt.scatter(testListX, testListY)
#   plt.show()

    #Normalize and plot normalized points
    testListXNorm = []
    testListYNorm = []
    Normlist, NormScale, var1 = Symmetry.normalizeLandmarks(points, VAR)
    for item in Normlist.items():
        testListXNorm.append(item[1]['X'])
        testListYNorm.append(item[1]['Y'])

    diff = abs(VAR - var1)
    if  diff > 0.005:
        return false

#   plt.scatter(testListXNorm, testListYNorm)
#   plt.show()

    return True

def runAllTests():

    result = True
    for i in range(1, 100):
        result &= testReflectPoint()
        result &= testVerticalSymmetryOfLine()
        result &= testHorizonatalSymmetryOfLine()
        result &= testNormalizeLandmarks()
        if(result == False):
            print('Tests Failed!')
            return

    print('Tests Passed!')

    return

#degugging code
#*******************************

#images = {}
#utils.extractImagesFromVideo('C:\\GIT\\Symmetry\\TestVideos', images, True)

#run All Tests Manually
runAllTests()
#testNormalizeLandmarks()
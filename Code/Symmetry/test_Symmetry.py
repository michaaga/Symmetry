import Symmetry
import utils
import random
import cv2
import math
import projectDefs
import matplotlib.pylab as plt


#return random values in H/W ranges
def randX(): return random.randint(5, projectDefs.IMAGE_WIDTH - 5)
def randY(): return random.randint(5, projectDefs.IMAGE_HEIGHT - 5)

#test Symmetry distance of points
testImgPath = 'C:\\GIT\\Symmetry\\Code\\Symmetry\\test_image.jpg'

def testReflectPoint():
    global img 
    img = cv2.imread(testImgPath)   
    random.seed()

    #first point is the center of image, the second is random.
    p1 = {'X': projectDefs.IMAGE_WIDTH / 2, 'Y': projectDefs.IMAGE_HEIGHT / 2}
    p2 = {'X': randX(), 'Y': randY()}

    #draw the symmetry line test points.
    img = cv2.drawMarker(img, ((int)(p1['X']), (int)(p1['Y'])) , (0, 255, 0), 0, 30)
    img = cv2.drawMarker(img, ((int)(p2['X']), (int)(p2['Y'])) , (0, 255, 0), 0, 30)

    #draw the actual symmetry line on the image.
    img = cv2.line(img, ((int)(p1['X']), (int)(p1['Y'])), ((int)(p2['X']), (int)(p2['Y'])), (0, 0, 0), 5)

    #test for 50 points
    for i in range(1, 10000):
        testPt = {'X': randX(), 'Y': randY()}
        reflectedPoint =  Symmetry.reflectPoint(p1, p2, testPt)
        refBackPoint = Symmetry.reflectPoint(p1, p2, reflectedPoint)
        diff = math.sqrt(Symmetry.pointsPairSqrDistance(testPt, refBackPoint))

        if(abs(diff) > 0.01):
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
    center = {'X': projectDefs.IMAGE_WIDTH / 2, 'Y': projectDefs.IMAGE_HEIGHT / 2}
    utils.annotatePoint(img, center, str('C'), (0, 255, 0))

    srcLinePoint = {'X': center['X'] - 400, 'Y': center['Y']}
    dstLinePoint = {'X': center['X'] + 400, 'Y': center['Y']}
    
    #draw Symmetry line
    utils.drawLineOnImage(img, srcLinePoint, dstLinePoint)

    #create test points across the horizontal line
    points = {}
    for pair in projectDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY:
        pt1 = pair[0]
        pt2 = pair[1]

        PointX =  random.randint(0, projectDefs.IMAGE_WIDTH)
        deltaY =  random.randint(0, projectDefs.IMAGE_HEIGHT /2)
        points[pt1] = {'X': PointX, 'Y': center['Y'] + deltaY}
        points[pt2] = {'X': PointX, 'Y': center['Y'] - deltaY}
        utils.annotatePoint(img, points[pt1])
        utils.annotatePoint(img, points[pt2])
        utils.drawLineOnImage(img, points[pt1], points[pt2])

    #calculate Symmetry line SD
    val = Symmetry.checkSymmetryOfLine(img, srcLinePoint, dstLinePoint, points, projectDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY )

    #if __debug__:
    #    utils.resize_and_show(img, True)

    if(abs(val) > 0.001):
        return False

    return True

def testHorizontalSymmetryOfLine():
    global img 
    random.seed(10)

    img = cv2.imread(testImgPath)
    center = {'X': projectDefs.IMAGE_WIDTH / 2, 'Y': projectDefs.IMAGE_HEIGHT / 2}
    utils.annotatePoint(img, center, str('C'), (0, 255, 0))

    srcLinePoint = {'X': center['X'], 'Y': center['Y']  - 400}
    dstLinePoint = {'X': center['X'], 'Y': center['Y']  + 400}
    
    #draw Symmetry line
    utils.drawLineOnImage(img, srcLinePoint, dstLinePoint)

    #create test points across the horizontal line
    points = {}
    for pair in projectDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY:
        pt1 = pair[0]
        pt2 = pair[1]

        PointY =  random.randint(0, projectDefs.IMAGE_HEIGHT)
        deltaX =  random.randint(0, projectDefs.IMAGE_WIDTH /2)
        points[pt1] = {'X': center['X'] + deltaX , 'Y': PointY}
        points[pt2] = {'X': center['X'] - deltaX , 'Y': PointY}
        utils.annotatePoint(img, points[pt1])
        utils.annotatePoint(img, points[pt2])
        utils.drawLineOnImage(img, points[pt1], points[pt2])

    #calculate Symmetry line SD
    val = Symmetry.checkSymmetryOfLine(img, srcLinePoint, dstLinePoint, points, projectDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY )

    #if __debug__:
    #    utils.resize_and_show(img, True)

    if(abs(val) > 0.001):
        return False

    return True

def testNormalizeLandmarks():
    global img 
    random.seed(10)
    VAR = 100
    numOfPoints = 1500

    srcPoints = {}
    dstPoints = {}
    
    testListX = []
    testListY = []

    testListXNorm = []
    testListYNorm = []

    #create & plot test points including center of mass
    for i in range(1,numOfPoints):
        pt = {'X': randX(), 'Y': randY()}
        srcPoints[i] = pt
        testListX.append(pt['X'])
        testListY.append(pt['Y'])

    center = Symmetry.centerMass(srcPoints)
    #plt.scatter([center['X']], [center['Y']], color = 'hotpink')
    #plt.scatter(testListX, testListY)
    #plt.show()

    #Normalize and plot normalized points
    Normlist, var1 = Symmetry.normalizeLandmarks(srcPoints, VAR)
    for item in Normlist.items():
        testListXNorm.append(item[1]['X'])
        testListYNorm.append(item[1]['Y'])

    normCenter = Symmetry.centerMass(Normlist)

    #plt.scatter([normCenter['X']], [normCenter['Y']], color = 'red')
    #plt.scatter(testListXNorm, testListYNorm)
    #plt.show()

    if abs(VAR - var1) > 0.005:
       return False

    if abs(center['X'] - normCenter['X']) > 0.005:
        return False

    if abs(center['Y'] - normCenter['Y']) > 0.005:
        return False
    
    return True

def runAllTests():

    result = True
    numOfIterations = 10
    for i in range(1, numOfIterations):
        result &= testReflectPoint()
        result &= testVerticalSymmetryOfLine()
        result &= testHorizontalSymmetryOfLine()
        result &= testNormalizeLandmarks()

        if(result == False):
            print('Test #' + str(i) + ' Failed!')
            return
        else:
            print('Iteration #' + str(i) + '/' + str(numOfIterations) + ' Completed Successfully.')


    print('All Tests Passed!')

    return

#debugging code
#*******************************

#images = {}
#utils.extractImagesFromVideo('C:\\GIT\\Symmetry\\TestVideos', images, True)

#run All Tests Manually
#runAllTests()
#testNormalizeLandmarks()
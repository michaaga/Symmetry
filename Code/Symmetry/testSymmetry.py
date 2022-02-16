import Symmetry
import utils
import random
import cv2
import math

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
    for i in range(1, 100):
        testPt = {'X': randX(), 'Y': randY()}
        reflectedPoint =  Symmetry.reflectPoint(p1, p2, testPt)
        refBackPoint = Symmetry.reflectPoint(p1, p2, reflectedPoint)
        diff = math.sqrt(Symmetry.pointsPairSqrDistance(testPt, refBackPoint))

        if(diff < 0.01):
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

def testCalcSD():
    global img
    global idx

    img = cv2.imread(testImgPath)
    random.seed()

    #first point is the center of image, the second is random.
    center = {'X': Symmetry.WIDTH / 2, 'Y': Symmetry.HEIGHT / 2}
    dst = {'X': randX(), 'Y': randY()}

    #draw the actual symmetry line on the image.
    img = cv2.line(img, ((int)(center['X']), (int)(center['Y'])), ((int)(dst['X']), (int)(dst['Y'])), (0, 0, 0), 5)

    totalDistance = 0
    #test for 50 points
    counts = 0
    idx = 0

    for i in range(100):
        p0 = {'X': randX(), 'Y': randY()}
        p1 = {'X': randX(), 'Y': randY()}

        #make sure points are on both sides of symmetry line
        totalDistance+= Symmetry.calcSD(p0, p1, center, dst)
        counts+=1

    print('found:' + str(counts) + ' counts')
    utils.resize_and_show(img, True)
    cv2.waitKey()

    return

def testCheckSymmetryOfLine():
    global img 
    global idx
    random.seed(10)

    #first point is the center of image, the second is random.
    center = {'X': Symmetry.WIDTH / 2, 'Y': Symmetry.HEIGHT / 2}
    #dst = {'X': randX(), 'Y': randY()}
    lineLength = 100

    points = {}

    #create test points
    for i in range(1,100):
        pt1 = {'X': randX(), 'Y': randY()}
        pt2 = {'X': randX(), 'Y': randY()}
        points[i[0]] = pt1
        points[i[1]] = pt2

    #reset the image to see the best SD
    img = cv2.imread(testImgPath)

    #calculate Dst point to angle
    angle = 30
    dstX = center['X'] + lineLength * math.cos(math.radians(angle))
    dstY = center['Y'] + lineLength * math.sin(math.radians(angle))
    dst = {'X': dstX, 'Y': dstY}

    #draw Symmetry line
    img = cv2.line(img, ((int)(center['X']), (int)(center['Y'])), ((int)(dst['X']), (int)(dst['Y'])), (0, 0, 0), 5)
    idx = 0

    #calculate Symmetry line SD
    val = Symmetry.checkSymmetryOfLine(img, center, dst, points)

    #if(VAR - val > 1.0):
    #    print("Value:" + str(val))
    #    return False

    return 

def testNormalizeLandmarks():
    global img 
    global idx
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

    if(VAR - std < 0.001):
        return False

    return True


#degugging code
#**************
#testReflectPoint()


#images = {}
#utils.extractImagesFromVideo('C:\\GIT\\Symmetry\\TestVideos', images, True)

#run Debug code

#testReflectPoint()
#testCalcSD()
testCheckSymmetryOfLine()
#testNormalizeLandmarks()
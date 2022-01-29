from xmlrpc.client import MAXINT
import cv2
from cv2 import minEnclosingTriangle
import utils
import random
import math

#Reflect point p along line through points p0 and p1
#param p point to reflect
#param p0 first point for reflection line
#param p1 second point for reflection line
def reflectPoint(p0, p1, p):
    #var dx, dy, a, b, x, y;
    dx = p1['X'] - p0['X']
    dy = p1['Y'] - p0['Y']
    a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
    b = 2 * dx * dy / (dx * dx + dy * dy)
    x = round(a * (p['X'] - p0['X']) + b * (p['Y'] - p0['Y']) + p0['X'])
    y = round(b * (p['X'] - p0['X']) - a * (p['Y'] - p0['Y']) + p0['Y'])

    return { 'X':x, 'Y':y }

WIDTH = 1080
HEIGHT = 1920

#return random values in H/W ranges
def randX(): return random.randint(5,1080 - 5)
def randY(): return random.randint(5,HEIGHT - 5)

#check if point is in the image range
def inRange(pt):
    if((pt['X'] > 0) and (pt['X'] < WIDTH) and (pt['Y'] > 0) and (pt['Y'] < HEIGHT)):
        return True
    else:
        return False

#return avg ot 2 (X,Y) Dict. points
def avgPts(p1,p2):
    return { 'X': (int)((p1['X'] + p2['X']) / 2.0), 'Y': (int)((p1['Y'] + p2['Y']) / 2.0) }

#lineSrc = line point 1; lineDst = line point 2; pt = point to check against.
def isPointLeftOfLine(lineSrc, lineDst, pt):
    return ((lineDst['X'] - lineSrc['X'])*(pt['Y'] - lineSrc['Y']) - (lineDst['Y'] - lineSrc['Y'])*(pt['X'] - lineSrc['X'])) > 0

#returns symmetry distance between any point and its Symmetry Transform
def pointsPairSD(p1, p2):
    return math.dist([p1['X'], p1['Y']], [p2['X'], p2['Y']])**2

#calculate the symmetry distance between a pair of points
def calcSD(p0, p1, centerPoint, dst):

#from the Alg:
#a) The two points {P0,P1} are folded to obtain {P0~,P1~}
#b) Points P0~ and P1~ are averaged to obtain P0^.
#c) P1^ is obtained by reflecting P0^ about the symmetry axis.
#d) return SD of the two original points vs (P0^,P1^) <-> (P0,P1)
  global img
  global idx 

  reflectedPoint = reflectPoint(centerPoint, dst, p1)
  
  p0_ = avgPts(p0, reflectedPoint)
  p1_ = reflectPoint(centerPoint, dst, p0_)

#   if(inRange(p0_) and inRange(p1_)):
#     #draw the symmetry line testpoints.
#     utils.annotatePoint(img, p0, str(idx), (0, 0, 0))
#     utils.annotatePoint(img, p1, str(idx+1), (0, 0, 0))

#     utils.annotatePoint(img, p0_, str(idx) + '*', (0, 0, 255))
#     utils.annotatePoint(img, p1_, str(idx+1) + '*', (0, 0, 255))
#     idx +=2

  return pointsPairSD(p0, p0_) + pointsPairSD(p1, p1_) 

#calculate SD for a specific symmetry line
def checkLineSD(center, dst, points):
    
    rightPts = []
    leftPts  = []

    #create test points
    for pt in points:
        if(isPointLeftOfLine(center, dst, pt)):
            leftPts.append(pt)
        else:
            rightPts.append(pt)

    assert len(points) == (len(leftPts) + len(rightPts))

    totalSD = 0
    for leftItem in leftPts:
        for rightItem in rightPts:
            totalSD += calcSD(leftItem, rightItem, center, dst)

    return totalSD / len(points)

def checkAllSymmetryLines(img, center, points):
    lineLength = 100
    minVal = 2**64
    minAngle = 0

    #run  10 lines across the circle and check for minimal SD
    for angle in range(0, 360, 1):

        #calculate Dst point to angle
        dstX = center['X'] + (int)(lineLength * math.cos(math.radians(angle)))
        dstY = center['Y'] + (int)(lineLength * math.sin(math.radians(angle)))
        dst = {'X': dstX, 'Y': dstY}

        #draw Symmetry line
        #img = cv2.line(img, ((int)(center['X']), (int)(center['Y'])), ((int)(dst['X']), (int)(dst['Y'])), (0, 0, 0), 5)

        #calculate Symmetry line SD
        val = checkLineSD(center, dst, points)
        #print("Angle: " + str(angle) + ", Value:" + str(val))
        
        #look for the min
        if(val < minVal):
            minVal = val
            minAngle = angle #save the min result corresponding angle
            #utils.resize_and_show(img, True)
            #print("Min Value:" + str(minAngle) + '\n')

    return minVal, minAngle

#degugging code
#**************
#testReflectPoint()

#test Symmetry distance of points
def testReflectPoint():
    global img 
    img = cv2.imread('C:\\GIT\\Symmetry\\TestImages\\N12_02_MS_20.jpg')
    
    random.seed()

    #first point is the center of image, the second is random.
    p1 = {'X': WIDTH / 2, 'Y': HEIGHT / 2}
    p2 = {'X': randX(), 'Y': randY()}

    #draw the symmetry line testpoints.
    img = cv2.drawMarker(img, ((int)(p1['X']), (int)(p1['Y'])) , (0, 255, 0), 0, 30)
    img = cv2.drawMarker(img, ((int)(p2['X']), (int)(p2['Y'])) , (0, 255, 0), 0, 30)

    #draw the actual symmetry line on the image.
    img = cv2.line(img, ((int)(p1['X']), (int)(p1['Y'])), ((int)(p2['X']), (int)(p2['Y'])), (0, 0, 0), 5)

    #test for 50 points
    for i in range(50):
        testPt = {'X': randX(), 'Y': randY()}
        reflectedPoint = reflectPoint(p1, p2, testPt)

        #only draw points that have both original and reflection on the image.
        if(inRange(testPt) and inRange(reflectedPoint)):
            img = cv2.drawMarker(img, ((int)(testPt['X']), (int)(testPt['Y'])) , (255, 0, 0), 0, 30)
            img = cv2.drawMarker(img, ((int)(reflectedPoint['X']), (int)(reflectedPoint['Y'])) , (0, 0, 0), 0, 30)

    #cv2.imshow('Image', img)
    utils.resize_and_show(img, True)
    cv2.waitKey()

    return

def testCalcSD():
    global img
    global idx

    img = cv2.imread('C:\\GIT\\Symmetry\\TestImages\\N12_02_MS_20.jpg')
    random.seed()

    #first point is the center of image, the second is random.
    center = {'X': WIDTH / 2, 'Y': HEIGHT / 2}
    dst = {'X': randX(), 'Y': randY()}

    #draw the actual symmetry line on the image.
    img = cv2.line(img, ((int)(center['X']), (int)(center['Y'])), ((int)(dst['X']), (int)(dst['Y'])), (0, 0, 0), 5)

    totalDistance = 0
    #test for 50 points
    counts = 0
    idx = 0

    for i in range(30):
        p0 = {'X': randX(), 'Y': randY()}
        p1 = {'X': randX(), 'Y': randY()}

        #make sure points are on both sides of symmetry line
        if(isPointLeftOfLine(center, dst, p0) != isPointLeftOfLine(center, dst, p1)): 
            totalDistance+= calcSD(p0, p1, center, dst)
            counts+=1

    print('found:' + str(counts) + ' counts')
    utils.resize_and_show(img, True)
    cv2.waitKey()

    return

def testCheckLineSD():
    global img 
    global idx
    random.seed(10)

    #first point is the center of image, the second is random.
    center = {'X': WIDTH / 2, 'Y': HEIGHT / 2}
    #dst = {'X': randX(), 'Y': randY()}
    lineLength = 100

    points = []

    #create test points
    for i in range(50):
        testPt = {'X': randX(), 'Y': randY()}
        points.append(testPt)

    minVal = 2**64

    #run  10 lines across the circle and check for minimal SD
    for angle in range(0, 360, 36):

        #reset the image to see the best SD
        img = cv2.imread('C:\\GIT\\Symmetry\\TestImages\\N12_02_MS_20.jpg')

        #calculate Dst point to angle
        dstX = center['X'] + (int)(lineLength * math.cos(math.radians(angle)))
        dstY = center['Y'] + (int)(lineLength * math.sin(math.radians(angle)))
        dst = {'X': dstX, 'Y': dstY}

        #draw Symmetry line
        img = cv2.line(img, ((int)(center['X']), (int)(center['Y'])), ((int)(dst['X']), (int)(dst['Y'])), (0, 0, 0), 5)
        idx = 0

        #calculate Symmetry line SD
        val = checkLineSD(center, dst, points)        
        print("Angle: " + str(angle) + ", Value:" + str(val))
        
        #look for the min
        if(val < minVal):
            minVal = val
            utils.resize_and_show(img, True)
            print("Min Value:" + str(minVal) + '\n')

    return minVal

#run Debug code

#testCalcSD()
#testReflectPoint()
#testCheckLineSD()
import cv2
import utils
import random

#brief Reflect point p along line through points p0 and p1
#param p point to reflect
#param p0 first point for reflection line
#param p1 second point for reflection line
#return object
def  reflectPoint(p0, p1, p):
    #var dx, dy, a, b, x, y;
    dx = p1['X'] - p0['X']
    dy = p1['Y'] - p0['Y']
    a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
    b = 2 * dx * dy / (dx * dx + dy * dy)
    x = round(a * (p['X'] - p0['X']) + b * (p['Y'] - p0['Y']) + p0['X'])
    y = round(b * (p['X'] - p0['X']) - a * (p['Y'] - p0['Y']) + p0['Y'])

    return { 'X':x, 'Y':y }
    
def testFunc():
    img = cv2.imread('C:\\GIT\\Symmetry\\TestImages\\N12_02_MS_20.jpg')
    
    p1 = {'X': 100.0, 'Y': 100.0}
    p2 = {'X': 200.0, 'Y': 200.0}

    testPt = {'X': 100.0, 'Y': 300.0}

    utils.drawLineOnImage(img,p1,p2)
    img = cv2.drawMarker(img, ((int)(testPt['X']), (int)(testPt['Y'])) , (255, 0, 0), 0, 10)
    reflectedPoint = reflectPoint(p1, p2, testPt)

    img = cv2.drawMarker(img, ((int)(reflectedPoint['X']), (int)(reflectedPoint['Y'])) , (0, 0, 255), 0, 10)


    cv2.imshow('Image', img)
    cv2.waitKey()

    return


WIDTH = 1080
HEIGHT = 1920

def randX(): return random.randint(5,1080 - 5)
def randY(): return random.randint(5,HEIGHT - 5)

def inRange(pt):
    if((pt['X'] > 0) and (pt['X'] < WIDTH) and (pt['Y'] > 0) and (pt['Y'] < HEIGHT)):
        return True
    else:
        return False

def testFunc2():
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

testFunc2()

import cv2
import landmarkDefs
import utils
import random
import math

WIDTH = 1080
HEIGHT = 1920
IMAGE_LOAD_SKIP_CNT = 1
IMAGE_WRITE_SKIP_CNT = 25

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
    x = a * (p['X'] - p0['X']) + b * (p['Y'] - p0['Y']) + p0['X']
    y = b * (p['X'] - p0['X']) - a * (p['Y'] - p0['Y']) + p0['Y']

    return { 'X':x, 'Y':y }

#check if point is in the image range
def inRange(pt):
    if((pt['X'] > 0) and (pt['X'] < WIDTH) and (pt['Y'] > 0) and (pt['Y'] < HEIGHT)):
        return True
    else:
        return False

#return avg ot 2 (X,Y) Dict. points
def avgPts(p1,p2):
    return { 'X': (p1['X'] + p2['X']) / 2.0, 'Y': (p1['Y'] + p2['Y']) / 2.0 }

#lineSrc = line point 1; lineDst = line point 2; pt = point to check against.
def isPointLeftOfLine(lineSrc, lineDst, pt):
    return ((lineDst['X'] - lineSrc['X'])*(pt['Y'] - lineSrc['Y']) - (lineDst['Y'] - lineSrc['Y'])*(pt['X'] - lineSrc['X'])) > 0

#returns symmetry distance between any point and its Symmetry Transform
def pointsPairSqrDistance(p1, p2):
    return math.dist([p1['X'], p1['Y']], [p2['X'], p2['Y']])**2

#calculate the symmetry distance between a pair of points
def calcSD(p0, p1, src, dst, img = ''):

#from the Alg:
#a) The two points {P0,P1} are folded to obtain {P0~,P1~}
#b) Points P0~ and P1~ are averaged to obtain P0^.
#c) P1^ is obtained by reflecting P0^ about the symmetry axis.
#d) return SD of the two original points vs (P0^,P1^) <-> (P0,P1)
  reflectedPoint = reflectPoint(src, dst, p1)
  
  p0_ = avgPts(p0, reflectedPoint)
  p1_ = reflectPoint(src, dst, p0_)

  showPoints = False

  if(inRange(p0_) and inRange(p1_) and showPoints == True):
     #draw the symmetry line testpoints.
     utils.annotatePoint(img, p0, str('0'), (0, 0, 0))
     utils.annotatePoint(img, p1, str('1'), (0, 0, 0))
     utils.annotatePoint(img, reflectedPoint, str('ref'), (0, 0, 0))

     utils.annotatePoint(img, p0_, str('avg'), (0, 0, 255))
     utils.annotatePoint(img, p1_, str('refBack'), (0, 0, 255))

  return (pointsPairSqrDistance(p0, p0_) + pointsPairSqrDistance(p1, p1_)) / 2.0 

#calculate SD for a specific symmetry line
def checkSymmetryOfLine(img, src, dst, points, symmetryIndexes):

    #go over all symmtery matche Indexes and calculate symmetry over the line (src,dst)
    totalSD = 0
    for pair in symmetryIndexes:
        pt1 = pair[0]
        pt2 = pair[1]

        #calculate Symmetry line SD
        totalSD += calcSD(points[pt1], points[pt2], src, dst, img)

    return totalSD

#find points center of mass
def centerMass(points):
      
  center_x = 0.0
  center_y = 0.0

  for p in points.items():
    center_x += p[1]['X']
    center_y += p[1]['Y']

  center_x = center_x / len(points)
  center_y = center_y / len(points)

  return {'X': center_x,'Y': center_y}

def normalizeLandmarks(landmarkList, var):

    mean = centerMass(landmarkList)

    #calculate the avg sqr distance to the center.
    totalDistance = 0
    for p in landmarkList.items():
        totalDistance += pointsPairSqrDistance(mean, p[1])

    #Square STD value
    stdSqr = totalDistance / len(landmarkList)

    normalizedList = {}
    scale = var / math.sqrt(stdSqr)
    for p in landmarkList.items():
        point = {'X': p[1]['X'] * scale, 'Y': p[1]['Y'] * scale}
        normalizedList[p[0]] =  point
 
    return normalizedList, scale



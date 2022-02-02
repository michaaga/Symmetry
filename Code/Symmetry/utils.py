import mediapipe as mp
import cv2
import os
import random
import math
import landmarkDefs

DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 1920
IMAGE_LOAD_SKIP_CNT = 20

#Extract images from video and save to disk (optional)
def extractImagesFromVideo(path, images, saveToDisk = False):

  i = 0 #Skip frams to shorten debugging.

  #create output folder if not exists
  outPath = os.path.join(path, 'Images')
  os.makedirs(outPath,exist_ok=True)

  #go over all files in directory and extract images by frame index
  for filename in os.listdir(path):  

      laodStr = os.path.join(path, filename)
      print('Loading Video:' + laodStr)

      # Opens the Video file
      cap= cv2.VideoCapture(laodStr)
      i=0

      while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == False:
              break

          i+=1
          if(i % IMAGE_LOAD_SKIP_CNT != 0):
            continue

          fName = os.path.splitext(filename)[0] +  '_' + str(i)
          images[fName] = frame    
          i+=1

          if(saveToDisk):
            cv2.imwrite(os.path.join(outPath, fName + '.jpg'), frame)
              
      cap.release()
      cv2.destroyAllWindows()

#load images from folder instead of video for debugging
def load_images_from_folder(folder, images):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
          images[filename] = img

#create random color [RGB]
def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

#resize image to desired ratio
def resize_and_show(image, showImages = False):
    h, w = image.shape[:2]
    if h < w:
      img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
      img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

    if(showImages):
        cv2.imshow('Image',img)
    
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
        
        # closing all open windows
        cv2.destroyAllWindows()

#add caption on an image
def addTextOnImage(image, text, pos):
  font = cv2.FONT_HERSHEY_SIMPLEX
  return cv2.putText(image, text, pos, font, 1, (0, 255, 0), 2, cv2.LINE_AA)    

#draw line from (x,y) to (x,y) on an image.
def drawLineOnImage(image, src, dest, color = (0,0,0)):
  height, width, channels = image.shape
  image = cv2.line(image, ((int)(src['X']), (int)(src['Y'])), ((int)(dest['X']), (int)(dest['Y'])), color, 2)

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

#convert point to image coordinates
def convertPoint(landmarks, idx):
  point = { 'X': (int)(landmarks.landmark[idx].x * DESIRED_HEIGHT), 'Y': (int)(landmarks.landmark[idx].y * DESIRED_WIDTH) }
  return point

#get relevant landmarks only (image coordinates)
def getFilteredLandmarkData(landmarks, filterSet):
  keypoints = {}
  for idx in filterSet:
      keypoints[idx] = convertPoint(landmarks, idx)

  return keypoints

#debug: test method to check position of landmark points..
def printLandmarkPoints(landmarkImagePoints, img):
  font = cv2.FONT_HERSHEY_SIMPLEX
  for x in landmarkDefs.LIPS_LANDMARK_SYMMTERY:
    img = cv2.drawMarker(img, ((int)(landmarkImagePoints[x[0]]['X']), (int)(landmarkImagePoints[x[0]]['Y'])) , (255,0,0), 0, 3)
    img = cv2.drawMarker(img, ((int)(landmarkImagePoints[x[1]]['X']), (int)(landmarkImagePoints[x[1]]['Y'])) , (0,0,255), 0, 3)
    #img = cv2.putText(img, str(x[0]), ((int)(x[1]['X']), (int)(x[1]['Y'])), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)  

def annotatePoint(img, pt, text = '', color = (0,0,0)):
  font = cv2.FONT_HERSHEY_SIMPLEX
  img = cv2.drawMarker(img, ((int)(pt['X']), (int)(pt['Y'])) , color, 0, 30)
  img = cv2.putText(img, text, ((int)(pt['X']), (int)(pt['Y'])), font, 0.7, color, 1, cv2.LINE_AA)  

def angleBetweenPoints(pt1, pt2):
    x1 = pt1['X']
    y1 = pt1['Y']
    x2 = pt2['X']
    y2 = pt2['Y']
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.degrees(math.acos(inner_product/(len1*len2)))

    """Get the angle of this line with the horizontal axis."""

    """Get the angle of this line with the horizontal axis."""

    """Get the angle of this line with the horizontal axis."""

#Get the angle of this line with the horizontal axis.
def get_angle(p1, p2):

    dx = p2['X'] - p1['X']
    dy = p2['Y']- p1['Y']
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)  # angle is in (-180, 180]
    if angle < 0:
        angle = 360 + angle
    return (int)(angle)
import mediapipe as mp
import cv2
import os
import random
import math
import landmarkDefs
import Symmetry

DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 1920

#Extract images from a single video and save to disk (optional)
def extractImagesFromVideo(videoPath, outPath, images, saveToDisk = False):

  i = 0 #Skip frams to shorten debugging.

  # Opens the Video file
  cap= cv2.VideoCapture(videoPath)
  
  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == False:
          break

      if(i % Symmetry.IMAGE_LOAD_SKIP_CNT != 0):
        continue

      fName = os.path.splitext(os.path.basename(videoPath))[0] + '_' + str(i)    
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

#fill the images dictionary
def getImages(path, outPath,  images, debugOption = 'VIDEO_FILE'):

#  debugOption = 'VIDEO_FILE'
#  debugOption = 'IMAGE_FOLDER'
#  debugOption = 'ONE_IMAGE'

  if (debugOption == 'VIDEO_FILE'):
        extractImagesFromVideo(path, outPath, images, False)
  elif (debugOption == 'IMAGE_FOLDER'):
    load_images_from_folder(path, images)
  else:
      img = cv2.imread(path)
      if img is not None:
        images['Test Image'] = img

    # Preview the images.
  for name, image in images.items():
    if(debugOption == 'ONE_IMAGE'):
      # Show the image loaded for single image test
      resize_and_show(image, True)
    else:
      resize_and_show(image) 

  return

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
  return cv2.putText(image, text, pos, font, 1, (0, 0, 0), 2, cv2.LINE_AA)    

#draw line from (x,y) to (x,y) on an image.
def drawLineOnImage(image, src, dest, scale, color = (0,0,0)):
  height, width, channels = image.shape
  image = cv2.line(image, ((int)(src['X'] / scale), (int)(src['Y'] / scale)), ((int)(dest['X'] / scale ), (int)(dest['Y'] / scale)), color, 2)

#convert point to image coordinates
def convertPoint(landmarks, idx):
  point = { 'X': landmarks.landmark[idx].x * DESIRED_HEIGHT, 'Y': landmarks.landmark[idx].y * DESIRED_WIDTH }
  return point

#get relevant landmarks only (image coordinates)
def getFilteredLandmarkData(landmarks, filterSet):
  keypoints = {}
  for idx in filterSet:
      keypoints[idx] = landmarks[idx]

  return keypoints

def getAllLandmarksData(landmarks):
  keypoints = {}
  for idx in range(0,468):
      keypoints[idx] = convertPoint(landmarks, idx)

  return keypoints

#debug: test method to check position of landmark points..
def printLandmarkPoints(landmarkImagePoints, scale, img):
  font = cv2.FONT_HERSHEY_SIMPLEX
  for x in landmarkDefs.LIPS_LANDMARK_SYMMTERY:
    img = cv2.drawMarker(img, ((int)(landmarkImagePoints[x[0]]['X'] / scale), (int)(landmarkImagePoints[x[0]]['Y'] / scale)) , (255,0,0), 0, 3)
    img = cv2.drawMarker(img, ((int)(landmarkImagePoints[x[1]]['X'] / scale), (int)(landmarkImagePoints[x[1]]['Y'] / scale)) , (0,0,255), 0, 3)
    #img = cv2.putText(img, str(x[0]),((int)(landmarkImagePoints[x[0]]['X']), (int)(landmarkImagePoints[x[0]]['Y'])), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)  
    #img = cv2.putText(img, str(x[0]),((int)(landmarkImagePoints[x[1]]['X']), (int)(landmarkImagePoints[x[1]]['Y'])), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)  

  return

#add text and marker on point
def annotatePoint(img, pt, text = '', color = (0,0,0)):
  font = cv2.FONT_HERSHEY_SIMPLEX
  img = cv2.drawMarker(img, ((int)(pt['X']), (int)(pt['Y'])) , color, 0, 30)
  img = cv2.putText(img, text, ((int)(pt['X']), (int)(pt['Y'])), font, 0.7, color, 1, cv2.LINE_AA)  

#calculate angle between two lines
def angleBetweenPoints(pt1, pt2):
    x1 = pt1['X']
    y1 = pt1['Y']
    x2 = pt2['X']
    y2 = pt2['Y']
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.degrees(math.acos(inner_product/(len1*len2)))

#Get the angle of this line with the horizontal axis.
def get_angle(p1, p2):
    dx = p2['X'] - p1['X']
    dy = p2['Y']- p1['Y']
    theta = math.atan2(dy, dx)
    angle = math.degrees(theta)  # angle is in (-180, 180]
    if angle < 0:
        angle = 360 + angle
    return angle

def createLandmarkList():
  landmarkList = []
  for x in landmarkDefs.LIPS_LANDMARK_SYMMTERY:
    if x[0] in landmarkList or x[1] in landmarkList:
      print ("duplicate found!!!!!!!!!!")
      return

    else:  
      landmarkList.append(x[0])
      landmarkList.append(x[1])
    
  return landmarkList
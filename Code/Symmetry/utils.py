import cv2
import os
import random
import math
import projectDefs

imageXtextLocation = 50
imageYtextLocation = 0

#Extract images from a single video and save to disk (optional)
def extractImagesFromVideo(videoPath, outPath, images, saveToDisk = False):

  # Opens the Video file
  cap= cv2.VideoCapture(videoPath)
  
  i = -1 #Skip frames to shorten debugging.
  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == False:
          break

      i+=1
      if(i % projectDefs.IMAGE_LOAD_SKIP_CNT != 0):
        continue

      fName = os.path.splitext(os.path.basename(videoPath))[0] + '_' + str(i)    

      # TODO: Rotate all videos manually and remove this code.
      # Using cv2.rotate() method
      # Using cv2.ROTATE_90_CLOCKWISE rotate
      # by 90 degrees clockwise
      frame = cv2.rotate(frame, cv2.ROTATE_180)
      images[fName] = frame

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
      img = cv2.resize(image, (projectDefs.IMAGE_WIDTH, math.floor(h/(w/projectDefs.IMAGE_WIDTH))))
    else:
      img = cv2.resize(image, (math.floor(w/(h/projectDefs.IMAGE_HEIGHT)), projectDefs.IMAGE_HEIGHT))

    if(showImages):
        cv2.imshow('Image',img)
    
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
        
        # closing all open windows
        cv2.destroyAllWindows()

#add caption on an image
def addTextOnImage(image, text, resetLocation = False):
  global imageXtextLocation
  global imageYtextLocation

  if(resetLocation):
    imageYtextLocation = 50
  else:
    imageYtextLocation = imageYtextLocation + 50 
  
  font = cv2.FONT_HERSHEY_SIMPLEX
  pos = (imageXtextLocation, imageYtextLocation)
  return cv2.putText(image, text, pos, font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)    

#draw line from (x,y) to (x,y) on an image.
def drawLineOnImage(image, src, dest, color = (0,0,0)):
  #height, width, channels = image.shape
  image = cv2.line(image, ((int)(src['X']), (int)(src['Y'])), ((int)(dest['X']), (int)(dest['Y'])), color, 1)

#convert point to image coordinates
def convertPointToImageDim(landmarks, idx):
  point = { 'X': landmarks.landmark[idx].x * projectDefs.IMAGE_WIDTH, 'Y': landmarks.landmark[idx].y * projectDefs.IMAGE_HEIGHT }
  return point

#get relevant landmarks only (image coordinates)
def getFilteredLandmarkData(landmarks, filterSet):
  keyPoints = {}
  for idx in filterSet:
      keyPoints[idx] = landmarks[idx]

  return keyPoints

def getAllLandmarksData(landmarks):
  keyPoints = {}
  for idx in range(0, len(landmarks.landmark)):
      keyPoints[idx] = convertPointToImageDim(landmarks, idx)

  return keyPoints

#method to check position of landmark points..
def printLandmarkPoints(landmarkImagePoints, img, normSet = False):
  font = cv2.FONT_HERSHEY_SIMPLEX
  fontSize = 0.4 # Was 0.3

  if(normSet):
    verticalColor = (255,0,0)
    horizontalColor = (0,0,255)
  else:
    verticalColor = (0,255,0)
    horizontalColor = (125,0,125)

  for x in projectDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY:
    img = cv2.drawMarker(img, ((int)(landmarkImagePoints[x[0]]['X']), (int)(landmarkImagePoints[x[0]]['Y'])) , verticalColor, 0, 3)
    img = cv2.drawMarker(img, ((int)(landmarkImagePoints[x[1]]['X']), (int)(landmarkImagePoints[x[1]]['Y'])) , verticalColor, 0, 3)
#   img = cv2.putText(img, str(x[0]),((int)(landmarkImagePoints[x[0]]['X']), (int)(landmarkImagePoints[x[0]]['Y'])), font, fontSize, (255, 0, 0), 1, cv2.LINE_AA)  
#   img = cv2.putText(img, str(x[1]),((int)(landmarkImagePoints[x[1]]['X']), (int)(landmarkImagePoints[x[1]]['Y'])), font, fontSize, (255, 0, 0), 1, cv2.LINE_AA)  

  for x in projectDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY:
    img = cv2.drawMarker(img, ((int)(landmarkImagePoints[x[0]]['X']), (int)(landmarkImagePoints[x[0]]['Y'])) , horizontalColor, 0, 3)
    img = cv2.drawMarker(img, ((int)(landmarkImagePoints[x[1]]['X']), (int)(landmarkImagePoints[x[1]]['Y'])) , horizontalColor, 0, 3)
#   img = cv2.putText(img, str(x[0]),((int)(landmarkImagePoints[x[0]]['X']), (int)(landmarkImagePoints[x[0]]['Y'])), font, fontSize, (255, 0, 0), 1, cv2.LINE_AA)  
#   img = cv2.putText(img, str(x[1]),((int)(landmarkImagePoints[x[1]]['X']), (int)(landmarkImagePoints[x[1]]['Y'])), font, fontSize, (255, 0, 0), 1, cv2.LINE_AA)  

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
#   if angle < 0:
#       angle = 360 + angle
    return angle

def createLandmarkList():
  landmarkList = []
  for x in projectDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY:
    if not (x[0] in landmarkList):
      landmarkList.append(x[0])
      
    if not (x[1] in landmarkList):
      landmarkList.append(x[1])
    
  for y in projectDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY:
    if not (y[0] in landmarkList):
      landmarkList.append(y[0])

    if not (y[1] in landmarkList):
      landmarkList.append(y[1])

  for y in projectDefs.LIPS_GUIDE_SYMMETRY_POINTS:
    if not (y[0] in landmarkList):
      landmarkList.append(y[0])

    if not (y[1] in landmarkList):
      landmarkList.append(y[1])    

  return landmarkList

#return normalized value in values from normMin to normMax
def normalizeList(values, normMin, normMax):
    valMin = min(values)
    valRange = max(values) - valMin

    for i in range(len(values)):
        values[i] = normMin + (((values[i] - valMin) * (normMax - normMin)) / valRange)

#filter list by ratio
def filterList(list, ratio):
  tempList = []
  for i in range(len(list)):
    if(i == 0): #skip first element
      tempList.append(list[i])
      continue

    tempList.append(list[i] * ratio + tempList[i-1] * (1 - ratio))

  return tempList

#filter Dictionary by prev Dict. and a given Ratio
def filterDictionary(dict, prevDict, ratio):
  for i in dict.keys():
    dict[i]['X']  = dict[i]['X'] * ratio + prevDict[i]['X'] * (1 - ratio)
    dict[i]['Y']  = dict[i]['Y'] * ratio + prevDict[i]['Y'] * (1 - ratio)
import mediapipe as mp
import cv2
import os
import random
import math

DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 1920

#Extract images from video and save to disk (optional)
def extractImagesFromVideo(path, images, saveToDisk = False):

  i = 0 #Skip frams to shorten debugging.
  imagesSkipCount = 20

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
          if(i%imagesSkipCount != 0):
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
def addTextOnImage(image, text):
  font = cv2.FONT_HERSHEY_SIMPLEX
  return cv2.putText(image, text, (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)    

#draw line from (x,y) to (x,y) on an image.
def drawLineOnImage(image, src, dest):
  height, width, channels = image.shape
  image = cv2.line(image, ((int)(src['X']), (int)(src['Y'])), ((int)(dest['X']), (int)(dest['Y'])), random_color(), 3)

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
  point = { 'X': landmarks.landmark[idx].x * DESIRED_HEIGHT, 'Y': landmarks.landmark[idx].y * DESIRED_WIDTH }
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
  for x in landmarkImagePoints.items():
    img = cv2.drawMarker(img, ((int)(x[1]['X']), (int)(x[1]['Y'])) , (255, 0, 0), 0, 10)
    img = cv2.putText(img, str(x[0]), ((int)(x[1]['X']), (int)(x[1]['Y'])), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA)  
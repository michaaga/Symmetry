import matplotlib.pyplot as plt
from logging import exception
import mediapipe as mp
import numpy as np
import cv2
import os

import utils
import Symmetry
import projectDefs

images = {} #global images dictionary

videoFolderPath = 'C:\\GIT\\Symmetry\\Data\\TestVideos' #Movement_sense_video
ImagesOutPath = 'C:\\GIT\\Symmetry\\Data\\TestImages' #create image landmarks output folder

# Load drawing_utils and drawing_styles
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

prevLandmarks = {}

#Run Face Mesh landmark detection on a single image.
def MpFaceMesh(image):

  circleDrawingSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

  #Run FaceMesh on each image to get landmarks
  with mp_face_mesh.FaceMesh(
      static_image_mode=False,        #Was True, unrelated images (True) or as a video stream (False)
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.8,
      min_tracking_confidence = 0.9) as face_mesh:

    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
      raise exception('missing faces')

    #draw landmarks and connections
    for face_landmarks in results.multi_face_landmarks:
      #Draw Mesh
      if(1):
          mp_drawing.draw_landmarks(
          image=image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())

      #draw frame lines
      if(0):
        mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=projectDefs.MY_FACE_CONNECTIONS,
        landmark_drawing_spec=circleDrawingSpec,#None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())

  return results.multi_face_landmarks[0]

#Calculate single image symmetry distance
def imageSymmetry(image, name, landmarkList):

  #run FaceMesh to get All landmark points
  #get image coordinates Landmarks
  imageLandmarks = utils.getAllLandmarksData(MpFaceMesh(image))
  selectedLandmarks = utils.getFilteredLandmarkData(imageLandmarks, landmarkList)
 
  if projectDefs.filterLandmarks:
    global prevLandmarks
    if len(prevLandmarks) > 0:
      utils.filterDictionary(selectedLandmarks, prevLandmarks, projectDefs.LANDMARK_FILTER_CONST)
    prevLandmarks = selectedLandmarks.copy()

  #convert Image points to Normalized Scale of NORM_VAR
  if projectDefs.useNormalizedLandmarks:
    selectedLandmarks, Var = Symmetry.normalizeLandmarks(selectedLandmarks, projectDefs.NORM_VAR)

  center = Symmetry.centerMass(selectedLandmarks)
  utils.annotatePoint(image, center, "", (255, 0, 0))

  leftRightRefLineSrc, leftRightRefLineDst = GetHorLineRefLinePoints(selectedLandmarks, center)

  #draw Vertical ref line from Image Landmarks
  #leftRightRefLineSrc = selectedLandmarks[projectDefs.LEFT_LIPS_MARKER]
  #leftRightRefLineDst = selectedLandmarks[projectDefs.RIGHT_LIPS_MARKER]
  
  
  leftRightRefLineAngle = utils.get_angle(leftRightRefLineSrc, leftRightRefLineDst)
  utils.drawLineOnImage(image, leftRightRefLineSrc, leftRightRefLineDst)

  #draw Horizontal ref line from Image Landmarks
  upDownRefLineSrc = selectedLandmarks[projectDefs.LOWER_LIP_MAX]
  upDownRefLineDst = selectedLandmarks[projectDefs.UPPER_LIP_MIN]
  upDownRefLineAngle = utils.get_angle(upDownRefLineSrc, upDownRefLineDst) + 90 #offset vertical line to zero
  utils.drawLineOnImage(image, upDownRefLineSrc, upDownRefLineDst)

  #find and print landmarks center
  ms = abs(selectedLandmarks[projectDefs.UPPER_LIP_MIN]['Y'] - selectedLandmarks[projectDefs.LOWER_LIP_MAX]['Y'])

  skipFrame = False

  HorizontalSD = Symmetry.checkSymmetryOfLine(image, upDownRefLineSrc, upDownRefLineDst, selectedLandmarks, projectDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY)
  VerticalSD = Symmetry.checkSymmetryOfLine(image, leftRightRefLineSrc, leftRightRefLineDst, selectedLandmarks, projectDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY)

  if(ms < projectDefs.MIN_MOUTH_SIZE_FOR_SD_CALC and projectDefs.ignoreSmallMouthSize):  
    skipFrame = True

  #Embedded Text labels On Images
  utils.addTextOnImage(image, name, True)
  utils.addTextOnImage(image, 'Mouth Size = ' + str(ms))
  utils.addTextOnImage(image, 'Vertical SD : ' + str(VerticalSD))
  utils.addTextOnImage(image, 'Horizontal Face Line Angle = ' + str(leftRightRefLineAngle))
  utils.addTextOnImage(image, 'Horizontal SD : ' + str(HorizontalSD))
  utils.addTextOnImage(image, 'Vertical Face Line Angle = ' + str(upDownRefLineAngle))

  #plot the landmarks on top of the image
  utils.printLandmarkPoints(selectedLandmarks, image)

  return ms, VerticalSD, HorizontalSD, leftRightRefLineAngle, upDownRefLineAngle, skipFrame

#Process all images of a video
def ProcessImages(videoPath, filename, outPath, images):
  
  #load all images from video/disk
  utils.getImages(videoPath, outPath, images )
  #utils.getImages('C:/GIT/Symmetry/TestImages/test_image.jpg', images, 'ONE_IMAGE')

  #create a list from the symmetry tuples
  landmarkList = utils.createLandmarkList()

  index = 0
  SD_DATA_VERT = []
  SD_DATA_HOR = []
  MOUTH_SIZE_DATA = []
  HOR_REF_ANGLE = []
  VER_REF_ANGLE = []
  xVal = list(range(0, len(images)))

 # choose codec according to format needed, natural video frame rate is ~30 fps
  if projectDefs.createVideoOutput:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(outPath + '\\' + filename, fourcc, 30 / projectDefs.IMAGE_WRITE_SKIP_CNT, (projectDefs.IMAGE_WIDTH, projectDefs.IMAGE_HEIGHT))

  global prevLandmarks
  prevLandmarks = {}

  for name, image in images.items():
    print('Processing Image: ' + name + ', '+ str(index) + '/' + str(len(images)))
    index +=1

    ms, VerticalSD, HorizontalSD, leftRightRefLineAngle, upDownRefLineAngle, skipFrame = imageSymmetry(image, name, landmarkList)
    if(index % projectDefs.IMAGE_WRITE_SKIP_CNT == 0):
      cv2.imwrite(os.path.join(outPath, name + '.jpg'), image)
      
      if projectDefs.createVideoOutput:
        video.write(image)

    if skipFrame == False:
      SD_DATA_VERT.append(VerticalSD)
      SD_DATA_HOR.append(HorizontalSD)
      MOUTH_SIZE_DATA.append(ms)
      HOR_REF_ANGLE.append(leftRightRefLineAngle)
      VER_REF_ANGLE.append(upDownRefLineAngle)
    elif len(SD_DATA_VERT) > 0 and len(SD_DATA_HOR) > 0 and len(MOUTH_SIZE_DATA) > 0 and len(HOR_REF_ANGLE) > 0 and len(VER_REF_ANGLE) > 0:  
      SD_DATA_VERT.append(SD_DATA_VERT[-1])
      SD_DATA_HOR.append(SD_DATA_HOR[-1])
      MOUTH_SIZE_DATA.append(MOUTH_SIZE_DATA[-1])
      HOR_REF_ANGLE.append(HOR_REF_ANGLE[-1])
      VER_REF_ANGLE.append(VER_REF_ANGLE[-1])
    else:
      SD_DATA_VERT.append(0)
      SD_DATA_HOR.append(0)
      MOUTH_SIZE_DATA.append(0)
      HOR_REF_ANGLE.append(0)
      VER_REF_ANGLE.append(0)

  #close video stream
  if projectDefs.createVideoOutput:
    video.release()

  cv2.destroyAllWindows()

  #filter data
  if projectDefs.filterSDOutputs:
    SD_DATA_VERT = utils.filterList(SD_DATA_VERT, projectDefs.SD_FILTER_CONST)
    SD_DATA_HOR = utils.filterList(SD_DATA_HOR, projectDefs.SD_FILTER_CONST)
    MOUTH_SIZE_DATA = utils.filterList(MOUTH_SIZE_DATA, projectDefs.SD_FILTER_CONST)

  if projectDefs.filterAngleOutputs:
      utils.filterList(HOR_REF_ANGLE, projectDefs.SD_FILTER_CONST)
      utils.filterList(VER_REF_ANGLE, projectDefs.SD_FILTER_CONST)

  #Normalize both lists together to keep the ratio between them.
  longList = SD_DATA_HOR + SD_DATA_VERT
  tmpNormLongList = list(filter(lambda  a: a != min(longList), longList))
  maxLongList = max(tmpNormLongList)
  minLongList = min(tmpNormLongList)

  utils.normalizeList(longList, projectDefs.SD_MIN_NORM_VALUE, projectDefs.SD_MAX_NORM_VALUE)
  tmpNormLongList = list(filter(lambda  a: a != min(longList), longList))
  maxNormLongList = max(tmpNormLongList)
  minNormLongList = min(tmpNormLongList)

  #print SD Normalized or Not
  if projectDefs.normalizeOutputSD:
    normHOR = [] #split the lists again for display after normalization
    normVERT = []
    for i in range (0,len(SD_DATA_HOR)):
        normHOR.append(longList[i])

    for j in range (i+1, len(SD_DATA_HOR) + len(SD_DATA_VERT)):
        normVERT.append(longList[j])

    ms_norm = MOUTH_SIZE_DATA.copy()
    utils.normalizeList(ms_norm, maxNormLongList - 5, maxNormLongList + 5)

    plt.plot(xVal, normHOR, label = "Horizontal SD Norm")
    plt.plot(xVal, normVERT, label = "Vertical SD Norm")
    plt.plot(xVal, ms_norm, label = "Mouth Opening (relative)")
    plt.xlabel('Image Frame')
    plt.ylabel('y - axis')
    plt.title('SD Norm Data')
    plt.legend()
    plt.ylim([minNormLongList  - 10, maxNormLongList  + 10])
    plt.savefig(outPath + '\\' + filename + '_SD_Norm.png')
    #plt.show()
    plt.close()

  #plot original values for comparison
  ms_raw = MOUTH_SIZE_DATA.copy()
  utils.normalizeList(ms_raw, maxLongList - 20, maxLongList + 20)
  plt.plot(xVal, SD_DATA_HOR, label = "SD Horizontal")
  plt.plot(xVal, SD_DATA_VERT, label = "SD Vertical")
  plt.plot(xVal, ms_raw, label = "Mouth Opening")
  plt.xlabel('Image Frame')
  plt.ylabel('y - axis')
  plt.title('Orig Data SD')
  plt.legend()
  plt.ylim([minLongList - 50, maxLongList + 50])

  plt.savefig(outPath + '\\' + filename + '_SD.png')
  #plt.show()
  plt.close()


  #print Angles
  ms_norm_angle = MOUTH_SIZE_DATA.copy()
  angleMaxValue = max(HOR_REF_ANGLE + VER_REF_ANGLE)
  angleMinValue = min(HOR_REF_ANGLE + VER_REF_ANGLE)

  utils.normalizeList(ms_norm_angle, angleMaxValue - 10, angleMaxValue - 5)
  plt.plot(xVal, HOR_REF_ANGLE, label = "Horizontal Ref line angle Norm")
  plt.plot(xVal, VER_REF_ANGLE, label = "Vertical Ref line angle Norm")
  plt.plot(xVal, ms_norm_angle, label = "Mouth Opening [2-5 from Max]")
  plt.xlabel('Image Frame')
  plt.ylabel('y - axis')
  plt.title('Ref Angles')
  plt.legend()
  plt.ylim([angleMinValue - 5, angleMaxValue + 5])

  plt.savefig(outPath + '\\' + filename + '_Angles.png')
  #plt.show()
  plt.close()

  return

#Process all Videos in a folder
def ProcessVideoFolder():

  #go over all files in directory and extract images by frame index
  files = [f for f in os.listdir(videoFolderPath) if os.path.isfile(os.path.join(videoFolderPath, f))]

  for filename in files:  
      images = {}
      videoFilePath = os.path.join(videoFolderPath, filename)

      if ".mp4" not in videoFilePath:
        print('Skipping file :' + videoFilePath)
        continue

      #create a folder for each video images
      videoOutputPath = os.path.join(ImagesOutPath, os.path.splitext(filename)[0])
      os.makedirs(videoOutputPath,exist_ok=True)

      print('Loading Video:' + videoFilePath)
      ProcessImages(videoFilePath, filename, videoOutputPath, images)
  return

def ProcessWebCam():

   #create a list from the symmetry tuples
  landmarkList = utils.createLandmarkList()

  # define a video capture object
  vid = cv2.VideoCapture(0)
  index = 0

  while(True):
        
      # Capture the video frame
      # by frame
      ret, image = vid.read()
      try:
        imageSymmetry(image, str(index), landmarkList)
        index = index + 1

        # Display the resulting frame
        cv2.imshow('image', image)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

      except:
        continue
    
  # After the loop release the cap object
  vid.release()
  # Destroy all the windows
  cv2.destroyAllWindows()

def GetHorLineRefLinePoints(landmarks, centerPoint):
  x = np.array([centerPoint['X'], landmarks[projectDefs.LEFT_LIPS_REF_1]['X'], landmarks[projectDefs.LEFT_LIPS_REF_2]['X'], landmarks[projectDefs.RIGHT_LIPS_REF_1]['X'], landmarks[projectDefs.RIGHT_LIPS_REF_2]['X']])
  y = np.array([centerPoint['Y'], landmarks[projectDefs.LEFT_LIPS_REF_1]['Y'], landmarks[projectDefs.LEFT_LIPS_REF_2]['Y'], landmarks[projectDefs.RIGHT_LIPS_REF_1]['Y'], landmarks[projectDefs.RIGHT_LIPS_REF_2]['Y']])
  a, b = np.polyfit(x, y, 1)

  rightX = landmarks[projectDefs.RIGHT_LIPS_MARKER]['X']
  right = {'X': rightX,'Y': a*rightX + b}

  leftX = landmarks[projectDefs.LEFT_LIPS_MARKER]['X']
  left = {'X': leftX,'Y': a*leftX + b}

  return  left, right 

# Run #
ProcessVideoFolder()
#ProcessWebCam()





          
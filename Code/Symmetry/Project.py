from logging import exception
import mediapipe as mp
import cv2
import os

import utils
import Symmetry
import landmarkDefs
import matplotlib.pyplot as plt

images = {} #global images dictionary
NORM_VAR = 100

videoFolderPath = 'C:\\GIT\\Symmetry\\TestVideos'
#videoFolderPath = 'C:\\GIT\\Symmetry\\Videos\\Movement_sense_video'
ImagesOutPath = 'C:\\GIT\\Symmetry\\TestImages' #create image landmarks output folder

# Load drawing_utils and drawing_styles
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

SD_FILTER_CONST = 0.5
LANDMARK_FILTER_CONST = 0.5

prevNormLandmarks = {}

debug = 1

#Run Face Mesh landmark detection on a single image.
def MpFaceMesh(image):

  circleDrawingSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

  #Run FaceMesh on each image to get landmarks
  with mp_face_mesh.FaceMesh(
      static_image_mode=True,        #Was True, unrelated images (True) or as a video stream (False)
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5) as face_mesh:

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
        connections=landmarkDefs.MY_FACE_CONNECTIONS,
        landmark_drawing_spec=circleDrawingSpec,#None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())

  return results.multi_face_landmarks[0]

#Calculate single image symmetry distance
def imageSymmetry(image, name, landmarkList, filterLandmarks = True):

  #run FaceMesh to get All landmark points
  rawLandmarkPoints = MpFaceMesh(image)
  
  #get image coordinates Landmarks
  imageLandmarks = utils.getAllLandmarksData(rawLandmarkPoints) 

  #if debug:
  #  utils.printLandmarkPoints(imageLandmarks, 1, image)
  #  utils.resize_and_show(image, true)

  #convert Image points to Normlized Scale of NORM_VAR
  normLandmarks, scale, var = Symmetry.normalizeLandmarks(imageLandmarks, NORM_VAR)

  if filterLandmarks:
    global prevNormLandmarks
    if len(prevNormLandmarks) > 0:
      utils.filterDictionary(normLandmarks, prevNormLandmarks, LANDMARK_FILTER_CONST)

    prevNormLandmarks = normLandmarks.copy()

  #debug code - remove later
  scale = 1

  #utils.printLandmarkPoints(normLandmarks, 1, image)
  #utils.resize_and_show(image, true)

    
  #get the image relevant (X,Y) points, hashmap of {idx: (X,Y)}
  landmarkImagePointsNorm = utils.getFilteredLandmarkData(normLandmarks, landmarkList)  
  guideImagePoints    = utils.getFilteredLandmarkData(imageLandmarks, landmarkDefs.FACE_GUIDE)  
  guideImagePointsNorm    = utils.getFilteredLandmarkData(normLandmarks, landmarkDefs.FACE_GUIDE)  

  #find face Norm reference line - Vertical
  verticalRefLineSrcNorm = guideImagePointsNorm[landmarkDefs.LEFT_MARKER]
  verticalLineDstNorm = guideImagePointsNorm[landmarkDefs.RIGHT_MARKER]
  verticalRefLineAngleNorm = utils.get_angle(verticalRefLineSrcNorm, verticalLineDstNorm)

  #draw Veritcal ref line from Image Landmarks
  verticalRefLineSrc = guideImagePoints[landmarkDefs.LEFT_MARKER]
  verticalLineDst = guideImagePoints[landmarkDefs.RIGHT_MARKER]
  verticalRefLineAngle = utils.get_angle(verticalRefLineSrc, verticalLineDst)
  utils.drawLineOnImage(image, verticalRefLineSrc, verticalLineDst, scale)


  #find and print face reference line - Horizontal
  horizontalRefLineSrcNorm = guideImagePointsNorm[landmarkDefs.UP_MARKER]
  horizontalLineDstNorm = guideImagePointsNorm[landmarkDefs.DOWN_MARKER]
  horizontalRefLineAngleNorm = utils.get_angle(horizontalRefLineSrcNorm, horizontalLineDstNorm)


  horizontalRefLineSrc = guideImagePoints[landmarkDefs.UP_MARKER]
  horizontalLineDst = guideImagePoints[landmarkDefs.DOWN_MARKER]
  horizontalRefLineAngle = utils.get_angle(horizontalRefLineSrc, horizontalLineDst)
  utils.drawLineOnImage(image, horizontalRefLineSrc, horizontalLineDst, scale)

  #find and print landmarks center
  #center = Symmetry.centerMass(landmarkImagePoints)
  #utils.annotatePoint(image, center, 'CM')

  #run Symmetry Alg while using the face ref line as the symmetry line.
  VerticalSDNorm = Symmetry.checkSymmetryOfLine(image, verticalRefLineSrcNorm, verticalLineDstNorm, landmarkImagePointsNorm, landmarkDefs.LIPS_VERTICAL_LANDMARK_SYMMTERY)
  HorizontalSDNorm = Symmetry.checkSymmetryOfLine(image, horizontalRefLineSrcNorm, horizontalLineDstNorm, landmarkImagePointsNorm, landmarkDefs.LIPS_HORIZONTAL_LANDMARK_SYMMTERY)
 
  VerticalSD = Symmetry.checkSymmetryOfLine(image, verticalRefLineSrc, verticalLineDst, imageLandmarks, landmarkDefs.LIPS_VERTICAL_LANDMARK_SYMMTERY)
  HorizontalSD = Symmetry.checkSymmetryOfLine(image, horizontalRefLineSrc, horizontalLineDst, imageLandmarks, landmarkDefs.LIPS_HORIZONTAL_LANDMARK_SYMMTERY)
 

  #get mouth size + SF
  msNorm = (guideImagePointsNorm[14]['Y'] - guideImagePointsNorm[13]['Y']) / scale
  ms = (guideImagePoints[14]['Y'] - guideImagePoints[13]['Y']) / scale


  #add Text Info On Images
  utils.addTextOnImage(image, name, (50, 50))
  utils.addTextOnImage(image, 'Mouth Size (Norm)= ' + str(msNorm), (50, 100))

  utils.addTextOnImage(image, 'Vertical SD   : ' + str(VerticalSDNorm), (50, 150))
  utils.addTextOnImage(image, 'Norm Vertical Face Line Angle = ' + str(verticalRefLineAngleNorm), (50, 200))
  utils.addTextOnImage(image, 'Vertical Face Line Angle = ' + str(verticalRefLineAngle), (50, 250))

  utils.addTextOnImage(image, 'Horizontal SD : ' + str(HorizontalSDNorm), (50, 300))
  utils.addTextOnImage(image, 'Norm Horizontal Face Line Angle = ' + str(horizontalRefLineAngleNorm), (50, 350))
  utils.addTextOnImage(image, 'Horizontal Face Line Angle = ' + str(horizontalRefLineAngle), (50, 400))

  #plot the landmarks on top of the image
  utils.printLandmarkPoints(landmarkImagePointsNorm, scale, image)
  utils.printLandmarkPoints(imageLandmarks, scale, image)

  return VerticalSDNorm, HorizontalSDNorm, msNorm

#Process all images of a video
def ProcessImages(videoPath, filename, outPath, images, filterLandmarks = False, filterSD = False, normSD = False):
  
  #load all images from video/disk
  utils.getImages(videoPath, outPath, images )
  #utils.getImages('C:/GIT/Symmetry/TestImages/test_image.jpg', images, 'ONE_IMAGE')

  #create a list from the symmetry tuples
  landmarkList = utils.createLandmarkList()

  index = 0
  SD_DATA_VERT = []
  SD_DATA_HOR = []
  MOUTH_SIZE_DATA = []
  xVal = list(range(0, len(images)))

 # choose codec according to format needed
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
  video = cv2.VideoWriter(outPath + '\\' + filename, fourcc, 30, (utils.DESIRED_HEIGHT, utils.DESIRED_WIDTH))

  prevNormLandmarks = {}
  for name, image in images.items():
    print('Processing Image: ' + name + ', '+ str(index) + '/' + str(len(images)))
    index +=1

    sdVert, sdHor, ms = imageSymmetry(image, name, landmarkList)

    if(index % Symmetry.IMAGE_WRITE_SKIP_CNT == 0):
      cv2.imwrite(os.path.join(outPath, name + '.jpg'), image)

    #add data for plot
    SD_DATA_VERT.append(sdVert)
    SD_DATA_HOR.append(sdHor)
    MOUTH_SIZE_DATA.append(ms)

    #add frame to output video.
    video.write(image)

  #close video stream
  cv2.destroyAllWindows()
  video.release()

  #filter data
  if filterSD:
    SD_DATA_VERT = utils.filterList(SD_DATA_VERT, SD_FILTER_CONST)
    SD_DATA_HOR = utils.filterList(SD_DATA_HOR, SD_FILTER_CONST)
    MOUTH_SIZE_DATA = utils.filterList(MOUTH_SIZE_DATA, SD_FILTER_CONST)

  if normSD:
      MAX_NORM_VALUE = 1000
      MIN_NORM_VALUE = 0
      
      #align a linear ratio for a visible graph
      ratio = max(SD_DATA_VERT) / max(MOUTH_SIZE_DATA)
      
      #Normalize mouth size values
      rangeMouthOpen = max(MOUTH_SIZE_DATA) - min(MOUTH_SIZE_DATA)
      minMouthOpen = min(MOUTH_SIZE_DATA)
      
      for i in range(len(MOUTH_SIZE_DATA)):
        MOUTH_SIZE_DATA[i] = utils.normalizeValue(MOUTH_SIZE_DATA[i], minMouthOpen, rangeMouthOpen, MIN_NORM_VALUE, MAX_NORM_VALUE)

      #normalize vertical values
      rangeSDVert = max(SD_DATA_VERT) - min(SD_DATA_VERT)
      minSDVert = min(SD_DATA_VERT)

      for i in range(len(SD_DATA_VERT)):
        SD_DATA_VERT[i] = utils.normalizeValue(SD_DATA_VERT[i], minSDVert, rangeSDVert, MIN_NORM_VALUE, MAX_NORM_VALUE)

      #normalize horizontal values
      rangeSDHor = max(SD_DATA_HOR) - min(SD_DATA_HOR)
      minSDHor = min(SD_DATA_HOR)

      for i in range(len(SD_DATA_HOR)):
        SD_DATA_HOR[i] = utils.normalizeValue(SD_DATA_HOR[i], minSDHor, rangeSDHor, MIN_NORM_VALUE, MAX_NORM_VALUE)


  plt.plot(xVal, SD_DATA_VERT, label = "SD Vertical")
  plt.plot(xVal, SD_DATA_HOR, label = "SD Horizontal")
  plt.plot(xVal, MOUTH_SIZE_DATA, label = "Mouth Opening")
  plt.xlabel('Image Frame')
  plt.ylabel('y - axis')
  plt.title('SD & Mouth Size per Frame')
  plt.legend()
  plt.savefig(outPath + '\\' + filename + '_plot.png')
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
      ProcessImages(videoFilePath, filename, videoOutputPath, images, False, False)
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
    
      imageSymmetry(image, str(index), landmarkList)
      index = index + 1

      # Display the resulting frame
      cv2.imshow('image', image)
         
      # the 'q' button is set as the
      # quitting button you may use any
      # desired button of your choice
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
  # After the loop release the cap object
  vid.release()
  # Destroy all the windows
  cv2.destroyAllWindows()

#Debug Only Code

  #degug: print All landmark markers + numbering on image.
  #utils.printLandmarkPoints(landmarkImagePoints, image)

#Run

#ProcessVideoFolder()

ProcessWebCam()





          
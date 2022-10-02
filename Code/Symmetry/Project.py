import matplotlib.pyplot as plt
from logging import exception
import mediapipe as mp
import cv2
import os

import utils
import Symmetry
import projectDefs

images = {} #global images dictionary

videoFolderPath = 'C:\\GIT\\Symmetry\\TestVideos'
#videoFolderPath = 'C:\\GIT\\Symmetry\\Videos\\Movement_sense_video'
ImagesOutPath = 'C:\\GIT\\Symmetry\\TestImages' #create image landmarks output folder

# Load drawing_utils and drawing_styles
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

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
        connections=projectDefs.MY_FACE_CONNECTIONS,
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

#  if debug:
#    utils.printLandmarkPoints(imageLandmarks, 1, image)
#    utils.resize_and_show(image, True)

  #convert Image points to Normalized Scale of NORM_VAR
  normLandmarks, scale, var = Symmetry.normalizeLandmarks(imageLandmarks, projectDefs.NORM_VAR)

  if filterLandmarks:
    global prevNormLandmarks
    if len(prevNormLandmarks) > 0:
      utils.filterDictionary(normLandmarks, prevNormLandmarks, projectDefs.LANDMARK_FILTER_CONST)

    prevNormLandmarks = normLandmarks.copy()

  #TODO: debug code - remove later
  scale = 1

  #utils.printLandmarkPoints(normLandmarks, 1, image)
  #utils.resize_and_show(image, true)

  #get the image relevant (X,Y) points, hashmap of {idx: (X,Y)}
  landmarkImagePointsNorm = utils.getFilteredLandmarkData(normLandmarks, landmarkList)  
  guideImagePoints        = utils.getFilteredLandmarkData(imageLandmarks, projectDefs.FACE_GUIDE)  
  guideImagePointsNorm    = utils.getFilteredLandmarkData(normLandmarks, projectDefs.FACE_GUIDE)  

  ## Vertical Section ##

  #find face Norm reference line
  verticalRefLineSrcNorm = guideImagePointsNorm[projectDefs.LEFT_MARKER]
  verticalRefLineDstNorm = guideImagePointsNorm[projectDefs.RIGHT_MARKER]
  verticalRefLineAngleNorm = utils.get_angle(verticalRefLineSrcNorm, verticalRefLineDstNorm)

  #draw Vertical ref line from Image Landmarks
  verticalRefLineSrc = guideImagePoints[projectDefs.LEFT_MARKER]
  verticalRefLineDst = guideImagePoints[projectDefs.RIGHT_MARKER]
  verticalRefLineAngle = utils.get_angle(verticalRefLineSrc, verticalRefLineDst)
  utils.drawLineOnImage(image, verticalRefLineSrc, verticalRefLineDst, scale)


  ## Horizontal Section ##

  #find and print face reference line
  horizontalRefLineSrcNorm = guideImagePointsNorm[projectDefs.UP_MARKER]
  horizontalRefLineDstNorm = guideImagePointsNorm[projectDefs.DOWN_MARKER]
  horizontalRefLineAngleNorm = utils.get_angle(horizontalRefLineSrcNorm, horizontalRefLineDstNorm)

  #draw Horizontal ref line from Image Landmarks
  horizontalRefLineSrc = guideImagePoints[projectDefs.UP_MARKER]
  horizontalRefLineDst = guideImagePoints[projectDefs.DOWN_MARKER]
  horizontalRefLineAngle = utils.get_angle(horizontalRefLineSrc, horizontalRefLineDst)
  utils.drawLineOnImage(image, horizontalRefLineSrc, horizontalRefLineDst, scale)

  #find and print landmarks center
  #center = Symmetry.centerMass(landmarkImagePoints)
  #utils.annotatePoint(image, center, 'CM')

  #run Symmetry Alg while using the face ref line as the symmetry line.
  VerticalSDNorm = Symmetry.checkSymmetryOfLine(image, horizontalRefLineSrcNorm, horizontalRefLineDstNorm, landmarkImagePointsNorm, projectDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY)
  HorizontalSDNorm = Symmetry.checkSymmetryOfLine(image, verticalRefLineSrcNorm, verticalRefLineDstNorm, landmarkImagePointsNorm, projectDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY)
 
  VerticalSD = Symmetry.checkSymmetryOfLine(image, horizontalRefLineSrc, horizontalRefLineDst, imageLandmarks, projectDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY)
  HorizontalSD = Symmetry.checkSymmetryOfLine(image, verticalRefLineSrc, verticalRefLineDst, imageLandmarks, projectDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY)
 
  #get mouth size + SF
  msNorm = abs((guideImagePointsNorm[projectDefs.MOUTH_UPPER_LIP_MIN_HEIGHT]['Y'] - guideImagePointsNorm[projectDefs.MOUTH_LOWER_LIP_MAX_HEIGHT]['Y']) / scale)
  ms = abs((guideImagePoints[projectDefs.MOUTH_UPPER_LIP_MIN_HEIGHT]['Y'] - guideImagePoints[projectDefs.MOUTH_LOWER_LIP_MAX_HEIGHT]['Y']) / scale)

  #Embedded Text labels On Images
  utils.addTextOnImage(image, name, True)
  utils.addTextOnImage(image, 'Mouth Size = ' + str(ms))
  utils.addTextOnImage(image, 'Mouth Size (Norm)= ' + str(msNorm))

  utils.addTextOnImage(image, 'Vertical SD : ' + str(VerticalSD))
  utils.addTextOnImage(image, 'Vertical SD (Norm) : ' + str(VerticalSDNorm))
  utils.addTextOnImage(image, 'Vertical Face Line Angle (Norm) = ' + str(verticalRefLineAngleNorm))
  utils.addTextOnImage(image, 'Vertical Face Line Angle = ' + str(verticalRefLineAngle))

  utils.addTextOnImage(image, 'Horizontal SD : ' + str(HorizontalSD))
  utils.addTextOnImage(image, 'Horizontal SD (Norm) : ' + str(HorizontalSDNorm))
  utils.addTextOnImage(image, 'Horizontal Face Line Angle (Norm) = ' + str(horizontalRefLineAngleNorm))
  utils.addTextOnImage(image, 'Horizontal Face Line Angle = ' + str(horizontalRefLineAngle))

  #plot the landmarks on top of the image
  utils.printLandmarkPoints(landmarkImagePointsNorm, scale, image, True)
  utils.printLandmarkPoints(imageLandmarks, scale, image)

  return VerticalSDNorm, HorizontalSDNorm, msNorm

#Process all images of a video
def ProcessImages(videoPath, filename, outPath, images, filterLandmarks = False, normSD = False, filterSD = False):
  
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

 # choose codec according to format needed, natural video frame rate is ~30 fps
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
  video = cv2.VideoWriter(outPath + '\\' + filename, fourcc, 30 / projectDefs.IMAGE_WRITE_SKIP_CNT, (projectDefs.IMAGE_WIDTH, projectDefs.IMAGE_HEIGHT))

  prevNormLandmarks = {}
  for name, image in images.items():
    print('Processing Image: ' + name + ', '+ str(index) + '/' + str(len(images)))
    index +=1

    sdVert, sdHor, ms = imageSymmetry(image, name, landmarkList, filterLandmarks)

    if(index % projectDefs.IMAGE_WRITE_SKIP_CNT == 0):
      cv2.imwrite(os.path.join(outPath, name + '.jpg'), image)
      video.write(image)    #add frame to output video.

    #add data for plot
    SD_DATA_VERT.append(sdVert)
    SD_DATA_HOR.append(sdHor)
    MOUTH_SIZE_DATA.append(ms)

  #close video stream
  cv2.destroyAllWindows()
  video.release()

  # Plot Raw Data
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

  if normSD:
    utils.normalizeList(MOUTH_SIZE_DATA, projectDefs.MOUTH_SIZE_MIN_NORM_VALUE, projectDefs.MOUTH_SIZE_MAX_NORM_VALUE)
    utils.normalizeList(SD_DATA_VERT, projectDefs.SD_MIN_NORM_VALUE, projectDefs.SD_MAX_NORM_VALUE)
    utils.normalizeList(SD_DATA_HOR, projectDefs.SD_MIN_NORM_VALUE, projectDefs.SD_MAX_NORM_VALUE)
 
    plt.plot(xVal, SD_DATA_VERT, label = "SD Vertical Norm")
    plt.plot(xVal, SD_DATA_HOR, label = "SD Horizontal Norm")
    plt.plot(xVal, MOUTH_SIZE_DATA, label = "Mouth Opening Norm")
    plt.xlabel('Image Frame')
    plt.ylabel('y - axis')
    plt.title('SD & Mouth Size per Frame')
    plt.legend()
    plt.savefig(outPath + '\\' + filename + '_plot_Norm.png')
    #plt.show()
    plt.close()

  #filter data
  if filterSD:
    SD_DATA_VERT = utils.filterList(SD_DATA_VERT, projectDefs.SD_FILTER_CONST)
    SD_DATA_HOR = utils.filterList(SD_DATA_HOR, projectDefs.SD_FILTER_CONST)
    MOUTH_SIZE_DATA = utils.filterList(MOUTH_SIZE_DATA, projectDefs.SD_FILTER_CONST)

    # Plot Filtered Data
    plt.plot(xVal, SD_DATA_VERT, label = "SD Vertical Filtered")
    plt.plot(xVal, SD_DATA_HOR, label = "SD Horizontal Filtered")
    plt.plot(xVal, MOUTH_SIZE_DATA, label = "Mouth Opening Filtered")
    plt.xlabel('Image Frame')
    plt.ylabel('y - axis')
    plt.title('SD & Mouth Size per Frame')
    plt.legend()
    plt.savefig(outPath + '\\' + filename + '_plot_filter.png')
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
      ProcessImages(videoFilePath, filename, videoOutputPath, images, True, True, False)
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

# Run #
ProcessVideoFolder()
#ProcessWebCam()





          
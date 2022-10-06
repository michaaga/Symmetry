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
  #get image coordinates Landmarks
  imageLandmarks = utils.getAllLandmarksData(MpFaceMesh(image))
  selectedLandmarks = utils.getFilteredLandmarkData(imageLandmarks, landmarkList)

  #convert Image points to Normalized Scale of NORM_VAR
  selectedLandmarksNorm, Var = Symmetry.normalizeLandmarks(selectedLandmarks, projectDefs.NORM_VAR)
  
  center = Symmetry.centerMass(selectedLandmarks)
  normCenter = Symmetry.centerMass(selectedLandmarksNorm)
  utils.annotatePoint(image, center, "", (255, 0, 0))
  utils.annotatePoint(image, normCenter, "", (0, 0, 255))
  centersDiff = Symmetry.pointsPairSqrDistance(center, normCenter)
  if abs(centersDiff) > 1:
    raise exception("centers do not align")

  # utils.printLandmarkPoints(selectedLandmarks, 1, image)
  # utils.printLandmarkPoints(selectedLandmarksNorm, 1, image)
  # utils.resize_and_show(image, True)

  if filterLandmarks:
    global prevNormLandmarks
    if len(prevNormLandmarks) > 0:
      utils.filterDictionary(selectedLandmarksNorm, prevNormLandmarks, projectDefs.LANDMARK_FILTER_CONST)

    prevNormLandmarks = selectedLandmarksNorm.copy()

  ## Vertical Section ##

  #draw Norm Vertical ref line from Image Landmarks
  verticalRefLineSrcNorm = selectedLandmarksNorm[projectDefs.LEFT_LIPS_MARKER]
  verticalRefLineDstNorm = selectedLandmarksNorm[projectDefs.RIGHT_LIPS_MARKER]
  verticalRefLineAngleNorm = utils.get_angle(verticalRefLineSrcNorm, verticalRefLineDstNorm) 
  utils.drawLineOnImage(image, verticalRefLineSrcNorm, verticalRefLineDstNorm)


  #draw Vertical ref line from Image Landmarks
  verticalRefLineSrc = selectedLandmarks[projectDefs.LEFT_LIPS_MARKER]
  verticalRefLineDst = selectedLandmarks[projectDefs.RIGHT_LIPS_MARKER]
  verticalRefLineAngle = utils.get_angle(verticalRefLineSrc, verticalRefLineDst)
  utils.drawLineOnImage(image, verticalRefLineSrc, verticalRefLineDst)


  ## Horizontal Section ##

  #draw Norm Horizontal ref line from Image Landmarks
  horizontalRefLineSrcNorm = selectedLandmarksNorm[projectDefs.LOWER_LIP_MIN]
  horizontalRefLineDstNorm = selectedLandmarksNorm[projectDefs.UPPER_LIP_MAX]
  horizontalRefLineAngleNorm = utils.get_angle(horizontalRefLineSrcNorm, horizontalRefLineDstNorm) + 90 #offset vertical line to zero
  utils.drawLineOnImage(image, horizontalRefLineSrcNorm, horizontalRefLineDstNorm)

  #draw Horizontal ref line from Image Landmarks
  horizontalRefLineSrc = selectedLandmarks[projectDefs.LOWER_LIP_MIN]
  horizontalRefLineDst = selectedLandmarks[projectDefs.UPPER_LIP_MAX]
  horizontalRefLineAngle = utils.get_angle(horizontalRefLineSrc, horizontalRefLineDst) + 90 #offset vertical line to zero
  utils.drawLineOnImage(image, horizontalRefLineSrc, horizontalRefLineDst)

  #find and print landmarks center
  #center = Symmetry.centerMass(landmarkImagePoints)
  #utils.annotatePoint(image, center, 'CM')

  #run Symmetry Alg while using the face ref line as the symmetry line.
  VerticalSDNorm = Symmetry.checkSymmetryOfLine(image, horizontalRefLineSrcNorm, horizontalRefLineDstNorm, selectedLandmarksNorm, projectDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY)
  HorizontalSDNorm = Symmetry.checkSymmetryOfLine(image, verticalRefLineSrcNorm, verticalRefLineDstNorm, selectedLandmarksNorm, projectDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY)
 
  VerticalSD = Symmetry.checkSymmetryOfLine(image, horizontalRefLineSrc, horizontalRefLineDst, selectedLandmarks, projectDefs.LIPS_VERTICAL_LANDMARK_SYMMETRY)
  HorizontalSD = Symmetry.checkSymmetryOfLine(image, verticalRefLineSrc, verticalRefLineDst, selectedLandmarks, projectDefs.LIPS_HORIZONTAL_LANDMARK_SYMMETRY)
 
  #get mouth size + SF
  msNorm = abs(selectedLandmarksNorm[projectDefs.UPPER_LIP_MIN]['Y'] - selectedLandmarksNorm[projectDefs.LOWER_LIP_MAX]['Y'])
  ms = abs(selectedLandmarks[projectDefs.UPPER_LIP_MIN]['Y'] - selectedLandmarks[projectDefs.LOWER_LIP_MAX]['Y'])

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
  utils.printLandmarkPoints(selectedLandmarksNorm, image, True)
  utils.printLandmarkPoints(selectedLandmarks, image)

  return ms, msNorm, VerticalSD, VerticalSDNorm, verticalRefLineAngleNorm, verticalRefLineAngle, HorizontalSD, HorizontalSDNorm, horizontalRefLineAngleNorm, horizontalRefLineAngle

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
  SD_DATA_VERT_NORM = []
  SD_DATA_HOR_NORM = []
  MOUTH_SIZE_DATA = []
  MOUTH_SIZE_DATA_NORM = []
  HOR_REF_ANGLE = []
  VER_REF_ANGLE = []
  HOR_REF_ANGLE_NORM = []
  VER_REF_ANGLE_NORM = []
  xVal = list(range(0, len(images)))

 # choose codec according to format needed, natural video frame rate is ~30 fps
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
  video = cv2.VideoWriter(outPath + '\\' + filename, fourcc, 30 / projectDefs.IMAGE_WRITE_SKIP_CNT, (projectDefs.IMAGE_WIDTH, projectDefs.IMAGE_HEIGHT))

  global prevNormLandmarks
  prevNormLandmarks = {}
  for name, image in images.items():
    print('Processing Image: ' + name + ', '+ str(index) + '/' + str(len(images)))
    index +=1

    ms, msNorm, VerticalSD, VerticalSDNorm, verticalRefLineAngleNorm, verticalRefLineAngle, HorizontalSD, HorizontalSDNorm, horizontalRefLineAngleNorm, horizontalRefLineAngle = imageSymmetry(image, name, landmarkList, filterLandmarks)

    if(index % projectDefs.IMAGE_WRITE_SKIP_CNT == 0):
      cv2.imwrite(os.path.join(outPath, name + '.jpg'), image)
      video.write(image)    #add frame to output video.

    #add data for plotting
    SD_DATA_VERT.append(VerticalSD)
    SD_DATA_HOR.append(HorizontalSD)
    SD_DATA_VERT_NORM.append(VerticalSDNorm)
    SD_DATA_HOR_NORM.append(HorizontalSDNorm)
    MOUTH_SIZE_DATA.append(ms)
    MOUTH_SIZE_DATA_NORM.append(msNorm)
    HOR_REF_ANGLE.append(horizontalRefLineAngle)
    VER_REF_ANGLE.append(verticalRefLineAngle)
    HOR_REF_ANGLE_NORM.append(horizontalRefLineAngleNorm)
    VER_REF_ANGLE_NORM.append(verticalRefLineAngleNorm)

  #close video stream
  cv2.destroyAllWindows()
  video.release()

  # # Plot Raw Data
  # plt.plot(xVal, SD_DATA_VERT, label = "SD Vertical")
  # plt.plot(xVal, SD_DATA_HOR, label = "SD Horizontal")
  # plt.plot(xVal, MOUTH_SIZE_DATA, label = "Mouth Opening")
  # plt.plot(xVal, HOR_REF_ANGLE, label = "Horizontal Ref line angle")
  # plt.plot(xVal, VER_REF_ANGLE, label = "Vertical Ref line angle")
  # plt.xlabel('Image Frame')
  # plt.ylabel('y - axis')
  # plt.title('Raw Data SD')
  # plt.legend()
  # plt.savefig(outPath + '\\' + filename + '_plot.png')
  # #plt.show()
  # plt.close()

  # Plot Normalized data
  plt.plot(xVal, SD_DATA_VERT_NORM, label = "SD Vertical Norm")
  plt.plot(xVal, SD_DATA_HOR_NORM, label = "SD Horizontal Norm")
  plt.plot(xVal, MOUTH_SIZE_DATA_NORM, label = "Mouth Opening Norm")
  plt.xlabel('Image Frame')
  plt.ylabel('y - axis')
  plt.title('Norm Data SD')
  plt.legend()
  plt.savefig(outPath + '\\' + filename + '_Norm_Points_plot.png')
  #plt.show()
  plt.close()


#  #plot angles
#   plt.plot(xVal, HOR_REF_ANGLE, label = "Horizontal Ref line angle Norm")
#   plt.plot(xVal, VER_REF_ANGLE, label = "Vertical Ref line angle Norm")
#   plt.xlabel('Image Frame')
#   plt.ylabel('y - axis')
#   plt.title('Norm Data Ref Angles')
#   plt.legend()
#   plt.savefig(outPath + '\\' + filename + '_Angles.png')
#   #plt.show()
#   plt.close()

  #plot angles Norm
  plt.plot(xVal, HOR_REF_ANGLE_NORM, label = "Horizontal Ref line angle Norm")
  plt.plot(xVal, VER_REF_ANGLE_NORM, label = "Vertical Ref line angle Norm")
  plt.xlabel('Image Frame')
  plt.ylabel('y - axis')
  plt.title('Norm Data Ref Angles')
  plt.legend()
  plt.savefig(outPath + '\\' + filename + '_Norm_Angles.png')
  #plt.show()
  plt.close()

  if normSD:
    utils.normalizeList(MOUTH_SIZE_DATA, projectDefs.MOUTH_SIZE_MIN_NORM_VALUE, projectDefs.MOUTH_SIZE_MAX_NORM_VALUE)

    longList = SD_DATA_HOR + SD_DATA_VERT #Normalize both lists together to keep the ratio between them.
    utils.normalizeList(longList, projectDefs.SD_MIN_NORM_VALUE, projectDefs.SD_MAX_NORM_VALUE)

    normHOR = [] #split the lists again for display after normalization
    normVERT = []
    for i in range (0,len(SD_DATA_HOR)):
        normHOR.append(longList[i])

    for j in range (i+1, len(SD_DATA_HOR) + len(SD_DATA_VERT)):
        normVERT.append(longList[j])
        
    plt.plot(xVal, normVERT, label = "SD Vertical Norm")
    plt.plot(xVal, normHOR, label = "SD Horizontal Norm")
    plt.plot(xVal, MOUTH_SIZE_DATA, label = "Mouth Opening Norm")
    plt.xlabel('Image Frame')
    plt.ylabel('y - axis')
    plt.title('SD & Mouth Size per Frame')
    plt.legend()
    plt.savefig(outPath + '\\' + filename + '_plot_Norm_Points_Norm_SD.png')
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





          
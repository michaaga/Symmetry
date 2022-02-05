from logging import exception
from turtle import xcor
import mediapipe as mp
import cv2
import os
import math 
import utils
import Symmetry
import landmarkDefs
import matplotlib.pyplot as plt


images = {} #global images dictionary

videoFolderPath = 'C:\\GIT\\Symmetry\\TestVideos'
#videoFolderPath = 'C:\\GIT\\Symmetry\\Videos\\Movement_sense_video'
ImagesOutPath = 'C:\\GIT\\Symmetry\\TestImages' #create image landmarks output folder

# Load drawing_utils and drawing_styles
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def runMpFaceMesh(image):

  circleDrawingSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

  #Run FaceMesh on each image to get landmarks
  with mp_face_mesh.FaceMesh(
      static_image_mode=False,        #Was True, unrelated images (True) or as a video stream (False)
      max_num_faces=2,
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

def getImages():
  if 0:
    utils.load_images_from_folder(videoFolderPath, images)
  else:
    utils.extractImagesFromVideo(videoFolderPath, images, False)

    # Preview the images.
  for name, image in images.items():
    print(name)   
    utils.resize_and_show(image)  

  return

def main():
  
  #load all images from video/disk
  getImages()

  #create a list from the list of tuples
  landmarkList  = utils.createLandmarkList()

  index = 0
  SD = []
  mouthSize = []
  xVal = list(range(0, len(images)))

  for name, image in images.items():
    print('Processing Image: ' + name + ', '+ str(index) + '/' + str(len(images)))
    index +=1

    #run FaceMesh to get landmark points
    landmarkPoints = runMpFaceMesh(image)
    allLandmarks = utils.getAllLandmarksData(landmarkPoints)

    normLandmarks, scale = Symmetry.normalizeLandmarks(allLandmarks, 100)
      
    #get the image relevant (X,Y) points, hashmap of {idx: (X,Y)}
    landmarkImagePoints = utils.getFilteredLandmarkData(normLandmarks, landmarkList)  
    guideImagePoints    = utils.getFilteredLandmarkData(normLandmarks, landmarkDefs.FACE_GUIDE)  

    #find and print face reference line
    refLineSrc = guideImagePoints[9]
    refLineDst = guideImagePoints[94]

    utils.drawLineOnImage(image, refLineSrc, refLineDst, scale)
    refLineAngle = utils.get_angle(refLineSrc, refLineDst)

    #find and print landmarks center
    center = Symmetry.centerMass(landmarkImagePoints)
    #utils.annotatePoint(image, center, 'CM')

    #run Symmetry Alg while using the face ref line as the symmetry line.
    SD.append(Symmetry.checkSymmetryOfLine(image, refLineSrc, refLineDst, landmarkImagePoints))
    
    angle = 0 # MISSING CALCULATION FOR ANGLE DERIVED FROM POINTS
    #keep the angles from 0 to 180.
    if(angle > 180):
      angle = angle - 180

    #plot the "best" symmetry line from the center
    #lineLength = 100
    #dstX = center['X'] + lineLength * math.cos(math.radians(angle))
    #dstY = center['Y'] + lineLength * math.sin(math.radians(angle))
    #dst = {'X': dstX, 'Y': dstY}
    #utils.drawLineOnImage(image, center, dst, (0,0,255))
    
    #get mouth size + SF
    ms = (guideImagePoints[14]['Y'] / scale - guideImagePoints[13]['Y'] / scale ) * 50
    mouthSize.append(ms)

    #add Text Info On Images
    utils.addTextOnImage(image, name, (50, 50))
    utils.addTextOnImage(image, 'SD = ' + str(SD), (50, 100))
    utils.addTextOnImage(image, 'MIN SD Angle = ' + str(angle), (50, 150))
    utils.addTextOnImage(image, 'Ref Line Angle = ' + str(refLineAngle), (50, 200))
    utils.addTextOnImage(image, 'Angle Diff = ' + str(refLineAngle - angle), (50, 250))
    utils.addTextOnImage(image, 'Mouth Size = ' + str(ms), (50, 300))


    #plot the landmarks
    utils.printLandmarkPoints(landmarkImagePoints, scale, image)

    #save
    #utils.resize_and_show(image, True)
    cv2.imwrite(os.path.join(ImagesOutPath, name + '.jpg'), image)

  #plt.plot(xVal, SD, label = "SD")
  plt.plot(xVal, mouthSize, label = "Mouth Size") 
  plt.xlabel('Image Frame')
  plt.ylabel('y - axis')
  plt.title('SD & Mouth Size per Frame')
  plt.legend()
  plt.show()

  return

#Debug Only Code

  #degug: print All landmark markers + numbering on image.
  #utils.printLandmarkPoints(landmarkImagePoints, image)


#Run

main()







          
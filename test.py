import mediapipe as mp
import cv2
import math
import os

images = {} #hold the images dictionary
DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 1920

#Extract images from video and save to disk
def extractImagesFromVideo(path, saveToDisk = False):

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

def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
          images[filename] = img
    return images

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

# Load drawing_utils and drawing_styles
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

circleDrawingSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

videoFolderPath = 'C:\\Users\\212574830\\Desktop\\Project symmetry\\Videos\\Movement_sense_video'
extractImagesFromVideo(videoFolderPath, False)

#load_images_from_folder('C:\\Users\\212574830\\Desktop\\Project symmetry\\Videos\\Movement_sense_video\\Images')

# Preview the images.
for name, image in images.items():
  print(name)   
  resize_and_show(image)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,        #Was True, Boolean indicating if the images it processes should be treated as unrelated images (True) or as a video stream (False)
    max_num_faces=2,
    min_detection_confidence=0.5) as face_mesh:


  for name, image in images.items():
    # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face landmarks of each face.
    print(f'Face landmarks of {name}:')
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    #create image landmarks output folder
    OutPath = 'C:\\Users\\212574830\\Desktop\\Project symmetry\\Images'
    os.makedirs(OutPath,exist_ok=True)

    for face_landmarks in results.multi_face_landmarks:
      #Draw Mesh
      if(0):
          mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())


      mp_drawing.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_LIPS ,
        landmark_drawing_spec=circleDrawingSpec,#None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())

      resize_and_show(annotated_image)    
      cv2.imwrite(os.path.join(OutPath, name + '.jpg'), annotated_image)
        

#def mirrorImage(img):

#1. Crop image to half the image
#2. clone & flip the remaining half
#3. merge both images together

#image_obj = Image.open(image_path)
 #   rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
 #   rotated_image.save(saved_location)
 #   rotated_image.show()
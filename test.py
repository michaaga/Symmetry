import mediapipe as mp
import cv2
import math
import os

images = {} #hold the images dictionary

def extractImagesFromVideo(path):
    # Opens the Video file
    cap= cv2.VideoCapture(path)
    i=0

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        fName = os.path.basename(path) +  '_' + str(i)
        #cv2.imwrite(r'C:\\Users\\212574830\\Desktop\\Project symmetry\\Images\\' + os.path.basename(path) + str(i) + '.jpg',frame)
        images[fName] = frame
        i+=1

    cap.release()
    cv2.destroyAllWindows()

DESIRED_HEIGHT = 1080
DESIRED_WIDTH = 1920
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

filePath = r'C:\Users\212574830\Desktop\Project symmetry\Videos\Movement sense video\N08_02_MS.mp4' #N05_01_MS.mp4, TMD05_MS.mp4, N08_02_MS.mp4

extractImagesFromVideo(filePath)

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
    OutPath = r'C:\\Users\\212574830\\Desktop\\Project symmetry\\Images\\' + os.path.basename(filePath)
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
        cv2.imwrite(OutPath +'\\' + name + '.jpg',annotated_image)


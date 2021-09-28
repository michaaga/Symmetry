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
        fName = os.path.basename(path) +  '_' + str(i) +'.jpg'
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

extractImagesFromVideo(r'C:\Users\212574830\Desktop\Project symmetry\Videos\Movement sense video\TMD05_MS.mp4')

# Preview the images.
for name, image in images.items():
  print(name)   
  resize_and_show(image)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
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
    for face_landmarks in results.multi_face_landmarks:
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
          connections=mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_contours_style())
    resize_and_show(annotated_image, True)


import cv2
import numpy as np
import screeninfo
import mediapipe as mp
from data.utils import get_features
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Coords
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# get monitor with and height
monitors = screeninfo.get_monitors()
width = monitors[0].width
height = monitors[0].height

# initialize circle pos
x = 0
y = 30

# set circle direction
direction = 10


# set radius & color
radius = 30
color = (0, 255, 0)  

# 영상 파일 저장을 위한 VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('circle_movement.avi', fourcc, 30, (width, height))

count = True

cap = cv2.VideoCapture(0)
f = open(f'result.csv','a', newline='')
wr = csv.writer(f)
wr.writerow([f'x', 'y','xi_hat','yi_hat','yl'])
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while True:
        # 검은색 배경 영상 생성
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if count:
            for i in range(3, 0, -1):
                cv2.rectangle(frame, (width // 2 - 50, height // 2 - 50), (width // 2 + 50, height // 2 + 50), (0, 0, 0), -1)
                cv2.putText(frame, str(i), (width // 2 - 30, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                cv2.imshow('Circle Movement', frame)
                cv2.waitKey(1000)  # 1초 대기
        count = False    
        
        x = x + direction
        # 동그라미가 화면 밖으로 나가면 방향 변경
        if x + radius > width and direction > 0:
            direction *= -1
            y = y + height//5
        elif x - radius < 0 and direction < 0:
            direction *= -1
            y = y + height//5

        # 카메라로부터 영상 프레임 읽기
        ret, camera_frame = cap.read()

        # 원본 영상과 동그라미를 함께 화면에 표시
        small_cam = cv2.resize(camera_frame,(300,300))
        frame[height - 300:height, width - 300:width] = small_cam
        y_min = 0 if y - radius < 0 else y - radius
        x_min = 0 if x - radius < 0 else x - radius
        frame[y_min:y + radius, x_min:x + radius] = color
        cv2.setWindowProperty('Circle Movement',cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Circle Movement', frame)

        # get features
        frame = cv2.flip(camera_frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        mesh_coords = np.array([(point.x, point.y) for point in results.multi_face_landmarks[0].landmark])
        xi_hat, yi_hat, yl = get_features(mesh_coords)

        epoch = [x-width//2,y-height//2,xi_hat,yi_hat,yl]
        wr.writerow(epoch)

        if y >= height + radius:
            break
        if cv2.waitKey(1) == ord('q'):
            break

    out.write(frame)
f.close()
# 사용한 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()
   


  

   
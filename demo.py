import cv2
import numpy as np
import mediapipe as mp
import time
import math

# Euclidean distance between two points
def euclaideanDistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Values to calculate average FPS
count = 0
fps_sum = 0

# Weight Sum & Bias
W = 0.2
B = 0.13

# Load the video
cap = cv2.VideoCapture(0)

# Initialize the FaceMesh model for face landmark detection
with mp.solutions.face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True, # refine landmarks 468 to 478
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as face_mesh:
    while True:
        # Mark the start time
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        # Double scaling for accuracy
        # You can remove this, if you don't want
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        # Since using a selfie camera, mirror the image horizontally
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            # Landmarks Detection
            mesh_coords = np.array([(point.x, point.y) for point in results.multi_face_landmarks[0].landmark])
            
            # Obtaining the x & y coordinates of the midpoint between the eyebrows
            glabella = (int((mesh_coords[6][0] + mesh_coords[168][0])/2*w), int((mesh_coords[6][1] + mesh_coords[168][1])/2*h))
            
            # Calculating the coordinates of the center for each eye
            # Defining the center of the eyes as the average of the left & right endpoints
            L_center = (mesh_coords[33] + mesh_coords[133])/2
            R_center = (mesh_coords[362] + mesh_coords[263])/2
            
            # Calculating the distance and direction of the iris from the center of the eye
            # Detecting the downward direction of the iris is challenging
            LCen2Iris_x, LCen2Iris_y = mesh_coords[468] - L_center
            RCen2Iris_x, RCen2Iris_y = mesh_coords[473] - R_center
            
            # Measuring the height and width of each eye
            # Height : Intended for detecting the downward direction
            #          Detecting the upward direction is challenging
            # Width : As the distance increases, the value decreases
            #         For correction of the error that occurs in this case
            #
            # Blink detection using the EAR method is possible with these values
            L_height = euclaideanDistance(mesh_coords[159], mesh_coords[145])
            L_width = euclaideanDistance(mesh_coords[33], mesh_coords[133])
            
            R_height = euclaideanDistance(mesh_coords[386], mesh_coords[374])
            R_width = euclaideanDistance(mesh_coords[362], mesh_coords[263])
            
            # Give more weight to the larger eye size
            # When the face rotates, the size of one eye becomes smaller 
            # It makes it difficult to trust the information coming from that eye
            L_weight = L_width/(L_width+R_width)
            R_weight = R_width/(L_width+R_width)
            
            # By dividing the corresponding variable, error correction based on distance is performed            
            # As the distance between the screen and the face increases, all parameter values decrease
            # Select the larger value between the widths of the eyes
            distance_correction = max(L_width, R_width)
            
            
            # Generating Features
            # Distance and head shaking direction compensation are applied
            # xi: x-axis coordinate based on iris location relative to the eye center
            # yi: y-axis coordinate based on iris location relative to the eye center
            # yl: y-axis coordinate based on the height of the eye
            xi = (LCen2Iris_x*L_weight + RCen2Iris_x*R_weight)/distance_correction
            yi = (LCen2Iris_y*L_weight + RCen2Iris_y*R_weight)/distance_correction
            yl = (L_height*L_weight+R_height*R_weight)/distance_correction
            
            # yl is fixed along the y-axis, while xi and yi vary based on head tilt
            # Aligning the axes of xi and yi with the yl axis
            # Determining the theta value using the vector connecting the two eyes
            l = euclaideanDistance(mesh_coords[133], mesh_coords[362])
            cos_theta = (mesh_coords[362][0]-mesh_coords[133][0])/l
            sin_theta = (mesh_coords[362][1]-mesh_coords[133][1])/l
            
            # Rotating by -theta angle
            # | x_hat|    |cos(theta)     sin(theta)|   | x |
            # |      | =  |                         | x |   |
            # | y_hat|    |-sin(theta)    cos(theta)|   | y |
            xi_hat = xi*cos_theta + yi*sin_theta
            yi_hat = -xi*sin_theta + yi*cos_theta
            
            # Viewing direction in the x-axis
            # Scaling for display
            # horiz < 0 : Looking towards the left
            # horiz > 0 : Looking towards the right
            horiz = 1000*(xi_hat)
            
            # Viewing direction in the y-axis
            # Scaling for display
            # Sum of eye height and iris position, weighted, plus bias
            # verti < 0 : Looking upwards
            # verti > 0 : Looking downwards
            verti = 1200*(yi_hat - W*yl + B)
            
            # Rotate by theta again
            # | h_rot|    |cos(theta)     -sin(theta)|   | h |
            # |      | =  |                          | x |   |
            # | v_rot|    |sin(theta)      cos(theta)|   | v |
            horiz_rot = horiz*cos_theta - verti*sin_theta
            verti_rot = horiz*sin_theta + verti*cos_theta
            
            # Display the direction of gaze on the screen
            pos_gaze = (int(glabella[0] + horiz_rot), int(glabella[1] + verti_rot))
            
            cv2.line(frame, glabella, pos_gaze, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, pos_gaze, 15, (0,255,255), 1, cv2.LINE_AA)
            
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        cv2.imshow('img', frame)
        
        # Calculate the FPS
        time_gap = time.time() - start_time
        fps = 1/time_gap
        
        count += 1
        fps_sum += fps
        

        
fps_avg = fps_sum/count
print('Average FPS : {:.2f}'.format(fps_avg))
cap.release()
cv2.destroyAllWindows()            






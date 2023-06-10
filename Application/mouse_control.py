#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import numpy as np
import mediapipe as mp
import time
import math
import win32api, win32con

# Euclidean distance between two points
def euclaideanDistance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# For mouth vertical / horizon
def vhRatio1(h_right, h_left, v_top, v_bottom):
    hDistance = euclaideanDistance(h_right, h_left)

    vDistance = euclaideanDistance(v_top, v_bottom)

    ratio = vDistance/hDistance
    return ratio

# Mouse click
def left_click_down(): # push the left
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def left_click_up():  # release the left
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    
def right_click_down(): # push the right
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
    
def right_click_up(): # release the right
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

# Move the mouse with win32api
def move_cursor(x, y):
    # win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y)

# Color
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

# Values to calculate average FPS
count = 0
fps_sum = 0

# Weight Sum & Bias
W = 0.2
B = 0.13

# For Lip
IsOpenLip = False
IsOpenLip_prev = False

# For Detect eye Blink
std_blink = 5.0
left_click = False
right_click = False

# State
state = 0 # 0: mouse move, 1: click, 2: scroll
state_str = ['Mouse Move', 'Click', 'Scroll']

size_show = 20





# Load the video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open the camera.")
    exit()

# Read a frame from the camera
ret, frame = cap.read()

# Get the height and width of the frame
h, w = frame.shape[:2]

# Initialize status window
show_height, show_width = 8 * size_show, 18 * size_show
h2w_ratio = h/w
resized_h = int(h2w_ratio * show_width)
show_height += resized_h
show_state = np.zeros((show_height, show_width, 3), dtype=np.uint8)

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
            
            # Viewing direction in the x-axis
            # horiz < 0 : Looking towards the left
            # horiz > 0 : Looking towards the right
            horiz = (LCen2Iris_x*L_weight + RCen2Iris_x*R_weight)/distance_correction
            
            # Viewing direction in the y-axis
            # Sum of eye height and iris position, weighted, plus bias
            # verti < 0 : Looking upwards
            # verti > 0 : Looking downwards
            verti = (LCen2Iris_y*L_weight + RCen2Iris_y*R_weight
                    - W*(L_height*L_weight+R_height*R_weight))/distance_correction + B
            
            # Display the direction of gaze on the screen
            pos_gaze = (int(glabella[0] +1000*horiz), int(glabella[1] + 1200*verti))
            
            cv2.line(frame, glabella, pos_gaze, GREEN, 2, cv2.LINE_AA)
            cv2.circle(frame, pos_gaze, 15, YELLOW, 1, cv2.LINE_AA)
            
            show_state[8*size_show:,] = cv2.resize(frame, (show_width, resized_h))
            
            
            # For mouse
            # Detect lip ratio
            IsOpenLip = 1.0 < vhRatio1(mesh_coords[91], mesh_coords[61], mesh_coords[13], mesh_coords[14])
            
            if IsOpenLip and not IsOpenLip_prev:
                state = (state + 1)%3
            
            # Operation as Mode
            if state == 0: # Mouse Move Mode
                move_cursor(int(80*horiz), int(80*verti)) # High fps but only Window
            # Click Mode    
            elif state == 1: 
                l_ratio = L_width/L_height
                r_ratio = R_width/R_height
                if r_ratio > std_blink and l_ratio < std_blink: # Right Blink, Right click
                    if ~right_click:
                        right_click_down()
                        right_click = True
                elif r_ratio < std_blink and l_ratio > std_blink: # Left Blink, Left click
                    if ~left_click:
                        left_click_down()
                        left_click = True
                
                # Opened eye, release
                if r_ratio < std_blink and l_ratio < std_blink:         
                        if right_click:
                            right_click_up()
                            right_click = False
                        if left_click:
                            left_click_up()
                            left_click = False
            # Scroll Mode                
            else: 
                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -int(500*verti), 0)
            
            # Upadate if mouse is open or close
            IsOpenLip_prev = IsOpenLip
            
            
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        # cv2.imshow('img', frame)
        cv2.imshow('status window', show_state)
        
        # Initialize status
        show_state[:,:,:] = 0
        # Set the top and bottom frame borders
        show_state[0:2, :] = WHITE  # Top border
        # Set the left and right frame borders
        show_state[:8*size_show, 0:2] = WHITE  # Left border
        show_state[:8*size_show, -2:] = WHITE  # Right border
        
        show_state[4 * size_show:4 * size_show + 2, :] = WHITE
        show_state[8 * size_show:8 * size_show + 2, :] = WHITE
        
        # Add the current state string in the first row
        text = state_str[state]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        text_size, _ = cv2.getTextSize(text, font, font_scale, 2)
        text_width = text_size[0]
        text_x = (show_width - text_width) // 2
        text_y = 4*size_show - (4*size_show - text_size[1])//2
        cv2.putText(show_state, text, (text_x, text_y), font, font_scale, WHITE, 2)
        
        # Add the next state string in the second row
        next_state = (state+1)%3
        text_next = 'Next:' + state_str[next_state]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        text_size, _ = cv2.getTextSize(text_next, font, font_scale, 2)
        text_x = (show_width - text_size[0]) // 2
        text_y = 8*size_show - (4*size_show - text_size[1])//2
        cv2.putText(show_state, text_next, (text_x, text_y), font, font_scale, WHITE, 2)

        
        
        # Calculate the FPS
        time_gap = time.time() - start_time
        fps = 1/time_gap
        
        count += 1
        fps_sum += fps
        
        
        

        
fps_avg = fps_sum/count
print('Average FPS : {:.2f}'.format(fps_avg))
cap.release()
cv2.destroyAllWindows()            


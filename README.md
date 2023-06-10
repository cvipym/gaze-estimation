# Gaze Estimation (Mediapipe)
> This is Gaze Estimation Project, here we will use coputer vision techniques to extracting the eyes and irises, mediapipe python modules will provide the landmarks
## Description
### 1. Collecting Data
- Saving images for feature extraction
- Dividing and saving into 9 classes
### 2. Data Analysis
- Landmark extraction using Mediapipe
- Parameter extraction from landmarks
- Feature Generation
- Complete fomula for determining gaze direction
### 3. Demo
- Provide a demonstration of the implemented gaze direction determination system
### 4. Application(Mouse Control)
- The controller operates through three distinct modes, each triggered by specific actions. First, by closing and opening the mouth, the mode changes seamlessly. Within the system, we have incorporated a total of three modes
1) Mouse Movement
> By tracking the direction of the user's gaze, the mouse cursor moves accordingly on the screen. You can use it like direction key
2) Clicking
> Leveraging blink detection, the system can execute mouse clicks based on the user's eye blinks. A left eye blink corresponds to a left-click action, while a right eye blink triggers a right-click action. This enables users to interact with various elements on the screen through eye movements.
3) Scrolling
> With the scrolling mode, users can effortlessly scroll through content by aligning their gaze with the desired scrolling direction. As they look up or down, the system detects the gaze direction and performs smooth scrolling accordingly
- Video

[![Video Label](http://img.youtube.com/vi/Rrw5OHCAx_4/0.jpg)](https://youtu.be/Rrw5OHCAx_4)

## Installation
### 1. opencv
```
pip install opencv-python
```
### 2. mediapipe
```
pip install mediapipe
```
## Acknowledgments
I received a lot of help from this repository: <https://github.com/Asadullah-Dal17>

## Contact
E-mail: lym051010@gmail.com

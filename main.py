import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

start_time = time.time()
frame_counter = 0

while True:
    isTrue, frame = capture.read()
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)


    results = hands.process(frameRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)

    
    frame_counter += 1
    fps = frame_counter/(time.time()-start_time)
    fps = int(fps)

    cv.putText(frame, f"{fps} FPS", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)


    cv.imshow("Hand LandMarks Detector", frame)
    cv.waitKey(1)

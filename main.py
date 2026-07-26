import cv2 as cv
import mediapipe as mp
import random

# base = mp.tasks.BaseOptions
# vis = mp.tasks.vision
# hand = vis.HandLandmarker
# hand_opt = vis.HandLandmarkerOptions
# run = mp.tasks.vision.RunningMode

# cv2.putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
x = 100
i = 0
font_size = 1.2
word = "I Love U"
while cap.isOpened():
    s, frame = cap.read()
    if not s:
        break
    img = cv.flip(cv.blur(frame,(20,20)), 1)
    h,w,_ = img.shape
    target_w = int(h * (9 / 16))
    start_x = (w - target_w) // 2
    end_x = start_x + target_w
    vertical_crop = img[0:h, start_x:end_x]
    i+=10
    if i >= h:
        i = 0
        x = random.randint(1, target_w-5)
        font_size = float(random.randint(9,13))/10
    for j in range(len(word)):
        print(j)
        cv.putText(vertical_crop,word[j],(x, i + j*40),cv.FONT_HERSHEY_PLAIN, font_size, (0,255,0),2, cv.LINE_AA)
        
    cv.imshow("test", vertical_crop)
    if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty("test", cv.WND_PROP_VISIBLE)<1:
        break 
cap.release()
cv.destroyAllWindows()
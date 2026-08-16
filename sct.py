import cv2, mediapipe as mp

base , vis = mp.tasks.BaseOptions, mp.tasks.vision

opt = vis.HandLandmarkerOptions(
    base_options=base(model_asset_path='hand_landmarker.task'),
    running_mode=vis.RunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

fps = 0

with vis.HandLandmarker.create_from_options(opt) as landmarker:
    while True:
        run, img = cap.read()
        if not run: break



        img = cv2.flip(img,1)

        fps+=33
        
        mp_img = mp.Image(mp.ImageFormat.SRGB,cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(mp_img, fps)

        h, w, _ = img.shape

        if res.hand_landmarks:
            for hand in res.hand_landmarks:
                dist = (((hand[4].x-hand[8].x)**2+(hand[4].y-hand[8].y)**2)**0.5)*100   #math.dist

                pt1 = (int(hand[4].x * w), int(hand[4].y * h))
                pt2 = (int(hand[8].x * w), int(hand[8].y * h))

                cv2.line(img, pt1, pt2, (0, 0, 0), 2)

                cent = (int((hand[4].x+hand[8].x)*w/2), int((hand[4].y+hand[8].y)*h/2)) 

                cv2.circle(img,cent,10,(255,255,255),-1)
                cv2.circle(img,cent,11,(0,0,0),1)

                result = 1 + (dist - 2) * 99 / 34   #can be switch to numpy.interp()
                result = max(1, min(result, 100))

                cv2.putText(img, str(int(result)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('title',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()


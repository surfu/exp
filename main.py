# Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
import cv2, mediapipe as mp
base, vis = mp.tasks.BaseOptions, mp.tasks.vision

opt = vis.HandLandmarkerOptions(
    base('hand_landmarker.task'),
    vis.RunningMode.VIDEO,
    1
)
fps = 0
cap  = cv2.VideoCapture(0)

with vis.HandLandmarker.create_from_options(opt) as det:
    while True:
        run, img = cap.read()

        img = cv2.flip(img ,1)

        fps +=30

        mp_img = mp.Image(mp.ImageFormat.SRGB,cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        res = det.detect_for_video(mp_img,fps)
        h,w = img.shape[:2]

        if res.hand_landmarks:
            for hand in res.hand_landmarks:

                dist =( ((hand[4].x-hand[8].x)**2+(hand[4].y-hand[8].y)**2)**0.5)*100
                res = 1 + (dist-2)*99/(35-2)
                res =int(max(1,min(res,100)))
                img = cv2.resize(cv2.resize(img, (w//res, h//res)),(w,h),interpolation=cv2.INTER_NEAREST)

                cv2.putText(img,str(res),(200,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)

        cv2.imshow("name", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
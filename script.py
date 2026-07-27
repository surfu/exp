import cv2,  time
import mediapipe as mp

model = 'info/holistic_landmarker.task'

baseopt = mp.tasks.BaseOptions
landm = mp.tasks.vision.HolisticLandmarker
landopt = mp.tasks.vision.HolisticLandmarkerOptions
run = mp.tasks.vision.RunningMode
Img = mp.Image
ImgF = mp.ImageFormat
mp_dw = mp.tasks.vision.drawing_utils
mp_st = mp.tasks.vision.drawing_styles
mp_con = mp.tasks.vision.PoseLandmarksConnections

opt = landopt(
    base_options=baseopt(model_asset_path= model),
    running_mode=run.VIDEO
)

cap = cv2.VideoCapture(0)

with landm.create_from_options(opt) as lm:
    while cap.isOpened():
        s, frame = cap.read()
        if not s:
            break
        rev = cv2.flip(frame,1)
        rgb = cv2.cvtColor(rev,cv2.COLOR_BGR2RGB)
        mp_img = Img(image_format=ImgF.SRGB,data=rgb)
        frame_timestamp_ms = int(time.time() * 1000)
        result = lm.detect_for_video(mp_img, frame_timestamp_ms)
        h, w, _ = rev.shape
        if result.pose_landmarks:
            mp_dw.draw_landmarks(
                rev,
                result.pose_landmarks,  # Pass the whole list directly
                mp_con.POSE_LANDMARKS,  # Make sure it's POSE_CONNECTIONS (plural)
                mp_st.get_default_pose_landmarks_style()
            )
            # nose = result.pose_landmarks[0]
            # cv2.circle(rev, (int(nose.x * w), int(nose.y * h)), 5, (0, 255, 0), -1)
            # sc = result.pose_landmarks[12]
            # cv2.circle(rev, (int(sc.x * w), int(sc.y * h)), 5, (0, 255, 0), -1)

        cv2.imshow('Holistic Landmarker', rev)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Holistic Landmarker", cv2.WND_PROP_VISIBLE) <1:
            break

cap.release()
cv2.destroyAllWindows()
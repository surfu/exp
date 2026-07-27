import cv2, os
import mediapipe as mp
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "hand_landmarker.task")
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=dataset_path),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "video.mp4")
latest_result = None
vid = cv2.VideoCapture(video_path)

with mp.tasks.vision.HandLandmarker.create_from_options(options) as det:
    while vid.isOpened():
        s , fr = vid.read()
        if not s:
            break

        resided_vid = cv2.resize(fr,(640,360))
        rgb_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(vid.get(cv2.CAP_PROP_POS_MSEC))
        res = det.detect_for_video(mp_image, frame_timestamp_ms)       
        if res and res.hand_landmarks: 
            hand_lst = res.hand_landmarks
            handedness = res.handedness
            for idx in range(len(hand_lst)):
                hand_landmarks = hand_lst[idx]
                ness = handedness[idx]
                mp_drawing.draw_landmarks(
                        resided_vid,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                ind = hand_landmarks[8]
                ind_2 = hand_landmarks[7]
                mid = hand_landmarks[12]
                mid_2 = hand_landmarks[11]
                thumb_tip = hand_landmarks[16]
                thumb_mcp = hand_landmarks[13]
                
                up = ind.y<ind_2.y and mid.y<mid_2.y
                dwn = thumb_tip.y >thumb_mcp.y
                if up and dwn:
                    blur = cv2.blur(resided_vid,(20,20))
                    resided_vid = blur

        cv2.imshow('test',resided_vid)
        
        cv2.moveWindow('test',0,0)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("test", cv2.WND_PROP_VISIBLE)<1:
            break
vid.release()
cv2.destroyAllWindows()

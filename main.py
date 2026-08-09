import cv2 as cv, mediapipe as mp, time, numpy as np

base, vis = mp.tasks.BaseOptions, mp.tasks.vision
options = vis.HandLandmarkerOptions(
    base_options=base(model_asset_path='hand_landmarker.task'),
    running_mode=vis.RunningMode.VIDEO, 
    num_hands=2
)

mp_draw = vis.drawing_utils
mp_hands = vis.HandLandmarksConnections
mp_styles = vis.drawing_styles

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

recorded_frames = []
start_time = None
frame_timestamp_ms = 0

with vis.HandLandmarker.create_from_options(options) as landmarker:
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            if start_time is None:
                start_time = time.time()

            img = cv.flip(frame, 1)
            h,w,_= img.shape

            cv.rectangle(img, (120, 50), (190, 120), (255, 0, 0), -1)
            cv.rectangle(img, (190, 50), (260, 120), (0, 255, 0), -1)
            cv.rectangle(img, (260, 50), (330, 120), (0, 0, 255), -1)

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(img, cv.COLOR_BGR2RGB))
            frame_timestamp_ms += 33
            res = landmarker.detect_for_video(mp_img, frame_timestamp_ms)

            if res.hand_landmarks:
                for hand in res.hand_landmarks:
                    mp_draw.draw_landmarks(
                        img, hand, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # Film 
            film = img.astype(np.float32)
            film[:, :, 0] *= 0.80
            film[:, :, 1] *= 0.95
            film[:, :, 2] *= 1.15
            film = np.clip(film, 0, 255).astype(np.uint8)
            grain = np.random.randint(-18, 18, (h, w, 3), dtype=np.int16)
            film = np.clip(film.astype(np.int16) + grain, 0, 255).astype(np.uint8)
            kernel_x = cv.getGaussianKernel(w, w / 2)
            kernel_y = cv.getGaussianKernel(h, h / 2)
            mask = (kernel_y * kernel_x.T) / (kernel_y * kernel_x.T).max()
            for i in range(3):
                film[:, :, i] = (film[:, :, i] * mask).astype(np.uint8)
            img = film
            img[:, :, 0] = img[:, :, 0] * 0.7  # Lower Blue
            img[:, :, 2] = cv.add(img[:, :, 2], 30) 
            recorded_frames.append(img)

            cv.imshow("test", img)
            if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty("test", cv.WND_PROP_VISIBLE) < 1:
                break
    finally:
        total_time = time.time() - start_time if start_time else 0
        cap.release()
        cv.destroyAllWindows()

if recorded_frames and total_time > 0:
    actual_fps = len(recorded_frames) / total_time

    fourcc = cv.VideoWriter.fourcc(*'mp4v')
    output = cv.VideoWriter("output.mp4", cv.CAP_FFMPEG, fourcc, actual_fps, (width, height))

    for frame in recorded_frames:
        output.write(frame)
        
    output.release()
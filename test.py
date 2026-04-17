import cv2

from src.utils.face_detection import FaceDetection
from src.utils.hand_detection import HandDetection
from src.utils.pose_detection import PoseDetection

video_path = r'datasets/raw/how2sign_raw\0-0kX3XoMPQ_6-3-rgb_front.mp4'
cap = cv2.VideoCapture(video_path)

hand_detection = HandDetection()
pose_detection = PoseDetection()
face_detection = FaceDetection()

while True:
    ret, frame = cap.read() # (720, 1280, 3)
    frame = cv2.resize(frame, (256, 256))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # frame = cv2.resize(frame, (224, 224))
    # results = hand_detection.detect_video(frame) # handedness, NormalizedLandmark, NormalizedLandmark
    # frame_result = hand_detection.draw_landmarks_on_image(frame, results)

    # results = pose_detection.detect_pose(frame)
    # frame_result = pose_detection.draw_landmarks_on_image(frame, results)

    results = face_detection.detect_face(rgb_frame)
    frame_result = face_detection.draw_landmarks_on_image(rgb_frame, results)

    cv2.imshow("frame", frame_result)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# 5. Always release the capture and close windows when finished
cap.release()
cv2.destroyAllWindows()
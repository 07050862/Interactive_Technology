import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)

        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)

        # Right hand
        # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Left Hand
        # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Pose Detections
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        '''
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w,c = image.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            print(id, lm, cx, cx)
            cv2.circle(image, (cx, cy), 5, (255,0,0), cv2.FILLED)
        '''
        # Nose is the 0 landmark
        h, w, c = image.shape
        cx = int(results.pose_landmarks.landmark[15].x * w)
        cy = int(results.pose_landmarks.landmark[15].y * h)
        print(cx, cy)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        cx = int(results.pose_landmarks.landmark[16].x * w)
        cy = int(results.pose_landmarks.landmark[16].y * h)
        print(cx, cy)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


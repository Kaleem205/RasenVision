import cv2
import mediapipe as mp
from recognition.pose_classifier import PoseClassifier
from recognition.motion_detector import MotionDetector

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2 
    )
    
    # Initialize both recognition modules
    pose_classifier = PoseClassifier()
    motion_detector = MotionDetector(buffer_size=15)

    cap = cv2.VideoCapture(1)

    print("Starting RasenVision camera... Press 'q' on your keyboard to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 1. Process Static Pose
                current_pose = pose_classifier.classify(hand_landmarks)
                
                # 2. Process Dynamic Motion
                wrist = hand_landmarks.landmark[0]
                motion_detector.update(wrist)
                current_motion = motion_detector.detect_motion()
                
                # Render results on screen
                h, w, c = frame.shape
                wrist_x = int(wrist.x * w)
                wrist_y = int(wrist.y * h)
                
                # Draw Pose in Green
                cv2.putText(frame, f"Pose: {current_pose}", (wrist_x - 50, wrist_y - 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                # Draw Motion in Orange
                cv2.putText(frame, f"Motion: {current_motion}", (wrist_x - 50, wrist_y - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.imshow('RasenVision - Motion Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
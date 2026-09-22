import cv2
import mediapipe as mp
from recognition.pose_classifier import PoseClassifier
from recognition.motion_detector import MotionDetector
from fsm.gesture_fsm import GestureFSM

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2 
    )
    
    # Initialize all three modules
    pose_classifier = PoseClassifier()
    motion_detector = MotionDetector(buffer_size=15)
    fsm = GestureFSM()

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

        # Default state if no hands are visible
        current_state = "IDLE"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract data
                current_pose = pose_classifier.classify(hand_landmarks)
                
                wrist = hand_landmarks.landmark[0]
                motion_detector.update(wrist)
                current_motion = motion_detector.detect_motion()
                
                # Feed data into the FSM Brain
                current_state = fsm.update(current_pose, current_motion)
                
                # Render debug text
                h, w, c = frame.shape
                wrist_x = int(wrist.x * w)
                wrist_y = int(wrist.y * h)
                cv2.putText(frame, f"Pose: {current_pose}", (wrist_x - 50, wrist_y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, f"Motion: {current_motion}", (wrist_x - 50, wrist_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw the MASTER STATE prominently at the top of the screen
        state_color = (0, 0, 255) if current_state == "ACTIVATED" else (0, 255, 255) if current_state == "CHARGING" else (255, 255, 255)
        cv2.putText(frame, f"SYSTEM STATE: {current_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3, cv2.LINE_AA)

        cv2.imshow('RasenVision - FSM Logic', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
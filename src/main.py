import cv2
import mediapipe as mp
import math
from recognition.pose_classifier import PoseClassifier
from recognition.motion_detector import MotionDetector
from fsm.gesture_fsm import GestureFSM
from rendering.procedural_renderer import ProceduralRenderer

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2 
    )
    
    pose_classifier = PoseClassifier()
    motion_detector = MotionDetector(buffer_size=15)
    fsm = GestureFSM()
    
    # Load the VFX video here
    renderer = ProceduralRenderer()

    cap = cv2.VideoCapture(1)
    print("Starting RasenVision camera... Press 'q' on your keyboard to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_state = "IDLE"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                current_pose = pose_classifier.classify(hand_landmarks)
                wrist = hand_landmarks.landmark[0]
                motion_detector.update(wrist)
                current_motion = motion_detector.detect_motion()
                
                current_state = fsm.update(current_pose, current_motion)
                h, w, c = frame.shape
                
                if current_state == "ACTIVATED":

                    # True palm center: average of wrist + all 4 knuckle bases
                    knuckle_ids = [0, 5, 9, 13, 17]
                    palm_x = int(sum(hand_landmarks.landmark[i].x for i in knuckle_ids) / len(knuckle_ids) * w)
                    palm_y = int(sum(hand_landmarks.landmark[i].y for i in knuckle_ids) / len(knuckle_ids) * h)

                    index_finger = hand_landmarks.landmark[5]
                    dist = math.sqrt((wrist.x - index_finger.x)**2 + (wrist.y - index_finger.y)**2)
                    scale = dist * 8.0   # bumped up from 5.0 for a bigger effect
                    
                    frame = renderer.draw_effect(frame, palm_x, palm_y, scale_factor=scale)
                else:
                    renderer.reset()
                
                cv2.putText(frame, f"Pose: {current_pose}", (int(wrist.x * w) - 50, int(wrist.y * h) - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        state_color = (0, 0, 255) if current_state == "ACTIVATED" else (0, 255, 255) if current_state == "CHARGING" else (255, 255, 255)
        cv2.putText(frame, f"SYSTEM STATE: {current_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3, cv2.LINE_AA)

        cv2.imshow('RasenVision - Video VFX', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
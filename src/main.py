import cv2
import mediapipe as mp
import math
from recognition.pose_classifier import PoseClassifier
from recognition.motion_detector import MotionDetector
from fsm.gesture_fsm import GestureFSM
from rendering.procedural_renderer import ProceduralRenderer

def main():
    mp_hands = mp.solutions.hands
    
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1 
    )
    
    pose_classifier = PoseClassifier()
    motion_detector = MotionDetector(buffer_size=15)
    fsm = GestureFSM()
    
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

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            wrist = hand_landmarks.landmark[0]
            
            # --- NEW FIX: Cross-Product Rotation Logic ---
            # Calculate vectors from wrist to Index MCP and wrist to Pinky MCP
            v_idx_x = hand_landmarks.landmark[5].x - wrist.x
            v_idx_y = hand_landmarks.landmark[5].y - wrist.y
            
            v_pnk_x = hand_landmarks.landmark[17].x - wrist.x
            v_pnk_y = hand_landmarks.landmark[17].y - wrist.y
            
            # 2D Cross product determines if the palm surface is facing the camera
            cross_z = (v_idx_x * v_pnk_y) - (v_idx_y * v_pnk_x)
            
            # Using a threshold (0.005) so if the hand is horizontal/edge-on 
            # (palm facing ceiling or floor), it won't falsely trigger as facing the camera.
            if handedness == "Left":
                is_palm_facing = cross_z > 0.005
            else:
                is_palm_facing = cross_z < -0.005
            # ---------------------------------------------
            
            current_pose = pose_classifier.classify(hand_landmarks)
            motion_detector.update(wrist)
            current_motion = motion_detector.detect_motion()
            
            current_state = fsm.update(current_pose, current_motion)
            
            # Dissipate if the palm IS firmly facing the camera
            if is_palm_facing:
                fsm.state = "IDLE"
                fsm.stable_frames = 0
                current_state = "IDLE"
                current_pose = "PALM_OF_HAND (HIDDEN)"
            
            h, w, c = frame.shape
            
            if current_state == "ACTIVATED":
                knuckle_ids = [0, 5, 9, 13, 17]
                palm_x = int(sum(hand_landmarks.landmark[i].x for i in knuckle_ids) / len(knuckle_ids) * w)
                palm_y = int(sum(hand_landmarks.landmark[i].y for i in knuckle_ids) / len(knuckle_ids) * h)

                index_finger = hand_landmarks.landmark[5]
                dist = math.sqrt((wrist.x - index_finger.x)**2 + (wrist.y - index_finger.y)**2)
                scale = dist * 8.0   
                
                frame = renderer.draw_effect(frame, palm_x, palm_y, scale_factor=scale)
            else:
                renderer.reset()
            
            cv2.putText(frame, f"Pose: {current_pose}", (int(wrist.x * w) - 50, int(wrist.y * h) - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            renderer.reset()

        state_color = (0, 0, 255) if current_state == "ACTIVATED" else (0, 255, 255) if current_state == "CHARGING" else (255, 255, 255)
        cv2.putText(frame, f"SYSTEM STATE: {current_state}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 3, cv2.LINE_AA)

        cv2.imshow('RasenVision - Video VFX', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
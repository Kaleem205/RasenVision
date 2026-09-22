import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Configure the hand tracking model
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=2 # We want to track up to 2 hands for Naruto hand signs!
    )

    # Open the default Mac webcam (1 is usually the built-in camera)
    cap = cv2.VideoCapture(1)

    print("Starting RasenVision camera... Press 'q' on your keyboard to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # Flip the frame horizontally so it acts like a mirror
        frame = cv2.flip(frame, 1)
        
        # OpenCV uses BGR colors, but MediaPipe needs RGB. Convert it:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # This is where the magic happens: find the hands!
        results = hands.process(rgb_frame)

        # Draw the 21 dots and connecting lines on the hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Show the video window
        cv2.imshow('RasenVision - Hand Tracking', frame)

        # Listen for the 'q' key to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

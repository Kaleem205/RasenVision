import math
from collections import deque

class MotionDetector:
    def __init__(self, buffer_size=15):
        # This deque acts as our sliding window, remembering the last 15 frames
        self.history = deque(maxlen=buffer_size)

    def update(self, wrist_landmark):
        # Add the current wrist X and Y coordinates to our memory
        self.history.append((wrist_landmark.x, wrist_landmark.y))

    def detect_motion(self):
        # We need a full buffer to accurately calculate movement over time
        if len(self.history) < self.history.maxlen:
            return "CALIBRATING..."

        # Get the oldest and newest positions in memory
        start_x, start_y = self.history[0]
        end_x, end_y = self.history[-1]

        # Calculate the total displacement (distance moved)
        dx = end_x - start_x
        dy = end_y - start_y
        distance = math.sqrt(dx**2 + dy**2)

        # MediaPipe coordinates are normalized from 0.0 to 1.0. 
        # A distance of 0.10 means the hand moved 10% across the screen.
        if distance > 0.10:
            if abs(dx) > abs(dy):
                return "SWIPING"
            else:
                if dy > 0:
                    return "MOVING DOWN"
                else:
                    return "THRUSTING UP"
        
        return "STABLE"

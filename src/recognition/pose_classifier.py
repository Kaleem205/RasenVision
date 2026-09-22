import math

class PoseClassifier:
    def __init__(self):
        pass

    def get_distance(self, point1, point2):
        """Calculate the 2D distance between two landmarks."""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def classify(self, hand_landmarks):
        # Point 0 is always the wrist
        wrist = hand_landmarks.landmark[0]
        
        # Landmark indices for the 4 main fingertips and their corresponding knuckles
        finger_tips = [8, 12, 16, 20] 
        finger_knuckles = [5, 9, 13, 17]
        
        fingers_open = 0
        
        # Compare the distance of each fingertip and knuckle to the wrist
        for tip, knuckle in zip(finger_tips, finger_knuckles):
            tip_dist = self.get_distance(wrist, hand_landmarks.landmark[tip])
            knuckle_dist = self.get_distance(wrist, hand_landmarks.landmark[knuckle])
            
            # If the fingertip is further from the wrist than the knuckle, the finger is extended
            if tip_dist > knuckle_dist:
                fingers_open += 1
                
        if fingers_open == 4:
            return "OPEN PALM"
        elif fingers_open == 0:
            return "CLOSED FIST"
        else:
            return "CHARGING..."

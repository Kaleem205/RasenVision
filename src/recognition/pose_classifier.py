import math

class PoseClassifier:
    def classify(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        
        # We check the 4 main fingers: Index, Middle, Ring, Pinky
        # Pairs of (Fingertip ID, Knuckle ID)
        fingers = [(8, 5), (12, 9), (16, 13), (20, 17)]
        extended_count = 0
        
        for tip_idx, knuckle_idx in fingers:
            tip = hand_landmarks.landmark[tip_idx]
            knuckle = hand_landmarks.landmark[knuckle_idx]
            
            # Calculate true 3D distance from the wrist to the fingertip
            dist_tip = math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
            
            # Calculate true 3D distance from the wrist to the base knuckle
            dist_knuckle = math.sqrt((knuckle.x - wrist.x)**2 + (knuckle.y - wrist.y)**2 + (knuckle.z - wrist.z)**2)
            
            # If the tip is further from the wrist than the knuckle, the finger is extended
            if dist_tip > dist_knuckle * 1.15:
                extended_count += 1
                
        # Determine the pose based on how many fingers are extended out
        if extended_count >= 3:
            return "OPEN PALM"
        elif extended_count <= 1:
            return "CLOSED FIST"
        else:
            return "CHARGING..."

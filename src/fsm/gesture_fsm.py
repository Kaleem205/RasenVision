class GestureFSM:
    def __init__(self):
        self.state = "IDLE"
        self.stable_frames = 0
        self.required_frames = 5  # How many consecutive frames to confirm a gesture

    def update(self, pose, motion):
        # 1. IDLE STATE: Waiting for the user to make a fist to start charging
        if self.state == "IDLE":
            if pose == "CLOSED FIST":
                self.stable_frames += 1
                if self.stable_frames >= self.required_frames:
                    self.state = "CHARGING"
                    self.stable_frames = 0
            else:
                self.stable_frames = 0
                
        # 2. CHARGING STATE: Waiting for the user to open their palm
        elif self.state == "CHARGING":
            if pose == "OPEN PALM":
                self.stable_frames += 1
                if self.stable_frames >= self.required_frames:
                    self.state = "ACTIVATED"
                    self.stable_frames = 0
            elif pose != "CLOSED FIST" and pose != "OPEN PALM":
                # If they do something else, cancel the charge
                self.state = "IDLE"
                self.stable_frames = 0

        # 3. ACTIVATED STATE: The Rasenshuriken is active! Waiting for a throw.
        elif self.state == "ACTIVATED":
            # If the user swipes their hand or closes their fist, cancel/throw the effect
            if motion == "SWIPING" or pose == "CLOSED FIST":
                self.state = "IDLE"
                
        return self.state

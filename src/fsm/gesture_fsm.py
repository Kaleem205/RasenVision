class GestureFSM:
    def __init__(self):
        self.state = "IDLE"
        self.stable_frames = 0
        self.required_frames = 5
        self.no_pose_frames = 0
        self.max_no_pose_frames = 20  # ~0.6s of "neither pose" before we give up

    def update(self, pose, motion):
        # 1. IDLE: waiting for a closed fist to start charging
        if self.state == "IDLE":
            if pose == "CLOSED FIST":
                self.stable_frames += 1
                if self.stable_frames >= self.required_frames:
                    self.state = "CHARGING"
                    self.stable_frames = 0
            else:
                self.stable_frames = 0

        # 2. CHARGING: waiting for the palm to open
        elif self.state == "CHARGING":
            if pose == "OPEN PALM":
                self.stable_frames += 1
                self.no_pose_frames = 0
                if self.stable_frames >= self.required_frames:
                    self.state = "ACTIVATED"
                    self.stable_frames = 0
            elif pose == "CLOSED FIST":
                # still fisted — stay charging, don't punish, just don't progress
                self.stable_frames = 0
                self.no_pose_frames = 0
            else:
                # transitional pose ("CHARGING..." — fingers partway open):
                # don't reset immediately, just don't count it as stable yet
                self.stable_frames = 0
                self.no_pose_frames += 1
                if self.no_pose_frames >= self.max_no_pose_frames:
                    # only cancel if this drags on way too long
                    self.state = "IDLE"
                    self.no_pose_frames = 0

        # 3. ACTIVATED: effect is live, waiting for cancel
        elif self.state == "ACTIVATED":
            # Removed the swipe cancellation. 
            # Fast movements will no longer destroy the Rasenshuriken.
            if pose == "CLOSED FIST":
                self.state = "IDLE"

        return self.state
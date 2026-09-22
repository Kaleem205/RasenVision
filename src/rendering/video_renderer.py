import cv2
import numpy as np

class VideoRenderer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f"ERROR: Could not load video at {video_path}")

        self.is_active = False
        self._mask_cache = {}

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.is_active = False

    def _radial_mask(self, size):
        """Soft circular falloff so the video's rectangular edges fade to
        nothing instead of showing a visible box around the effect."""
        if size in self._mask_cache:
            return self._mask_cache[size]
        y, x = np.ogrid[:size, :size]
        cx = cy = size / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (size / 2)
        mask = np.clip(1.0 - dist, 0, 1) ** 1.5  # smooth falloff, full bright center
        mask_3ch = np.dstack([mask, mask, mask]).astype(np.float32)
        self._mask_cache[size] = mask_3ch
        return mask_3ch

    def draw_effect(self, frame, center_x, center_y, scale_factor=1.0):
        self.is_active = True

        success, vfx_frame = self.cap.read()
        if not success:
            # loop from a point where the effect is already large/formed
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
            success, vfx_frame = self.cap.read()
            if not success:
                return frame

        vh, vw = vfx_frame.shape[:2]
        zoom_factor = 1.6  # lower than before — less aggressive crop, keeps more of the swirl visible
        crop_h, crop_w = int(vh / zoom_factor), int(vw / zoom_factor)
        cy, cx = vh // 2, vw // 2
        vfx_cropped = vfx_frame[cy - crop_h // 2: cy + crop_h // 2,
                                 cx - crop_w // 2: cx + crop_w // 2]

        h, w = frame.shape[:2]
        box_size = int(700 * scale_factor)
        box_size = min(box_size, h - 10, w - 10)
        if box_size < 20:
            box_size = 20
        if box_size % 2 == 0:
            box_size += 1
        half_box = box_size // 2

        y1 = center_y - half_box
        y2 = y1 + box_size
        x1 = center_x - half_box
        x2 = x1 + box_size
        if y1 < 0:
            y2 -= y1; y1 = 0
        if x1 < 0:
            x2 -= x1; x1 = 0
        if y2 > h:
            y1 -= (y2 - h); y2 = h
        if x2 > w:
            x1 -= (x2 - w); x2 = w
        y1, x1 = max(0, y1), max(0, x1)
        box_size = min(y2 - y1, x2 - x1)
        if box_size < 10:
            return frame

        vfx_resized = cv2.resize(vfx_cropped, (box_size, box_size))
        mask = self._radial_mask(box_size)

        # apply the radial fade to the VFX before blending, so the far
        # edges of the video frame contribute nothing near the boundary
        vfx_masked = (vfx_resized.astype(np.float32) * mask)

        roi = frame[y1:y2, x1:x2].astype(np.float32)
        blended = np.clip(cv2.add(roi, vfx_masked), 0, 255).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended

        return frame
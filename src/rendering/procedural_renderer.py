import cv2
import numpy as np
import math
import random

class ProceduralRenderer:
    def __init__(self):
        self.anim_frame = 0
        self.particles = []
        for _ in range(150):
            self.particles.append([
                random.uniform(0, 2 * math.pi),
                random.uniform(1.2, 4.0),  
                random.uniform(10, 25),    
                random.uniform(0, 2 * math.pi)      
            ])

    def reset(self):
        self.anim_frame = 0

    def draw_effect(self, frame, center_x, center_y, scale_factor=1.0):
        self.anim_frame += 1
        
        h, w = frame.shape[:2]
        
        box_size = int(650 * scale_factor)
        box_size = max(30, min(box_size, h - 10, w - 10))
        if box_size % 2 == 0: 
            box_size += 1
            
        half_box = box_size // 2

        # FIXED BOUNDARY LOGIC: Shifts the box inward if it hits the screen edge 
        # so the array size always perfectly matches the effect canvas
        y1 = center_y - half_box
        y2 = y1 + box_size
        x1 = center_x - half_box
        x2 = x1 + box_size
        
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w
            
        y1, x1 = max(0, y1), max(0, x1)
        
        if y2 - y1 < 10 or x2 - x1 < 10: 
            return frame

        canvas = np.zeros((box_size, box_size, 3), dtype=np.uint8)
        center = (box_size // 2, box_size // 2)
        
        growth = min(1.0, self.anim_frame / 12.0)
        base_r = int((1 - (1 - growth) ** 3) * 75 * scale_factor) 
        
        # Fade in to prevent the blue flash
        fade = min(1.0, self.anim_frame / 5.0)
        b = int(255 * fade)
        g = int(100 * fade)
        r_color = int(20 * fade)
        
        cv2.circle(canvas, center, int(base_r * 1.15), (b, g, r_color), -1)
        canvas = cv2.GaussianBlur(canvas, (51, 51), 0)

        # --- 3D Particle Wind Simulation ---
        wind_layer = np.zeros_like(canvas)
        tilt = 0.45 
        
        for p in self.particles:
            p[0] += math.radians(p[2])
            pulse = math.sin(self.anim_frame * 0.1 + p[3]) * 0.3
            current_r = base_r * (p[1] + pulse)
            
            x = int(center[0] + math.cos(p[0]) * current_r)
            y = int(center[1] + math.sin(p[0]) * current_r * tilt)
            
            depth_intensity = (math.sin(p[0]) + 1) / 2 
            brightness = int(50 + (205 * depth_intensity))
            color = (255, 230, brightness)
            
            tail_x = int(center[0] + math.cos(p[0] - 0.15) * current_r)
            tail_y = int(center[1] + math.sin(p[0] - 0.15) * current_r * tilt)
            
            cv2.line(wind_layer, (tail_x, tail_y), (x, y), color, max(1, int(3 * scale_factor)))

        wind_layer = cv2.GaussianBlur(wind_layer, (7, 7), 0)
        
        # --- Fast-spinning internal structure ---
        lines_layer = np.zeros_like(canvas)
        for i in range(8):
            angle = self.anim_frame * (15 + i) + (i * 45)
            axes = (int(base_r * 0.95), int(base_r * 0.35))
            cv2.ellipse(lines_layer, center, axes, angle, 0, 360, (255, 230, 80), max(1, int(2*scale_factor)))
            
            angle2 = -self.anim_frame * (18 + i) + (i * 30)
            axes2 = (int(base_r * 0.8), int(base_r * 0.2))
            cv2.ellipse(lines_layer, center, axes2, angle2, 0, 360, (255, 255, 200), max(1, int(1.5*scale_factor)))

        # --- Turbulent bright core ---
        core_wobble = int(math.sin(self.anim_frame * 0.8) * 3 * scale_factor)
        core_r = int(base_r * 0.35) + core_wobble
        
        cv2.circle(lines_layer, center, core_r, (255, 255, 255), -1)
        cv2.circle(lines_layer, center, int(core_r * 1.3), (255, 200, 50), max(1, int(4 * scale_factor)))
        lines_layer = cv2.GaussianBlur(lines_layer, (5, 5), 0)
        
        final_effect = cv2.add(canvas, lines_layer)
        final_effect = cv2.add(final_effect, wind_layer)

        # Additive Blending
        roi = frame[y1:y2, x1:x2].astype(np.float32)
        blended = np.clip(cv2.add(roi, final_effect.astype(np.float32)), 0, 255).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended
        
        return frame
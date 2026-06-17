import cv2
import numpy as np
import time
import math

class Cup:
    def __init__(self):
        self.is_present = False
        self.is_guarding = False
        self.area_threshold = 30000
        self.steady_start_time = 0.0
        self.confirm_delay = 1.0
        self.falling_threshold = 50

        self.box = None
        self.center = None
        self.prev_center = None
        self.occupied_cells = set()
        self.locked_mask = None

        self.cup_body_mask = None
        self.original_cup_area = 0

    def check_stability(self, candidate_box, candidate_center):
        if candidate_box is not None:
            if self.steady_start_time == 0.0:
                self.steady_start_time = time.time()
            elif time.time() - self.steady_start_time > self.confirm_delay:
                self.is_present = True
                self.box = candidate_box
                self.center = candidate_center
                self.prev_center = candidate_center
                self.steady_start_time = 0.0
                print("Cup!!!")
                return True
        else:
            self.steady_start_time = 0.0

        return False

    def calculate_defense_wall(self, frame_shape, grid_start_x, grid_start_y, cell_w, cell_h):
        cup_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(cup_mask, [self.box], 255)

        self.cup_body_mask = cup_mask.copy()
        self.original_cup_area = cv2.countNonZero(self.cup_body_mask)

        inv_cup_mask = cv2.bitwise_not(cup_mask)

        # 畫出超厚的邊界線
        guard_thickness = 180
        cup_line_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.polylines(cup_line_mask, [self.box], True, 255, guard_thickness)

        # 魔法交集：只取外側
        self.locked_mask = cv2.bitwise_and(cup_line_mask, inv_cup_mask)

        # 檢查 16 個格子並分發指令
        self.occupied_cells.clear()
        cell_area = cell_w * cell_h

        for row in range(4):
            for col in range(4):
                cx = grid_start_x + col * cell_w
                cy = grid_start_y + row * cell_h

                cell_cup_guard = self.locked_mask[cy:cy+cell_h, cx:cx+cell_w]
                cell_cup_body = cup_mask[cy:cy+cell_h, cx:cx+cell_w]

                # 絕對互斥條件：防護區 > 15%，且實體覆蓋 < 5%
                if (cv2.countNonZero(cell_cup_guard) / cell_area) > 0.15 and \
                   (cv2.countNonZero(cell_cup_body) / cell_area) < 0.2:
                    self.occupied_cells.add(row * 4 + col)

        print(f"✅ 計算完成！隨時準備啟動外圍馬達防波堤: {list(self.occupied_cells)}")

    def check_spill(self, current_center):
        if not self.is_guarding and current_center is not None and self.prev_center is not None:
            dx = current_center[0] - self.prev_center[0]
            dy = current_center[1] - self.prev_center[1]

            if math.hypot(dx, dy) > self.falling_threshold: # sqrt(x^2+y^2)
                self.is_guarding = True
                print("Spilled")
                return True

            self.prev_center = current_center

        return False

    def reset(self):
        print("Cup is reseting")
        self.is_present = False
        self.is_guarding = False
        self.occupied_cells.clear()
        self.locked_mask = None
        self.prev_center = None
        self.box = None
        self.center = None
        self.cup_body_mask = None
        self.original_cup_area = 0
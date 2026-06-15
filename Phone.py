import cv2
import numpy as np
import time
import math

class Phone:
    def __init__(self):
        self.is_present = False
        self.area_threshold = 80000
        self.steady_start_time = 0.0
        self.confirm_delay = 1.0

        self.box = None
        self.high_occupied = set()
        self.low_occupied = set()
        self.locked_high_mask = None
        self.locked_low_mask = None
        self.recent_hand_center = None

        self.phone_body_mask = None
        self.original_phone_area = 0

    def update_hand_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            self.recent_hand_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    def check_stability(self, candidate_box): # 持續一秒偵測到手機才算是真的有手機
        if candidate_box is not None:
            if self.steady_start_time == 0.0:
                self.steady_start_time = time.time()
            elif time.time() - self.steady_start_time > self.confirm_delay:
                self.is_present = True
                self.box = candidate_box
                self.steady_start_time = 0.0
                print("Phone!!!")
                return True  # 觸發計算訊號
        else:
            self.steady_start_time = 0.0  
            
        return False

    def calculate_support(self, frame_shape, grid_start_x, grid_start_y, cell_w, cell_h):
        center_x = int(np.mean([p[0] for p in self.box]))
        hx, hy = self.recent_hand_center if self.recent_hand_center else (center_x, frame_shape[0])

        phone_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(phone_mask, [self.box], 255)

        self.phone_body_mask = phone_mask.copy()
        self.original_phone_area = cv2.countNonZero(self.phone_body_mask)

        inv_phone_mask = cv2.bitwise_not(phone_mask)

        p0, p1, p2, p3 = self.box
        dist01 = math.hypot(p0[0]-p1[0], p0[1]-p1[1])
        dist12 = math.hypot(p1[0]-p2[0], p1[1]-p2[1])

        if dist01 > dist12: # 選長邊
            edgeA, edgeB = (p0, p1), (p2, p3)
        else:
            edgeA, edgeB = (p1, p2), (p3, p0)

        midA = ((edgeA[0][0] + edgeA[1][0])/2, (edgeA[0][1] + edgeA[1][1])/2)
        midB = ((edgeB[0][0] + edgeB[1][0])/2, (edgeB[0][1] + edgeB[1][1])/2)

        if math.hypot(midA[0] - hx, midA[1] - hy) < math.hypot(midB[0] - hx, midB[1] - hy):
            front_edge, back_edge = edgeA, edgeB
        else:
            front_edge, back_edge = edgeB, edgeA

        # 繪製粗線並取交集
        high_support_thickness = 90
        low_support_thickness = 270
        high_line_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        low_line_mask = np.zeros(frame_shape[:2], dtype=np.uint8)

        cv2.line(high_line_mask, (int(back_edge[0][0]), int(back_edge[0][1])), (int(back_edge[1][0]), int(back_edge[1][1])), 255, high_support_thickness)
        cv2.line(low_line_mask, (int(front_edge[0][0]), int(front_edge[0][1])), (int(front_edge[1][0]), int(front_edge[1][1])), 255, low_support_thickness)

        self.locked_high_mask = cv2.bitwise_and(high_line_mask, phone_mask)
        self.locked_low_mask = cv2.bitwise_and(low_line_mask, inv_phone_mask)

        # 檢查 16 個格子
        cell_area = cell_w * cell_h
        self.high_occupied.clear()
        self.low_occupied.clear()

        for row in range(4):
            for col in range(4):
                cx = grid_start_x + col * cell_w
                cy = grid_start_y + row * cell_h
                m_id = row * 4 + col

                cell_high = self.locked_high_mask[cy:cy+cell_h, cx:cx+cell_w]
                if (cv2.countNonZero(cell_high) / cell_area) > 0.2:
                    self.high_occupied.add(m_id)

                cell_phone = phone_mask[cy:cy+cell_h, cx:cx+cell_w]
                cell_low = self.locked_low_mask[cy:cy+cell_h, cx:cx+cell_w]
                if (cv2.countNonZero(cell_low) / cell_area) > 0.15 and (cv2.countNonZero(cell_phone) / cell_area) < 0.05:
                    self.low_occupied.add(m_id)

        print(f"🔺 升起 (粉紅靠背): {list(self.high_occupied)}")
        print(f"🔻 微升 (淺藍擋板): {list(self.low_occupied)}")

    def reset(self):
        print("Phone is reseting")
        self.is_present = False
        self.high_occupied.clear()
        self.low_occupied.clear()
        self.locked_high_mask = None
        self.locked_low_mask = None
        self.recent_hand_center = None
        self.box = None
        self.phone_body_mask = None
        self.original_phone_area = 0
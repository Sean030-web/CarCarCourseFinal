import cv2
import numpy as np
import time
from picamera2 import Picamera2
# from servo_test2 import Servo
import math

# ==========================================
# 1. 初始化相機與硬體
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

print("啟動 Dynamic Shape Display 核心視覺引擎...")
# my_servo = Servo() # TODO: 實機測試時解開

time.sleep(3)

picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False
})

# ==========================================
# 2. 初始化 MOG2 動態大腦與全域變數
# ==========================================
backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=True)

print("✅ 引擎啟動完成！")
print("⌨️  操作提示：按下 'r' 鍵可手動 切換/凍結 背景學習，按下 'q' 離開。")

FALLING_THRESHOLD = 30
prev_cup_center = None

# 手機狀態變數
is_phone_present = False
phone_area_threshold = 80000
update_bg = True
isFirst_object = True
phone_steady_start_time = 0.0
phone_confirm_delay = 1.0

# 🌟 進階記憶變數：高低差與手部追蹤
high_occupied = set()
low_occupied = set()
locked_high_mask = None
locked_low_mask = None
recent_hand_center = None  # 記憶手退開前的最後位置

# 網格校正參數 (請依實體微調)
GRID_START_X = 90
GRID_START_Y = 10
GRID_WIDTH = 440
GRID_HEIGHT = 440
CELL_W = GRID_WIDTH // 4
CELL_H = GRID_HEIGHT // 4

# ==========================================
# 3. 核心辨識迴圈
# ==========================================
try:
    while True:
        frame = picam2.capture_array()

        # 根據 update_bg 的狀態決定是否學習新背景
        current_learning_rate = -1 if update_bg else 0
        fgMask = backSub.apply(frame, learningRate=current_learning_rate)

        _, thresh = cv2.threshold(fgMask, 254, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        phone_detected_now = False
        phone_box_this_frame = None
        phone_candidate_this_frame = None

        # ==========================================
        # 🟢 階段一：尋找目標與手部追蹤
        # ==========================================
        if not is_phone_present:
            update_bg = True
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 15000 or area > 200000:
                    continue


                update_bg = False

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0

                if solidity < 0.8:
                    continue

                rect = cv2.minAreaRect(contour)
                (center_x, center_y), (width, height), angle = rect

                # 🛠️ 內縮校正：扣除膨脹誤差，讓判定更貼合實體
                shrink_margin = 30
                adj_w = max(1, width - shrink_margin)
                adj_h = max(1, height - shrink_margin)
                adjusted_rect = ((center_x, center_y), (adj_w, adj_h), angle)

                box = cv2.boxPoints(adjusted_rect)
                box = np.intp(box)

                if height == 0 or width == 0:
                    continue

                aspect_ratio = max(width, height) / min(width, height)
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

                # 確認是否為手機
                if 1.5 < aspect_ratio < 2.5 and area > 80000:
                    cv2.putText(frame, "Phone", (int(center_x), int(center_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    phone_candidate_this_frame = box  # 暫存這個畫面裡的框框
                else:
                    phone_candidate_this_frame = None
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        recent_hand_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # 確認是否為水杯 (保留原本功能)
                if aspect_ratio <= 1.3:
                    cv2.putText(frame, "Cup", (int(center_x), int(center_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if prev_cup_center is not None:
                        dx = center_x - prev_cup_center[0]
                        dy = center_y - prev_cup_center[1]
                        if np.sqrt(dx**2 + dy**2) > FALLING_THRESHOLD:
                            cv2.putText(frame, "SPILL WARNING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    prev_cup_center = (center_x, center_y)

            # ==========================================
            # 🎯 決策層：方向判斷與 3D 高低差分配
            # ==========================================

            if not is_phone_present:
                if phone_candidate_this_frame is not None:
                    # 如果是剛看到手機，開始記錄時間
                    if phone_steady_start_time == 0.0:
                        phone_steady_start_time = time.time()
                        print("⏳ 發現手機輪廓，等待畫面穩定中...")

                    # 如果看著它的時間已經超過設定的延遲時間 (1秒)
                    elif time.time() - phone_steady_start_time > phone_confirm_delay:
                        phone_detected_now = True
                        phone_box_this_frame = phone_candidate_this_frame
                        phone_steady_start_time = 0.0  # 觸發後將計時器歸零
                else:
                    # 如果這一個 Frame 突然沒看到手機 (可能是手揮過去造成的雜訊)，計時器立刻歸零重算
                    phone_steady_start_time = 0.0


            if phone_detected_now and not is_phone_present:
                print("\n📱 [系統觸發] 偵測到手機！計算 3D 傾斜姿態...")
                update_bg = False
                is_phone_present = True

                if phone_box_this_frame is not None:
                    # 1. 取得手的最後位置 (若抓不到，預設手從畫面正下方伸進來)
                    hx, hy = recent_hand_center if recent_hand_center else (center_x, 480)

                    # 建立內外遮罩
                    phone_mask = np.zeros((480, 640), dtype=np.uint8)
                    cv2.fillPoly(phone_mask, [phone_box_this_frame], 255)
                    inv_phone_mask = cv2.bitwise_not(phone_mask)

                    # 找出四個角與邊長
                    p0, p1, p2, p3 = phone_box_this_frame
                    dist01 = math.hypot(p0[0]-p1[0], p0[1]-p1[1])
                    dist12 = math.hypot(p1[0]-p2[0], p1[1]-p2[1])

                    # 找出兩條長邊
                    if dist01 > dist12:
                        edgeA, edgeB = (p0, p1), (p2, p3)
                    else:
                        edgeA, edgeB = (p1, p2), (p3, p0)

                    # 比較哪一條邊離「手」比較近
                    midA = ((edgeA[0][0] + edgeA[1][0])/2, (edgeA[0][1] + edgeA[1][1])/2)
                    midB = ((edgeB[0][0] + edgeB[1][0])/2, (edgeB[0][1] + edgeB[1][1])/2)
                    distA_to_hand = math.hypot(midA[0] - hx, midA[1] - hy)
                    distB_to_hand = math.hypot(midB[0] - hx, midB[1] - hy)

                    if distA_to_hand < distB_to_hand:
                        front_edge = edgeA  # 離手近 -> 擋板 (外側, 低)
                        back_edge = edgeB   # 離手遠 -> 靠背 (內側, 高)
                    else:
                        front_edge = edgeB
                        back_edge = edgeA

                    # 2. 繪製高低支架的初始粗線
                    support_thickness = 90 * 2
                    high_line_mask = np.zeros((480, 640), dtype=np.uint8)
                    low_line_mask = np.zeros((480, 640), dtype=np.uint8)

                    cv2.line(high_line_mask, (int(back_edge[0][0]), int(back_edge[0][1])), (int(back_edge[1][0]), int(back_edge[1][1])), 255, support_thickness)
                    cv2.line(low_line_mask, (int(front_edge[0][0]), int(front_edge[0][1])), (int(front_edge[1][0]), int(front_edge[1][1])), 255, support_thickness)

                    # 🎯 魔法交集：高的取內部，低的取外部
                    locked_high_mask = cv2.bitwise_and(high_line_mask, phone_mask)
                    locked_low_mask = cv2.bitwise_and(low_line_mask, inv_phone_mask)

                    # 3. 檢查 16 個格子並分發指令
                    cell_area = CELL_W * CELL_H
                    for row in range(4):
                        for col in range(4):
                            cx = GRID_START_X + col * CELL_W
                            cy = GRID_START_Y + row * CELL_H

                            # 檢查是否為「高靠背」
                            cell_high = locked_high_mask[cy:cy+CELL_H, cx:cx+CELL_W]
                            if (cv2.countNonZero(cell_high) / cell_area) > 0.2:
                                high_occupied.add(row * 4 + col)

                            cell_phone = phone_mask[cy:cy+CELL_H, cx:cx+CELL_W] # 手機實體的純白遮罩

                            # 檢查是否為「低擋板」
                            cell_low = locked_low_mask[cy:cy+CELL_H, cx:cx+CELL_W]
                            if (cv2.countNonZero(cell_low) / cell_area) > 0.15 and (cv2.countNonZero(cell_phone) / cell_area) < 0.05:
                                low_occupied.add(row * 4 + col)

                print(f"🔺 升起 (粉紅靠背): {list(high_occupied)}")
                print(f"🔻 微升 (淺藍擋板): {list(low_occupied)}")

                # TODO: 傳送指令給馬達
                # commands = {}
                # for m in high_occupied: commands[m] = 1.0   # 全速上升
                # for m in low_occupied: commands[m] = 0.5    # 半速/短時間上升
                # my_servo.go_up_and_down(**commands)

        # ==========================================
        # 🔴 階段二：看守模式
        # ==========================================
        else:
            max_area = 0
            largest_box = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    rect = cv2.minAreaRect(contour)
                    largest_box = np.intp(cv2.boxPoints(rect))

            if largest_box is not None:
                cv2.drawContours(frame, [largest_box], 0, (0, 0, 255), 2)
                cv2.putText(frame, f"Guarding... Area: {int(max_area)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 觸發動作：手機移開
            if max_area < phone_area_threshold:
                print("\n👋 手機已移開！全體復位。")
                is_phone_present = False
                update_bg = True

                high_occupied.clear()
                low_occupied.clear()
                locked_high_mask = None
                locked_low_mask = None
                recent_hand_center = None
                isFirst_object = True

                # my_servo.go_back()

        # ==========================================
        # 🎨 UI 繪製：高低實體遮罩與動態網格
        # ==========================================
        if locked_high_mask is not None:
            frame[locked_high_mask > 0] = [255, 0, 255, 255] # 粉紅色 (靠背)
        if locked_low_mask is not None:
            frame[locked_low_mask > 0] = [255, 255, 0, 255]  # 淺藍色 (前端擋板)

        for i in range(5):
            vx = GRID_START_X + i * CELL_W
            cv2.line(frame, (vx, GRID_START_Y), (vx, GRID_START_Y + GRID_HEIGHT), (0, 255, 255), 2)
            hy = GRID_START_Y + i * CELL_H
            cv2.line(frame, (GRID_START_X, hy), (GRID_START_X + GRID_WIDTH, hy), (0, 255, 255), 2)

        for row in range(4):
            for col in range(4):
                motor_id = row * 4 + col
                cx = GRID_START_X + col * CELL_W
                cy = GRID_START_Y + row * CELL_H

                # 根據陣列變換格子的顏色
                if motor_id in high_occupied:
                    cv2.rectangle(frame, (cx, cy), (cx + CELL_W, cy + CELL_H), (255, 0, 255), 4) # 粉紅框
                    text_color = (255, 0, 255)
                elif motor_id in low_occupied:
                    cv2.rectangle(frame, (cx, cy), (cx + CELL_W, cy + CELL_H), (255, 255, 0), 4) # 淺藍框
                    text_color = (255, 255, 0)
                else:
                    text_color = (0, 255, 255) # 預設黃色

                cv2.putText(frame, str(motor_id), (cx + 10, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # 顯示狀態與畫面
        status_text = "Background Updating: ON" if update_bg else "Background Updating: OFF"
        color = (0, 255, 0) if update_bg else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Smart Table Vision', frame)
        cv2.imshow('Debug Mask', thresh)

        # 鍵盤事件偵測
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            update_bg = not update_bg
            if update_bg:
                print("🧠 [手動模式] 背景學習：開啟")
            else:
                print("🧊 [手動模式] 背景學習：凍結")

except KeyboardInterrupt:
    print("\n程式已手動中斷")

finally:
    # my_servo.go_back()
    picam2.stop()
    cv2.destroyAllWindows()
    print("相機已安全關閉")
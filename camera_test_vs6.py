import cv2
import numpy as np
import time
from picamera2 import Picamera2
from servo_test2 import Servo
import math
from Phone import Phone
from Cup import Cup
from Clock import Clock

# ==========================================
# 1. 初始化相機與硬體
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

print("啟動 Dynamic Shape Display 核心視覺引擎...")
my_servo = Servo()

time.sleep(3)

picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False
})

# ==========================================
# 2. 初始化 MOG2 動態大腦與全域變數
# ==========================================
backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=200, detectShadows=True)

print("✅ 引擎啟動完成！")
print("⌨️  操作提示：按下 'r' 鍵可手動 切換/凍結 背景學習，按下 'q' 離開。")



# 手機
my_phone = Phone()

#杯子狀態變數
my_cup = Cup()
cup_center_this_frame = (0, 0)

# 計時器
my_clock = Clock()
clock_center_this_frame = (0, 0)


# 網格校正參數
GRID_START_X = 90
GRID_START_Y = 10
GRID_WIDTH = 440
GRID_HEIGHT = 440
CELL_W = GRID_WIDTH // 4
CELL_H = GRID_HEIGHT // 4


# 'r'
is_r = False
update_bg = True

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

        phone_candidate_this_frame = None
        cup_candidate_this_frame = None
        cup_box_this_frame = None
        clock_candidate_this_frame = None

        # ==========================================
        # 🟢 階段一：尋找目標與手部追蹤
        # ==========================================
        if not my_phone.is_present and not my_cup.is_present and not my_clock.is_present and not my_clock.is_counting:
            update_bg = True
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 9000 or area > 200000:
                    continue

                if not is_r:
                    update_bg = False

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0

                if solidity < 0.8:
                    continue

                rect = cv2.minAreaRect(contour)
                (center_x, center_y), (width, height), angle = rect

                # 內縮校正
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
                    # phone_candidate_this_frame = None
                    my_phone.update_hand_center(contour)

                # 確認是否為水杯
                if aspect_ratio <= 1.4 and 80000 > area > 20000:
                    cv2.putText(frame, "Cup", (int(center_x), int(center_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cup_candidate_this_frame = box
                    cup_center_this_frame = (center_x, center_y)
                else:
                    # cup_candidate_this_frame = None
                    pass

                if area < 20000 and aspect_ratio <= 1.3:
                    cv2.putText(frame, "Clock", (int(center_x), int(center_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    clock_candidate_this_frame = box
                    clock_center_this_frame = (center_x, center_y)

            # ==========================================
            # 🎯 決策層：方向判斷與 3D 高低差分配
            # ==========================================

            if not my_phone.is_present:
                if my_phone.check_stability(phone_candidate_this_frame):
                    my_phone.calculate_support(frame.shape, GRID_START_X, GRID_START_Y, CELL_W, CELL_H)
                    update_bg = False
                    commands = {}
                    for m in my_phone.high_occupied: commands[m] = -1.0
                    for m in my_phone.low_occupied: commands[m] = -0.1
                    my_servo.go_up_and_down(commands)


            if not my_cup.is_present:
                if my_cup.check_stability(cup_candidate_this_frame, cup_center_this_frame):
                    my_cup.calculate_defense_wall(frame.shape, GRID_START_X, GRID_START_Y, CELL_W, CELL_H)
                    update_bg = False

            if not my_clock.is_present:
                if my_clock.check_stability(clock_candidate_this_frame, clock_center_this_frame):
                    my_clock.lock_position(frame.shape, GRID_START_X, GRID_START_Y, CELL_W, CELL_H)
                    update_bg = False


        # ==========================================
        # 🔴 階段二：看守模式
        # ==========================================
        elif my_phone.is_present:
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
            if my_phone.phone_body_mask is not None:
                # 擷取當下畫面中，嚴格對應「手機原本位置」的像素
                current_phone_pixels = cv2.bitwise_and(thresh, my_phone.phone_body_mask)
                remaining_area = cv2.countNonZero(current_phone_pixels)

            if remaining_area < (my_phone.original_phone_area * 0.3):
                my_phone.reset()
                update_bg = True
                my_servo.go_back()

        elif my_cup.is_present:
            max_area = 0
            largest_box = None
            current_center = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    rect = cv2.minAreaRect(contour)
                    largest_box = np.intp(cv2.boxPoints(rect))
                    current_center = (rect[0][0], rect[0][1])

            if largest_box is not None:
                cv2.drawContours(frame, [largest_box], 0, (0, 0, 255), 2)
                status_msg = "Guarding Cup..."
                cv2.putText(frame, f"{status_msg} Area: {int(max_area)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 🚨 水杯專屬：瞬間傾倒偵測
            if my_cup.check_spill(current_center):
                commands = {}
                for m in my_cup.occupied_cells: commands[m] = -1.0
                my_servo.go_up_and_down(commands)

            if my_cup.cup_body_mask is not None:
                # 擷取當下畫面中，嚴格對應「水杯原本位置」的像素
                current_cup_pixels = cv2.bitwise_and(thresh, my_cup.cup_body_mask)
                remaining_area = cv2.countNonZero(current_cup_pixels)

            if remaining_area < my_cup.original_cup_area * 0.7:
                print("\n👋 桌面已清空！全體復位。")
                my_cup.reset()
                update_bg = True
                my_servo.go_back()

        elif my_clock.is_present:
            cx, cy = my_clock.center
            # 顯示預備時間
            cv2.putText(frame, f"Ready: {my_clock.time_str}", (int(cx) - 40, int(cy) - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            if my_clock.clock_body_mask is not None:
                current_clock_pixels = cv2.bitwise_and(thresh, my_clock.clock_body_mask)
                remaining_area = cv2.countNonZero(current_clock_pixels)

                # 方塊被拿開，觸發倒數！
                if remaining_area < (my_clock.original_clock_area * 0.7):
                    my_clock.start_countdown()

        elif my_clock.is_counting:
            # update_bg = False
            is_time_up = my_clock.update_countdown()

            # 1. 取得需要動態改變的馬達指令 (Delta Commands)
            commands = my_clock.calculate_go_up_and_down_servo()

            # 2. 如果有馬達狀態改變，才送出指令給硬體
            if len(commands) > 0:
                my_servo.go_up_and_down(commands)

            # 3. 在畫面上繪製超大倒數文字
            cv2.putText(frame, my_clock.time_str, (150, 240), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 255), 5)

            # 4. 時間到：復位
            if is_time_up:
                my_servo.go_up_and_down({15: 1.0})
                my_clock.reset()
                update_bg = True
                my_servo.go_back()





        # ==========================================
        # 🎨 UI 繪製：高低實體遮罩與動態網格
        # ==========================================
        if my_phone.locked_high_mask is not None:
            frame[my_phone.locked_high_mask > 0] = [255, 0, 255, 255] # 粉紅色 (靠背)
        if my_phone.locked_low_mask is not None:
            frame[my_phone.locked_low_mask > 0] = [255, 255, 0, 255]  # 淺藍色 (前端擋板)

        if my_cup.locked_mask is not None:
            if my_cup.is_guarding:
                frame[my_cup.locked_mask > 0] = [0, 165, 255, 255]   # 🚨 觸發防護：亮橘色 (警告)
            else:
                frame[my_cup.locked_mask > 0] = [255, 100, 0, 255]   # 🛡️ 待命狀態：深藍色框框

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
                if my_clock.is_counting:
                    cv2.rectangle(frame, (cx, cy), (cx + CELL_W, cy + CELL_H), (255, 255, 0), 4)
                    text_color = (255, 255, 0)
                if motor_id in my_cup.occupied_cells:
                    color = (0, 165, 255) if my_cup.is_guarding else (255, 100, 0)
                    cv2.rectangle(frame, (cx, cy), (cx + CELL_W, cy + CELL_H), color, 4)
                    text_color = color
                if motor_id in my_phone.high_occupied:
                    cv2.rectangle(frame, (cx, cy), (cx + CELL_W, cy + CELL_H), (255, 0, 255), 4) # 粉紅框
                    text_color = (255, 0, 255)
                elif motor_id in my_phone.low_occupied:
                    cv2.rectangle(frame, (cx, cy), (cx + CELL_W, cy + CELL_H), (255, 255, 0), 4) # 淺藍框
                    text_color = (255, 255, 0)
                else:
                    text_color = (0, 255, 255) # 預設黃色

                cv2.putText(frame, str(motor_id), (cx + 10, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # 顯示狀態與畫面
        status_text = "Background Updating: ON" if update_bg else "Background Updating: OFF"
        if is_r: status_text += " (Forced, press 'r' again to cancel)"
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
            is_r = not is_r
            if update_bg:
                print("🧠 [手動模式] 背景學習：開啟")
            else:
                print("🧊 [手動模式] 背景學習：凍結")

except KeyboardInterrupt:
    print("\n程式已手動中斷")

finally:
    my_servo.go_back()
    picam2.stop()
    cv2.destroyAllWindows()
    print("相機已安全關閉")
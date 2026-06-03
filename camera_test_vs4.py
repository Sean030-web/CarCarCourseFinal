import cv2
import numpy as np
import time
from picamera2 import Picamera2
# from servo_test2 import Servo  # 暫時不需要馬達，先註解掉

# ==========================================
# 1. 初始化相機
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

print("啟動 Dynamic Shape Display 視覺引擎 (純視覺測試版)...")
# my_servo = Servo() # TODO: 實機測試時解開

time.sleep(3)

picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False
})

# ==========================================
# 2. 初始化 MOG2 與網格設定
# ==========================================
backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=True)
print("✅ MOG2 動態引擎啟動！")

prev_cup_center = None
FALLING_THRESHOLD = 30

# 狀態變數
is_phone_present = False 
phone_area_threshold = 10000
update_bg = True 

# 網格設定 (請依實體慢慢微調)
GRID_START_X = 80   
GRID_START_Y = 32   
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
        
        current_learning_rate = -1 if update_bg else 0
        fgMask = backSub.apply(frame, learningRate=current_learning_rate)
        
        _, thresh = cv2.threshold(fgMask, 254, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        phone_detected_now = False
        phone_box_this_frame = None # 用來記錄這幀手機的形狀，方便後續算網格

        # ==========================================
        # 🟢 階段一：尋找手機模式 (嚴格審查)
        # ==========================================
        if not is_phone_present: 
            has_large_object = False 
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 只要面積夠大，就標記有大型物體 (預防性凍結背景用)
                if 15000 < area < 200000:
                    has_large_object = True 
                    
                    # 嚴格的「手機形狀審查」
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    if solidity < 0.8: 
                        continue

                    rect = cv2.minAreaRect(contour)
                    (center_x, center_y), (width, height), angle = rect
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    
                    if height == 0 or width == 0: 
                        continue
                        
                    aspect_ratio = max(width, height) / min(width, height)
                    cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                    
                    if 1.5 < aspect_ratio < 2.5:
                        cv2.putText(frame, "Phone", (int(center_x), int(center_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        phone_detected_now = True 
                        phone_box_this_frame = box # 存起來給網格計算用
                        
                    elif aspect_ratio <= 1.3:
                        cv2.putText(frame, "Cup", (int(center_x), int(center_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        # (水杯防傾倒邏輯保留)
            
            # 決策觸發點 1
            if phone_detected_now:
                print("\n📱 [系統觸發] 嚴格審查通過，確認為手機！")
                is_phone_present = True
                update_bg = False # 進入看守模式，大腦永久凍結
                
                # TODO: 實機測試時，在這裡加入馬達升起程式碼
                # my_servo.go_up_and_down(**motor_commands) 
                
            else:
                # 預防性凍結：如果有大東西(例如手)在畫面上，暫停學習；畫面空了才恢復
                update_bg = not has_large_object

        # ==========================================
        # 🔴 階段二：看守手機模式 (只看面積)
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
                phone_box_this_frame = largest_box # 存起來給網格計算用
                cv2.drawContours(frame, [largest_box], 0, (0, 0, 255), 2) 
                cv2.putText(frame, f"Guarding... Area: {int(max_area)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 決策觸發點 2：確認手機拿走
            if max_area < phone_area_threshold:
                print("\n👋 [系統觸發] 手機已移開！")
                is_phone_present = False
                update_bg = True # 解凍背景，重新適應空桌子
                
                # TODO: 實機測試時，在這裡加入馬達降下復位程式碼
                # my_servo.go_back()

        # ==========================================
        # 🎨 網格計算與繪圖層
        # ==========================================
        active_motors = set()
        
        # 1. 如果畫面上確定有手機，計算它壓到了哪些格子
        if phone_box_this_frame is not None:
            # 取得手機的正向外接矩形 (用來做最簡單快速的碰撞測試)
            px, py, pw, ph = cv2.boundingRect(phone_box_this_frame)
            
            for row in range(4):
                for col in range(4):
                    cx = GRID_START_X + col * CELL_W
                    cy = GRID_START_Y + row * CELL_H
                    
                    # 判斷手機矩形有沒有跟這個網格重疊 (碰撞檢測)
                    if not (px > cx + CELL_W or px + pw < cx or py > cy + CELL_H or py + ph < cy):
                        active_motors.add(row * 4 + col)

        # 2. 畫出 4x4 網格並根據狀態變色
        for row in range(4):
            for col in range(4):
                motor_id = row * 4 + col
                cx = GRID_START_X + col * CELL_W
                cy = GRID_START_Y + row * CELL_H
                
                # 如果這格被壓到，畫粗線亮綠色；沒壓到畫細線黃色
                if motor_id in active_motors:
                    cv2.rectangle(frame, (cx, cy), (cx + CELL_W, cy + CELL_H), (0, 255, 0), 4)
                    text_color = (0, 255, 0)
                else:
                    cv2.rectangle(frame, (cx, cy), (cx + CELL_W, cy + CELL_H), (0, 255, 255), 1)
                    text_color = (0, 255, 255)
                    
                cv2.putText(frame, str(motor_id), (cx + 10, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # 3. 如果是剛剛放下去的那一幀，印出需要升起的馬達編號清單
        if phone_detected_now and not is_phone_present:
            print(f"🎯 [馬達模擬] 準備升起以下區塊: {list(active_motors)}")

        # ==========================================
        # 顯示狀態與畫面
        # ==========================================
        status_text = "Background: ON (Learning)" if update_bg else "Background: OFF (Frozen)"
        color = (0, 255, 0) if update_bg else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Smart Table Vision', frame)
        cv2.imshow('Debug Mask', thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n程式已手動中斷")

finally:
    # my_servo.go_back() # TODO: 關閉前確保實體馬達歸零
    picam2.stop()
    cv2.destroyAllWindows()
    print("相機已安全關閉")
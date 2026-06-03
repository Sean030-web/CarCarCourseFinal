import cv2
import numpy as np
import time
from picamera2 import Picamera2
# from servo_test2 import Servo
import math




# ==========================================
# 1. 初始化相機
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

print("相機啟動中，等待光線穩定...")
# my_servo = Servo() # 馬達

time.sleep(3)

picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False
})

# ==========================================
# 2. 初始化 MOG2 動態背景減除器
# ==========================================
backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=True)

print("✅ MOG2 動態引擎啟動！")
print("⌨️  操作提示：請在影像視窗上按下 'r' 鍵來 切換/凍結 背景學習！")

prev_cup_center = None
FALLING_THRESHOLD = 30
is_phone_present = False 
phone_area_threshold = 50000
phone_occupied = set()

# 🌟 新增：手動控制背景更新的開關 (預設為 True，代表持續學習)
update_bg = True 

# 請根據你實際架設的鏡頭角度，慢慢微調這四個數字，直到網格完美貼合實體桌面
GRID_START_X = 122   # 網格左上角的 X 座標
GRID_START_Y = 30   # 網格左上角的 Y 座標
GRID_WIDTH = 440    # 網格的總寬度
GRID_HEIGHT = 440   # 網格的總高度

# 根據總寬高，算出每一格的大小
CELL_W = GRID_WIDTH // 4
CELL_H = GRID_HEIGHT // 4

# ==========================================
# 3. 核心辨識迴圈
# ==========================================
try:
    while True:
        frame = picam2.capture_array()
        
        # ==========================================
        # 🧠 核心邏輯：手動套用學習率
        # ==========================================
        # 根據 update_bg 的狀態決定學習率
        current_learning_rate = -1 if update_bg else 0
        
        # 將學習率套用到 MOG2
        fgMask = backSub.apply(frame, learningRate=current_learning_rate)
        
        # 過濾陰影並進行二值化
        _, thresh = cv2.threshold(fgMask, 254, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        phone_detected_now = False

        if not is_phone_present: # 如果沒有手機才一直看有沒有手機
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
                
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                
                if height == 0 or width == 0: 
                    continue
                    
                aspect_ratio = max(width, height) / min(width, height)
                
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                
                if 1.5 < aspect_ratio < 2.5:
                    cv2.putText(frame, "Phone", (int(center_x), int(center_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    phone_detected_now = True 
                    phone_box_this_frame = box
                    
                elif aspect_ratio <= 1.3:
                    cv2.putText(frame, "Cup", (int(center_x), int(center_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    if prev_cup_center is not None:
                        dx = center_x - prev_cup_center[0]
                        dy = center_y - prev_cup_center[1]
                        displacement = np.sqrt(dx**2 + dy**2)
                        
                        if displacement > FALLING_THRESHOLD:
                            cv2.putText(frame, "SPILL WARNING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                            
                    prev_cup_center = (center_x, center_y)
            
            # ==========================================
            # 4. 決策層
            # ==========================================
            if phone_detected_now and not is_phone_present:
                print("📱 [模擬觸發] 偵測到手機！(馬達升起)")
                update_bg = False
                is_phone_present = True
                #看一下有哪些馬達會被手機覆蓋
                if phone_box_this_frame is not None:
                    # 1. 建立一張跟畫面一樣大的全黑畫布
                    support_mask = np.zeros((480, 640), dtype=np.uint8)
                    
                    # 2. 找出短邊並畫上「粗線」當作支架遮罩
                    p0, p1, p2, p3 = phone_box_this_frame
                    dist01 = math.hypot(p0[0]-p1[0], p0[1]-p1[1])
                    dist12 = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                    
                    # 🔧 參數調整：支架吃進去手機的「深度」(像素)
                    # 數字越大，升起的面積越往手機中間延伸
                    support_thickness = 45 
                    
                    if dist01 < dist12:
                        # 0-1 和 2-3 是短邊
                        cv2.line(support_mask, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), 255, support_thickness)
                        cv2.line(support_mask, (int(p2[0]), int(p2[1])), (int(p3[0]), int(p3[1])), 255, support_thickness)
                    else:
                        # 1-2 和 3-0 是短邊
                        cv2.line(support_mask, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 255, support_thickness)
                        cv2.line(support_mask, (int(p3[0]), int(p3[1])), (int(p0[0]), int(p0[1])), 255, support_thickness)
                    
                    # 3. 檢查 16 個格子 (使用覆蓋率)
                    cell_area = CELL_W * CELL_H 
                    for row in range(4):
                        for col in range(4):
                            cx = GRID_START_X + col * CELL_W
                            cy = GRID_START_Y + row * CELL_H
                            
                            # 把這格的範圍從黑畫布上「剪」下來
                            cell_region = support_mask[cy:cy+CELL_H, cx:cx+CELL_W]
                            overlap_pixels = cv2.countNonZero(cell_region)
                            coverage_ratio = overlap_pixels / cell_area
                            
                            if coverage_ratio > 0.1:
                                phone_occupied.add(row * 4 + col)
                
                print(f"🎯 phone 覆蓋到的地方: {list(phone_occupied)}")

                # 這邊要寫一些馬達的程式
                # my_servo.go_up_and_down(**{0: 1.0}) # 這邊是測試版

        else: # 只需要看手機有沒有被拿走
            max_area = 0
            largest_box = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    
                    # 順便抓個外框來畫圖，讓你知道電腦還在盯著它
                    rect = cv2.minAreaRect(contour)
                    largest_box = np.intp(cv2.boxPoints(rect))
            
            # 畫出目前盯著的色塊
            if largest_box is not None:
                cv2.drawContours(frame, [largest_box], 0, (0, 0, 255), 2) # 紅框代表看守中
                cv2.putText(frame, f"Guarding... Area: {int(max_area)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 觸發動作：如果最大面積小於門檻 (代表手機被拿走，只剩小雜訊)
            if max_area < phone_area_threshold:
                print("\n👋 手機已移開！降下支架。")
                is_phone_present = False
                update_bg = True # 🧠 解凍背景：重新適應空桌子
                phone_occupied.clear()
                
                # 呼叫馬達復位
                # my_servo.go_back()

            
        



        for i in range(5):
            # 垂直線
            vx = GRID_START_X + i * CELL_W
            cv2.line(frame, (vx, GRID_START_Y), (vx, GRID_START_Y + GRID_HEIGHT), (0, 255, 255), 2)
            # 水平線
            hy = GRID_START_Y + i * CELL_H
            cv2.line(frame, (GRID_START_X, hy), (GRID_START_X + GRID_WIDTH, hy), (0, 255, 255), 2)

        # 2. 在每一格印上 0~15 的編號 (方便對齊馬達通道)
        for row in range(4):
            for col in range(4):
                motor_id = row * 4 + col  # 算出 0~15 的編號
                # 計算文字要放的座標 (放在每一格的左上角稍微偏移一點)
                text_x = GRID_START_X + col * CELL_W + 10
                text_y = GRID_START_Y + row * CELL_H + 30
                cv2.putText(frame, str(motor_id), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        # ==========================================
        # 顯示狀態與畫面
        # ==========================================
        # 在畫面上顯示目前的學習狀態，方便你辨識
        status_text = "Background Updating: ON (Learning)" if update_bg else "Background Updating: OFF (Frozen)"
        color = (0, 255, 0) if update_bg else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Smart Table Vision', frame)
        cv2.imshow('Debug Mask', thresh)

        # ----------------------------------------
        # ⌨️ 鍵盤事件偵測
        # ----------------------------------------
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 按下 'r' 鍵，反轉開關狀態 (True 變 False，False 變 True)
            update_bg = not update_bg
            if update_bg:
                print("🧠 [手動模式] 背景更新：開啟 (開始適應新光影與物件)")
            else:
                print("🧊 [手動模式] 背景更新：凍結 (停止學習，鎖定當前背景)")

except KeyboardInterrupt:
    print("\n程式已手動中斷")

finally:
    # my_servo.go_back()
    picam2.stop()
    cv2.destroyAllWindows()
    print("相機已安全關閉")
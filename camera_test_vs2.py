import cv2
import numpy as np
import time
from picamera2 import Picamera2

# ==========================================
# 1. 初始化相機
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

print("相機啟動中，等待光線穩定...")
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

# 🌟 新增：手動控制背景更新的開關 (預設為 True，代表持續學習)
update_bg = True 

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
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 15000 or area > 200000: 
                continue
                
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
            is_phone_present = True
            
        elif not phone_detected_now and is_phone_present:
            print("👋 [模擬觸發] 手機移開了。(馬達降下)")
            is_phone_present = False

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
    picam2.stop()
    cv2.destroyAllWindows()
    print("相機已安全關閉")
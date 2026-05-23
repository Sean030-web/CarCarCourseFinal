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

print("相機啟動中，請稍候...")
time.sleep(2)

# ==========================================
# 2. 擷取背景底圖 (零記憶體消耗絕招)
# ==========================================
print("⚠️ 請保持桌面淨空！正在擷取背景...")
background = picam2.capture_array()
bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
bg_gray = cv2.GaussianBlur(bg_gray, (21, 21), 0)
print("✅ 背景擷取完成！請將手機或水杯放入畫面測試。")

prev_cup_center = None
FALLING_THRESHOLD = 30
# 軟體狀態變數：紀錄目前系統是否認為手機存在
is_phone_present = False 

# ==========================================
# 3. 核心辨識迴圈 (搭載傾斜辨識 + 三重極致濾網)
# ==========================================
try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        diff = cv2.absdiff(bg_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        phone_detected_now = False
        
        for contour in contours:
            # --- 【濾網一：面積上下限】 ---
            # 濾除太小的雜點或太大的手臂干擾 (數值可依實際鏡頭高度微調)
            area = cv2.contourArea(contour)
            if area < 4000 or area > 30000: 
                continue
                
            # --- 【濾網二：飽滿度 (Solidity) 檢查】 ---
            # 計算凸包面積，藉此判斷物體是否為實心幾何形狀
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # 如果形狀不規則 (例如張開的手掌或傳輸線)，飽滿度會很低，直接跳過
            if solidity < 0.85:
                continue

            # --- 取得可旋轉的最小外接矩形 ---
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            
            # 取得矩形的四個頂點座標並畫圖
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            if height == 0 or width == 0: 
                continue
                
            # 計算真實的長寬比 (確保大邊除以小邊，永遠大於等於 1)
            aspect_ratio = max(width, height) / min(width, height)
            
            # 畫出完美貼合的傾斜追蹤綠框
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            
            # --- 【濾網三：長寬比分類邏輯】 ---
            # 手機：長寬比通常落在 1.6 到 2.5 之間
            if 1.6 < aspect_ratio < 2.5:
                cv2.putText(frame, "Phone", (int(center_x), int(center_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                phone_detected_now = True # 做上記號：這個畫面有看到手機！
                
            # 水杯：正上方看下去接近圓形，長寬比小於 1.3
            elif aspect_ratio <= 1.3:
                cv2.putText(frame, "Cup", (int(center_x), int(center_y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # 傾倒偵測邏輯
                if prev_cup_center is not None:
                    dx = center_x - prev_cup_center[0]
                    dy = center_y - prev_cup_center[1]
                    displacement = np.sqrt(dx**2 + dy**2)
                    
                    if displacement > FALLING_THRESHOLD:
                        cv2.putText(frame, "SPILL WARNING!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        
                prev_cup_center = (center_x, center_y)
        
        # ==========================================
        # 4. 模擬決策層 (用 Print 取代馬達動作)
        # ==========================================
        if phone_detected_now and not is_phone_present:
            print("📱 [模擬觸發] 偵測到手機！(未來這裡馬達會轉到 90 度)")
            is_phone_present = True
            
        elif not phone_detected_now and is_phone_present:
            print("👋 [模擬觸發] 手機移開了。(未來這裡馬達會降回 0 度)")
            is_phone_present = False

        # ----------------------------------------
        # 顯示畫面
        cv2.imshow('Smart Table Vision Test', frame)

        # 按 'q' 鍵離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("程式已手動中斷")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("相機已安全關閉")
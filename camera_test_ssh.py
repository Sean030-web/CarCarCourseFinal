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
print("🛑 提示：因為目前處於 SSH 無頭模式，不會顯示影像視窗。請按 Ctrl+C 來結束程式。")

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
            area = cv2.contourArea(contour)
            if area < 10000 or area > 60000: 
                continue
                
            # --- 【濾網二：飽滿度 (Solidity) 檢查】 ---
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            if solidity < 0.85:
                continue

            # --- 取得可旋轉的最小外接矩形 ---
            rect = cv2.minAreaRect(contour)
            (center_x, center_y), (width, height), angle = rect
            
            if height == 0 or width == 0: 
                continue
                
            # 計算真實的長寬比 
            aspect_ratio = max(width, height) / min(width, height)
            
            # --- 【濾網三：長寬比分類邏輯】 ---
            if 1.6 < aspect_ratio < 2.5:
                phone_detected_now = True 
                
            elif aspect_ratio <= 1.3:
                # 傾倒偵測邏輯
                if prev_cup_center is not None:
                    dx = center_x - prev_cup_center[0]
                    dy = center_y - prev_cup_center[1]
                    displacement = np.sqrt(dx**2 + dy**2)
                    
                    if displacement > FALLING_THRESHOLD:
                        # 這裡改成印在終端機上
                        print("⚠️ [警告] 偵測到水杯傾倒位移！")
                        
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
        # ⚠️ 【SSH 修改重點】: 註解掉所有圖形介面與按鍵偵測
        # cv2.imshow('Smart Table Vision Test', frame)
        # cv2.imshow('Debug Mask', thresh)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # ----------------------------------------

except KeyboardInterrupt:
    # 當你在終端機按下 Ctrl+C 時，會觸發這裡，安全結束程式
    print("\n程式已手動中斷 (Ctrl+C)")

finally:
    picam2.stop()
    # 雖然沒開視窗，但保留這行避免潛在的資源殘留
    cv2.destroyAllWindows() 
    print("相機已安全關閉")
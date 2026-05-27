# 這是馬達轉不轉的測試
import time
from adafruit_servokit import ServoKit

print("初始化 PCA9685 驅動板...")
kit = ServoKit(channels=16)

# ==========================================
# ⚠️ 嚴重警告：以下寫法專屬 360 度連續旋轉馬達
# ==========================================

try:
    servo_current_angle = [0.0]
    print("啟動 360 度馬達...")
    
    # 針對 360 度馬達，我們使用 continuous_servo 和 throttle (油門)
    # throttle 的數值範圍是 -1.0 到 1.0
    
    # 數值 1.0 代表「全速順時針旋轉」 (實際方向依不同廠牌可能相反)
    print("this servo will spin 2 sec, rest 2 sec, repeatedly")
    while True:
        kit.continuous_servo[0].throttle = 1.0 
        time.sleep(0.3)
        servo_current_angle[0] += 0.3
        kit.continuous_servo[0].throttle = 0
        time.sleep(2)
        kit.continuous_servo[0].throttle = -1.0
        time.sleep(0.3)
        servo_current_angle[0] -= 0.3
        kit.continuous_servo[0].throttle = 0
        time.sleep(2)


except KeyboardInterrupt:
    # 當你按下 Ctrl + C 時，執行煞車動作
    print("\n收到停止指令，緊急煞車！")
    if servo_current_angle[0] >= 0:
        kit.continuous_servo[0].throttle = -1.0 
        time.sleep(servo_current_angle[0])
    else:
        kit.continuous_servo[0].throttle = 1.0 
        time.sleep(-servo_current_angle[0])
    # 將油門設為 0，馬達就會完全停止
    kit.continuous_servo[0].throttle = 0
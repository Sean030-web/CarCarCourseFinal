# 這是馬達轉不轉的測試
import time
from adafruit_servokit import ServoKit


# 馬達校正係數
k = [1.0] * 16
k[0] = 0.75
k[1] = 0.75
k[2] = 0.75
k[3] = 0.9
k[4] = 0.65
k[5] = 0.55
k[6] = 0.7
k[7] = 0.7
k[8] = 0.9
k[9] = 0.65
k[11] = 0.9
k[12] = 0.7
k[13] = 0.8
k[14] = 0.7
k[15] = 0.4

class Servo:
    def __init__(self):
        print("初始化 PCA9685 驅動板...")
        self.kit = ServoKit(channels=16)
        # 🌟 改為記錄每顆馬達「全速運作的累積時間」(秒)，正代表上升，負代表下降
        self.servo_accumulated_time = [0.0] * 16

    def go_up_and_down(self, motors):
        """
        傳入字典，例如 {0: 1.0, 1: -1.0}
        """
        # 1. 一律以全速 (1.0 或 -1.0) 輸出，確保最大扭力，排除非線性速度問題
        for key, value in motors.items():
            calibrated_speed = value * k[key]
            if key == 15 and value > 0: calibrated_speed *= 1.2

            self.kit.continuous_servo[key].throttle = calibrated_speed

        # 2. 固定作動時間
        run_time = 0.7
        time.sleep(run_time)

        # 3. 煞車並精準記錄通電時間
        for key, value in motors.items():
            self.kit.continuous_servo[key].throttle = 0
            # 累積該馬達運作的時間軸
            self.servo_accumulated_time[key] += value * run_time
            if key == 15: print(self.servo_accumulated_time[key])

    def go_back(self):
        print("\n執行高精度安全復位（全速時間對等機制）...")

        # 複製一份當前的時間記憶，避免迴圈內動態修改導致混亂
        remaining_times = [(3.5 * t) for t in self.servo_accumulated_time]
        remaining_times[0] *= 1.2
        remaining_times[4] *= 1.2
        remaining_times[3] *= 1.2

        # 1. 讓所有需要歸位的馬達「同時」以全速反向啟動
        active_home = False
        for i in range(16):
            if remaining_times[i] > 0:
                # 之前是上升的，現在全速下降
                self.kit.continuous_servo[i].throttle = -0.3 * k[i]
                active_home = True
            elif remaining_times[i] < 0:
                # 之前是下降的，現在全速上升
                self.kit.continuous_servo[i].throttle = 0.3 * k[i]
                active_home = True
            if i == 15: self.kit.continuous_servo[i].throttle *= 3.0
            if i == 15: print("15 is going down")

        if not active_home:
            print("所有馬達皆在原點，無需復位。")
            return

        # 2. 建立高精度計時監控迴圈 (精準分流關閉各個馬達)
        start_time = time.time()
        # 找出哪顆馬達需要最長的復位時間
        max_back_time = max([abs(t) for t in remaining_times])

        while time.time() - start_time < max_back_time:
            elapsed = time.time() - start_time
            for i in range(16):
                # 如果該馬達的反轉時間已經達標，立刻切斷電源
                if remaining_times[i] > 0 and elapsed >= remaining_times[i]:
                    self.kit.continuous_servo[i].throttle = 0
                elif remaining_times[i] < 0 and elapsed >= abs(remaining_times[i]):
                    self.kit.continuous_servo[i].throttle = 0
            # 稍作微小延遲，釋放樹莓派 CPU 負擔
            time.sleep(0.005)

        # 3. 最終保險防護：確保所有馬達斷電，並徹底清空大腦記憶
        for i in range(16):
            self.kit.continuous_servo[i].throttle = 0
            self.servo_accumulated_time[i] = 0.0

        print("✅ 全體矩陣精準歸位完成！")

def main():
    try:
        my_servo = Servo()
        #print("this servo will spin 2 sec, rest 2 sec, repeatedly!")
        while True:
            commands = {}
            for i in range(16): commands[i] = 1.0
            my_servo.go_up_and_down({4: 1.0, 5: 1.0})
            time.sleep(2)
            #my_servo.go_back()
            #my_servo.go_up_and_down({15: -1.0})
            #time.sleep(2)
            break

            #my_servo.go_up_and_down({0: 1.0, 1: 1.0, 8: 1.0, 14: 1.0})
            #time.sleep(2)


    except KeyboardInterrupt:
        # 當你按下 Ctrl + C 時，執行煞車動作
        my_servo.go_back()



if __name__ == '__main__':
    main()
# 這是馬達轉不轉的測試
import time
from adafruit_servokit import ServoKit


class Servo:
    def __init__(self):
        print("初始化 PCA9685 驅動板...")
        self.kit = ServoKit(channels=16)
        self.servo_current_angle = [0.0] * 16 # 應該是拿來復位用

    def go_up_and_down(self, **kwargs): # 存要上升的馬達以及各個要上升的速度
        for key, value in kwargs.items():
            self.kit.continuous_servo[key].throttle = value
        time.sleep(0.3)
        for key, value in kwargs.items():
            self.kit.continuous_servo[key].throttle = 0
            self.servo_current_angle[key] += value * 0.3

    def go_back(self):
        for i in range(16):
            self.kit.continuous_servo[i].throttle = -self.servo_current_angle[i] # 理論上來說angle is bounded by 1
        time.sleep(1)
        for i in range(16):
            self.kit.continuous_servo[i].throttle = 0


def main():
    try:
        my_servo = Servo()
        print("this servo will spin 2 sec, rest 2 sec, repeatedly!")
        while True:
            my_servo.go_up_and_down({0: 1.0})
            time.sleep(2)
            my_servo.go_up_and_down({0: -1.0})
            time.sleep(2)


    except KeyboardInterrupt:
        # 當你按下 Ctrl + C 時，執行煞車動作
        my_servo.go_back()


if __name__ == '__main__':
    main()
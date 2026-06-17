import time
import math
import cv2
import numpy as np

class Clock:
    def __init__(self):
        self.is_present = False
        self.is_counting = False

        # 持續一秒
        self.steady_start_time = 0.0
        self.confirm_delay = 1.0
        self.box = None
        self.center = None

        # 只看他那邊去判斷木塊是否有移開
        self.clock_body_mask = None
        self.original_clock_area = 0

        # 計時器專用變數
        self.set_seconds = 0
        self.time_str = "00:00"
        self.countdown_start_time = 0.0
        self.remaining_seconds = 0
        self.previous_binary_state = [0] * 16

    def check_stability(self, candidate_box, candidate_center):
        if candidate_box is not None:
            if self.steady_start_time == 0.0:
                self.steady_start_time = time.time()
            elif time.time() - self.steady_start_time > self.confirm_delay:
                self.is_present = True
                self.box = candidate_box
                self.center = candidate_center
                self.steady_start_time = 0.0
                print("Clock")
                return True
        else:
            self.steady_start_time = 0.0
        return False

    def lock_position(self, frame_shape, grid_start_x, grid_start_y, cell_w, cell_h):
        # 2. 記錄核心遮罩，並映射網格座標為時間
        self.clock_body_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.clock_body_mask, [self.box], 255)
        self.original_clock_area = cv2.countNonZero(self.clock_body_mask)

        cx, cy = self.center

        # 計算網格 X, Y
        col = max(0, min(3, (cx - grid_start_x) // cell_w))
        row = max(0, min(3, (cy - grid_start_y) // cell_h))
        col = int(col)
        row = int(row)

        TIME_MAP = [
            [5, 10, 15, 20],
            [25, 30, 35, 40],
            [45, 50, 55, 60],
            [65, 70, 75, 80]
        ]

        # 將對應到的秒數存入系統變數中，供後續倒數使用
        self.set_seconds = TIME_MAP[row][col]

        # 將秒數轉換成漂亮的人類可讀字串 (HH:MM:SS)
        mins, secs = divmod(self.set_seconds, 60)
        hours, mins = divmod(mins, 60)

        if hours > 0: # 這個是加爽的搞不好之後會用到
            self.time_str = f"{hours}h {mins}m {secs}s"
        elif mins > 0:
            self.time_str = f"{mins}m {secs}s"
        else:
            self.time_str = f"{secs}s"

        print(f"位置({col}, {row})={self.set_seconds}秒")
        print(f"總設定時間為：{self.time_str}")

    def start_countdown(self):
        print(f"\n開始倒數 {self.time_str}！")
        self.is_present = False
        self.is_counting = True
        self.countdown_start_time = time.time()

    def update_countdown(self):
        if not self.is_counting:
            return False

        elapsed = time.time() - self.countdown_start_time
        self.remaining_seconds = max(0, math.ceil(self.set_seconds - elapsed))

        # 倒數計時大字體格式 (MM:SS)
        mins, secs = divmod(self.remaining_seconds, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            self.time_str = f"{hours:02d}:{mins:02d}:{secs:02d}"
        else:
            self.time_str = f"{mins:02d}:{secs:02d}"

        if self.remaining_seconds <= 0:
            print("\n時間到, 陣列復位。")
            self.is_counting = False
            return True # 代表倒數結束

        return False # 還在倒數中

    def get_binary_state(self):
        mins, secs = divmod(self.remaining_seconds, 60)
        min_tens, min_units = divmod(mins, 10) # 分鐘的十位數與個位數
        sec_tens, sec_units = divmod(secs, 10) # 秒鐘的十位數與個位數

        state = [0] * 16

        # 利用 bitwise right shift (>>) 來把 0~9 的數字轉成 4 個 bit (0或1)
        # 例如數字 5 (二進位 0101): >>3 為 0, >>2 為 1, >>1 為 0, >>0 為 1

        # 第一橫列 (馬達 0~3): 分鐘十位數
        for i in range(4): state[0*4 + i] = (min_tens >> (3 - i)) & 1
        # 第二橫列 (馬達 4~7): 分鐘個位數
        for i in range(4): state[1*4 + i] = (min_units >> (3 - i)) & 1
        # 第三橫列 (馬達 8~11): 秒鐘十位數
        for i in range(4): state[2*4 + i] = (sec_tens >> (3 - i)) & 1
        # 第四橫列 (馬達 12~15): 秒鐘個位數
        for i in range(4): state[3*4 + i] = (sec_units >> (3 - i)) & 1

        return state

    def calculate_go_up_and_down_servo(self):
        if not self.is_counting:
            return {}

        current_binary_state = self.get_binary_state()
        commands = {}

        for i in range(16):
            if current_binary_state[i] != self.previous_binary_state[i]:
                if current_binary_state[i] == 1:
                    commands[i] = -1.0
                elif current_binary_state[i] == 0:
                    commands[i] = 1.0
            self.previous_binary_state[i] = current_binary_state[i]

        return commands


    def reset(self):
        self.is_present = False
        self.is_counting = False
        self.box = None
        self.center = None
        self.clock_body_mask = None
        self.original_clock_area = 0
        self.set_seconds = 0
        self.time_str = "00:00"
        self.previous_binary_state = [0] * 16



def main():
    print("========== 🤖 模擬器啟動 ==========")
    my_clock = Clock()

    # 1. 假裝我們有這些桌面的網格參數
    frame_shape = (480, 640, 3)
    grid_start_x, grid_start_y = 90, 10
    cell_w, cell_h = 110, 110

    # 2. 假裝我們用相機抓到了一個方塊
    # 設定在 (col=0, row=0) 的位置 -> 預期會抓到 TIME_MAP[0][0] = 5 秒
    fake_center = (140, 60)
    fake_box = np.array([[100, 20], [180, 20], [180, 100], [100, 100]], dtype=np.int32)

    # 3. 模擬「方塊放上去超過一秒」的防手震邏輯
    print("\n[模擬] 放上方塊...")
    my_clock.check_stability(fake_box, fake_center)
    time.sleep(1.1) # 假裝過了 1.1 秒

    if my_clock.check_stability(fake_box, fake_center):
        my_clock.lock_position(frame_shape, grid_start_x, grid_start_y, cell_w, cell_h)
    else:
        print("❌ 鎖定失敗，請檢查邏輯")
        return

    # 4. 模擬「方塊被拿開」
    print("\n[模擬] 玩家拿開了方塊！")
    my_clock.start_countdown()

    # 5. 模擬「時間流逝的主迴圈」 (每 0.5 秒更新一次)
    print("\n--- 進入監視迴圈 ---")
    while True:
        is_time_up = my_clock.update_countdown()

        # 呼叫你的核心函式，取得馬達指令
        commands = my_clock.calculate_go_up_and_down_servo()

        # 如果字典裡面有東西 (代表有位數改變)，我們就印出來看看！
        if commands:
            print(f"\n[{my_clock.time_str}] 偵測到位數改變！發送馬達指令: {commands}")

            # 把 16 顆馬達的二進位陣列，印成 4 橫排方便人類閱讀
            state = my_clock.previous_binary_state
            print(f"目前 BCD 陣列狀態:")
            print(f"   分鐘(十位): {state[0:4]}")
            print(f"   分鐘(個位): {state[4:8]}")
            print(f"   秒鐘(十位): {state[8:12]}")
            print(f"   秒鐘(個位): {state[12:16]}")
            print("-" * 40)

        if is_time_up:
            print("✅ 模擬結束。")
            break

        time.sleep(0.5) # 模擬 CPU 在跑其他事情


if __name__ == '__main__':
    main()
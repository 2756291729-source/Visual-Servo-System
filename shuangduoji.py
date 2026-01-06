# -*- coding: utf-8 -*-
import time
import serial
import serial.tools.list_ports


class STM32SafeController:
    def __init__(self):
        self.serial_port = None

    def connect_without_reset(self, port_name=None):
        """完全避免复位的连接方法"""
        try:
            # 1. 先创建虚拟端口
            self.serial_port = serial.Serial(None, 115200, timeout=0.1)

            # 2. 关键设置（必须在open之前）
            self.serial_port.dtr = None  # 完全释放DTR控制
            self.serial_port.rts = False  # 禁用RTS

            # 3. 指定端口并打开
            if port_name:
                self.serial_port.port = port_name
            else:
                # 自动检测STM32端口
                for p in serial.tools.list_ports.comports():
                    if 'CH340' in p.description:
                        self.serial_port.port = p.device
                        break

            self.serial_port.open()

            # 4. 发送激活脉冲
            self.serial_port.write(b'\x00')
            time.sleep(0.5)  # 必须的稳定时间

            print(f"安全连接成功 {self.serial_port.port}")
            return True

        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def send_cmd(self, cmd):
        """发送不会导致复位的指令"""
        if not self.serial_port:
            return False

        try:
            # STM32专用指令格式
            full_cmd = f"{cmd}\r".encode('ascii')

            # 分块发送避免干扰
            for i in range(0, len(full_cmd), 2):  # 每次发2字节
                self.serial_port.write(full_cmd[i:i + 2])
                time.sleep(0.01)  # 10ms间隔

            return True
        except:
            return False


def main():
    ctrl = STM32SafeController()

    # 连接阶段
    if not ctrl.connect_without_reset():
        print("""
        连接失败，请检查：
        1. 是否已断开DTR与NRST的物理连接
        2. BOOT0引脚是否已接地
        3. 开发板供电是否稳定
        """)
        return

    # 使用示例
    print("\n指令格式说明:")
    print("单舵机: <ID> <PWM值> (如: 001 1500)")
    print("双舵机: <ID1> <PWM1> <ID2> <PWM2> (如: 000 1500 001 1800)")
    print("输入 'exit' 退出")

    while True:
        try:
            cmd = input("ST32> ").strip()
            if cmd.lower() == 'exit':
                break

            parts = cmd.split()

            if len(parts) == 2:
                # 单舵机模式
                servo_id, value = parts
                cmd_str = f"#{servo_id}P{value}T1000!"
                if ctrl.send_cmd(cmd_str):
                    print(f"已发送单舵机指令: {cmd_str}")
                else:
                    print("发送失败")

            elif len(parts) == 4:
                # 双舵机模式
                servo_id1, value1, servo_id2, value2 = parts
                cmd_str = f"{{#{servo_id1}P{value1}T1000!#{servo_id2}P{value2}T1000!}}"
                if ctrl.send_cmd(cmd_str):
                    print(f"已发送双舵机指令: {cmd_str}")
                else:
                    print("发送失败")

            else:
                print("错误: 无效指令格式")

        except KeyboardInterrupt:
            print("\n程序退出")
            break
        except Exception as e:
            print(f"错误: {e}")

    # 关闭连接
    if ctrl.serial_port and ctrl.serial_port.is_open:
        ctrl.serial_port.close()
    print("串口已关闭")


if __name__ == "__main__":
    print("""
    ===== STM32安全控制程序 =====
    使用前必须：
    1. 断开CH340的DTR与STM32 NRST的连接
    2. 确保BOOT0引脚接地
    3. 使用质量可靠的USB线缆
    """)
    main()
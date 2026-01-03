# 🎯 Visual Servo Tracking System (基于视觉伺服的二维云台跟踪系统)

> 一个高精度、抗震荡的二自由度视觉跟踪系统。
>
> 采用 模糊控制 (Fuzzy Control) 与 脉冲积分 (Integral Pulse) 策略，有效解决了低成本舵机的机械死区与抖动问题。

(建议替换为你论文里的上位机运行截图)

## ✨ 核心特性 (Features)

* 👀 多模式视觉采集：支持海康威视工业相机（SDK集成）与普通 USB Webcam（OpenCV）。

* 🧠 智能控制算法：

  * 模糊控制：替代传统 PID，根据误差大小动态调整响应策略。

  * 脉冲积分策略 (Integral Pulse)：独创的抗震荡算法，将高频抖动转化为低频微调，实现 <4px 的稳态精度。

  * 卡尔曼滤波：平滑目标坐标，抵抗光照干扰。

* 🎮 交互式上位机：基于 PyQt5 开发，支持实时调参、ROI 锁定、波形显示。

* 🛡️ 硬件保护：内置舵机软限位与异常状态自动熔断机制。

## 🛠️ 系统架构 (Architecture)

系统采用 PC 上位机 + STM32 下位机的分层控制架构：

(建议替换为你论文里的系统框图 Fig.1)

## 🚀 快速开始 (Quick Start)

### 1. 硬件准备

* 相机: 海康威视 MV-CA013-20GC 或 任意 USB 摄像头。

* 云台: 二自由度舵机云台 (SG90 或类似)。

* 控制器: STM32F103 (或其他支持串口通信的单片机)。

### 2. 环境安装

```
# 克隆仓库
git clone [https://github.com/YourUsername/Visual-Servo-System.git](https://github.com/YourUsername/Visual-Servo-System.git)

# 安装依赖
pip install -r requirements.txt


```

### 3. 运行系统

连接好硬件（确认串口号），运行主程序：

```
python src/main.py


```

## ⚙️ 算法亮点 (Algorithm Highlight)

针对低成本舵机物理分辨率不足（最小步进对应画面 ~8px）导致无法精确对中的问题，本项目实现了 脉冲积分策略：

```
# 核心逻辑片段
if abs(fuzzy_output) <= 1.0: 
    # 微动蓄力模式
    self.pulse_accum += fuzzy_output  
    if abs(self.pulse_accum) >= THRESHOLD:
        action = 1 if self.pulse_accum > 0 else -1
        self.pulse_accum = 0 # 释放脉冲
else:
    # 大动响应模式
    action = round(fuzzy_output)


```


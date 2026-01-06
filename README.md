# 🎯 Visual Servo Tracking System (基于模糊控制的二自由度视觉伺服系统)

> 🚀 一个针对低成本硬件优化的“高精度、抗震荡”视觉跟踪解决方案。
>
> 本项目采用独创的 “模糊控制 (Fuzzy Control) + 脉冲积分 (Integral Pulse)” 混合策略，在低帧率相机（<10fps）和低精度舵机（存在机械死区）的非理想条件下，成功实现了稳态误差 <4px 的高精度跟踪。

## 🖼️ 效果演示 (Demo)

## ✨ 核心特性 (Key Features)

* 👁️ 双模式视觉采集引擎：

  * 工业级：深度集成 海康威视 (Hikvision) 工业相机 SDK (MvCameraControl)，支持参数软触发与曝光调节。

  * 消费级：兼容任意 USB Webcam (OpenCV)，实现低成本部署。

* 🧠 智能抗震荡控制算法：

  * 模糊逻辑：替代传统 PID，解决增益调度难题，根据误差非线性调整响应。

  * 脉冲积分策略 (Integral Pulse)：针对舵机物理死区设计的蓄力机制，将高频震荡转化为低频微调呼吸。

* ⚡ 高性能视觉处理：

  * ROI 动态追踪：基于上一帧位置锁定搜索区域，大幅降低算力消耗。

  * 卡尔曼滤波 (Kalman Filter)：平滑目标坐标，有效抑制光照突变和卷帘快门效应带来的干扰。

* 📊 交互式调试上位机：基于 PyQt5 开发，支持实时波形显示、参数在线整定、手动/自动模式切换。

## 🛠️ 系统架构 (Architecture)

系统采用典型的 PC上位机 + STM32下位机 分层架构：

```
graph LR
    A[工业相机/USB相机] -->|USB3.0| B(PC上位机)
    B -->|图像处理 & 模糊运算| B
    B -->|UART串口指令| C[STM32控制器]
    C -->|PWM信号| D[二自由度云台]
    D -->|机械运动| A


```

1. 感知层：采集原始图像，进行畸变校正（calibrate.py 提供参数）。

2. 决策层：计算像素偏差，通过模糊控制器输出修正量。

3. 执行层：STM32 解析 {xxP1500T1000!} 指令驱动舵机。

## ⚙️ 算法亮点 (Algorithm Highlight)

针对低成本舵机物理分辨率不足（最小步进对应画面位移 ~8px）导致的“死区无法对中”或“目标点震荡”问题，本项目实现了以下创新：

### 1. 脉冲积分策略 (Integral Pulse Strategy)

传统 PID 在微小误差下容易因积分项累积过快导致超调（Overshoot）。本策略引入时间维度的蓄力机制：

```
# controller.py 中的核心逻辑简述
if abs(fuzzy_output) <= 1.0: 
    # [微动蓄力模式]
    # 当输出极小时，不立即驱动舵机，而是存入累积器
    self.pulse_accum += fuzzy_output  
    
    # 只有当蓄力值超过物理死区阈值 (THRESHOLD) 时，才释放一次最小步进脉冲
    if abs(self.pulse_accum) >= THRESHOLD:
        action = 1 if self.pulse_accum > 0 else -1
        self.pulse_accum = 0 # 释放能量
else:
    # [大动响应模式]
    action = round(fuzzy_output)


```

### 2. 模糊集合设计

将误差映射为 {NB, NS, ZE, PS, PB} 五个模糊集，特别压缩了 ZE (Zero) 区域的范围，配合脉冲策略实现高灵敏度捕捉。

## 🚀 快速开始 (Quick Start)

### 1. 硬件准备

* 相机: 海康威视 MV-CA013-20GC (推荐) 或 普通 USB 摄像头。

* 云台: SG90 / MG996R 双轴舵机云台。

* 控制器: STM32F103C8T6 (或其他串口设备)。

### 2. 环境依赖

本项目依赖 Python 3.8+。

特别注意：如果你使用海康相机，必须先安装 MVS 客户端 (Machine Vision Studio) 以获取驱动支持。

```
# 1. 克隆仓库
git clone [https://github.com/YourUsername/Visual-Servo-System.git](https://github.com/YourUsername/Visual-Servo-System.git)
cd Visual-Servo-System

# 2. 安装 Python 依赖
pip install -r requirements.txt


```

### 3. 运行系统

连接好硬件，确认串口号（如 COM3），然后运行：

```
python src/main.py


```

## 📂 项目结构 (File Structure)

```
Visual-Servo-System/
├── src/
│   ├── main.py                    # 主程序入口 (GUI & 逻辑)
│   ├── shuangduoji.py             # STM32舵机通信封装
│   ├── calibrate.py               # 相机标定脚本
│   ├── camera_calibration_results.npz # 标定参数文件
│   ├── MvCameraControl_class.py   # 海康威视 SDK 核心
│   ├── CamOperation_class.py      # 相机操作封装
│   ├── MvErrorDefine_const.py     # 错误码定义
│   └── PixelType_header.py        # 像素格式定义
├── assets/                        # 演示图片与文档资源
├── requirements.txt               # 依赖库列表
└── README.md                      # 项目说明文档


```

## 📄 论文与引用 (Reference)

本项目的详细理论分析、实验数据及 PID 与模糊控制的对比实验，请参阅课程报告：

* 基于模糊控制的视觉伺服跟踪系统设计.pdf&#x20;

## 🤝 致谢 (Acknowledgments)

* 感谢 深圳技术大学 (SZTU) 工程物理学院提供的实验设备支持。

* 感谢课程《高级项目研究及劳动教育》指导教师的建议。

## 📜 许可证 (License)

本项目遵循 MIT 许可证。详见 LICENSE 文件。


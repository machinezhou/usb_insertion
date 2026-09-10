项目介绍：固定SO101机械臂桌面抓取USB A数据线插入固定USB插口
1 环境：

机械臂固定
插座固定
top / side / wrist camera 固定

USB：
放在桌面限定区域内
位置随机
yaw 可以随机
正反面可能变化
有明显正反标识

USB 线：
自由
有足够松弛
不打结
不缠机械臂
不绷紧

2 实现方案：
NORMAL HOME / RESET
        │
        ▼
┌──────────────────────┐
│ ACT PICK + APPROACH  │
│                      │
│ 找 USB               │
│ 抓 USB               │
│ 抬起                 │
│ 粗调方向              │
│ 搬到插座附近          │
└──────────┬───────────┘
           │
           ▼
   Vision Capture Check
           │
           ▼
┌──────────────────────┐
│ VISUAL ALIGNING      │
│                      │
│ top + side           │
│ USB tip / axis       │
│ reference pose       │
│ image error          │
│ visual servo         │
└──────────┬───────────┘
           │
           ▼
       PREINSERT
           │
           ▼
┌──────────────────────┐
│ VISUAL INSERTING     │
│                      │
│ 沿插入轴慢速前进       │
│ 持续检查横向/姿态误差   │
└──────────┬───────────┘
           │
           ▼
     SUCCESS DETECTOR
       │          │
     SUCCESS     FAILURE
       │          │
      DONE      RETRACT
                   │
                   ▼
             VISUAL ALIGN
                   │
                 RETRY


3 模块职责划分：
ACT
= 处理随机 USB 初始状态、抓取和长距离搬运

Vision
= 测量 USB 相对理想插入姿态的误差

Visual Servo
= 做确定性的精密闭环对准

Insertion Controller
= 做最后慢速插入

Success Detector
= 独立确认是否真的插到底

State Machine
= 管理所有阶段和 retry





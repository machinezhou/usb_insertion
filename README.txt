# USB Insertion

基于 **SO-101 机械臂 + LeRobot ACT + 多相机视觉闭环** 的 USB-A
自动抓取、对准与插入系统。

> **当前状态（2026-09-10）：V1 CODE FREEZE**
>
> 纯软件开发与测试阶段已完成。下一阶段进入实验环境固定、硬件配置、视觉/插入标定、Preflight
> 和低风险实机联调。除非实机验证证明算法或接口本身存在问题，否则暂不继续修改核心源码。

## 1. 项目目标

固定 SO-101、USB 插座以及 `top / side / wrist` 三路相机。USB
位于桌面限定区域，位置和 yaw 可随机，正反面可能变化并具有可识别标记。USB
线保持松弛，不打结、不缠机械臂、不绷紧。

系统目标：

1.  ACT 寻找并抓取 USB；
2.  抬起、粗调方向并搬运到插座附近；
3.  Vision 判断是否进入视觉接管区域；
4.  Visual Servo 完成精密对准；
5.  Plug-Face Classifier 判断正反面；
6.  Insertion Controller 沿插入轴低速推进并持续检查误差；
7.  Success Detector 独立判断是否插入成功；
8.  失败时 RETRACT → ALIGN → RETRY。

## 2. 工作流

``` text
HOME / RESET
    │
    ▼
ACT PICK + APPROACH
    │
    ▼
Vision Capture Check
    │
    ▼
VISUAL ALIGNING
    │
    ▼
PREINSERT
    │
    ▼
VISUAL INSERTING
    │
    ▼
SUCCESS DETECTOR
   │          │
SUCCESS    FAILURE
   │          │
 DONE      RETRACT
              │
              ▼
         VISUAL ALIGN
              │
            RETRY
```

模块职责：

-   **ACT**：处理随机 USB
    初始状态、抓取和长距离搬运，不承担最终毫米级插入精度。
-   **Vision**：测量 USB 相对理想插入姿态的视觉误差。
-   **Visual Servo**：利用 image Jacobian 做确定性的精密闭环对准。
-   **Insertion Controller**：完成最后的低速插入，并持续执行 visual
    guard。
-   **Success Detector**：独立确认 USB 是否真正插入成功。
-   **State Machine / Orchestrator**：管理
    HOME、ACT、ALIGN、INSERT、RETRACT、RETRY、DONE/ABORT。

## 3. 代码结构

``` text
src/usb_insertion/
├── adapters/          # LeRobot ACT、SO-101、camera 等适配
├── cli/               # dry-run、真实运行、标定、inspection
├── config/            # hardware / run 等配置
├── control/           # ActionGuard、轨迹、控制循环、scripted motion
├── core/              # datatypes、protocols、基础工具
├── fakes/             # 无硬件测试组件
├── runtime/           # preflight
├── vision/
│   ├── alignment/     # alignment、plug-face、visual servo
│   └── insertion/     # insertion、success detection
└── workflow/          # state machine、orchestrator

tests/                 # 单元/集成测试
```

真实入口由 `run_task` 将 hardware/run
config、Preflight、ACT、Vision、Visual Servo、Insertion
Controller、Success Detector、SO-101 adapter、ActionGuard、scripted
motion 和 Orchestrator 连接成完整闭环。

## 4. V1 Code Freeze 验证结果

截至 2026-09-10：

-   [x] 未发现遗留 `TODO` / `FIXME` / `NotImplementedError`
-   [x] 空 Python 文件仅为正常的 `__init__.py`
-   [x] `compileall` 通过
-   [x] 核心 import smoke test 通过
-   [x] RunConfig / Preflight / ACT Adapter 最终边界测试通过
-   [x] 全量测试 **173 / 173 PASS**
-   [x] Fake end-to-end dry run 通过
-   [x] 最终状态 `DONE`
-   [x] retry 路径已在 dry run 中实际走通

验证结果：

``` text
USB INSERTION V1
====================================
Source completeness       PASS
No unfinished TODO        PASS
Python compile            PASS
Core imports              PASS
RunConfig tests           PASS
Preflight tests           PASS
ACT adapter tests         PASS
Full unit tests           PASS  173 / 173
Fake end-to-end dry run   PASS
------------------------------------
V1 SOFTWARE STATUS        CODE FREEZE
====================================
```

Fake dry run 基线：

``` text
========== DRY RUN RESULT ==========
success          : True
final_state      : DONE
attempts         : 2
ACT steps        : 4
alignment steps  : 4
insertion steps  : 6
====================================
```

## 5. 开发环境

当前开发环境：

``` text
Conda environment: lerobot_usb
Project directory: ~/projects/usb_insertion
```

仓库保留环境基线：

``` text
environment.yml
conda-explicit.txt
pip-freeze.txt
software_baseline.txt
pyproject.toml
```

进入环境：

``` bash
conda activate lerobot_usb
cd ~/projects/usb_insertion
```

## 6. 软件回归测试

修改核心源码后至少执行：

``` bash
cd ~/projects/usb_insertion

python -m compileall -q src/usb_insertion tests

python -m unittest discover   -s tests   -p 'test_*.py'   -v

python -m usb_insertion.cli.dry_run
```

当前基线为
`173 tests / OK`。以后测试数量可以增加；判断标准是**全部通过**，不是永远固定为
173。

Dry run 至少应得到：

``` text
success     : True
final_state : DONE
```

## 7. Code Freeze 后的排查原则

后续实机阶段如果出现 USB 识别不稳定、ROI 不合适、正反面误判、capture
region 不合适、servo 方向/增益不合适、插入距离不合适、success threshold
不合适或 ACT→Vision 切换时机不合适，优先按以下顺序排查：

``` text
实验环境
  ↓
hardware 配置
  ↓
标定资产
  ↓
运行参数 / 阈值
  ↓
单模块实机验证
  ↓
算法 / 源码
```

只有实机证明确实存在算法假设、接口或实现问题时，才解除 Code Freeze。

## 8. 配置与标定资产

代码完成不等于真实机器人已经可以直接执行完整任务。

当前已有：

``` text
configs/hardware.yaml
```

后续需要根据真实环境生成或完善的运行资产包括：

``` text
configs/vision.yaml
configs/plug_face.yaml
outputs/vision_alignment/...
```

以及 image Jacobian、PREINSERT/INSERTED pose、insertion joint
delta、success reference、ACT checkpoint/run 参数等。

这些属于**实机配置与标定阶段**，不是 V1 源码缺失。

### Plug-Face

正反面分类代码已经完成，但真实 USB 模板尚未标定。

``` bash
python -m usb_insertion.cli.calibrate_plug_face_reference
```

预期生成：

``` text
configs/plug_face.yaml
outputs/vision_alignment/plug_face/marked.png
outputs/vision_alignment/plug_face/unmarked.png
```

之后使用：

``` bash
python -m usb_insertion.cli.inspect_plug_face_live
```

检查 MARKED / UNMARKED / NO USB 三种情况。

模板尚未生成时，不应直接把 `inspect_plug_face_live`
的配置文件缺失视为代码错误。

## 9. Preflight 与实机安全

真实完整运动之前必须通过 Preflight。它用于确认
hardware/run/vision/plug-face 配置、ACT checkpoint/dataset
contract、robot port、camera、insertion delta、guard/correction
keys、success reference 等关键依赖已经准备完整。

安全原则：

-   标定和 inspection 优先使用只读连接；
-   不需要运动的 CLI 不应发送动作；
-   初次运动使用保守速度、动作增量和 timeout；
-   先验证单模块，再运行完整任务；
-   保留 ActionGuard、visual guard、retry 和 abort；
-   机械臂、USB 线、插座和相机周围必须无碰撞风险；
-   异常运动立即停止，不通过放宽安全阈值强行继续。

# TODO --- 下一阶段：实机配置与标定

## Phase 1 --- 固定实验环境

-   [ ] 固定 SO-101 底座和 USB 插座
-   [ ] 固定 top / side / wrist 三路相机
-   [ ] 确定 USB 随机放置区域
-   [ ] 确认 USB 线松弛且不会缠绕机械臂
-   [ ] 尽量固定照明条件

## Phase 2 --- Hardware 核对

-   [ ] 核对 SO-101 follower port / calibration
-   [ ] 核对三路 camera device、resolution、FPS
-   [ ] 更新并验证 `configs/hardware.yaml`
-   [ ] 验证 robot read-only observation
-   [ ] 独立验证 top / side / wrist camera

## Phase 3 --- Vision Alignment

-   [ ] 固定理想 PREINSERT 几何关系
-   [ ] 采集 alignment reference
-   [ ] 确认 top / side ROI
-   [ ] 验证 USB tip / axis 特征
-   [ ] 验证 capture region
-   [ ] 生成/完善 `configs/vision.yaml`
-   [ ] 标定并验证 image Jacobian
-   [ ] live inspection 验证视觉误差和 servo 方向

## Phase 4 --- Plug-Face

-   [ ] 采集 `marked.png`
-   [ ] 采集 `unmarked.png`
-   [ ] 生成 `configs/plug_face.yaml`
-   [ ] 验证 MARKED / UNMARKED / NO USB
-   [ ] 必要时调整
    ROI、`min_good_matches`、`ratio_test`、`min_score_margin`

## Phase 5 --- Insertion / Success

-   [ ] 采集 PREINSERT pose
-   [ ] 采集 INSERTED pose
-   [ ] 生成/验证 insertion joint delta
-   [ ] 确认 gripper delta 为安全值
-   [ ] 采集 success reference
-   [ ] 验证 success detector
-   [ ] 验证 retract 路径

## Phase 6 --- ACT / Run 配置

-   [ ] 确认 ACT checkpoint
-   [ ] 核对 dataset state/action joint order
-   [ ] 核对 ACT 使用的 camera names
-   [ ] 确认 task prompt
-   [ ] 完善 `configs/run.yaml`
-   [ ] 使用保守的 action delta、速度、timeout、retry 参数

## Phase 7 --- Preflight

-   [ ] 所有配置文件存在
-   [ ] 所有 reference / `.npy` 标定资产存在
-   [ ] camera / robot / ACT contract 全部通过
-   [ ] Preflight 无 error
-   [ ] 记录最终硬件和标定基线

## Phase 8 --- 分模块实机验收

建议顺序：

``` text
Camera read-only
    ↓
Vision live inspection
    ↓
Plug-face live inspection
    ↓
Success detector
    ↓
Robot observation
    ↓
Scripted HOME / RETRACT（低速）
    ↓
Visual Servo（小步）
    ↓
Insertion Controller（低速、短距离）
    ↓
ACT checkpoint
```

## Phase 9 --- 首次整机联调

-   [ ] 初次使用最低风险速度/动作范围
-   [ ] ACT 只负责送入 capture region
-   [ ] 确认 ACT → Vision handoff
-   [ ] 确认 Visual Align 收敛
-   [ ] 确认 PREINSERT
-   [ ] 确认 Visual Insertion guard
-   [ ] 确认 Success Detector
-   [ ] 确认 failure → retract → retry
-   [ ] 确认最终 `DONE`
-   [ ] 保存日志并复盘

------------------------------------------------------------------------

## 10. 开发恢复点

``` text
当前阶段：V1 CODE FREEZE 已完成
软件基线：compile PASS / imports PASS / 173 tests PASS / dry_run DONE
当前任务：不要继续补功能代码
下一步：Phase 1 实验环境固定 + Phase 2 hardware.yaml 实机核对
之后：Vision → Plug-Face → Jacobian → Insertion → Success → ACT/Run → Preflight → 实机联调
```

**恢复原则：先读取本 README 和仓库当前代码，再从 TODO
中第一个未完成项继续。**
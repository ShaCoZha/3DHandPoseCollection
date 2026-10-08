> Layout note: this document preserves historical protocol and operating notes. For commands in the reorganized checkout, use [the repository README](../README.md).

# 15 分钟 Apple Watch IMU + Hand Pose

## 新模式：四相机替代 Quest（Linux 采集机）

接收端新增 `--pose-source multicamera`；默认不加参数仍为原 Quest 模式。
新模式复用手机现有的 `capture_prepare → capture_prepared → capture_commit → capture_committed → capture_stop/capture_finished` 协议，**不要求 Quest 连接**。
当前目录未包含 `PhoneRelayModel.swift` / `PhoneRootView.swift`：这里改的是 server 的采集模式，没有修改或重新编译手机 UI。
在使用本文既有同步协议的 iPhone app 上，仍从 **15 min IMU + Hand Pose** 入口开始；若手机 UI 文案仍显示 Quest，仅文案没有更改。
不要使用旧的普通 IMU 录制入口，那个入口不发送同步采集控制消息。

### 启动

关闭占用相机的 SpinView / 预览程序，然后在这台电脑运行：

```bash
cd /home/spice-lab/Desktop/pc_receiver
python3 server.py --host 0.0.0.0 --port 8765 \
  --pose-source multicamera \
  --camera-config multicamera_config.json \
  --dataset-dir dataset_multicamera
```

手机连接这台电脑的可达 IP、端口 8765，保持前台；Watch 按原流程校时和录制。

若需要自动运行最新版逐关节加权后处理，使用单独的 `multicamera_config_weighted.json`：

```bash
cd /home/spice-lab/Desktop/pc_receiver
/home/spice-lab/miniconda3/bin/python3 -u server.py --host 0.0.0.0 --port 8765 \
  --pose-source multicamera \
  --camera-config multicamera_config_weighted.json \
  --dataset-dir dataset_multicamera
```

此配置启用 `weightedHandpose`、显式指定 `reliabilityPython`，并使用 35 ms 的四视角配对阈值。
原配置保持 30 ms；后一次约 93 秒录像的实际四视角跨度约 33 ms，曾因 30 ms 阈值无法配对。
35 ms 仅放宽离线帧组的接受范围，不代表同时曝光或改善了跨设备同步精度。
Watch 上传未完成时，不应把 `capture_finished` 当作完整数据已到齐；加权回放需要 IMU 文件可用。

点击 Start 后，server 检查四台相机、打开时间戳数据、确认实际图像和编码尺寸，再与手机校时。
只有 Watch 已确认且手机发出 commit，才在预定窗口内保存 RGB。未 commit、迟到、初始化失败均中止。
相机进程自身持有 900 秒结束期限；手机 Stop、连接中断或 server 管道 EOF 也会停止。
每台相机独立采集，某路缺帧不会把其它三路的原始数据一起丢弃。

### 配置与标定

`multicamera_config.json` 指向已验证的 PySpin、WiLoR、Anipose 三个 Python 环境，不合并或重装环境。
当前默认选用经本次录像几何对照后更匹配的：
`pc_receiver/calibration_captures/2026-10-02_20-36-58/calibration/calibration.toml`。
这是 2026-10-02 新拍 120 秒 ChArUco 板后联合优化的内外参。153 组低运动画面用于
求解，86 组按时间块留出验证；跨视角误差中位数 0.61 px、P90 1.72 px。
这属于同一录像内的验证（也用于比较两种求解方法），不是独立拍摄的验收，也不是手部关节精度保证。
旧默认 9/19 参数保留，配置备份在新标定目录的 `previous_multicamera_config.json`。
如需另一份已有标定，启动前更改配置中的 `calibration` 路径即可。

新标定的检测缓存、选中画面、报告及板角点叠加图均保存在上述 `calibration` 目录。
无桌面离线脚本为 `calibrate_multicamera_recording.py`（检测、相机时钟漂移修正、板运动估计）
及 `solve_multicamera_calibration.py --refine-intrinsics`（求解及跨视角验证），均使用
`Camera Calibration Orlyse/test_pyspin/bin/python`，位置参数为完整标定录像会话目录。
求解脚本只生成候选文件，不自动部署。板配置为 10×8、22 mm 方格、16 mm 标记、DICT_4X4_50。
固定内参版本也可运行；完整参数版本包含对旧内参的软约束。
四台相机按原硬件序列号对应 cam01–cam04；坐标单位米，cam01 为世界原点。
新标定只能用于相同机位和镜头设置的采集；重建旧录像前必须核对这一点。
每个 session 复制该文件、保存 SHA-256 及以下固定映射：

| 视频标识 | 相机序列号 |
|---|---|
| cam01 | 25132928 |
| cam02 | 25132918 |
| cam03 | 25132909 |
| cam04 | 25132908 |

配置 `hand` 默认 `right`，左手实验改为 `left`。当前姿态处理针对共同视野中**一只指定左右侧的手**，没有实现多人的跨视角身份关联。
配置 `fps` 默认 20；保存实际时间戳，不用 `frame_index / 20` 作为时间。
RGB 使用 mp4v 压缩并按约 60 秒切段，保留相机实际分辨率；这里的 RGB 指彩色图像，不是无损 Bayer 原始传感器数据。
本模块不写相机的持久 UserSet，结束后尝试还原本次修改的易失采集设置。

### 新文件

每次手机 UUID 对应一个目录，原 Watch 文件名保持不变：

```text
dataset_multicamera/<sessionId>/
  capture.json
  clock_sync.jsonl
  imudata.jsonl                    # Watch 经手机在结束后上传
  timestamp.json
  multicamera_hand_pose_aligned.jsonl
  multicamera/
    config.json
    calibration.toml
    metadata.json                 # 标定哈希、相机映射、时钟方式
    status.json                   # 实际帧数、缺帧及停止状态
    worker.log
    cam01_clock.jsonl ... cam04_clock.jsonl
    cam01_frames.jsonl ... cam04_frames.jsonl
    cam01_frames_aligned.jsonl ... cam04_frames_aligned.jsonl
    rgb/cam01/segment_0000.mp4 ...
    frame_sets.jsonl
    alignment_report.json
    postprocess.log
    handpose/
      processing_status.json
      wilor_2d.npz
      pose_3d.npz
      wilor_report.json
      pose_report.json
      wilor.log
      triangulate.log
```

`camXX_frames.jsonl` 一行严格对应一帧已提交给编码器的图像，包含视频路径、段内帧号、相机 FrameID、相机原始纳秒时间戳、曝光时长、PC 收到图像的单调时间、初始映射时间及估计时钟不确定度。
播放 MP4 的时间只用于播放，分析使用 JSONL；异常中断后应检查段文件能否解码，不把 sidecar 行数自动等同于已成功落盘的图像数。

### 时间对齐方法和边界

1. 每台相机用 `TimestampLatch` / `TimestampLatchValue` 测量相机时钟到 PC 单调时钟的偏移。每组 8 次，选主机调用区间最短的一次，保留全部原始测量。开始、约每 30 秒、结束时追加测量。
2. PC 与 iPhone 沿用四时间戳 ping/pong；开始和约每 60 秒测量，保存 `device="pc"` 的 PC–Phone 偏移。
3. 在线映射冻结初始 offset；离线在原始相机时间坐标插值 camera–PC offset，再在 PC 时间坐标插值 PC–Phone offset，最后加手机冻结 epoch。

```text
pcMs = cameraTimestampNs / 1e6 - cameraMinusPcMs
phoneMs = pcMs - pcMinusPhoneMs
unixTimeMs = phoneMs + phoneEpochOffsetMs
```

没有把帧到达 PC 的时间冒充相机时间戳。时间戳或 Latch 不可用时会失败，不静默退化成回调时间。
相机 SDK 的图像时间戳接口单位为纳秒，具体图像事件时刻仍取决于设备；此实现没有宣称已测量曝光中心、Watch 传感器到图像的端到端延迟。
依据：[FLIR 图像时间戳接口](https://softwareservices.flir.com/Spinnaker/latest/cpp/class_spinnaker_1_1_image.html)、[本机型号 Timestamp Latch](https://softwareservices.flir.com/BFS-PGE-23S3/latest/Model/public/DeviceControl.html)。

**本模式做时钟对齐，不启用 GPIO 同步触发或 PTP 同步曝光。** 四台相机自由运行，`hardwareExposureSynchronized=false`。
离线按校正时间就近配对，每帧最多使用一次；默认四视角时间跨度超过 30 ms 的组合不用于三角化，阈值可改 `maxPairSkewMs`。
姿态时间采用 cam01 的实际时间，各视角真实时间及 `cameraSkewMs` 同时保存。该时间差不是网络校时不确定度，两者需要分别评估。
估计不确定度来自相机 Latch 主机调用区间和 PC–Phone RTT，不含 Watch–Phone RTT、网络非对称、曝光/传感器延迟或未观测漂移。

### 姿态处理和 IMU 配对

正常到期或手机 Stop 后，默认 `autoProcess=true`，后台依次运行：
时间漂移校正 → 四视角时间匹配 → WiLoR 2D → aniposelib RANSAC 3D。
手机收到 `capture_finished` 表示采集停止，不表示 Watch 上传或 GPU 后处理已经完成。
查看 `capture.json.handPoseProcessing` 或 `multicamera/handpose/processing_status.json`。
同一 server 的后处理任务串行；若需连续采集时避免 GPU 后处理争用资源，将 `autoProcess` 设为 `false`，最后统一处理。
失败/断线的 partial session 不自动推理，检查原始数据后可手动运行：

```bash
python3 process_multicamera.py "dataset_multicamera/<sessionId>"
```

这条命令也用于恢复失败的后处理；不要在同一 session 的后台任务仍运行时重复执行。
WiLoR 按指定 handedness 选最大检测；侧面/背面被误判为另一只手时会缺失，不把该视角错误关联到另一只手。
三角化至少需要两个有效视角；当前以 5 px 重投影残差筛除关节，失败关节为 `valid=false, position=null`，不沿用旧姿态。
输出 21 个具名关节、米制位置、重投影残差、使用相机数及来源帧。它不是 Quest 的 26 骨骼/quaternion schema，不直接输入原来的 Quest 专用 viewer 或数据集脚本。

Watch 上传完成后：

```bash
python3 align_capture.py "dataset_multicamera/<sessionId>"
```

该命令识别新模式，生成 `imudata_aligned.jsonl` 和相机对齐报告，不要求 Quest 的 `hand_pose.jsonl`。
分析用 `imudata_aligned.jsonl` 与 `multicamera_hand_pose_aligned.jsonl` 的 `unixTimeMs`；按时间配对，不按行号配对。
IMU 轴与标定世界坐标仍不是共同的空间坐标，时间校准不替代空间旋转标定。

### 无桌面机器的回放

```bash
/home/spice-lab/Desktop/HandPoseComparison/poem_env/bin/python visualize_multicamera.py "dataset_multicamera/<sessionId>"
python3 serve_visualization.py "dataset_multicamera/<sessionId>/visualization" --port 8766
```

在本地 VS Code / Cursor 的 Ports 面板转发远端 8766，使用本地浏览器打开
`http://localhost:8766/viewer.html`。回放服务仅绑定远端 localhost，并支持视频字节范围请求，便于拖动时间轴。
页面顶部显示所用标定；重新生成后刷新页面。`8765` 是手机采集 TCP 服务，`8766` 是浏览器回放服务，两者独立。

### 骨长与时间约束

`multicamera_config.json` 的 `regularizeHandpose: true` 会在 Anipose 三角化后运行
`regularize_handpose.py`，使用 `regularizationPython` 指定的环境。当前默认使用现有
`HandPoseComparison/poem_env/bin/python`，不修改 WiLoR/Anipose 环境。

每段录像从至少三个视角较一致的关节估计固定骨长，再联合优化三维观测锚点、软骨长约束、
按真实时间戳计算的加速度约束。当前时间断开阈值 0.12 秒；每关节相邻有效观测间隔
在此阈值内时，即使中间缺一帧也保持时间约束，但缺失关节本身仍不插值。
2026-10-03 将 `accelerationScale` 从 8 调为 3，数值越小时间约束越强。
这可能压低快速真实动作，需结合视频贴合程度评估，不能只用加速度变小判断效果。
较可靠观测范围外的极端空间点会剔除，报告记录数量和范围；这是基于本段录像的范围估计，
不是通用的手部活动边界。模型身份错误与不同步曝光仍可能产生错误轨迹。

结果输出保持 `handpose/pose_3d.npz` 和 `multicamera_hand_pose_aligned.jsonl` 接口；
原始结果保留为 `pose_3d_unconstrained.npz`、`pose_report_unconstrained.json` 和
`multicamera_hand_pose_unconstrained.jsonl`。重新运行三角化会刷新这些原始输入；仅重跑约束
始终从原始输入开始，避免重复平滑。参数和前后指标见 `regularization_report.json`。
约束后 `reprojectionErrorPixels` 表示全部可用视角的误差中位数；旧的两视角误差单独保存在
`rawSubsetReprojectionErrorPixels`，不能将旧的 5px 筛选标准当成优化后坐标的质量保证。

仅重跑约束（在 pc_receiver 目录）：

```bash
/home/spice-lab/Desktop/HandPoseComparison/poem_env/bin/python process_multicamera.py "dataset_multicamera/<sessionId>" --stage regularize
```

此次 B2117BB9 会话提供 `/constraints/comparison.html` 同步前后对比。骨长更恒定、
加速度更小不是实际精度证明；页面同时显示原来可靠视角的投影误差变化，说明平滑的代价。
另有 `/stabilization/comparison.html` 对比 10/03 进一步减抖前后。新版独立 3D 窗口使用
整段固定中心和尺度，不随当前帧缺失关节重新缩放；滚轮仍可手动调整缩放。

### POEM 离线比较

已有 WiLoR 二维结果的右手会话，可以用现有 POEM OakInk 预训练权重运行对照：

```bash
/home/spice-lab/Desktop/HandPoseComparison/poem_env/bin/python /home/spice-lab/Desktop/HandPoseComparison/run_poem_capture.py "dataset_multicamera/<sessionId>"
/home/spice-lab/Desktop/HandPoseComparison/poem_env/bin/python visualize_multicamera.py "dataset_multicamera/<sessionId>" --method poem
```

推理读取 `frame_sets.jsonl` 指定的各路实际视频帧和会话标定，不按相同帧号假定同步。
结果另存 `multicamera/poem/`，包括关节、778 顶点网格、时间戳、裁剪信息及失败帧；
POEM 自带的无效投影检查失败时该帧留空，不插值三维结果。
浏览器回放位于 `http://localhost:8766/poem/viewer.html`。此次 B2117BB9 会话还提供
`/poem/comparison.html`，用于两个方法在同一时间的并排回放。
POEM 裁剪来自 WiLoR；输出完整度不等于置信度，不能与 Anipose 的筛选通过率直接比较。
两种方法之间的距离是分歧，不是真值误差；对 WiLoR 二维点的重投影指标会偏向 WiLoR 基线。

### 已完成的验证

- 原 Quest 协议回归及新模式无 Quest 启动、commit 前不录制、来源校验、Stop/断线释放、两级时钟映射、按时间配对/不复用帧的自动测试。
- 四台实机以 1920×1200、约 20 fps 录制约 4 秒，经模拟 iPhone 完整 prepare/commit/stop 流程；得到 80/81/80/80 帧，记录窗口内 FrameID 无缺口，SDK 正常释放。
- 该次实机测试配出 80 组，最大四视角时间跨度约 16.45 ms；这是自由运行相机的实测时间差，不是 iPhone 同步精度。
- 已有手部录像的 10 帧通过新离线 WiLoR＋Anipose 流程；测试 fixture 的时间戳是合成的，仅验证推理和文件关联。
- 尚未完成真实 iPhone＋Watch 的 15 分钟联合录制，不能据以上测试宣称跨设备同步精度已验证。

```bash
python3 -m unittest -v test_synchronized_capture test_multicamera_capture
```

以下为原 Quest 模式说明。

## 使用

1. 在 Mac/Xcode 重新编译、安装本 workspace 的 iPhone 和 Watch app。选择新模式 **15 min IMU + Hand Pose**，时长固定为 900 秒。
2. 用 Unity 6000.0.52f1 打开 `Z:\unity_projects\QuestBodyAPI`，重新 Build And Run 到 Quest。现有 `samplescene.apk` 没有包含此次修改。新脚本是现有 `PinchEventToServer` 的 partial class，已有场景无需添加组件；会自动发现 active 的 OVR/OpenXR 左右手 skeleton。
3. 在 PC receiver 目录运行：

   ```powershell
   python server.py --host 0.0.0.0 --port 8765 --dataset-dir dataset
   ```

   新模式不需要 `--quest`；原来的 Quest cue/event 模式仍可用 `--quest`。新模式要求默认的 session 文件夹布局，不能配合独立 `--imudata-dir` / `--timestamp-dir`。不要启用 IMU downsampling，除非实验确实需要。
4. iPhone 的 Host 和 Quest 场景中 `PinchEventToServer.serverIp` 都设为 PC 在同一网络上的地址，TCP 端口为 8765。打开 Watch app，确认 HealthKit/workout 权限；启动 Quest app 并启用 hand tracking。
5. 等待正在进行的旧文件上传结束。在手机连接 PC 后点击 Start。只有一个 pose-capable Quest TCP client 可参与；旧版本 Quest 会被拒绝，避免假同步。
6. 手机依次完成 Watch 校时、PC 对 Phone/Quest 校时、Quest arm、Watch start ACK、Quest commit ACK。预计开始时间设置在 PC 校时完成后 10 秒。失败或超时会中止，界面不会回退为“假设已经开始”。
7. 保持 iPhone app 在前台（本模式自动禁止闲置锁屏），Quest 保持佩戴、手在视野内。到时间后 Watch 和 Quest **各自按本地单调时钟自动结束**，无需依赖手机每秒发命令。也可在手机提前 Stop，结果标记为 partial/stopped。
8. 等待 Watch 文件经 iPhone 上传到 PC。Watch 仍采用原来的本地录制、结束后 `WCSession.transferFile` 方式，采集期间 PC 不会实时收到 IMU。
9. 在文件上传完成后运行漂移校正：

   ```powershell
   python align_capture.py "dataset\<sessionId>"
   ```

   输出 `imudata_aligned.jsonl`、`hand_pose_aligned.jsonl` 和 `alignment_report.json`。分析时使用这两个 aligned 文件的 `unixTimeMs`；它们采样率不同，按时间插值/近邻配对，不按行号配对。

## 数据和时间约定

每个手机生成的 UUID 对应 `dataset/<sessionId>/`：

| 文件 | 内容 |
| --- | --- |
| `capture.json` | 预定 900 秒窗口、冻结的 iPhone epoch、采集状态和 Quest 结束信息；`finished` 指 PC 控制窗口结束，不代表 Watch 上传已完成 |
| `clock_sync.jsonl` | 开始前及大约每 60 秒的时钟测量，包含原始 PC 往返测量、offset、RTT/估计不确定度；失败也会记录 |
| `hand_pose.jsonl` | 每 Unity rendered frame 在 LateUpdate 读取一次骨骼，含左右手、bone ID/name、世界坐标 position(m)、quaternion(xyzw)、tracking/confidence、原始 Quest 时间和初始对齐时间 |
| `imudata.jsonl` | 原有 IMU 内容，额外保留 CoreMotion `deviceMonotonicMs`、`rawWatchUnixTimeMs`、`watchMinusPhoneMs`、`clockSyncRttMs` |
| `timestamp.json` | 原有 iPhone 录制 metadata |
| `pose_status.jsonl` | Quest 本地备份路径、帧数、网络队列丢帧数及停止原因 |
| `alignment_report.json` | 样本数、序号缺口、tracking 无效数量、最大采样间隔、首尾覆盖缺口及校时覆盖间隔 |

Watch 请求频率仍为 100 Hz。Quest 频率跟随 Unity 实际帧率，不假定为 100 Hz。手丢失时输出 `trackingValid=false` 和空 bones，避免把旧骨骼当成当前有效位置。

统一时间轴采用 **iPhone 在准备时冻结的 Unix epoch + iPhone 单调时间**：

```text
phoneEpochOffsetMs = phone Date.now - phone systemUptime
deviceMinusPhoneMs = device monotonic - phone monotonic
alignedUnixTimeMs = deviceMonotonicMs - deviceMinusPhoneMs + phoneEpochOffsetMs
```

PC 不要求系统时间已与手机同步。它使用自己的 monotonic clock 分别与 Phone、Quest 做四时间戳往返测量，再相减得到 Quest–Phone offset。Watch 与 Phone 直接通过 WatchConnectivity 测量。每组收集 8 次有效回复，采用最小 RTT 的样本。Watch 最多尝试 24 次、总预算约 25 秒；PC 对每台设备最多尝试 16 次、总预算约 20 秒。超时可重试，收齐 8 次有效回复后结束测量，不再按 RTT 大小拒绝启动。手机在回复回调入口记录接收时刻，不把切回 UI 主线程的等待时间计入 RTT：

```text
offset = ((t1 - t0) + (t2 - t3)) / 2
RTT = (t3 - t0) - (t2 - t1)
```

已移除 Watch–Phone 原来的 100 ms 和 PC–Phone/Quest 原来的 50 ms RTT 启动门限，Watch 的启动命令校验及 PC 的初始/周期校时接收也不再限制 RTT 上限。仍拒绝非有限数值、负 RTT 和无效时间交换，并保留请求超时、总校时预算与启动截止时间。RTT 和偏移继续显示/保存；Quest 误差估计为两个 RTT 之和的一半，Watch 的 RTT/2 也应计入跨设备误差。

手机的 Clock sync 一行保留实际最小 RTT 和回复次数。此更改需要重新编译安装 iPhone **和 Watch** app，并重启 PC receiver；旧 Watch 仍可能拒绝 RTT 大于 100 ms 的启动命令。Quest 代码无需因此更改。PC 超时会显示 iPhone/Quest 名称和失败阶段，并写入 `clock_sync.jsonl`。手机总体启动超时为 75 秒；已经拿到预定开始时间后仍按原来的开始截止时间检查，不会迟到后偷偷开始。

记录期间不跳变初始时间映射。每分钟追加校时点；`align_capture.py` 在设备单调时间坐标中线性插值 offset，修正长期漂移。校时点范围之外使用最近 offset。如果手机被挂起/校时失败，可能只有初始或稀疏校时；查看 `calibrationCount` 和 `maximumCalibrationGapSeconds`，不要假定漂移已充分校正。

CoreMotion 时间使用实际 sensor sample timestamp，不以回调到达时刻作 IMU 时间；Apple 将此字段定义为自设备启动以来的秒数：https://developer.apple.com/documentation/coremotion/cmlogitem/timestamp 。Quest 时间标记的是 **Unity 读取骨骼的 observation time**，不是相机曝光或底层 tracking sample 的硬件时间；SDK tracking/prediction 延迟没有被网络校时消除。正式实验仍需真机测量跨模态残余延迟。

这次只统一时间。Watch 保持原有 FLU/单位转换，Quest 保持 Unity world 坐标；两者不是已标定的共同空间坐标。Quest recenter 会改变世界坐标，需要实验时避免或另行处理。

## 查看校正后的 hand pose 与 IMU

在 `align_capture.py` 完成后，从 `pc_receiver` 目录运行：

```bash
python3 visualize_aligned_capture.py "../IMUPoseDC/Jiawei/<sessionId>"
```

在 Chrome / Edge 中打开 session 目录生成的 `aligned_capture_viewer.html`。脚本只需 Python 3 标准库；HTML 包含完整数据，打开后无需网络或本地服务器。可以通过 `--output /path/to/viewer.html` 指定输出位置。

- 拖动完整时间轴或点击波形定位；支持播放、变速、按秒跳转，以及 2–60 秒局部窗口。页面自动从活动较多的一个区间开始显示。
- 左侧是可旋转和缩放的双手 3D 骨架，支持跟随双手、腕部居中分开展示、固定世界视角；空格播放/暂停，左右方向键逐 pose 帧查看。
- 右侧显示原始数值的三轴 `userAcceleration`（m/s²）、`rotationRate`（rad/s），以及由 pose 关节位置计算的拇指–食指尖距离（cm）。IMU 轴与 pose 的世界轴没有做空间标定。
- 两路分别使用 aligned 文件的 `unixTimeMs`，不做跨流重采样、不额外平移时间，也不写出匹配样本。骨架显示距游标最近的实际 pose 帧并标出时间差；超过 30 ms 没有邻近帧时留空。无效 tracking 不沿用旧骨架。
- 保留全部 IMU 样本、pose 帧和有效手的 26 个关节位置；数据打包为 float32，900 秒内相对时间量化误差最多约 0.031 ms。主波形显示窗口内全部样本；全程概览为 0.5 秒内的最大角速度模长。骨架不显示关节朝向。
- 骨架连线按 `skeletonType=XRHandLeft/XRHandRight` 和数字 `id`，遵循 [Meta OpenXR 26 关节定义](https://developers.meta.com/horizon/documentation/unity/unity-handtracking-interactions/)，不使用可能混有其他骨架枚举别名的 `name`。

## 中断和恢复

- Quest 始终在 `Application.persistentDataPath/synchronized_capture/<sessionId>/hand_pose_raw.jsonl` 另存全量数据，每 60 帧 flush。主线程每帧读取，后台 TCP 发送；发送队列最多约 300 个 pose 待发条目，超限只丢网络副本并计数。本地磁盘写失败会停止并报告。
- PC 连接断开会将 session 标为 `interrupted`；重新连接不会自动续接/重新开始旧 session。Watch 与离线 Quest 保留已设定的本地结束期限。
- 手机手动 Stop 或失败时尽力发送停止指令。Watch 暂不可达时可能继续到原定结束时间；保留这段 partial 数据，并按 `capture.json` / `timestamp.json` 判断有效窗口。
- 可以用 ADB 从 Quest 的上述持久化目录取回备份（实际路径见 `pose_status.jsonl`），再运行：

  ```powershell
  python align_capture.py "dataset\<sessionId>" --pose-source "<downloaded hand_pose_raw.jsonl>"
  ```

- 脚本只生成 aligned 派生文件，不覆盖原始文件。开始前/结束后的样本被本地采集窗口过滤为 `[start, end)`；漂移校正后的边界可能有小幅偏移。提前停止的 session 不应按完整 900 秒样本解释。
- PC 异常退出、设备进程被杀或磁盘故障可能留下不完整文件；自动停止不能等同于所有样本均已成功落盘。用报告检查最大采样间隔及首尾覆盖，不仅检查 sampleIndex。

## 修改与验证

- iPhone: `PhoneRelayModel.swift`、`PhoneRootView.swift`。
- 共享 schema / Watch: `IMUModels.swift`、`WatchMotionModel.swift`。
- Quest: `PinchEventToServer.cs` + `PinchEventToServer.Pose.cs`。
- PC: `server.py` + `synchronized_capture.py` + `align_capture.py`。
- 接收器测试：`python -m unittest -v test_synchronized_capture`。覆盖 offset/RTT、漂移、校时/arm/commit、无 Quest、未 commit 超时、900 秒模拟截止、窗口边界、提前停止、断线、来源校验及旧模式回归。
- Quest 已在本机 Unity 6000.0.52f1 batchmode 编译通过；iPhone/Watch 只有 Swift 语法解析检查，本 Windows 环境不能进行 Xcode 编译。
- 尚未进行 iPhone + Watch + Quest 真机 15 分钟实验。部署后请先检查左右手是否都有有效 bones、Watch 文件能否完整上传、网络/时钟报告及实际采样率，再用于正式采集。

### 逐关节可靠性与带权二维拟合（可选离线流程）

完成普通 WiLoR＋Anipose＋骨长/时间约束后，可复用同一段原始 RGB：

```bash
cd /home/spice-lab/Desktop/pc_receiver
python3 process_weighted_handpose.py "dataset_multicamera/<sessionId>"
```

使用 `reliabilityPython`（未设置则使用 `regularizationPython`），当前现有
`HandPoseComparison/poem_env/bin/python` 可运行。此流程不会更改自动采集模式，
也不会覆盖 `multicamera/handpose` 的基线输出。新文件在
`multicamera/weighted_handpose`，回放在 `visualization/weighted/viewer.html`，
同步对比在 `visualization/weighted/comparison.html`。

也可从原入口运行 `python3 process_multicamera.py "dataset_multicamera/<sessionId>" --stage weighted`。
若希望以后每次采集自动运行，在相机配置中显式设置 `weightedHandpose: true`；
它会在常规三角化和骨长/时间约束后运行本流程。当前没有修改自动采集配置。
4 次裁剪扰动会增加离线处理时间。

- 每帧每视角保留选定右手的检测框/整手检测分数、4 次裁剪扰动的 21 个二维关节、
  MANO 网格和参数、虚拟相机、源帧和手机时间戳。候选框与选手理由在 `cache/*.json`。
  手关联优先使用其他三路一致的投影位置，否则沿用原选手结果作为位置引导。
- 用归一化裁剪扰动、固定解剖皮肤区域的网格自遮挡、整手检测分数生成图像权重。
  在其他三路都一致时，再用留一视角预测修正权重；未知几何参考不当作遮挡。
  网格投影使用 WiLoR 自己的虚拟相机，不能直接套用标定相机的平移。
- 权重在优化前固定。最小化带权、鲁棒的**标定相机二维重投影**，并保留骨长、
  实际时间间隔加速度约束和较弱的原三维位置约束。缺少两路支持时保留更强深度先验；
  缺失基线关节仍留空，不把推断补点伪装成测量。
- 所有权重都是启发式可靠性，**不是校准后的逐关节置信度或可见性概率**。
  稳定但错误的预测、外物遮挡、选错手和自由曝光时间差仍可能导致错误。
  原始二维点本身也是 WiLoR 三维预测的投影，不是独立的二维真值。
- 页面中的二维点由红（低权重）到绿（高权重）；青色为优化后三维，黄色关节/圆圈
  表示少于两路支持。右侧可选择关节，查看四路的权重、扰动像素和网格可见率。
- `review.html` 提供固定时间原图/权重点对照和人工标注导出；
  `reliability_proxy_audit.json` 按**仅图像权重**分组评估留一视角一致性，避免拿包含
  同一几何误差的最终权重作循环验证。这仍不是独立真值精度验证。
- 推理每 64 帧保存一次缓存，输入校验通过后可继续。若标定、配对或原二维结果变动，
  缓存会拒绝复用；更换模型/修改算法或视频时，应使用新的输出目录或先归档旧结果。
  初次基线副本固定在 `baseline_pose_3d.npz`，重新选择基线也需先归档输出。

数值测试（无需相机）：

```bash
/home/spice-lab/Desktop/HandPoseComparison/poem_env/bin/python -m unittest test_hand_reliability test_regularize_handpose
```

# LeRobot 项目全解析 —— 初学者友好指南

> **对应论文**: [LeRobot: An Open-Source Library for End-to-End Robot Learning](https://arxiv.org/abs/2602.22818) (ICLR 2026)
> **代码版本**: v0.6.2 · Python 3.12+ · PyTorch
> **本文目标**: 帮助初学者 ① 理解机器人学习的核心原理 ② 看懂代码架构 ③ 最快速度跑通第一次训练

---

## 目录

- [1. LeRobot 是什么?](#1-lerobot-是什么)
- [2. 十分钟搞懂核心原理(论文精要)](#2-十分钟搞懂核心原理论文精要)
- [3. 必须先懂的 6 个概念](#3-必须先懂的-6-个概念)
- [4. 整体架构与数据流](#4-整体架构与数据流)
- [5. 仓库目录导览](#5-仓库目录导览)
- [6. 核心模块逐一详解](#6-核心模块逐一详解)
- [7. 代码走读:一次训练的完整旅程](#7-代码走读一次训练的完整旅程)
- [8. 快速上手:三条路径](#8-快速上手三条路径)
- [9. 训练实践指南(避坑手册)](#9-训练实践指南避坑手册)
- [10. 评估你的策略](#10-评估你的策略)
- [11. 七天学习路线图](#11-七天学习路线图)
- [附录A 术语表](#附录a-术语表)
- [附录B 论文 ↔ 代码对应表](#附录b-论文--代码对应表)
- [附录C 常见问题 FAQ](#附录c-常见问题-faq)

---

## 1. LeRobot 是什么?

**一句话**:LeRobot 是 Hugging Face 出品的**开源机器人学习库**,把"采集数据 → 存储数据集 → 训练策略模型 → 评估 → 部署到真机"这条完整链路,全部统一在一套 Python API 和 CLI 工具里。

**它解决什么问题?**(论文 §1)

机器人学习领域长期被"碎片化"困扰:

| 痛点 | LeRobot 的答案 |
|---|---|
| 每种机器人都有自己的控制接口,互不兼容 | 统一的 `Robot` 抽象类,一套 API 控制所有硬件 |
| 数据集格式五花八门,无法复用 | 标准化 `LeRobotDataset` 格式(Parquet + MP4),托管在 HF Hub |
| SOTA 算法实现散落各处,难以复现 | 纯 PyTorch 实现的算法库(ACT、Diffusion、π0、SmolVLA…) |
| 推理部署需要自己搭通信 | 异步推理栈(policy server + robot client) |

**社区规模**(截至 2025.09,论文数据):HF Hub 上已有 **16,000+ 个** LeRobot 格式数据集、**2,200+ 位**贡献者。其中 50%+ 的数据集来自 SO-100/SO-101 这类百美元级的开源机械臂——这正是 LeRobot "降低门槛"理念的体现。

---

## 2. 十分钟搞懂核心原理(论文精要)

### 2.1 显式模型 vs 隐式模型(论文 §2.1)

这是理解整个项目的**思想基石**:

```
传统机器人(显式模型)                    机器人学习(隐式模型)
┌─────────────────────────┐            ┌─────────────────────────┐
│ 感知模块(手工设计)      │            │                         │
│   ↓                     │            │   神经网络策略 π          │
│ 规划模块(运动学方程)    │    ──→     │   观测 o ──→ 动作 a      │
│   ↓                     │            │   (端到端,一体的)       │
│ 控制模块(PID等)        │            │                         │
└─────────────────────────┘            └─────────────────────────┘
专家手工推导,误差逐级累积              从数据中学习,数据越多性能越好
只适用于结构化环境(工厂)              适应非结构化环境(家庭)
```

**LeRobot 的信仰**:性能随数据和算力扩展的方法才能赢。所以它把重心放在"让数据采集更容易、让训练流程更标准、让模型可复用",而不是手工调参。

### 2.2 模仿学习(Imitation Learning)—— 本库最常用的范式

```
人类专家 ──遥操作──→ 机器人 ──记录──→ 数据集(观测 o, 动作 a)
                                              │
                                        训练策略 π
                                               ↓
                    新观测 o ──→ π(o) ──→ 动作 a ──→ 机器人执行
```

- **观测 (observation)**:机器人看到的世界 —— 相机图像、关节角度(本体感知)
- **动作 (action)**:机器人该做什么 —— 各关节的目标角度
- **策略 (policy)**:就是那个神经网络,`π(observation) → action`
- 训练目标很简单:**让策略模仿人类专家**。50 条高质量演示就足以训练出一个能用的 ACT 策略。

### 2.3 论文的四大支柱(§3)与代码的对应

| 论文章节 | 特性 | 代码位置 |
|---|---|---|
| §3.1 Accessible Real-world Robots | 统一硬件 API | `src/lerobot/robots/`, `motors/`, `cameras/`, `teleoperators/` |
| §3.2 Datasets | 标准数据格式 + 流式加载 | `src/lerobot/datasets/` |
| §3.3 Models | SOTA 算法纯 PyTorch 实现 | `src/lerobot/policies/` |
| §3.4 Inference | 异步推理(策略与控制解耦) | `src/lerobot/async_inference/`, `rollout/` |

---

## 3. 必须先懂的 6 个概念

1. **Policy(策略)**:核心神经网络。输入观测(图像+状态),输出动作。所有策略继承 `PreTrainedPolicy`(`nn.Module` + `HubMixin`),所以都能像 HF 模型一样 `save_pretrained()` / `from_pretrained()` 上传下载。

2. **LeRobotDataset**:一种自包含的数据集格式 —— Parquet 存状态/动作数值,MP4 存视频帧,`meta/` 目录存元信息(任务描述、数据统计、episode 索引)。

3. **Episode(回合)**:一次完整的任务演示,比如"抓起方块放到碗里"的 30 秒录制。一个数据集 = 几十到几千个 episode。

4. **Processor(处理器)**:数据在"数据集 → 策略 → 机器人"之间流动时的变换链(归一化、图像缩放、动作分块…),像 HF Transformers 的 tokenizer 一样可组合、可保存。

5. **Env(环境)**:仿真环境(LIBERO、MetaWorld 等),遵循 Gymnasium 接口,用于在仿真里训练和评估。

6. **Teleoperation(遥操作)**:人通过主臂/手柄/手机控制从臂,同时录制数据 —— 这是采集训练数据的主要方式。

---

## 4. 整体架构与数据流

### 4.1 全链路数据流

```
┌─────────────────────────── 数据采集(真机) ───────────────────────────┐
│                                                                      │
│  人 ──主臂(teleoperators)──从臂(robots/motors)                      │
│         │                            │                               │
│         └──────── cameras ───────────┤                                │
│                                      ↓                               │
│                          lerobot-record 录制                        │
│                                      ↓                               │
│                    LeRobotDataset(Parquet + MP4)──→ 推送到 HF Hub   │
└──────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────── 离线训练 ──────────────────────────────────┐
│                                                                      │
│  lerobot-train                                                       │
│    ├─ LeRobotDataset + EpisodeAwareSampler(按回合采样)              │
│    ├─ preprocessor(归一化/图像变换)                                 │
│    ├─ policy(ACT / Diffusion / SmolVLA / π0 …)                      │
│    ├─ optimizer + scheduler(optim/)                                 │
│    ├─ accelerator(分布式/FSDP2/混合精度,distributed/)              │
│    └─ checkpoint 定期保存 → 可推送 HF Hub                            │
└──────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────── 评估与部署 ────────────────────────────────┐
│                                                                      │
│  仿真评估: lerobot-eval + envs/(LIBERO/MetaWorld/…)                 │
│  真机评估: lerobot-record --policy.path=…                            │
│  异步部署: async_inference/(policy_server ←grpc→ robot_client)      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 训练时的单步数据流(最重要的一张图)

```
dataset[i] ──→ collate 成 batch ──→ preprocessor ──→ policy.forward ──→ loss
   │                                   │                (含 backward)
   │  原始 uint8 图像、                │  float 图像、
   │  未归一化的 state/action          │  归一化张量          ↓ optimizer.step()
   │                                                        权重更新
   └── 推理时反向:obs → preprocessor → policy.select_action → postprocessor → 机器人动作
```

**记住这个对称结构**:训练时 `preprocessor → forward`,推理时 `preprocessor → select_action → postprocessor`。postprocessor 把模型输出的归一化动作**反归一化**回真实物理量。

---

## 5. 仓库目录导览

```
lerobot/
├── src/lerobot/          # ★ 核心库(下节详解)
├── tests/                # pytest 测试套件(tests/outputs/ 存 E2E 产物)
├── docs/source/          # 官方文档(.mdx):每个 policy、每种硬件都有专页
├── examples/             # 按使用场景组织的教程脚本
├── benchmarks/           # 性能基准测试脚本
├── docker/               # 用户镜像 + 各仿真 benchmark 的专用镜像
├── scripts/              # 顶层辅助脚本
├── pyproject.toml        # 依赖、CLI 入口、工具配置的唯一真相源
├── Makefile              # E2E 测试目标(make test-end-to-end)
├── AGENT_GUIDE.md        # ★ 强烈推荐:面向用户的实操速查手册
└── AGENTS.md / CLAUDE.md # 面向 AI 助手的开发上下文
```

---

## 6. 核心模块逐一详解

所有模块都在 `src/lerobot/` 下。按"重要性 × 初学者接触顺序"排列:

### 6.1 `policies/` —— 策略模型(★ 最核心)

**每个策略一个子目录**,结构高度统一,以 ACT 为例:

```
policies/act/
├── configuration_act.py   # ACTConfig:超参数(dataclass,draccus 注册)
├── modeling_act.py        # ACTPolicy:网络结构与训练/推理逻辑
├── processor_act.py       # 该策略专属的数据处理器
└── README.md              # 原论文链接与说明
```

**策略家族一览**(按上手难度排序):

| 类别 | 策略 | 参数量 | 显存(训练) | 适合谁 |
|---|---|---|---|---|
| 模仿学习 | **ACT** | 52M | ~1 GB | ★ 初学者首选,单任务,笔记本可训 |
| 模仿学习 | Diffusion Policy | 263M | ~5 GB | 动作分布多模态的任务 |
| 模仿学习 | VQ-BeT | — | 中 | 离散 token 化动作 |
| VLA | **SmolVLA** | 450M | ~4 GB | 语言条件、多任务,中等 GPU |
| VLA | π0 / π0.5 / GR00T / X-VLA / Wall-X / EVO1… | 3.5B+ | 15-16 GB+ | 大 GPU,进阶 |
| 世界模型 | VLA-JEPA / LingBot-VA / FastWAM | — | 大 | 研究:预测未来观测 |
| 奖励模型 | SARM / TOPReward / Robometer | — | — | 为 RL/评估提供奖励信号 |
| 强化学习 | HIL-SERL / TD-MPC | — | — | 在线 RL(见 `rl/`) |

**关键基类** `policies/pretrained.py` 中的 `PreTrainedPolicy`:

```python
class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    # 子类必须定义:
    config_class = ACTConfig        # 配置类
    name = "act"                    # 注册名(CLI 用 --policy.type=act)

    # 子类必须实现的核心方法:
    def forward(self, batch): ...           # 训练:算 loss
    def select_action(self, obs): ...       # 推理:出动作
    def reset(self): ...                    # 每个新 episode 前清空内部队列
```

**工厂模式** `policies/factory.py`:`make_policy(cfg, ds_meta)` 根据 `--policy.type=act` 这样的字符串**懒加载**对应类(重依赖只在用到时才 import)。新增策略只需用 `@register_subclass("my_policy")` 装饰配置类即可被 CLI 发现。

### 6.2 `datasets/` —— 数据集(★ 第二核心)

| 文件 | 职责 |
|---|---|
| `lerobot_dataset.py` | `LeRobotDataset` 主类:加载/创建数据集,处理 `delta_timestamps`(取历史帧),视频解码 |
| `streaming_dataset.py` | `StreamingLeRobotDataset`:不下载整个数据集,边流式边训练(论文 §C.1) |
| `dataset_writer.py` / `dataset_reader.py` | 录制时写数据 / 训练时读数据的底层引擎 |
| `dataset_metadata.py` | 元信息:fps、特征形状、任务表、**stats(归一化统计量)** |
| `sampler.py` | `EpisodeAwareSampler`:保证同一 batch 来自同一 episode(时序模型需要) |
| `compute_stats.py` | 计算各特征的 min/max/mean/std,存入 meta/stats.json |
| `aggregate.py` / `dataset_tools.py` | 合并、拆分、编辑数据集的工具 |
| `video_utils.py` | MP4 编解码(torchcodec / PyAV 双后端) |

**数据集在磁盘上的样子**(v3.0 格式):

```
my_dataset/
├── data/
│   └── chunk-000/
│       ├── file-000.parquet    # 每帧一行:timestamp, episode_index, 观测, 动作…
│       └── file-001.parquet
├── videos/
│   └── chunk-000/
│       └── observation.images.front/
│           ├── episode_000000.mp4
│           └── episode_000001.mp4
└── meta/
    ├── info.json               # fps、特征定义、总帧数…
    ├── stats.json              # 归一化用的统计量 ★
    ├── tasks.jsonl             # 任务文本描述(语言条件策略用)
    └── episodes/
        └── chunk-000/*.parquet # 每个 episode 的长度、任务 id…
```

**为什么视频存 MP4 而不是图片?** 存储 省 10-50 倍,且 LeRobot 会按时间戳精确抽帧,训练时和 Parquet 里的状态数据自动对齐。

### 6.3 `processor/` —— 数据处理管线(新手最容易忽略但必须懂)

类比:HF Transformers 里 tokenizer 之于 LLM,processor 之于机器人策略。

```
ProcessorStep(抽象基类,可注册)
    ↓ 组合
DataProcessorPipeline(通用管线,HubMixin 可上传 Hub)
    ↓ 特化
PolicyProcessorPipeline(策略专属:训练前处理 + 推理后处理)
```

常用步骤(`processor/` 下每个文件一个):

- `normalize_processor.py` —— 用数据集 stats 把 state/action 归一化到 [-1,1]
- `device_processor.py` —— 搬运张量到 GPU
- `delta_action_processor.py` / `relative_action_processor.py` —— 把绝对动作转成增量/相对动作
- `tokenizer_processor.py` —— VLA 模型的文本 tokenization
- `observation_processor.py` / `env_processor.py` —— 观测/环境输出的适配

**设计精髓**:preprocessor 和 postprocessor 会随 checkpoint 一起保存/加载,保证"训练时怎么变换,推理时就怎么逆变换",杜绝 train/inference skew。

### 6.4 `configs/` —— 配置系统(一切 CLI 参数的来源)

基于 **draccus**(dataclass 驱动的配置库)+ 注册表模式实现多态:

```python
# configs/train.py
@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig          # --dataset.repo_id=…
    policy: PreTrainedConfig | None # --policy.type=act --policy.chunk_size=100
    env: EnvConfig | None           # --env.type=libero
    batch_size: int = 8
    steps: int = 100_000
    save_freq: int = 20_000
    optimizer: OptimizerConfig | None
    parallelism: ParallelismConfig  # 多卡拓扑
    accelerator: AcceleratorConfig  # 混合精度/梯度累积
    wandb: WandBConfig              # 实验跟踪
    ...
```

**多态机制**:`PreTrainedConfig` 是 `ChoiceRegistry`,`@register_subclass("act")` 把 `ACTConfig` 注册为 `act` 选项。于是 `--policy.type=act` 会实例化 `ACTConfig`,**每个策略的超参数都自动变成 CLI 参数**(`--policy.xxx`)。环境、机器人、处理器同理。

**可复现性**:每次训练的完整配置存为 checkpoint 里的 `train_config.json`,`--resume=true` 或 `--config_path=` 可原样恢复。

### 6.5 `scripts/` —— CLI 入口(你每天敲的命令)

`pyproject.toml [project.scripts]` 把这些脚本注册为命令行工具:

| 命令 | 用途 | 使用频率 |
|---|---|---|
| `lerobot-train` | 训练策略 | ★★★ |
| `lerobot-eval` | 仿真环境评估 | ★★★ |
| `lerobot-record` | 遥操作录制数据集 / 真机跑策略 | ★★★ |
| `lerobot-teleoperate` | 手动遥操作(调试硬件) | ★★ |
| `lerobot-calibrate` | 校准机械臂关节 | ★★(装好臂后一次) |
| `lerobot-setup-motors` | 设置电机 ID/波特率 | ★(每臂一次) |
| `lerobot-find-port` / `lerobot-find-cameras` | 找 USB 口/相机 | ★ |
| `lerobot-replay` | 回放数据集里的 episode 到真机 | ★ |
| `lerobot-dataset-viz` / `lerobot-edit-dataset` | 可视化/编辑数据集 | ★ |
| `lerobot-rollout` / `lerobot-annotate` / `lerobot-info` | 部署 rollout / 数据标注 / 环境信息 | 按需 |

### 6.6 `robots/` + `motors/` + `cameras/` + `teleoperators/` —— 硬件抽象层

**四层抽象,全部"配置类 + 基类 + 具体实现"的模式**:

```
robots/robot.py          motors/motors_bus.py       cameras/camera.py        teleoperators/teleoperator.py
┌──────────────┐         ┌──────────────┐          ┌──────────────┐        ┌──────────────┐
│ Robot (ABC)  │         │ MotorsBus    │          │ Camera (ABC) │        │Teleoperator  │
│ connect()    │         │ (总线协议)   │          │ connect()    │        │ connect()    │
│get_observation│        │ read()/write()│         │ read()→帧    │        │ get_action() │
│ send_action()│         └──────┬───────┘          │ async_read() │        │ send_feedback│
│ calibrate()  │                │                  └──────┬───────┘        └──────┬───────┘
└──────┬───────┘    ┌───────────┴──────────┐        ┌──────┴────────┐          │
       │            │ feetech  dynamixel   │        │ opencv realsense│         │
  so_follower      │ damiao   robstride    │        │ zmq  reachy2    │    so_leader
  koch  lekiwi     └──────────────────────┘        └───────────────┘    gamepad phone
  reachy2 unitree_g1 …                                                     keyboard …
```

**统一契约**(论文 §B):任何机器人只要实现 `get_observation()` / `send_action()`,就能直接用 LeRobot 的录制、训练、可视化全套工具。第三方硬件通过 `lerobot_robot_<name>` 插件包自动被发现。

**校准(calibration)**:电机原始读数 → 物理角度的映射(偏移+范围),存 `~/.cache/huggingface/lerobot/calibration/robots/<type>/<id>.json`。**没校准的臂不能安全运行。**

### 6.7 `envs/` —— 仿真环境

- `configs.py`:`EnvConfig` 基类(同样是 ChoiceRegistry),`factory.py` 的 `make_env()` 按类型创建
- 内置 benchmark:**LIBERO**、**MetaWorld**、RoboCasa、RoboMME、RoboTwin、VLABench(每个有专属 Dockerfile,见 `docker/Dockerfile.benchmark.*`)
- 仿真环境遵循 **Gymnasium** 接口(`reset()` / `step()`),与真机共用同一套 processor 适配层 —— **代码从仿真换到真机几乎不用改**

### 6.8 `optim/` —— 优化器与调度器

`factory.py` 的 `make_optimizer_and_scheduler(cfg, policy)`:
- 策略可通过 `get_optim_params()` 声明自定义参数组(如 ACT 对 backbone 用不同学习率)
- 支持 AdamW/SGD + cosine/warmup 等调度器,全部由 config 驱动

### 6.9 `distributed/` —— 多卡训练

- `make_accelerator()`:构建 HF Accelerate 实例(混合精度 bf16、梯度累积、FSDP2 分片)
- `ParallelDims`:声明并行拓扑(`--parallelism.dp_shard=8` 等)
- checkpoint 支持 safetensors(兼容)与 DCP 分片(快速恢复)双格式
- **初学者单卡完全不用碰这个模块**,但要知道 `--accelerator.mixed_precision=bf16` 能省一半显存

### 6.10 `async_inference/` + `rollout/` —— 部署与推理(论文 §3.4)

**异步推理**解决"策略推理慢 vs 控制频率高"的矛盾:

```
┌─ policy_server.py(可跑在强 GPU 机器)─┐        ┌─ robot_client.py(跑在机器人上)─┐
│  收观测 → select_action → 动作块      │ ←gRPC→ │  30Hz 控制循环 + 动作插值        │
└──────────────────────────────────────┘        └────────────────────────────────┘
```

策略一次预测**一整块动作(action chunk)**,客户端逐帧执行并插值,推理与控制**并行**。`rollout/` 是更上层的 rollout 控制器(环形缓冲、推理策略、机器人包装)。

### 6.11 `rl/` + `rewards/` —— 强化学习与奖励模型

- `rl/`:HIL-SERL(人类干预的样本高效 RL)的 actor-learner 架构、replay buffer、训练入口 `train_rl.py`
- `rewards/`:奖励模型(SARM、TOPReward、Robometer),可离线预计算奖励或在线打分
- **建议**:先掌握模仿学习,再碰 RL

### 6.12 其余支撑模块(速览)

| 模块 | 职责 |
|---|---|
| `common/` | 训练工具(checkpoint 保存/恢复、WandB 封装、控制循环工具) |
| `utils/` | 大杂烩工具箱:日志、随机种子、collate、导入守卫、可视化(rerun/foxglove) |
| `transforms/` | 图像增强变换(随机裁剪/颜色抖动,训练时用) |
| `jobs/` | 提交训练到 Hugging Face Jobs 云端(`--job.target=a10g-small`) |
| `model/` | 运动学工具(逆运动学等) |
| `annotations/` | 数据标注管线(SARM 相关) |
| `templates/` | HF model card 模板 |

---

## 7. 代码走读:一次训练的完整旅程

以这条命令为例,追踪它经过的每一站:

```bash
lerobot-train --policy.type=act --dataset.repo_id=lerobot/svla_so101_pickplace --steps=30000
```

**第 1 站:入口** `scripts/lerobot_train.py::main()`
→ draccus 把 CLI 参数解析成 `TrainPipelineConfig` 并 `validate()`(所有快速失败检查在此)。

**第 2 站:引擎** `distributed/factory.py::make_accelerator()`
→ 构建 Accelerator(单卡时近乎透明),设置随机种子。

**第 3 站:数据** `datasets/factory.py::make_train_eval_datasets()`
→ 主进程下载 Hub 数据集,构建 `LeRobotDataset` + `EpisodeAwareSampler` + DataLoader(`num_workers=4` 后台解码视频)。

**第 4 站:策略** `policies/factory.py::make_policy()`
→ 按 `act` 找到 `ACTPolicy`,用数据集元信息自动推断输入输出特征形状(图像通道数、动作维度),注入 stats。

**第 5 站:处理器** `make_pre_post_processors()`
→ 组装归一化/设备搬运/图像变换管线,与策略绑定。

**第 6 站:优化器** `optim/factory.py::make_optimizer_and_scheduler()`
→ ACT 声明了分组学习率(backbone 更小)。

**第 7 站:训练循环**(核心,约 60 行,`lerobot_train.py` L735 起):

```python
for _ in range(step, cfg.steps):
    batch = next(dl_iter)                                  # ① 取一个 batch
    batch = preprocessor(batch)                            # ② 归一化/图像变换
    train_tracker, _ = update_policy(                      # ③ 前向+反向+优化器步进
        train_tracker, policy, batch, optimizer, ...)
    step += 1
    if step % cfg.log_freq == 0:        ...                # ④ 日志/WandB
    if step % cfg.eval_steps == 0:      ...                # ⑤ 验证集 loss
    if should_save_checkpoint(step):    ...                # ⑥ 存 checkpoint
    if cfg.env and step % cfg.env_eval_freq == 0:          # ⑦ 仿真环境评估(可选)
        eval_policy_all(...)
```

`update_policy()` 内部:`policy.train()` → `accelerator.accumulate()`(梯度累积)→ `policy(batch)` 算 loss → `loss.backward()` → 梯度裁剪 → `optimizer.step()`。

**第 8 站:产出** `outputs/train/<job_name>/`

```
outputs/train/act_my_task/
├── checkpoints/
│   └── 030000/
│       ├── pretrained_model/        # ★ config.json + model.safetensors + processors
│       │   ├── config.json
│       │   ├── model.safetensors
│       │   ├── preprocessor/…
│       │   └── postprocessor/…
│       └── training_state/          # 优化器/调度器状态(--resume 用)
└── train_config.json                # 完整训练配置(可复现)
```

---

## 8. 快速上手:三条路径

### 路径 A:没有机器人,先用公开数据集训练(零硬件,30 分钟)

```bash
# 1. 安装(推荐 uv;pip 也行)
uv sync --locked --extra training        # 或: pip install 'lerobot[training]'

# 2. 训练 ACT(最轻量,CPU/笔记本 GPU 都能跑)
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --batch_size=8 \
  --steps=30000 \
  --output_dir=outputs/train/act_so101 \
  --job_name=act_so101 \
  --wandb.enable=true                    # 可选:免费实验跟踪

# 3. (可选)在仿真里评估
lerobot-eval --policy.path=outputs/train/act_so101/checkpoints/030000/pretrained_model \
  --env.type=<对应环境> --eval.n_episodes=50
```

**没有 GPU?** 一行上云:`lerobot-train ... --job.target=a10g-small`(用 `hf jobs hardware` 查看可用机型)。

### 路径 B:SO-101 全流程(有硬件,一个周末)

完整命令见 [`AGENT_GUIDE.md`](./AGENT_GUIDE.md) §4,流程概览:

```
① uv sync --locked --extra feetech          # 装依赖
② lerobot-find-port                          # 找 USB 口
③ lerobot-setup-motors                       # 电机 ID(每臂一次)
④ lerobot-calibrate                          # 关节校准(每臂一次)
⑤ lerobot-teleoperate                        # 手动试玩 ✓
⑥ lerobot-record --dataset.num_episodes=50   # 录 50 个 episode
⑦ (在线可视化检查数据质量)                    # huggingface.co/spaces/lerobot/visualize_dataset
⑧ lerobot-train --policy.type=act …          # 训练
⑨ lerobot-record --policy.path=…             # 真机评估 10 次,统计成功率
```

**数据质量黄金法则**(AGENT_GUIDE §5,比任何调参都重要):
- 先录 50 个**受限版本**任务的 episode(固定物体位置、固定相机、同一操作者)
- 训练一版 ACT → 看哪里失败 → 针对性补录 10-20 条 → 再扩展多样性
- **"仅凭相机画面,你自己能完成这个任务吗?"** —— 不能就先改相机位/灯光,别急着加数据

### 路径 C:理解代码(转岗/科研/贡献者)

按此顺序读(由浅入深):

1. `AGENT_GUIDE.md` —— 全局直觉
2. `policies/act/` 三个文件 —— 最小完整策略样本(configuration → modeling → processor)
3. `datasets/lerobot_dataset.py` 的 `__getitem__` —— 数据如何被取出
4. `processor/pipeline.py` 的 `ProcessorStep` —— 变换如何组合
5. `scripts/lerobot_train.py` 的 `train()` —— 训练主循环
6. `configs/train.py` —— 参数如何流动
7. `robots/robot.py` —— 硬件契约

---

## 9. 训练实践指南(避坑手册)

### 9.1 按显存选策略(AGENT_GUIDE §6 实测数据)

| 你的 GPU | 推荐 | 说明 |
|---|---|---|
| < 8 GB / M 系 Mac | **ACT** | 单次更新 84ms、峰值 0.94GB,最稳 |
| 12-16 GB | **SmolVLA** 或 ACT 大 batch | SmolVLA 解冻视觉编码器收益大 |
| 24 GB (3090/4090) | 任意;多任务选 SmolVLA | 可试 π0/π0.5(小 batch+梯度累积) |
| 80 GB (A100/H100) | 任意,大 batch | X-VLA / Wall-X 舒适 |
| 仅 CPU | 别本地训 | 用 Colab / `--job.target` 上云 |

### 9.1b NVIDIA DGX Spark 专项评估

**硬件规格**:GB10 Grace Blackwell Superchip · 128GB LPDDR5x 统一内存 · 273 GB/s 带宽 · 1 PFLOP (FP4) · ARM64 (aarch64) · 20 核 Grace CPU。

**两个关键特性对训练的影响**:

- ✅ **128GB 统一内存**:LeRobot 最大模型训练峰值仅 ~16GB,全部模型都装得下,还能开远超 RTX 4090 的大 batch。
- ⚠️ **273 GB/s 带宽**:仅为 RTX 4090(1008 GB/s)的 1/4 —— 训练是带宽敏感负载,单步速度预计慢 2-4 倍。
- ⚠️ **ARM64 平台**:torchcodec 需 torch≥2.11 的 aarch64 wheel,否则自动回退 PyAV 解码(功能等价)。

**各模型可训性**:

| 模型 | 训练峰值 | 结论 | 预期体验 |
|---|---|---|---|
| `act` | 0.9GB | ✅ 轻松 | 30k 步约 1.5-3h(4090 约 40min),流畅 |
| `diffusion` | 4.9GB | ✅ 轻松 | 良好 |
| `smolvla` | 3.9GB+ | ✅ **推荐** | 解冻视觉编码器 + 大 batch,Spark 甜点 |
| `vqbet` / `multi_task_dit` / `tdmpc` | 中 | ✅ 可训 | RL 环境交互偏 CPU,20 核 Grace 够用 |
| `pi0` / `pi0_fast` / `pi05` | 15-16GB | ✅ 内存充裕 | 微调可行,大 batch 弥补速度;全量训练需耐心 |
| `xvla` / `wall_x` / `evo1` / `eo1` / `molmoact2` | 15-16GB+ | ✅ 装得下 | 建议 PEFT/LoRA 微调 |
| `vla_jepa` / `lingbot_va` | 大 | ⚠️ 可训但慢 | 世界模型训练计算密集 |
| `fastwam` | 视频DiT | ❌ **不推荐** | 视频生成训练对算力/带宽要求远超 Spark 定位 |

**Spark 上的推荐配置**:

```bash
# 必开 bf16(Blackwell 原生支持)
lerobot-train ... --accelerator.mixed_precision=bf16

# 大模型微调加 PEFT,大幅省时省内存
lerobot-train ... --policy.type=pi0 --peft.type=lora
```

**一句话结论**:除 `fastwam` 外所有 20+ 模型都能训。最佳定位是 `act`/`smolvla`/`diffusion` 全量训练 + π0 级大模型微调;128GB 大内存 + `async_inference/` 也使 Spark 非常适合本地部署推理(policy server)。

### 9.1c 多台 DGX Spark 级联训练 LingBot-VA 评估

**级联能力**:NVIDIA 官方规格 —— ConnectX-7 网络支持**最多 4 台** DGX Spark 级联(200 Gbps ≈ 25 GB/s 节点间带宽)。

**LingBot-VA 模型构成**(源码 `policies/lingbot_va/` 实测):可训练 DiT 主干 **~5B 参数**(Wan2.2 双流 transformer,30 层)+ 冻结权重 ~20GB(Wan2.2 VAE + UMT5-XXL 文本编码器,统一内存同一物理池)。

**全量微调内存账**(AdamW + bf16 混合精度):

| 项目 | 占用 |
|---|---|
| fp32 权重(Accelerate bf16 模式保留 fp32 主权重) | 20 GB |
| fp32 梯度 | 20 GB |
| AdamW 动量 (m+v) | 40 GB |
| 冻结 VAE + UMT5 | ~20 GB |
| 激活值(30 层视频 DiT,长序列) | ~10-20 GB |
| **合计** | **~110-130 GB** |

**结论**:

| 配置 | 总内存 | 判定 |
|---|---|---|
| 1 台 Spark | 128 GB | ❌ 贴着上限,实际 batch 必 OOM |
| **2 台 Spark(FSDP2 分片)** | 256 GB | ✅ **最低可行配置**,每节点 ~55GB,从容 |
| 4 台 Spark(官方上限) | 512 GB | ✅ 富余,可加大 batch/关闭激活检查点 |

**答案:全量微调至少 2 台,推荐 2-4 台。**

```bash
# 双机 FSDP2 分片(LeRobot 原生支持)
torchrun --nnodes=2 --nproc-per-node=1 $(which lerobot-train) \
  --policy.type=lingbot_va \
  --parallelism.dp_shard=2 \
  --accelerator.mixed_precision=bf16 ...
```

**两个现实警告**:

1. **速度**:节点间 ConnectX-7 为 200 Gbps(~25 GB/s),FSDP 每步需 all-gather 20GB 参数,仅通信就 ~1-2s/步;叠加本地 273 GB/s 带宽瓶颈,视频扩散模型训练会**非常慢**(预计比 8×A100 慢一个数量级)。
2. **更务实的选择**:单台 Spark + `--peft.type=lora` —— 可训练状态从 80GB 骤降至 <5GB,轻松装下,速度也可接受。全量微调只在确有必要时才上集群。

### 9.2 训练步数怎么定?(以 epoch 为思考单位)

模仿学习通常 **5-10 个 epoch** 收敛,不是几十万步蛮跑:

```
总帧数       = 所有 episode 帧数之和        # 50 回合 × 30s × 30fps ≈ 45,000
每epoch步数  = ceil(总帧数 / batch_size)    # 45,000 / 8 ≈ 5,625
总步数       = epoch 数 × 每epoch步数       # 5 epoch ≈ 28k 步
```

| 数据规模 | batch=8 时 5 epoch | 10 epoch |
|---|---|---|
| 50 回合(45k 帧) | ~28k 步 | ~56k 步 |
| 100 回合(90k 帧) | ~56k 步 | ~113k 步 |
| 300 回合(270k 帧) | ~169k 步 | ~338k 步 |

### 9.3 三个最常见的坑

**坑 1:短训练没调调度器。** 默认 `scheduler_decay_steps=30000` 是为长跑设计的;如果你只训 5k 步,学习率全程贴着峰值不衰减:

```bash
lerobot-train ... --steps=5000 --policy.scheduler_decay_steps=5000 --save_freq=1000
# 法则:scheduler_decay_steps ≈ steps,save_freq = 想要的评估粒度
```

**坑 2:大策略 batch 太小。** π0/π0.5 显存受限只能 batch=1-4,务必开梯度累积把**有效 batch** 提到 16+:

```bash
--accelerator.gradient_accumulation.steps=16
```

**坑 3:SmolVLA 忘了解冻视觉编码器。** 默认 `freeze_vision_encoder=true`,专用任务上解冻通常大幅提升:

```bash
--policy.type=smolvla --policy.freeze_vision_encoder=false --policy.train_expert_only=false
```

### 9.4 何时停?

- 训练 loss 平台期 → 停,评估 checkpoint
- loss 还在降且 < 10 epoch → 继续
- **记住:loss 低 ≠ 成功率高**,最终以仿真/真机成功率为准

---

## 10. 评估你的策略

### 10.1 真机评估(SO-101 等)

复用 `lerobot-record` 挂上策略,跑 10 个 episode 统计成功率:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<PORT> --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/eval_my_task \
  --dataset.single_task="<与训练时相同的任务描述>" \
  --dataset.num_episodes=10 \
  --policy.path=${HF_USER}/act_my_task
```

### 10.2 仿真 benchmark 评估

```bash
lerobot-eval \
  --policy.path=${HF_USER}/diffusion_pusht \
  --env.type=pusht --eval.n_episodes=50 --eval.batch_size=10 --policy.device=cuda
```

- `n_episodes ≥ 50` 才有统计意义
- benchmark 依赖重 → 用官方 Dockerfile:`docker/Dockerfile.benchmark.<name>`

### 10.3 基准线

50 条干净演示的单任务抓放:**ACT 应达 > 70% 成功率**。达不到 → 先怀疑数据(§8 路径B的黄金法则),别急着换模型。

---

## 11. 七天学习路线图

| 天 | 目标 | 动作 |
|---|---|---|
| Day 1 | 建立直觉 | 读本文 §1-4 + [README](./README.md) + 论文 §1-2;跑 `lerobot-info` |
| Day 2 | 跑通训练 | 路径 A:公开数据集训 ACT;开 WandB 观察 loss 曲线 |
| Day 3 | 理解数据 | 读 `datasets/lerobot_dataset.py`;用 `lerobot-dataset-viz` 和在线可视化器检查数据;学 delta_timestamps |
| Day 4 | 读懂策略 | 精读 `policies/act/` 三个文件 + ACT 原论文;对照 `processor_act.py` 理解归一化 |
| Day 5 | 训练机制 | 精读 `scripts/lerobot_train.py` 的 `train()` 循环 + `configs/train.py`;理解 checkpoint 结构;试 `--resume` |
| Day 6 | 进阶策略 | 训 SmolVLA(解冻视觉编码器);对比 ACT;理解 VLA 与语言条件 |
| Day 7 | 评估与部署 | `lerobot-eval` 仿真评估;读 `async_inference/` 理解部署架构;(有硬件则走路径 B) |

---

## 附录A 术语表

| 术语 | 含义 |
|---|---|
| Policy(策略) | 观测→动作的神经网络 |
| Observation / Action | 机器人感知的世界 / 要执行的关节目标 |
| Episode(回合) | 一次完整任务演示 |
| Chunk(动作块) | 策略一次预测的多步动作(ACT 的核心技巧,抑制抖动) |
| Proprioception(本体感知) | 机器人自己的关节状态 |
| VLA | Vision-Language-Action,视觉-语言-动作多模态大模型 |
| BC / IL | Behavior Cloning / Imitation Learning,模仿学习 |
| Teleoperation(遥操作) | 人远程操控机器人(采集数据用) |
| FSDP / DDP | 分布式训练的两种分片/复制策略 |
| EMA | 指数滑动平均权重,评估时常比实时权重更稳 |
| delta_timestamps | 采样时同时取历史/未来帧的机制(时序模型需要) |
| draccus | 本库的 dataclass→CLI 配置解析库 |
| HF Hub | Hugging Face 模型/数据集托管平台 |

## 附录B 论文 ↔ 代码对应表

| 论文位置 | 内容 | 代码入口 |
|---|---|---|
| §3.1 / Fig.4 | 低成本硬件(SO-10X 等) | `robots/so_follower/`, `teleoperators/so_leader/` |
| §3.1 / App.B | Robot API | `robots/robot.py`(`Robot` 基类) |
| §3.2 / App.C | LeRobotDataset 格式与流式 | `datasets/lerobot_dataset.py`, `streaming_dataset.py` |
| §3.3 / Tab.2 | 模型显存/延迟对比 | `policies/`(ACT 52M / Diffusion 263M / π0 3.5B / SmolVLA 450M) |
| §3.3 / App.D | <100 行训练 / <40 行推理示例 | `examples/`, `scripts/lerobot_train.py` |
| §3.4 / App.E | 异步推理 server/client | `async_inference/policy_server.py`, `robot_client.py` |
| §4 | LIBERO / Meta-World 实验 | `envs/libero.py`, `envs/metaworld.py` |

## 附录C 常见问题 FAQ

**Q: 训练前必须下载数据集吗?**
A: 不必。`StreamingLeRobotDataset` 支持边流式边训练,大数据集不用整份下载。

**Q: 图像以什么格式进模型?**
A: 数据集存 uint8;`_preprocess_dataset_batch` 会转 float32 并除以 255,再由 processor 做归一化/增强。

**Q: 为什么同一 batch 必须来自同一 episode?**
A: `EpisodeAwareSampler` 保证时序连续性 —— ACT 等模型需要同一演示内的相邻帧来预测动作块。

**Q: checkpoint 里 `pretrained_model/` 和 `training_state/` 的区别?**
A: 前者是可发布可推理的模型(权重+配置+处理器);后者是优化器/调度器状态,只有 `--resume` 续训才需要。

**Q: 如何恢复中断的训练?**
A: `lerobot-train --resume=true --config_path=<checkpoint>/train_config.json`(Hub 上的 checkpoint 也可,自动找最新)。

**Q: 我的策略在真机上抖动怎么办?**
A: ① 检查演示数据一致性(犹豫的演示教出犹豫的策略);② 确认 `chunk_size` 与 temporal ensemble 配置;③ rollout 时开启动作插值。

**Q: 想加自己的机器人/策略?**
A: 机器人:实现 `Robot` 抽象类(或发 `lerobot_robot_<name>` 插件包);策略:写 config(`@register_subclass("my_policy")`)+ modeling(继承 `PreTrainedPolicy`),参考 `policies/act/` 的三文件结构。详见 [docs/source/bring_your_own_policies](https://huggingface.co/docs/lerobot/bring_your_own_policies)。

---

*本报告基于 LeRobot v0.6.2 源码与论文 arXiv:2602.22818v1 (ICLR 2026) 撰写。实操细节以 [`AGENT_GUIDE.md`](./AGENT_GUIDE.md) 和 [官方文档](https://huggingface.co/docs/lerobot/index) 为准。*

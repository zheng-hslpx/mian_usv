# USV（无人水面艇）任务调度系统

基于强化学习的USV多智能体任务调度系统，使用PPO算法和图神经网络实现最优调度策略。

## 项目结构

```
A/
├── env/                           # 环境模块目录
│   ├── __init__.py
│   ├── usv_env.py                 # USV环境核心实现
│   ├── usv_case_generator.py      # USV案例生成器
│   ├── usv_load_data.py          # USV数据加载器
│   ├── usv_constraint_validator.py # USV约束验证器
│   ├── charging_station_manager.py # 充电站管理器
│   └── generate_usv_validation_data.py # 验证数据生成脚本
├── graph/                         # 图神经网络模块
│   └── usv_hgnn.py               # USV异构图神经网络实现
├── utils/                         # 工具模块
│   ├── __init__.py
│   ├── my_utils.py               # 通用工具函数和常量
│   ├── save_manager.py          # 模型保存管理器
│   ├── model_tools.py           # 模型相关工具
│   └── color_config.json        # 颜色配置文件
├── usv_data_dev/                 # 验证数据目录
│   ├── 40_2/                    # 40任务2USV组合数据
│   ├── 40_4/                    # 40任务4USV组合数据
│   ├── ...                      # 其他组合数据
│   └── 120_8/                   # 120任务8USV组合数据
├── config_usv.json              # USV配置文件
├── usv_train.py                  # 训练主程序
├── usv_ppo.py                   # PPO算法实现
├── usv_mlp.py                   # MLP网络模块
├── usv_validate.py              # 模型验证程序
└── README.md                    # 项目说明文件（本文件）
```

## 核心模块功能详解

### 1. 环境模块 (env/)

#### usv_env.py - USV环境核心
**主要类：**
- **USVState**: 环境状态类，包含任务特征、USV特征、时间信息等
- **USVEvent**: 事件类，支持事件驱动的环境更新
- **USVEnv**: Gymnasium兼容的环境主类

**功能特点：**
- 支持多USV、多任务的并行调度
- 集成充电站管理，支持USV充电决策
- 实现事件驱动的状态更新机制
- 提供8大约束验证（电量、任务分配、时间约束等）
- 支持批次并行处理，提高训练效率

#### usv_case_generator.py - 案例生成器
**主要类：**
- **USVCaseData**: USV案例数据结构
- **USVCaseGenerator**: 案例生成器核心类

**功能特点：**
- 支持内存优先和文件保存两种模式
- 提供低、中、高三种随机化程度
- 支持20种预定义任务-USV组合（2/4/6/8 USV × 40/60/80/100/120 任务）
- 生成包含位置、类型、执行时间的完整任务数据
- 构建任务-USV邻接矩阵和任务前驱关系矩阵

#### usv_load_data.py - 数据加载器
**主要类：**
- **USVLoadedData**: 加载后的数据结构
- **USVDataLoader**: 数据加载器核心类

**功能特点：**
- 将USVCaseData转换为环境所需张量格式
- 构建空间邻接矩阵（任务-USV和任务-任务）
- 实现特征标准化和格式转换
- 支持单案例和批次处理

#### usv_constraint_validator.py - 约束验证器
**主要类：**
- **ConstraintResult**: 约束验证结果类
- **USVConstraintValidator**: 约束验证器核心类

**功能特点：**
- 验证8大约束条件：电池容量、任务时间、任务分配、单次出航等
- 提供详细的违反约束信息
- 支持批次验证
- 模块化设计，便于扩展新的约束类型

#### charging_station_manager.py - 充电站管理器
**主要类：**
- **ChargingRecord**: 充电记录数据结构
- **ChargingStationManager**: 充电站管理器核心类

**功能特点：**
- 支持多USV同时充电（无限充电能力）
- 维护USV充电时长和可用状态
- 提供充电统计和历史记录
- 支持充电特征提取用于决策

### 2. 图神经网络模块 (graph/)

#### usv_hgnn.py - 异构图神经网络
**主要类：**
- **USVBlock**: USV节点嵌入模块，实现"任务→USV"异构注意力更新
- **TaskBlock**: 任务节点嵌入模块，实现η近邻任务聚合更新

**功能特点：**
- 两阶段HGNN架构：Stage-1更新USV嵌入，Stage-2更新任务嵌入
- 支持单头和多头注意力机制
- 使用三元注意力：源节点特征 + 目标节点特征 + 边特征
- 完全兼容FJSP的graph/hgnn.py架构设计

### 3. 强化学习模块

#### usv_ppo.py - PPO算法实现
**主要类：**
- **Memory**: 经验存储类，适配USV二维动作格式
- **MLPs**: MLP模块整合，包含策略头和价值头
- **HGNNScheduler**: HGNN调度器，整合两阶段图神经网络
- **PPO**: PPO-Clip算法主类

**功能特点：**
- 完整的PPO-Clip算法实现
- 支持N+1动作空间（包含充电站动作）
- 集成GAE优势估计
- 支持批次训练和minibatch更新
- 提供模型保存和加载功能

#### usv_mlp.py - MLP网络模块
**主要类：**
- **USVPairFeature**: 成对特征构造器，支持N+1动作空间
- **USVActionHead**: 策略头，继承MLPActor
- **USVValueHead**: 价值头，继承MLPCritic

**功能特点：**
- 严格遵循mlp.py架构，确保训练一致性
- 支持充电站特征生成
- 使用tanh激活函数，无BatchNorm层
- 完全独立实现，不依赖外部mlp模块

### 4. 工具模块 (utils/)

#### my_utils.py - 通用工具
**功能特点：**
- 集中定义模型常量（嵌入维度、隐藏层维度等）
- 提供JSON读写工具函数
- 定义数值稳定性常量

#### save_manager.py - 模型保存管理器
**主要类：**
- **SaveManager**: 保存管理器核心类

**功能特点：**
- 配置化的实验目录创建和管理
- 最佳模型保存和管理（支持多种指标）
- 训练检查点保存和恢复
- 配置文件备份和版本管理
- 实验结果记录和查询

### 5. 训练和验证模块

#### usv_train.py - 训练主程序
**功能特点：**
- 完整的训练流程管理
- 支持visdom可视化训练过程
- 集成USV案例生成器和验证环境
- 支持训练日志记录和模型保存
- 提供训练统计和实验总结

#### usv_validate.py - 模型验证程序
**主要函数：**
- **get_usv_validate_env()**: 创建USV验证环境
- **usv_validate()**: USV策略验证函数

**功能特点：**
- 使用固定验证数据集评估模型性能
- 支持makespan指标计算
- 集成USV约束验证
- 提供验证结果统计

## 文件间的联动关系

### 1. 数据流向

```
案例生成 → 数据加载 → 环境初始化 → 智能体交互 → 模型训练 → 性能验证
```

**详细流程：**
1. **usv_case_generator.py** 生成USV案例数据
2. **usv_load_data.py** 将案例数据转换为环境张量格式
3. **usv_env.py** 初始化环境状态，提供Gymnasium接口
4. **usv_ppo.py** 中的智能体与环境交互，收集经验
5. **usv_hgnn.py** 和 **usv_mlp.py** 处理状态特征，输出动作
6. **usv_train.py** 管理整个训练流程
7. **usv_validate.py** 使用固定数据集评估模型性能

### 2. 核心组件协作

**环境-智能体交互：**
- `usv_env.py` 提供标准Gymnasium接口（reset、step、render）
- `usv_ppo.py` 的PPO类实现动作选择和策略更新
- `usv_constraint_validator.py` 实时验证调度决策的合法性

**特征处理流水线：**
- `usv_env.py` 的USVState类管理环境状态
- `usv_hgnn.py` 的USVBlock和TaskBlock处理图结构特征
- `usv_mlp.py` 的USVPairFeature构造成对特征
- `usv_ppo.py` 的MLPs模块输出策略和价值

**充电管理集成：**
- `usv_env.py` 集成充电站决策（N+1动作空间）
- `charging_station_manager.py` 管理充电状态和历史
- `usv_ppo.py` 增强USV特征，支持充电决策

### 3. 配置和工具支持

**配置管理：**
- `config_usv.json` 提供完整的配置参数
- `utils/my_utils.py` 定义模型常量
- `utils/save_manager.py` 管理实验结果和模型保存

## 项目使用方法

### 1. 环境准备

**依赖安装：**
```bash
pip install torch gymnasium numpy pandas visdom
```

**生成验证数据：**
```bash
cd env/
python generate_usv_validation_data.py
```

### 2. 训练模型

**启动训练：**
```bash
python usv_train.py
```

训练过程包括：
- 自动生成USV案例用于训练
- 使用PPO算法优化调度策略
- 定期验证和保存最佳模型
- 可视化训练过程（如启用visdom）

### 3. 模型验证

**验证已训练模型：**
```bash
python usv_validate.py
```

验证功能：
- 使用固定验证数据集评估性能
- 计算makespan等关键指标
- 验证调度约束满足情况

### 4. 配置说明

**主要配置项 (config_usv.json)：**

**环境参数 (env_paras)：**
- `num_tasks`: 任务数量（默认40）
- `num_usvs`: USV数量（默认4）
- `batch_size`: 批次大小（默认20）
- `map_size`: 地图尺寸（默认[800, 800]）
- `battery_capacity`: 电池容量（默认1200）
- `usv_speed`: USV航速（默认5）
- `enable_pre_validation`: 启用预验证（默认true）

**模型参数 (model_paras)：**
- `d`: 嵌入维度（默认32）
- `hidden_dim`: 隐藏层维度（默认128）
- `num_layers`: HGNN层数（默认2）
- `eta`: 任务近邻数量（默认3）

**训练参数 (train_paras)：**
- `lr`: 学习率（默认0.0002）
- `max_iterations`: 最大迭代次数（默认1000）
- `K_epochs`: PPO更新轮数（默认3）
- `gamma`: 折扣因子（默认1.0）

**保存参数 (save_paras)：**
- `base_path`: 保存基础路径（默认"./save"）
- `experiment_name`: 实验名称（默认"usv_experiment"）
- `auto_version`: 自动版本管理（默认true）

### 5. 扩展和定制

**添加新的任务类型：**
1. 修改 `usv_case_generator.py` 中的任务类型定义
2. 更新 `usv_env.py` 中的任务执行时间配置
3. 调整 `usv_ppo.py` 中的特征维度（如需要）

**调整约束条件：**
1. 在 `usv_constraint_validator.py` 中添加新的验证方法
2. 更新 `USVConstraintValidator.validate_all()` 方法
3. 修改环境参数以支持新约束

**自定义网络架构：**
1. 修改 `usv_hgnn.py` 中的图神经网络结构
2. 调整 `usv_mlp.py` 中的MLP架构
3. 更新配置文件中的维度参数

## 技术特点

1. **模块化设计**: 环境生成、数据处理、模型训练完全解耦
2. **图神经网络**: 使用异构图神经网络处理任务-USV关系
3. **强化学习**: 基于PPO算法的端到端训练
4. **约束验证**: 实时验证8大约束条件，确保调度可行性
5. **充电管理**: 集成充电站决策，支持长时间调度
6. **可扩展性**: 支持不同规模的任务-USV组合
7. **实验管理**: 完整的实验配置、模型保存和结果记录

## 性能指标

- **优化目标**: 最小化最大完成时间（makespan）
- **约束满足**: 8大约束条件100%满足
- **训练效率**: 支持批次并行训练
- **泛化能力**: 支持20种不同任务-USV组合


该系统为USV任务调度提供了完整的强化学习解决方案，具有良好的模块化设计和扩展性。

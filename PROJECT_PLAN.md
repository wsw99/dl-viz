# Project Plan: DL Algorithm Visualizer（深度学习算法可视化器）

## 1. 项目概述

**目标：** 构建一组交互式 HTML 页面，逐步展示深度学习基础算法的每个计算步骤，让学习者通过点击"下一步"直观理解算法内部机制。

**交付格式：** GitHub 仓库，`main` 分支。每个算法独立文件夹，克隆后 `cd` 进入目标文件夹，启动本地服务器即可运行，无需切换分支。

**核心体验：** 用户点击 **Next Step** 推进算法，每步更新可视化、显示纯文字说明、高亮伪代码对应行。

---

## 2. 仓库结构

### 顶层布局

```
dl-visualizer/
  README.md                   ← 项目简介、算法表格、运行说明
  PROJECT_PLAN.md             ← 本文档
  mlp-backprop/               ← MLP 前向传播 + 反向传播
  cnn-convolution/            ← CNN 卷积 + 池化 + 特征图
  optimizer/                  ← 优化器对比（SGD / Momentum / Adam）
  rnn-lstm/                   ← RNN/LSTM 门控与隐状态演化
  attention/                  ← 自注意力机制
  transformer-block/          ← Transformer 编码器块
```

### 每个算法文件夹结构（完全自包含）

```
<algorithm>/
  index.html            ← 入口，浏览器直接打开
  shared/
    controls.js         ← 步骤控制器（Next / Prev / Reset / Auto Play）
    layout.css          ← 统一布局与动画样式
  js/
    algorithm.js        ← 算法逻辑（纯 JS，不依赖 ML 库）
    visualization.js    ← 算法专属渲染（D3 / Canvas / SVG）
    main.js             ← 页面初始化
  data/
    sample_data.js      ← 该算法专用数据集（全局变量，无 fetch）
```

### 算法文件夹表

| 文件夹 | 算法 | 核心可视化 |
|--------|------|------------|
| `mlp-backprop/` | 多层感知机反向传播 | 神经元激活流、梯度反向流、权重更新热图 |
| `cnn-convolution/` | 卷积神经网络 | 卷积核滑动动画、特征图生成、Max Pooling |
| `optimizer/` | 优化器对比 | 损失曲面等高线 + 三条优化路径 |
| `rnn-lstm/` | LSTM 门控机制 | 展开时序图、三门激活值、细胞状态流 |
| `attention/` | 自注意力机制 | Q/K/V 矩阵、注意力权重热图、加权聚合 |
| `transformer-block/` | Transformer 编码器块 | 位置编码、多头注意力、Add&Norm、FFN |

### 共享代码策略

`shared/controls.js` 和 `shared/layout.css` **复制**进每个算法文件夹（与 ML Visualizer 相同策略）：
- 先在 `mlp-backprop/shared/` 开发并定型
- 创建新算法文件夹时复制进去
- 共享代码有改动时手动同步

---

## 3. 技术栈

**无后端，完全在浏览器内运行。**

| 库 | 用途 | 方式 |
|---|---|---|
| D3.js v7 | 网络图布局、等高线图、坐标轴、矩阵热图 | 本地文件 |
| Chart.js | 损失曲线、折线图 | 本地文件 |
| KaTeX | 数学公式渲染（∂L/∂w, σ(z), Softmax 等） | 本地文件 |
| Native Canvas API | 卷积滑动动画、MLP 网络绘制 | 内置 |
| Plain CSS animations | 步骤过渡、高亮效果 | 本地文件 |

> 所有算法逻辑从零用纯 JavaScript 实现，不依赖 TensorFlow.js / Brain.js 等 ML 库，确保代码直接对应可视化步骤。

### 如何在本地运行

```bash
git clone <repo-url>
cd dl-visualizer/mlp-backprop    # 进入目标算法文件夹
python -m http.server 8000
# 打开 http://localhost:8000
```

### 数据约定

每个算法有独立的 `data/sample_data.js`，以全局变量方式加载，避免 CORS 问题：

```js
// data/sample_data.js
window.DATA = { inputs: [[0,0],[0,1],[1,0],[1,1]], labels: [0,1,1,0] };
```

**数据不跨算法共享。** 各算法用专门设计的小型 toy dataset：
- MLP：XOR 问题（4个点）或二分类（20个点）
- CNN：8×8 灰度合成图像（手动构造的简单纹理）
- Optimizer：2D Rosenbrock 函数或 bowl 形二次函数
- LSTM：长度为 6 的二进制序列（奇偶预测任务）
- Attention：4个 token 的句子（手动设定 embedding）
- Transformer：4个 token 的序列，完整走一遍编码器

---

## 4. 页面布局（统一）

与 ML Visualizer 完全一致：

```
┌─────────────────────────────────────────────────────┐
│  [Algorithm Name]                      Step 3 / 12  │
├────────────────────────┬────────────────────────────┤
│                        │  Step caption（纯文字说明）  │
│   Main visualization   │  ─────────────────────────  │
│   (D3 / Canvas / SVG)  │  Pseudocode（当前行高亮）    │
│                        │  ─────────────────────────  │
│                        │  Parameter panel            │
│                        │  w₁=0.32  ∂L/∂w₁=-0.08     │
└────────────────────────┴────────────────────────────┤
│  [⏮ Reset]  [← Prev]  [▶ Next]  [⏩ Auto Play]      │
└─────────────────────────────────────────────────────┘
```

---

## 5. 各算法步骤设计

### 5.1 MLP 反向传播（`mlp-backprop/`）

**网络结构：** 2-3-1 全连接网络（输入层2节点，隐藏层3节点，输出层1节点），解决 XOR 问题。

| 步骤 | 展示内容 |
|------|----------|
| Step 0 | 静态网络图：节点（圆）+ 边（权重线），随机初始权重显示在边上 |
| Step 1 | 输入层激活：高亮输入节点，显示 x₁=1, x₂=0 |
| Step 2 | 前向传播到隐藏层：逐条边高亮，显示 z=w·x+b，节点颜色随激活值变化（深蓝→深红） |
| Step 3 | ReLU 激活：显示 ReLU(z) 曲线，节点更新为 a=ReLU(z) |
| Step 4 | 前向传播到输出层：同Step2，sigmoid 激活，输出 ŷ |
| Step 5 | 计算损失：Binary Cross-Entropy，在参数面板显示 L 值，损失曲线追加第一个点 |
| Step 6 | 反向传播：输出层梯度 ∂L/∂ŷ，红色箭头从右向左流动 |
| Step 7 | 隐藏层梯度：链式法则展示 ∂L/∂w 的计算过程，边上显示梯度值（蓝=负，红=正） |
| Step 8 | 权重更新：所有权重同时更新，边的粗细/颜色变化表示权重变化幅度 |
| Step N | 重复多轮，损失曲线持续下降 |
| Final | 收敛后的网络 + 最终决策边界（2D scatter plot） + 损失曲线 |

**参数面板显示：** 当前 epoch、损失值 L、选中权重的 w / ∂L/∂w / Δw。

---

### 5.2 CNN 卷积与池化（`cnn-convolution/`）

**输入：** 8×8 灰度合成图像（手工构造，内有简单边缘纹理）。  
**架构：** 一层 Conv（3×3 kernel，1个滤波器）→ ReLU → 2×2 MaxPooling。

| 步骤 | 展示内容 |
|------|----------|
| Step 0 | 显示 8×8 输入图像（像素值网格，颜色编码 0-255） |
| Step 1 | 展示 3×3 卷积核（数值显示在每格），说明参数量 |
| Step 2 | 卷积核移动到左上角第一个位置，对应输入区域高亮（蓝色框） |
| Step 3 | 逐元素相乘动画：输入 × 核 → 中间结果，求和得 z₀₀，写入输出特征图第(0,0)格 |
| Step 4-N | 卷积核逐格滑动（stride=1），每步高亮当前感受野，填入特征图 |
| Step K | 全部卷积完成，显示完整 6×6 特征图 |
| Step K+1 | ReLU 激活：逐格置负值为 0，颜色变化 |
| Step K+2 | MaxPooling：2×2 窗口滑动，取最大值，6×6 → 3×3 |
| Final | 三列并排：输入图 / 特征图（卷积后）/ 池化图，颜色热图对比 |

**可调参数：** 卷积核权重（点击格子编辑）、stride、是否使用 padding。

---

### 5.3 优化器对比（`optimizer/`）

**损失函数：** 2D Beale 函数或 Rosenbrock 函数（有鞍点和山谷，充分体现优化器差异）。  
**对比：** SGD / SGD+Momentum / Adam 三条路径，从相同初始点出发。

| 步骤 | 展示内容 |
|------|----------|
| Step 0 | D3 等高线图（contour plot），颜色表示损失高度，标注全局最优点 |
| Step 1 | 三个初始点（相同位置，三色圆点）放置在曲面上 |
| Step 2 | 第一次梯度计算：显示各算法的梯度向量 ∂L/∂w₁, ∂L/∂w₂ |
| Step 3 | SGD 更新：w ← w - lr·g，画出移动步骤，轨迹线 |
| Step 4 | Momentum 更新：v ← β·v + g；w ← w - lr·v，对比步长差异 |
| Step 5 | Adam 更新：显示 m₁（一阶矩）、m₂（二阶矩）、bias correction，步骤动画 |
| Step 6-N | 三者同步推进，轨迹线不断延伸，颜色区分（红/绿/蓝） |
| Final | 三条完整轨迹 + 收敛迭代次数对比 + 各算法最终损失柱状图 |

**参数面板显示：** 当前迭代、每个算法的 lr / β / ε、当前损失、步长大小。

---

### 5.4 LSTM 门控机制（`rnn-lstm/`）

**任务：** 奇偶预测（输入二进制序列 [1,0,1,1,0,1]，预测 1 的个数是奇还是偶）。  
**结构：** 单个 LSTM Cell，展开 6 个时间步。

| 步骤 | 展示内容 |
|------|----------|
| Step 0 | 展示输入序列 x₁...x₆，LSTM Cell 结构图（三门 + cell state） |
| Step 1 | t=1：输入 x₁ 进入，前一隐状态 h₀=0, c₀=0 |
| Step 2 | **遗忘门**：f = σ(W_f·[h,x]+b_f)，滑块动画显示遗忘程度（0=完全忘记，1=完全记住） |
| Step 3 | **输入门**：i = σ(W_i·[h,x]+b_i)，候选值 c̃ = tanh(W_c·[h,x]+b_c) |
| Step 4 | **细胞状态更新**：c₁ = f⊙c₀ + i⊙c̃，流程图动画（旧状态 → 遗忘 → 加入新信息） |
| Step 5 | **输出门**：o = σ(W_o·[h,x]+b_o)，h₁ = o⊙tanh(c₁) |
| Step 6-N | 推进到 t=2...6，每步高亮当前时间步，显示 h 和 c 的数值变化热图 |
| Final | 展开图的所有时间步 + h/c 随时间变化折线图 + 最终预测输出 |

**参数面板显示：** 当前时间步、f/i/o/c̃ 的数值（0-1 色条）、c 和 h 向量热图。

---

### 5.5 自注意力机制（`attention/`）

**任务：** 4 个 token 的简单句子（"The cat sat here"），embedding 维度 d=4。  
**展示：** 单头 Self-Attention 完整计算过程。

| 步骤 | 展示内容 |
|------|----------|
| Step 0 | 4×4 输入矩阵 X（行=token，列=embedding 维度），列名显示 |
| Step 1 | 投影到 Q、K、V：分别乘以 W_Q / W_K / W_V，三个矩阵并排显示 |
| Step 2 | 计算 QKᵀ：矩阵乘法动画，逐格高亮计算，得到 4×4 得分矩阵 |
| Step 3 | 缩放：除以 √d_k，防止梯度消失，显示缩放前后值对比 |
| Step 4 | Softmax：逐行做 softmax，4×4 矩阵转化为注意力权重（0-1），热图着色 |
| Step 5 | 加权聚合：Attention × V，高亮"The"关注"cat"的权重，动态加权求和 |
| Step 6 | 输出矩阵 Z：4×4，每个 token 的新表示 |
| Final | 注意力热图（"谁关注谁"）+ 输入与输出向量变化对比 |

**参数面板显示：** 当前 token pair 的注意力分数、softmax 前后值对比、d_k 值。

---

### 5.6 Transformer 编码器块（`transformer-block/`）

**任务：** 4 个 token，完整走一遍 Transformer Encoder Layer（d_model=8，2头注意力，FFN 隐层=16）。

| 步骤 | 展示内容 |
|------|----------|
| Step 0 | 输入序列，逐位置展示 Token Embedding |
| Step 1 | **位置编码**：PE(pos,2i) = sin(pos/10000^(2i/d))，显示 sin/cos 波形如何叠加到 embedding |
| Step 2 | **多头注意力（头1）**：走一遍 attention/的完整流程（快速版） |
| Step 3 | **多头注意力（头2）**：不同 W_Q/W_K/W_V，得到不同注意力模式，两个热图对比 |
| Step 4 | **拼接+线性变换**：concat(head1, head2)·W_O |
| Step 5 | **Add & Norm（第一个）**：残差连接 + Layer Normalization，显示归一化前后分布 |
| Step 6 | **FFN 前向传播**：Linear(d→d×4)→ReLU→Linear(d×4→d)，节点图 |
| Step 7 | **Add & Norm（第二个）**：最终输出矩阵 |
| Final | 完整流程图 + 输入/输出向量热图对比 + 每步维度变化说明 |

---

## 6. 步骤状态管理（共享核心机制）

与 ML Visualizer 完全相同的 **快照数组 + 索引渲染** 模式，在 `shared/step-engine.js` 中实现：

```js
// LSTM 示例 snapshot
steps.push({
  timestep: 1,
  forgetGate: 0.82,         // f 门激活值
  inputGate: 0.41,          // i 门激活值
  outputGate: 0.67,         // o 门激活值
  candidateCell: 0.23,      // c̃ 候选细胞值
  cellState: [...],          // c 向量
  hiddenState: [...],        // h 向量
  codeLine: 4,              // 伪代码高亮行
  caption: "遗忘门：决定从细胞状态中丢弃多少信息"
});
```

渲染层是纯函数：`render(steps[currentIndex])`，不保存内部状态。Next/Prev 只改变 `currentIndex`。

> **优势：** 回退免费（读取已存储快照），算法逻辑与渲染完全解耦，控制器在全部 6 个算法中不变。

---

## 7. 算法正确性验证

| 算法 | 验证方法 |
|------|----------|
| MLP 反向传播 | 与 NumPy 手动实现逐步对比，验证梯度数值（numerical gradient check） |
| CNN 卷积 | 与 Python `scipy.signal.convolve2d` 对比输出特征图 |
| 优化器 | 与 PyTorch 的 `torch.optim.SGD/Adam` 在相同初值下对比收敛路径 |
| LSTM | 与 PyTorch `nn.LSTMCell` 对比单步输出 |
| Self-Attention | 手算 4×4 小矩阵，验证 softmax 输出 |
| Transformer | 逐模块验证，位置编码用解析公式交叉校验 |

---

## 8. 评估指标与术语表

每个缩写和公式在可视化中必须显示**全称 + 中文译名**。

### 8.1 MLP 反向传播

| 缩写 | 全称 | 中文名称 | 说明 |
|---|---|---|---|
| BCE | Binary Cross-Entropy | 二元交叉熵 | 二分类损失。公式：−(y log ŷ + (1−y) log(1−ŷ)) |
| ∂L/∂w | Gradient w.r.t. weight | 损失对权重的梯度 | 链式法则求偏导，指示权重更新方向 |
| lr | Learning Rate | 学习率 | 梯度下降步长 |
| ReLU | Rectified Linear Unit | 线性整流函数 | ReLU(z) = max(0, z) |
| σ | Sigmoid Function | Sigmoid 函数 | σ(z) = 1/(1+e^−z) |
| Epoch | Training Epoch | 训练轮次 | 遍历完一次完整训练集 |

### 8.2 CNN

| 缩写 | 全称 | 中文名称 | 说明 |
|---|---|---|---|
| Conv | Convolution | 卷积 | 卷积核在输入上滑动，逐位置做内积 |
| Kernel / Filter | Convolutional Kernel | 卷积核 / 滤波器 | 可学习的权重矩阵，提取局部特征 |
| Feature Map | Feature Map | 特征图 | 卷积操作的输出，代表输入的某种特征 |
| Stride | Stride | 步长 | 卷积核每次移动的像素数 |
| Padding | Zero Padding | 填充 | 在输入边缘补零，控制输出尺寸 |
| Max Pooling | Max Pooling | 最大池化 | 取局部区域最大值，降采样 |
| Receptive Field | Receptive Field | 感受野 | 输出特征图上一点对应的输入区域大小 |
| FLOPs | Floating Point Operations | 浮点运算次数 | 衡量计算量 |

### 8.3 优化器

| 缩写 | 全称 | 中文名称 | 说明 |
|---|---|---|---|
| SGD | Stochastic Gradient Descent | 随机梯度下降 | w ← w − lr·g |
| Momentum | Gradient Descent with Momentum | 动量梯度下降 | v ← βv + g；w ← w − lr·v |
| Adam | Adaptive Moment Estimation | 自适应矩估计 | 结合一阶矩（动量）和二阶矩（梯度平方均值） |
| β₁, β₂ | Exponential Decay Rates | 指数衰减率 | Adam 超参数，通常 β₁=0.9, β₂=0.999 |
| ε | Epsilon | 数值稳定项 | 防止除零，Adam 中通常取 1e-8 |
| m̂, v̂ | Bias-Corrected Moments | 偏差修正矩 | m̂ = m/(1−β₁ᵗ)，修正初始偏差 |
| Loss Landscape | Loss Landscape | 损失曲面 | 参数空间中损失函数的几何形状 |
| Saddle Point | Saddle Point | 鞍点 | 某些方向极小、另一些方向极大的临界点 |

### 8.4 LSTM

| 缩写 | 全称 | 中文名称 | 说明 |
|---|---|---|---|
| LSTM | Long Short-Term Memory | 长短期记忆网络 | 能处理长程依赖的 RNN 变体 |
| f | Forget Gate | 遗忘门 | 决定丢弃多少旧的细胞状态。f=σ(W_f·[h,x]+b) |
| i | Input Gate | 输入门 | 决定写入多少新信息。i=σ(W_i·[h,x]+b) |
| o | Output Gate | 输出门 | 决定输出多少细胞状态。o=σ(W_o·[h,x]+b) |
| c | Cell State | 细胞状态 | LSTM 的"长期记忆"，跨时间步传递 |
| h | Hidden State | 隐藏状态 | LSTM 的"短期输出"，传递给下一步和输出层 |
| c̃ | Candidate Cell State | 候选细胞状态 | tanh(W_c·[h,x]+b)，新信息候选值 |
| tanh | Hyperbolic Tangent | 双曲正切 | 输出范围 [−1,1]，压缩细胞状态 |
| Vanishing Gradient | Vanishing Gradient | 梯度消失 | 梯度在反向传播时指数级衰减，LSTM 的设计动机 |

### 8.5 自注意力机制

| 缩写 | 全称 | 中文名称 | 说明 |
|---|---|---|---|
| Q, K, V | Query, Key, Value | 查询、键、值 | 注意力机制的三个线性投影 |
| d_k | Key Dimension | 键向量维度 | 缩放因子 √d_k 用于防止点积过大 |
| Softmax | Softmax Function | Softmax 函数 | 将分数归一化为概率分布 |
| Attention Score | Attention Score | 注意力分数 | QKᵀ/√d_k，表示 token 对的相关度 |
| Attention Weight | Attention Weight | 注意力权重 | 经 Softmax 后的分数，对 V 加权求和 |
| Self-Attention | Self-Attention | 自注意力 | Q/K/V 来自同一序列，token 互相关注 |
| Context Vector | Context Vector | 上下文向量 | 注意力加权后每个 token 的新表示 |

### 8.6 Transformer 编码器块

| 缩写 | 全称 | 中文名称 | 说明 |
|---|---|---|---|
| PE | Positional Encoding | 位置编码 | sin/cos 函数，将位置信息注入 embedding |
| MHA | Multi-Head Attention | 多头注意力 | 并行多组注意力，拼接后投影，捕获多种关联 |
| FFN | Feed-Forward Network | 前馈神经网络 | 两层线性变换 + ReLU，逐位置独立 |
| LN | Layer Normalization | 层归一化 | 对每个样本的特征维度归一化，稳定训练 |
| Residual | Residual Connection | 残差连接 | x + Sublayer(x)，防止梯度消失，便于训练深层网络 |
| d_model | Model Dimension | 模型维度 | Transformer 的 embedding 维度 |
| d_ff | FFN Hidden Dimension | FFN 隐层维度 | 通常为 4×d_model |
| Head | Attention Head | 注意力头 | MHA 中每个独立的注意力子空间 |

---

## 9. 实现顺序与里程碑

| 周次 | 任务 | 输出 |
|------|------|------|
| 第 1 周 | 搭建仓库 + 完成 `mlp-backprop`（确定共享框架） | `mlp-backprop` 可运行，step engine 定型 |
| 第 2 周 | 完成 `cnn-convolution` + `optimizer` | 两个文件夹可运行 |
| 第 3 周 | 完成 `rnn-lstm` + `attention` | 两个文件夹可运行 |
| 第 4 周 | 完成 `transformer-block` + 完善 README + 截图 | 全部 6 个算法完成 |

> **为何先做 MLP Backprop：** 它包含神经网络图渲染、梯度流可视化、参数面板三个核心组件，是所有后续算法的基础架构验证台。CNN 的网络图、LSTM 的门控图、Transformer 的 FFN 图都复用这里建立的渲染模式。

---

## 10. 开发难度与各算法工作量估算

| 算法 | 算法逻辑难度 | 可视化难度 | 预计工时 | 主要挑战 |
|------|-------------|-----------|---------|---------|
| MLP Backprop | ★★★☆☆ | ★★★☆☆ | 3-4天 | 梯度箭头方向与大小的视觉编码 |
| CNN Convolution | ★★☆☆☆ | ★★★☆☆ | 2-3天 | 卷积核滑动动画流畅度 |
| Optimizer | ★★☆☆☆ | ★★★★☆ | 2-3天 | D3 等高线绘制 + 三路径同步 |
| LSTM | ★★★★☆ | ★★★★☆ | 3-4天 | 门控数据流图的布局与动画 |
| Self-Attention | ★★★☆☆ | ★★★☆☆ | 2-3天 | 矩阵热图 + 加权聚合动画 |
| Transformer | ★★★★☆ | ★★★★★ | 4-5天 | 多个子模块组合，步骤多 |

---

## 11. 风险与备选方案

| 风险 | 备选方案 |
|------|---------|
| Transformer 步骤太多，页面太复杂 | 拆分为 `transformer-attn`（多头注意力）和 `transformer-ffn`（FFN+LayerNorm），分两个页面 |
| LSTM 门控数据流布局调试耗时 | 简化为 Vanilla RNN（只有 tanh + 隐状态），跳过三门设计 |
| D3 等高线图在 Optimizer 模块调试困难 | 改用预计算的 100×100 网格数据，用 Canvas 绘制热图替代 D3 等高线 |
| MLP 网络图节点边数多时性能差 | 限制最大网络规模（2-4-1），Canvas 绘制替代 SVG |
| Self-Attention 矩阵乘法动画太慢 | 简化为 token-by-token 顺序显示，不做完整矩阵动画 |
| `shared/` 代码在多个文件夹中出现分叉 | 维护 diff checklist，每次修改都打 tag 记录版本 |

---

## 12. 与 ML Visualizer 的复用关系

此项目是 ML Visualizer 的深度学习续集，可直接复用：

| 组件 | 复用策略 |
|------|---------|
| `shared/controls.js`（步骤控制器） | 直接复制，无需修改 |
| `shared/layout.css`（统一布局） | 直接复制，按需添加 DL 专属样式 |
| Step engine 快照模式 | 完全复用，只改 snapshot 数据结构 |
| 伪代码高亮面板 | 直接复制组件 |
| Chart.js 损失曲线 | 直接复用 linear-regression 中的实现 |
| KaTeX 公式渲染 | 直接复用，更换公式内容 |

---

## 13. 参考资料

### 顶校课程参考
- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/) — Assignment 2 含网络可视化、BatchNorm、Dropout；Assignment 3 含 Transformer、自监督学习
- [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/) — 包含 RNN、CNN、注意力机制的逐步讲解

### 高质量开源可视化参考
- [CNN Explainer (Polo Club, Georgia Tech)](https://github.com/poloclub/cnn-explainer) — 交互式 CNN 可视化，发表于 IEEE TVCG 2020，Svelte + D3 实现，可参考其卷积动画设计
- [GAN Lab (Polo Club)](https://github.com/poloclub/ganlab) — TensorFlow.js 浏览器内训练 + D3 可视化，参考其训练循环可视化结构
- [TensorFlow Playground](https://playground.tensorflow.org/) — 经典神经网络交互工具，参考其参数调节面板设计
- [Emilien Dupont: Optimization Visualization](https://emiliendupont.github.io/2018/01/24/optimization-visualization/) — 优化器等高线轨迹对比的简洁实现参考
- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Transformer 逐步图解，是本项目 Transformer 步骤设计的主要参考
- [BertViz](https://github.com/jessevig/bertviz) — Transformer 注意力头可视化，参考其多头热图布局
- [AttentionViz](https://attentionviz.com/) — 全局注意力可视化，参考其 Q/K 关联展示方式

### 关键论文
- Hochreiter & Schmidhuber (1997) — Long Short-Term Memory（LSTM 原论文）
- Vaswani et al. (2017) — Attention Is All You Need（Transformer 原论文）
- Wang et al. (2020) — CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization

---

## 14. 最终交付物

- GitHub 仓库（`main` 分支，单次 checkout）
- 6 个算法文件夹：`mlp-backprop/`、`cnn-convolution/`、`optimizer/`、`rnn-lstm/`、`attention/`、`transformer-block/`
- 每个文件夹：`index.html` + `shared/` + `js/` + `data/`
- 根目录 `README.md`：项目描述、算法表格、运行说明、截图预览
- 可选：每个算法录制 GIF 嵌入 README

---

*计划日期：2026-05-27*

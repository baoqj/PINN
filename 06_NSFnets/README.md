# Physics-Informed Neural Network (PINN) for Navier-Stokes Equations


https://colab.research.google.com/drive/1heLEvTUHN7O3MN65NQuI7ueGolD7psG9?usp=sharing

## 物理背景

**Navier-Stokes 方程**是描述流体运动的基本方程，用于模拟速度场和压力场的演化。二维不可压缩流体的 Navier-Stokes 方程可表示为：

1. **动量方程**（描述速度场的变化）：
   \[
   \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu \nabla^2 u
   \]
   \[
   \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \nu \nabla^2 v
   \]

2. **不可压缩条件**（描述流体的连续性）：
   \[
   \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
   \]

其中：
- \( u(x, y, t) \): x 方向速度分量。
- \( v(x, y, t) \): y 方向速度分量。
- \( p(x, y, t) \): 压力场。
- \( \nu \): 流体的运动粘性系数。

**问题目标**：
- 从给定的空间和时间点数据 \((x, y, t)\) 中，通过 PINN 预测速度场 \( u(x, y, t) \)、\( v(x, y, t) \) 和压力场 \( p(x, y, t) \)。


## Physics-Informed Neural Network (PINN) 方法

PINN 是结合物理约束（如偏微分方程）的深度学习方法。其核心思想是：
1. 用神经网络建模未知的物理量（如 \( u, v, p \)）。
2. 将物理方程（如 Navier-Stokes 方程）嵌入到损失函数中，作为约束条件。
3. 通过优化损失函数，求解满足物理约束的解。

### **PINN 的组成部分**
1. **神经网络表示**：用神经网络 \( NN(x, y, t) \) 表示速度和压力：
   \[
   NN(x, y, t) \to [u(x, y, t), v(x, y, t), p(x, y, t)]
   \]

2. **损失函数**：
   - **数据损失**（与观测数据的误差）：
     \[
     L_{\text{data}} = \| u_{\text{pred}} - u_{\text{true}} \|^2 + \| v_{\text{pred}} - v_{\text{true}} \|^2
     \]
   - **方程残差**（物理约束）：
     \[
     L_{\text{physics}} = \| f_u \|^2 + \| f_v \|^2 + \| f_{\text{continuity}} \|^2
     \]
     其中 \( f_u, f_v \) 是 Navier-Stokes 动量方程的残差，\( f_{\text{continuity}} \) 是不可压缩条件的残差。

3. **优化目标**：
   \[
   L = L_{\text{data}} + L_{\text{physics}}
   \]


## 代码解析

### 1. 生成数据：`generate_mat_file()`
```python
def generate_mat_file():
    # 生成空间点
    x = np.linspace(0, 10, N)  # x 方向的坐标
    y = np.linspace(0, 2, N)   # y 方向的坐标
    X, Y = np.meshgrid(x, y)   # 创建二维网格
    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))  # 网格点坐标

    # 时间点
    t = np.linspace(0, 10, T).reshape(-1, 1)  # 时间点

    # 生成速度场和压力场
    U_x = np.sin(2 * np.pi * X.flatten()[:, None] / 10) * np.cos(2 * np.pi * t.T / 10)
    U_y = np.cos(2 * np.pi * Y.flatten()[:, None] / 10) * np.sin(2 * np.pi * t.T / 10)
    U_star = np.stack((U_x, U_y), axis=1)

    p_star = np.sin(2 * np.pi * X.flatten()[:, None] / 10) * np.cos(2 * np.pi * t.T / 10)

    # 保存到 .mat 文件
    scipy.io.savemat('cylinder_nektar_wake.mat', {
        'U_star': U_star,
        'p_star': p_star,
        't': t,
        'X_star': X_star
    })
```

- 创建一个模拟的速度场 U(x,y,t)、V(x,y,t) 和压力场 P(x,y,t)，并将其保存到 .mat 文件中。


### 2. PINN 模型定义：VPNSFnet
```Python
class VPNSFnet:
    def __init__(self, x, y, t, u, v, layers):
        self.lowb = np.array([x.min(), y.min(), t.min()]).astype(np.float32)
        self.upb = np.array([x.max(), y.max(), t.max()]).astype(np.float32)

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.t = t.astype(np.float32)
        self.u = u.astype(np.float32)
        self.v = v.astype(np.float32)

        self.layers = layers

        # 初始化权重和偏置
        self.weights, self.biases = self.initialize_NN(layers)

        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
```
- 初始化 PINN 模型，包括：
1. 数据归一化范围 lowb 和 upb。
2. 输入数据 x,y,t 和目标数据 u,v。
3. 神经网络的架构（层数和单元数）。


### 3. 神经网络初始化：initialize_NN()
```Python
def initialize_NN(self, layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers - 1):
        W = self.xavier_init(size=[layers[l], layers[l + 1]])  # Xavier 初始化
        b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
    return weights, biases
```

- 初始化 PINN 的权重和偏置。
- 使用 Xavier 方法初始化权重，保证训练的稳定性。

### 4. 训练模型：train()
```Python
def train(self, epochs):
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            u_pred, v_pred, p_pred = self.net_NS(self.x, self.y, self.t)
            loss = tf.reduce_mean(tf.square(self.u - u_pred)) + tf.reduce_mean(tf.square(self.v - v_pred))
        gradients = tape.gradient(loss, self.weights + self.biases)
        self.optimizer.apply_gradients(zip(gradients, self.weights + self.biases))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy():.3e}")
```

- 使用 GradientTape 自动计算梯度，优化损失函数（包括数据匹配损失）。


## 总结
- 核心思想：
    - 使用 PINN 方法结合物理方程和数据约束，学习流体的速度场和压力场分布。

- 代码流程：
1. 创建模拟数据（速度场和压力场）。
2. 定义 PINN 模型，通过神经网络预测 u,v,p。
3. 训练模型，优化损失函数（数据损失 + 物理约束）。
4. 可视化预测结果，并保存计算结果。

- 优势：
    - PINN 不需要大量标注数据，仅需少量数据点即可通过物理约束学习解的分布。
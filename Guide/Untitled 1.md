## Epoch、Batch 和 Step 之间的关系

### Epoch

- **定义**：一个 **epoch** 表示模型对**整个训练集进行一次完整的遍历**，即所有样本数据都经历了前向传播和反向传播的过程。

  当我们说 “训练了 10 个 epoch”，意味着模型已经从头到尾扫过了训练集 10 次。

### Batch

- **定义**：训练时通常不会将整个数据集一次性送入模型，而是将数据分成若干小批量（mini-batch）逐步进行训练，其中**每个 batch 包含一定数量的样本（batch size）**。

- **公式**：假设数据集大小为 $N$，batch size 为 $B$，则一个 epoch 内的 batch 数量为：
  $$
  \text{Number of Batches per Epoch} = \left\lceil \frac{N}{B} \right\rceil
  $$
  这里使用向上取整 $\lceil \cdot \rceil$ 是因为数据集大小 $N$ 可能不被 $B$ 整除。大多数深度学习框架在加载数据时可以自动处理最后一个不完整的 batch。例如，在 PyTorch 的 `DataLoader` 中，通过设置参数 `drop_last` 决定是否丢弃最后那个不完整的 batch（如果 `drop_last=True`，则会丢弃最后不足一个 batch 的样本，以确保所有 batch 大小一致）。

### Step

- **定义**：在训练中，**一次对参数的更新过程被称为一个 step**。更具体地说，处理完一个 batch 后，对模型参数执行一次前向传播（forward）、反向传播（backward）以及参数更新（optimizer.step()）的整个过程即为 1 个 step。

- **公式**：对应于上面的定义，一个 epoch 内 step 的数量与该 epoch 内的 batch 数量相同。当训练了 $E$ 个 epoch 时，总的 step 数为：
  $$
  \text{Total Steps} = \text{Number of Batches per Epoch} \times E = \left\lceil \frac{N}{B} \right\rceil \times E
  $$

### 关系总结

| **概念**  | **定义**                                                 | **公式**                                                     |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| **Epoch** | 整个训练集完整遍历一次                                   | -                                                            |
| **Batch** | 一小组样本，用于一次参数更新前的前/后向传播              | $\text{Number of Batches per Epoch} = \left\lceil \frac{N}{B} \right\rceil$ |
| **Step**  | 一次完整的参数更新过程（前向传播 + 反向传播 + 权重更新） | $\text{Total Steps} = \left\lceil \frac{N}{B} \right\rceil \times E$ |

---

### 举例说明

假设：

- 数据集大小 $N = 10,000$
- batch size $B = 32$
- epoch 数 $E = 5$

1. 计算 1 个 epoch 中的 batch 数量：
   $$
   \text{Number of Batches per Epoch} = \left\lceil \frac{10,000}{32} \right\rceil = \left\lceil 312.5 \right\rceil = 313
   $$

2. 计算总步数：
   $$
   \text{Total Steps} = \left\lceil \frac{10,000}{32} \right\rceil \times 5 = 313 \times 5 = 1565
   $$

**图示**（以前 2 个 epoch 为例）：

```bash
Epoch 1
  └── Batch 1 → Step 1  (前向传播 + 反向传播 + 权重更新)
  └── Batch 2 → Step 2
  └── Batch 3 → Step 3
  └── ... 
  └── Batch 313 → Step 313
  
Epoch 2
  └── Batch 1 → Step 314
  └── Batch 2 → Step 315
  └── Batch 3 → Step 316
  └── ... 
  └── Batch 313 → Step 626

```

- **每个 epoch** 包含多个 batch，每个 batch 进行 1 次权重更新（即 1 个 step）。
- **多个 epoch** 连续训练时，也可以让 step 累积，这样除了用 `epochs` 控制训练回合数，还可以设置 `max_steps` 控制模型结束训练的时间（AI 画图的 UI 界面中常出现这个超参数选项）。

#### 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设数据集大小为 N=10000
N = 10000
B = 32   # batch_size
E = 5    # epochs

# 创建一个示例数据集
X = torch.randn(N, 10)        # 假设输入维度为 10
y = torch.randint(0, 2, (N,)) # 二分类标签

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=B, shuffle=True, drop_last=False)

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

steps_per_epoch = len(dataloader)  # batch 的数量
total_steps = steps_per_epoch * E

current_step = 0

# 这里设置成从 1 开始只是为了不在 print 中额外设置，实际训练的时候不需要纠结这一点
for epoch in range(1, E + 1):
    for batch_idx, (inputs, targets) in enumerate(dataloader, start=1): 
        # 清零梯度（这一步放在反向传播前和参数更新之后都可以）
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播（计算梯度）
        loss.backward()

        # 参数更新
        optimizer.step(
        
        # 如果不需要累积 step，可以去除 current_step 项直接打印 batch_idx
        current_step += 1
        # 每 50 步打印一次
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch}/{E}], Batch [{batch_idx}/{steps_per_epoch}], "
                  f"Step [{current_step}/{total_steps}], Loss: {loss.item():.4f}")

# 可以看到:
# - epoch 从 1 到 E
# - batch 从 1 到 steps_per_epoch
# - step 累计从 1 到 total_steps
```

**输出**：

```
Epoch [1/5], Batch [50/313], Step [50/1565], Loss: 0.7307
Epoch [1/5], Batch [100/313], Step [100/1565], Loss: 0.6950
Epoch [1/5], Batch [150/313], Step [150/1565], Loss: 0.7380
Epoch [1/5], Batch [200/313], Step [200/1565], Loss: 0.7046
Epoch [1/5], Batch [250/313], Step [250/1565], Loss: 0.6798
Epoch [1/5], Batch [300/313], Step [300/1565], Loss: 0.7319
Epoch [2/5], Batch [50/313], Step [363/1565], Loss: 0.7058
Epoch [2/5], Batch [100/313], Step [413/1565], Loss: 0.7026
Epoch [2/5], Batch [150/313], Step [463/1565], Loss: 0.6650
Epoch [2/5], Batch [200/313], Step [513/1565], Loss: 0.6923
Epoch [2/5], Batch [250/313], Step [563/1565], Loss: 0.6889
Epoch [2/5], Batch [300/313], Step [613/1565], Loss: 0.6896
...
```



### 实践相关基础知识

1. **学习率调度器（Scheduler）**

   常见的学习率更新方式有两种：

   - **以 step 为基础**：在每个 step 结束后更新学习率。

     ```python
     scheduler = ...
     
     for epoch in range(E):
         for batch_idx, (inputs, targets) in enumerate(dataloader):
             optimizer.zero_grad()
             outputs = model(inputs)
             loss = criterion(outputs, targets)
             loss.backward()
             optimizer.step()
             
             # 在每个 step 后更新学习率
             scheduler.step()
     ```

   - **以 epoch 为基础**：在每个 epoch 结束后更新学习率。

     ```python
     scheduler = ...
     
     for epoch in range(E):
         for batch_idx, (inputs, targets) in enumerate(dataloader):
             optimizer.zero_grad()
             outputs = model(inputs)
             loss = criterion(outputs, targets)
             loss.backward()
             optimizer.step()
         
         # 在每个 epoch 后更新学习率
         scheduler.step()
     ```

   

2. **早停（Early Stopping）**

   可以基于 epoch 或 step 来监控验证集性能，若在一定 patience（耐心值）内验证性能没有提高，则提前停止训练来避免过拟合。

   ```python
   best_val_loss = float('inf')
   patience_counter = 0
   patience = 5
   
   for epoch in range(E):
       train_one_epoch(model, dataloader, optimizer, criterion)
       
       val_loss = validate(model, val_dataloader, criterion)
       
       if val_loss < best_val_loss:
           best_val_loss = val_loss
           patience_counter = 0
           # 保存最佳模型参数
           torch.save(model.state_dict(), "best_model.pth")
       else:
           patience_counter += 1
           if patience_counter > patience:
               print(f"No improvement in validation for {patience} epochs, stopping early.")
               break
   ```

3. **Batch Size 与显存**

   更大的 batch size 意味着每个 step 会处理更多数据，占用更多显存。当遇到 GPU 内存不足（Out of Memory，OOM）错误时，可以尝试减小 batch size。

## Q：SGD、BGD、MBGD 三者的区别是什么？

它们的本质区别在于**每次更新使用的样本数量（batch size）**不同。

### SGD（Stochastic Gradient Descent，随机梯度下降）

**定义**：每次参数更新使用**单个样本**计算梯度（batch_size = $1$）。 

**参数更新**：
$$
\theta := \theta - \eta \nabla_\theta \ell(f_\theta(x_i), y_i)
$$

其中：

- $\theta$：模型的参数向量。
- $\eta$：学习率（learning rate）。
- $\ell(\cdot,\cdot)$：损失函数。
- $f_\theta(x_i)$：当前参数 $\theta$ 对第 $i$ 个样本 $x_i$ 的前向预测输出。
- $(x_i, y_i)$：第 $i$ 个训练样本及其对应的标签 $y_i$。
- $\nabla_\theta \ell(f_\theta(x_i), y_i)$：损失函数 $\ell$ 关于参数 $\theta$ 的梯度。
- **$:=$**：赋值符号，表示将右侧的结果**更新**到左侧变量中（即用新的 $\theta$ 替换旧的 $\theta$）。

### BGD（Batch Gradient Descent，批量梯度下降）

**定义**：每次参数更新都使用**整个训练集**计算梯度（batch_size = $N$）。

**参数更新**：
$$
\theta := \theta - \eta \nabla_\theta J(\theta)
$$
其中：

- $\theta$：模型的参数向量。

- $\eta$：学习率。

- $J(\theta)$：整个数据集的平均损失函数，定义为：
  $$
  J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(f_\theta(x_i), y_i)
  $$
  其中 $N$ 是数据集中的样本总数。

- $\nabla_\theta J(\theta)$：损失函数 $J(\theta)$ 关于参数 $\theta$ 的梯度。

### MBGD（Mini-Batch Gradient Descent，小批量梯度下降）

**定义**：每次参数更新使用**一小批样本（mini-batch）**计算梯度（batch_size = $B$）。

**参数更新**：

1. 将训练集划分为若干 mini-batch：
   $$
   \mathcal{B}_1, \mathcal{B}_2, \dots, \mathcal{B}_K
   $$
   其中每个 $\mathcal{B}_k$ 包含 $B$ 个样本，即：
   $$
   \mathcal{B}_k = \{(x_{k_1}, y_{k_1}), (x_{k_2}, y_{k_2}), \dots, (x_{k_B}, y_{k_B})\}.
   $$

2. 在第 $k$ 个 mini-batch 上定义平均损失函数：
   $$
   J_{\mathcal{B}_k}(\theta) = \frac{1}{B} \sum_{(x_{k_j},y_{k_j}) \in \mathcal{B}_k} \ell(f_\theta(x_{k_j}), y_{k_j}).
   $$

3. 利用该 mini-batch 的平均损失对参数进行更新：
   $$
   \theta := \theta - \eta \nabla_\theta J_{\mathcal{B}_k}(\theta)
   $$

其中：

- $\theta$：模型的参数向量。
- $\eta$：学习率。
- $B$：mini-batch 的大小。
- $\mathcal{B}_k$：第 $k$ 个 mini-batch，包含 $B$ 个样本。
- $\nabla_\theta J_{\mathcal{B}_k}(\theta)$：损失函数 $J_{\mathcal{B}_k}(\theta)$ 关于参数 $\theta$ 的梯度。

其实，**SGD 和 BGD 只是 MBGD 中不同 batch_size 下的特例**，切换方法只需要修改 batch_size 的值：

- **SGD**：batch_size = 1，即每次只使用 1 个样本计算梯度。  
- **BGD**：batch_size = N，即每次使用**整个数据集**计算梯度。  
- **MBGD**：batch_size = B（1 < B < N），即每次使用 mini-batch 计算梯度。。

**举例说明**：

假设我们有 100 个样本的训练数据集（N=100），使用以下不同 batch size 训练 2 个 epoch：

- **SGD (batch_size=1)**：每次更新参数使用 1 个样本，更新 200 次（2 个 epoch，100 个样本）。  
- **BGD (batch_size=100)**：每次使用 100 个样本，更新 2 次（2 个 epoch，每个 epoch 1 次）。  
- **MBGD (batch_size=20)**：每次使用 20 个样本，更新 10 次（2 个 epoch，100 个样本分成 5 个 batch，每个 epoch 更新 5 次，2 epoch 共 10 次）。  


# Epoch、Batch 和 Step 之间的关系

## Epoch

- **定义**：一个 **epoch** 表示模型对**整个训练集进行一次完整的遍历**，即所有样本数据都经历了前向传播和反向传播的过程。

  当我们说 “训练了 10 个 epoch”，意味着模型已经从头到尾扫过了训练集 10 次。

## Batch

- **定义**：训练时通常不会将整个数据集一次性送入模型，而是将数据分成若干小批量（mini-batch）逐步进行训练，其中**每个 batch 包含一定数量的样本（batch size）**。

- **公式**：假设数据集大小为 $N$，batch size 为 $B$，则一个 epoch 内的 batch 数量为：
  $$
  \text{Number of Batches per Epoch} = \left\lceil \frac{N}{B} \right\rceil
  $$
  这里使用向上取整 $\lceil \cdot \rceil$ 是因为数据集大小 $N$ 可能不被 $B$ 整除。大多数深度学习框架在加载数据时可以自动处理最后一个不完整的 batch。例如，在 PyTorch 的 `DataLoader` 中，通过设置参数 `drop_last` 决定是否丢弃最后那个不完整的 batch（如果 `drop_last=True`，则会丢弃最后不足一个 batch 的样本，以确保所有 batch 大小一致）。

## Step

- **定义**：在训练中，**一次对参数的更新过程被称为一个 step**。更具体地说，处理完一个 batch 后，对模型参数执行一次前向传播（forward）、反向传播（backward）以及参数更新（optimizer.step()）的整个过程即为 1 个 step。

- **公式**：对应于上面的定义，一个 epoch 内 step 的数量与该 epoch 内的 batch 数量相同。当训练了 $E$ 个 epoch 时，总的 step 数为：
  $$
  \text{Total Steps} = \text{Number of Batches per Epoch} \times E = \left\lceil \frac{N}{B} \right\rceil \times E
  $$

## 关系总结

| **概念**  | **定义**                                                 | **公式**                                                     |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| **Epoch** | 整个训练集完整遍历一次                                   | -                                                            |
| **Batch** | 一小组样本，用于一次参数更新前的前/后向传播              | $\text{Number of Batches per Epoch} = \left\lceil \frac{N}{B} \right\rceil$ |
| **Step**  | 一次完整的参数更新过程（前向传播 + 反向传播 + 权重更新） | $\text{Total Steps} = \left\lceil \frac{N}{B} \right\rceil \times E$ |

---

## 举例说明

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
- **多个 epoch** 训练时，也可以让 step 累积，这样除了用 `epochs` 控制训练回合数，还可以设置 `max_steps` 控制模型结束训练的时间（AI 画图的 UI 界面中常出现这个参数选项）。

### 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设数据集大小为 N=10000
N = 10000
B = 32   # batch_size
E = 5    # epochs

# 随机创建一个示例数据集
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
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播（计算梯度）
        loss.backward()

        # 参数更新
        optimizer.step()

        # 清零梯度（这一步放在反向传播前和参数更新之后都可以）
        optimizer.zero_grad()
        
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
Epoch [1/5], Batch [50/312], Step [50/1560], Loss: 0.7024
Epoch [1/5], Batch [100/312], Step [100/1560], Loss: 0.6983
Epoch [1/5], Batch [150/312], Step [150/1560], Loss: 0.7116
Epoch [1/5], Batch [200/312], Step [200/1560], Loss: 0.6915
Epoch [1/5], Batch [250/312], Step [250/1560], Loss: 0.7033
Epoch [1/5], Batch [300/312], Step [300/1560], Loss: 0.7051
Epoch [2/5], Batch [50/312], Step [362/1560], Loss: 0.6932
Epoch [2/5], Batch [100/312], Step [412/1560], Loss: 0.7037
Epoch [2/5], Batch [150/312], Step [462/1560], Loss: 0.6851
Epoch [2/5], Batch [200/312], Step [512/1560], Loss: 0.6887
Epoch [2/5], Batch [250/312], Step [562/1560], Loss: 0.7028
Epoch [2/5], Batch [300/312], Step [612/1560], Loss: 0.6646
...
```



---

## 实践相关基础知识

1. **学习率调度器（Scheduler）**

   - 有些学习率调度器（如 `torch.optim.lr_scheduler.StepLR`）会基于 step 而非 epoch 更新学习率。这意味着在每个 batch 完成训练并进行参数更新后，再调用 `scheduler.step()`。

     ```python
     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
     
     for epoch in range(E):
         for batch_idx, (inputs, targets) in enumerate(dataloader):
             outputs = model(inputs)
             loss = criterion(outputs, targets)
             loss.backward()
             optimizer.step()
     		optimizer.zero_grad()
             
             # 在每个 step 后更新学习率
             scheduler.step()
     ```

   - 也有一些调度器是以 epoch 为周期更新学习率，这时在每个 epoch 训练结束后调用一次 `scheduler.step()`。

2. **早停（Early Stopping）**

   - 早停策略用于在验证集指标不再改善时提前停止训练，以防止过拟合于训练集。

   - 可以基于 epoch 或 step 来执行。当数据集非常大时，1 个 epoch 的时间较长，这时可以基于 step 监控训练进程。

3. **Batch Size 对显存的影响**：

   更大的 batch size 意味着每个 step 会处理更多数据，导致显存消耗增大。当遇到 GPU 内存不足（Out of Memory）错误时，可以尝试减小 batch size。


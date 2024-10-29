# 交叉熵损失函数 nn.CrossEntropyLoss() 详解和要点提醒（PyTorch）

> 在阅读论文过程中，我们时常遇到复杂的损失函数。然而，当你打开源码深入分析时，会发现很多模型最终都依赖于交叉熵损失。
>
> 本文将深入探讨交叉熵损失的数学原理及 PyTorch 实现，并结合代码示例，揭示其内在的联系和差异，并指出初次使用时需要注意的地方。

## 目录

- [前言](#前言)
   - [什么是 Logits？](#什么是-logits)
      - [Logits 和 Softmax](#logits-和-softmax)
   - [什么是 One-Hot 编码？](#什么是-one-hot-编码)
      - [类别不是整数怎么办？](#类别不是整数怎么办)
- [nn.CrossEntropyLoss() 交叉熵损失](#nncrossentropyloss-交叉熵损失)
   - [参数](#参数)
   - [数学公式](#数学公式)
      - [带权重的公式（weight）](#带权重的公式weight)
      - [标签平滑（label_smoothing）](#标签平滑label_smoothing)
      - [同时带权重和标签平滑](#同时带权重和标签平滑)
   - [要点](#要点)
- [附录](#附录)
- [参考链接](#参考链接)

# 前言

## 什么是 Logits？

Logits 是指神经网络的最后一个线性层（全连接层）的未经过任何激活函数（例如 softmax 或 sigmoid）处理的输出，可以是任意实数，在分类的任务中，logits 通常是在进行多类别分类任务时的原始输出。

### Logits 和 Softmax

在多类别分类问题中，logits 通常会被传递给 softmax 函数，softmax 函数将这些 logits 转换为概率分布: 将任意实数的 logits 转换为 [0, 1] 之间的概率值，并且这些概率值的和为 1。

**代码示例**

为了更好地理解 logits 和 softmax 之间的关系，下面是一个简单的代码示例: 

```python
import torch
import torch.nn.functional as F

# 样例: 分类神经网络，便于对照理解
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=3):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)  # ReLU 激活函数
        logits = self.fc2(out)  # 输出层，不经过 softmax
        return logits

# 假设这是分类神经网络的输出 logits
logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])

# 使用 softmax 函数将 logits 转换为概率分布
probabilities = F.softmax(logits, dim=1)

print("Logits:")
print(logits)
print("\nProbabilities after applying softmax:")
print(probabilities)
```

**输出**：
```sql
Logits:
tensor([[2.0000, 1.0000, 0.1000],
        [1.0000, 3.0000, 0.2000]])

Probabilities after applying softmax:
tensor([[0.6590, 0.2424, 0.0986],
        [0.1131, 0.8360, 0.0508]])
```

**解释**：

1. **Logits**: `[[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]]` 是神经网络的输出，未经过 softmax 处理。
2. **Softmax**: softmax 函数将 logits 转换为概率分布，每个分布的概率值和为 1。

## 什么是 One-Hot 编码？

初入深度学习领域的人大多都会有这个疑问: 这些所说的类别，究竟是怎么表示成向量的？

One-Hot 是一个很直观的形容，但我当时看到并猜测到相应概念的时候，还是不敢确定，因为太直白了，总觉得编码成向量的过程应该没有这么简单，然而 One-Hot 就是如此，深度学习不是一蹴而就的，看似复杂的概念最初也是由一个个直白的想法发展得来。

具体来说，One-Hot 编码对于每个类别，使用一个与类别数**相同长度**的**二进制向量**，每个位置对应一个类别。其中，**只有一个**位置的值为 1（这就是 “One-Hot” 的含义），表示属于该类别，其余位置的值为 0。

例如，对于三个类别的分类问题（类别 A、B 和 C），使用 One-Hot 编码可得: 

- 类别 A: [1, 0, 0]
- 类别 B: [0, 1, 0]
- 类别 C: [0, 0, 1]

**代码示例**

```python
import torch

# 假设我们有三个类别: 0, 1, 2
num_classes = 3

# 样本标签
labels = torch.tensor([0, 2, 1, 0])

# 将标签转换为 One-Hot 编码
one_hot_labels = torch.nn.functional.one_hot(labels, num_classes)

print("Labels:")
print(labels)
print("\nOne-Hot Encoded Labels:")
print(one_hot_labels)
```

**输出**：

```sql
Labels:
tensor([0, 2, 1, 0])

One-Hot Encoded Labels:
tensor([[1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]])
```

**解释**：

1. **Labels**: `[0, 2, 1, 0]` 是我们初始的类别标签。
2. **One-Hot Encoded Labels**: `[[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]` 是将标签转换为 One-Hot 编码后的结果。每个向量中只有一个位置的值为 1（One-Hot）。

### 类别不是整数怎么办？

看了代码示例，可能会有一个疑问: 类别大多不会是整数而是字符，应该怎么编码？或许你心中已经有了一个很直白的答案: 那就做一个映射，将类别用整数编码，然后再将这些整数标签转换为 One-Hot 编码。

的确可以这样。

**代码示例**

```python
import torch

# 类别映射: A -> 0, B -> 1, C -> 2
category_map = {'A': 0, 'B': 1, 'C': 2}

# 样本类别标签
labels = ['A', 'C', 'B', 'A']

# 将类别标签转换为整数标签
integer_labels = torch.tensor([category_map[label] for label in labels])

# 将整数标签转换为 One-Hot 编码
num_classes = len(category_map)
one_hot_labels = torch.nn.functional.one_hot(integer_labels, num_classes)

print("Labels:")
print(labels)
print("\nInteger Labels:")
print(integer_labels)
print("\nOne-Hot Encoded Labels:")
print(one_hot_labels)
```

**输出**：

```sql
Labels:
['A', 'C', 'B', 'A']

Integer Labels:
tensor([0, 2, 1, 0])

One-Hot Encoded Labels:
tensor([[1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]])
```

**解释**：

1. **Labels**: `['A', 'C', 'B', 'A']` 是我们初始的类别标签。
2. **Integer Labels**: `[0, 2, 1, 0]` 是将类别标签映射到整数后的结果。`A` 对应 0，`B` 对应 1，`C` 对应 2。
3. **One-Hot Encoded Labels**: `[[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]` 是将整数标签转换为 One-Hot 编码后的结果。每个向量中只有一个位置的值为 1，表示该样本的类别，其余位置的值为 0。

# nn.CrossEntropyLoss() 交叉熵损失

> `torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0)`
>
> This criterion computes the cross entropy loss between input logits and target.
>
> 该函数计算输入 logits 和目标之间的交叉熵损失。

## 参数

- **weight** (Tensor, 可选): 一个形状为 $(C)$ 的张量，表示每个类别的权重。如果提供了这个参数，损失函数会根据类别的权重来调整各类别的损失，适用于类别不平衡的问题。默认值是 `None`。
- **size_average** (bool, 可选): 已弃用。如果 `reduction` 不是 `'none'`，则默认情况下损失是取平均（`True`）；否则，是求和（`False`）。默认值是 `None`。
- **ignore_index** (int, 可选): 如果指定了这个参数，则该类别的索引会被忽略，不会对损失和梯度产生影响。默认值是 `-100`。
- **reduce** (bool, 可选): 已弃用。请使用 `reduction` 参数。默认值是 `None`。
- **reduction** (str, 可选): 指定应用于输出的归约方式。可选值为 `'none'`、`'mean'`、`'sum'`。`'none'` 表示不进行归约，`'mean'` 表示对所有样本的损失求平均，`'sum'` 表示对所有样本的损失求和。默认值是 `'mean'`。
- **label_smoothing** (float, 可选): 标签平滑值，范围在 [0.0, 1.0] 之间。默认值是 `0.0`。标签平滑是一种正则化技术，通过在真实标签上添加一定程度的平滑来避免过拟合。

## 数学公式

> 附录部分会验证下述公式和代码的一致性。

假设有 $N$ 个样本，每个样本属于 $C$ 个类别之一。对于第 $i$ 个样本，它的真实类别标签为 $y_i$, 模型的输出 logits 为 $`\mathbf{x}_i = (x_{i1}, x_{i2}, \ldots, x_{iC})`$, 其中 $x_{ic}$ 表示第 $i$ 个样本在第 $c$ 类别上的原始输出分数（logits）。

交叉熵损失的计算步骤如下: 

1. **Softmax 函数**: 

   对 logits 进行 softmax 操作，将其转换为概率分布: 

   $`p_{ic} = \frac{\exp(x_{ic})}{\sum_{j=1}^{C} \exp(x_{ij})}`$

   其中 $p_{ic}$ 表示第 $i$ 个样本属于第 $c$ 类别的预测概率。

2. **负对数似然（Negative Log-Likelihood）**: 

   计算负对数似然: 

   $`\ell_i = -\log(p_{iy_i})`$

   其中 $\ell_i$ 是第 $i$ 个样本的损失, $p_{iy_i}$ 表示第 $i$ 个样本在真实类别 $y_i$ 上的预测概率。

3. **总损失**: 

   计算所有样本的平均损失（ `reduction` 参数默认为 `'mean'`）: 

   $`\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \ell_i = \frac{1}{N} \sum_{i=1}^{N} -\log(p_{iy_i})`$
   
   如果 `reduction` 参数为 `'sum'`，总损失为所有样本损失的和: 

   $`\mathcal{L} = \sum_{i=1}^{N} \ell_i = \sum_{i=1}^{N} -\log(p_{iy_i})`$
   
   如果 `reduction` 参数为 `'none'`，则返回每个样本的损失 $\ell_i$ 组成的张量: 
   
   $`\mathcal{L} = [\ell_1, \ell_2, \ldots, \ell_N] = [-\log(p_{1 y_1}), -\log(p_{2 y_2}), \ldots, -\log(p_{N y_N})]`$

### 带权重的公式（weight）

如果指定了类别权重 $\mathbf{w} = (w_1, w_2, \ldots, w_C)$, 则每个样本的损失会根据其真实类别的权重进行调整：

$$
\ell_i = w_{y_i} \cdot (-\log(p_{iy_i}))
$$

其中 $w_{y_i}$ 是第 $i$ 个样本真实类别的权重。

**总损失**: 

1. `reduction='mean'`：

    $`
    \mathcal{L} = \frac{\sum_{i=1}^{N} \ell_i}{\sum_{i=1}^{N} w_{y_i}} =\frac{\sum_{i=1}^{N} w_{y_i} \cdot (-\log(p_{iy_i}))}{\sum_{i=1}^{N} w_{y_i}}
    `$
    
    当所有类别的权重均为 1 时，分母就是样本数量 $N$。

2. `reduction='sum'`
   
   $`
   \mathcal{L} = \sum_{i=1}^{N} w_{y_i} \cdot (-\log(p_{iy_i}))
   `$

3.  `reduction='none'`
   
    $`
    \mathcal{L} = [w_{y_1} \cdot (-\log(p_{1 y_1})), \ldots, w_{y_N} \cdot (-\log(p_{N y_N}))]
    `$

### 标签平滑（label_smoothing）

如果标签平滑（label smoothing）参数 $\alpha$ 被启用，目标标签 $\mathbf{y}_i$ 会被平滑处理: 

$$
\mathbf{y}_i' = (1 - \alpha) \cdot \mathbf{y}_i + \frac{\alpha}{C}
$$

其中， $\mathbf{y}_i$ 是原始的 one-hot 编码目标标签, $\mathbf{y}_i'$ 是平滑后的标签。

样本损失会相应调整: 

$$
\ell_i = - \sum_{c=1}^{C} y_{ic}' \cdot \log(p_{ic})
$$

其中， $y_{ic}$ 是第 $i$ 个样本在第 $c$ 类别上的标签，为原标签 $y_i$ 经过 one-hot 编码后 $`\mathbf{y}_i`$ 中的值。对于一个 one-hot 编码标签向量, $`y_{ic}`$ 在样本属于类别 $c$ 时为 1，否则为 0。

### 同时带权重和标签平滑

将权重和标签平滑结合起来，样本损失函数的计算为：

$$
\ell_i = w_{y_i} \cdot \left( - \sum_{c=1}^{C} y_{i c}' \cdot \log(p_{i c}) \right)
$$

其中, $y_{i c}' = (1 - \alpha) y_{i c} + \frac{\alpha}{C}$。

损失（`reduction='mean'`）为：

$$
\mathcal{L} = \frac{\sum_{i=1}^{N} w_{y_i} \cdot \left( - \sum_{c=1}^{C} y_{i c}' \cdot \log(p_{i c}) \right)}{\sum_{i=1}^{N} w_{y_i}}
$$



## 要点

1. `nn.CrossEntropyLoss()` 接受的输入是 **logits**，这说明分类的输出不需要提前经过 softmax。如果提前经过 softmax，则需要使用 `nn.NLLLoss()`（负对数似然损失）。

   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   
   # 定义输入和目标标签
   logits = torch.tensor([[2.0, 0.5], [0.5, 2.0]])  # 未经过 softmax 的 logits
   target = torch.tensor([0, 1])  # 目标标签
   
   # 使用 nn.CrossEntropyLoss 计算损失（接受 logits）
   criterion_ce = nn.CrossEntropyLoss()
   loss_ce = criterion_ce(logits, target)
   
   # 使用 softmax 后再使用 nn.NLLLoss 计算损失
   log_probs = F.log_softmax(logits, dim=1)
   criterion_nll = nn.NLLLoss()
   loss_nll = criterion_nll(log_probs, target)
   
   print(f"Loss using nn.CrossEntropyLoss: {loss_ce.item()}")
   print(f"Loss using softmax + nn.NLLLoss: {loss_nll.item()}")
   
   # 验证两者是否相等
   assert torch.allclose(loss_ce, loss_nll), "The losses are not equal, which indicates a mistake in the assumption."
   print("The losses are equal, indicating that nn.CrossEntropyLoss internally applies softmax.")
   ```

   **输出**：

   ```sql
   Loss using nn.CrossEntropyLoss: 0.2014133334159851
   Loss using softmax + nn.NLLLoss: 0.2014133334159851
   The losses are equal, indicating that nn.CrossEntropyLoss internally applies softmax.
   ```

   **拓展:  F.log_softmax()**

   `F.log_softmax` 等价于先应用 `softmax` 激活函数，然后对结果取对数 log()。它是将 `softmax` 和 `log` 这两个操作结合在一起，以提高数值稳定性和计算效率。具体的数学定义如下: 

   $`\text{log\_softmax}(x_i) = \log\left(\text{softmax}(x_i)\right) = \log\left(\frac{\exp(x_i)}{\sum_j \exp(x_j)}\right) = x_i - \log\left(\sum_j \exp(x_j)\right)`$

   在代码中，`F.log_softmax` 的等价操作可以用以下步骤实现: 

   1. 计算 `softmax`。
   2. 计算 `softmax` 的结果的对数。

   ```python
   import torch
   import torch.nn.functional as F
   
   # 定义输入 logits
   logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]])
   
   # 计算 log_softmax
   log_softmax_result = F.log_softmax(logits, dim=1)
   
   # 分开计算 softmax 和 log
   softmax_result = F.softmax(logits, dim=1)
   log_result = torch.log(softmax_result)
   
   print("Logits:")
   print(logits)
   
   print("\nLog softmax (using F.log_softmax):")
   print(log_softmax_result)
   
   print("\nSoftmax result:")
   print(softmax_result)
   
   print("\nLog of softmax result:")
   print(log_result)
   
   # 验证两者是否相等
   assert torch.allclose(log_softmax_result, log_result), "The results are not equal."
   print("\nThe results are equal, indicating that F.log_softmax is equivalent to softmax followed by log.")
   ```

   **输出**：

   ```python
   Logits:
   tensor([[2.0000, 1.0000, 0.1000],
           [1.0000, 3.0000, 0.2000]])
   
   Log softmax (using F.log_softmax):
   tensor([[-0.4170, -1.4170, -2.3170],
           [-2.1791, -0.1791, -2.9791]])
   
   Softmax result:
   tensor([[0.6590, 0.2424, 0.0986],
           [0.1131, 0.8360, 0.0508]])
   
   Log of softmax result:
   tensor([[-0.4170, -1.4170, -2.3170],
           [-2.1791, -0.1791, -2.9791]])
   
   The results are equal, indicating that F.log_softmax is equivalent to softmax followed by log.
   ```

   从结果中可以看到 `F.log_softmax` 的结果等价于先计算 softmax 再取对数。

2. `nn.CrossEntropyLoss()` 实际上默认（reduction='mean'）计算的是每个样本的**平均损失**，已经做了归一化处理，所以不需要对得到的结果进一步除以 batch_size 或其他某个数，除非是用作 loss_weight。下面是一个简单的例子: 

   ```python
   import torch
   import torch.nn as nn
   
   # 定义损失函数
   criterion = nn.CrossEntropyLoss()
   
   # 定义输入和目标标签
   input1 = torch.tensor([[2.0, 0.5], [0.5, 2.0]], requires_grad=True)  # 批量大小为 2
   target1 = torch.tensor([0, 1])  # 对应的目标标签
   
   input2 = torch.tensor([[2.0, 0.5], [0.5, 2.0], [2.0, 0.5], [0.5, 2.0]], requires_grad=True)  # 批量大小为 4
   target2 = torch.tensor([0, 1, 0, 1])  # 对应的目标标签
   
   # 计算损失
   loss1 = criterion(input1, target1)
   loss2 = criterion(input2, target2)
   
   print(f"Loss with batch size 2: {loss1.item()}")
   print(f"Loss with batch size 4: {loss2.item()}")
   ```
   
   **输出**：

   ```python
   Loss with batch size 2: 0.2014133334159851
   Loss with batch size 4: 0.2014133334159851
   ```

   可以看到这里的 `input2` 实际上等价于 `torch.cat([input1, input1], dim=0)`，`target2` 等价于 `torch.cat([target1, target1], dim=0)`，简单拓展了 batch_size 大小但最终的 Loss 没变，这也就验证了之前的说法。

3. 目标标签  `target` 期望两种格式: 

   - **类别索引**: 类别的整数索引，而不是 one-hot 编码。范围在 $[0, C)$ 之间，其中 $C$​ 是类别数。如果指定了 `ignore_index`，则该类别索引也会被接受（即便可能不在类别范围内）
     使用示例: 

     ```python
     # Example of target with class indices
     import torch
     import torch.nn as nn
     
     loss = nn.CrossEntropyLoss()
     input = torch.randn(3, 5, requires_grad=True)
     target = torch.empty(3, dtype=torch.long).random_(5)
     output = loss(input, target)
     output.backward()
     ```

   - **类别概率**: 类别的概率分布，适用于需要每个批次项有多个类别标签的情况，如标签平滑等。
     使用示例: 

     ```python
     # Example of target with class probabilities
     import torch
     import torch.nn as nn
     
     loss = nn.CrossEntropyLoss()
     input = torch.randn(3, 5, requires_grad=True)
     target = torch.randn(3, 5).softmax(dim=1)
     output = loss(input, target)
     output.backward()
     ```

   
   > The performance of this criterion is generally better when target contains **class indices**, as this allows for optimized computation. Consider providing target as **class probabilities** only when a single class label per minibatch item is too **restrictive**.
   >
   > 通常情况下，当目标为**类别索引**时，该函数的性能更好，因为这样可以进行优化计算。只有在每个批次项的单一类别标签过于**限制**时，才考虑使用**类别概率**。
   

# 附录

> 用于验证数学公式和函数实际运行的一致性

```python
import torch
import torch.nn.functional as F

# 假设有两个样本，每个样本有三个类别
logits = torch.tensor([[1.5, 2.0, 0.5], [1.0, 0.5, 2.5]], requires_grad=True)
targets = torch.tensor([1, 2])

# 根据公式实现 softmax
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim=1, keepdim=True)

# 根据公式实现 log-softmax
def log_softmax(x):
    return x - torch.log(torch.exp(x).sum(dim=1, keepdim=True))

# 根据公式实现负对数似然损失（NLLLoss）
def nll_loss(log_probs, targets):
    N = log_probs.size(0)
    return -log_probs[range(N), targets].mean()

# 根据公式实现交叉熵损失
def custom_cross_entropy(logits, targets):
    log_probs = log_softmax(logits)
    return nll_loss(log_probs, targets)

# 使用 PyTorch 计算交叉熵损失
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
loss_torch = criterion(logits, targets)

# 使用根据公式实现的交叉熵损失
loss_custom = custom_cross_entropy(logits, targets)

# 打印结果
print("PyTorch 计算的交叉熵损失:", loss_torch.item())
print("根据公式实现的交叉熵损失:", loss_custom.item())

# 验证结果是否相等
assert torch.isclose(loss_torch, loss_custom), "数学公式验证失败"

# 带权重的交叉熵损失
weights = torch.tensor([0.7, 0.2, 0.1])
criterion_weighted = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
loss_weighted_torch = criterion_weighted(logits, targets)

# 根据公式实现带权重的交叉熵损失
def custom_weighted_cross_entropy(logits, targets, weights):
    log_probs = log_softmax(logits)
    N = logits.size(0)
    weighted_loss = -log_probs[range(N), targets] * weights[targets]
    return weighted_loss.sum() / weights[targets].sum()

loss_weighted_custom = custom_weighted_cross_entropy(logits, targets, weights)

# 打印结果
print("PyTorch 计算的带权重的交叉熵损失:", loss_weighted_torch.item())
print("根据公式实现的带权重的交叉熵损失:", loss_weighted_custom.item())

# 验证结果是否相等
assert torch.isclose(loss_weighted_torch, loss_weighted_custom, atol=1e-6), "带权重的数学公式验证失败"

# 标签平滑的交叉熵损失
alpha = 0.1
criterion_label_smoothing = torch.nn.CrossEntropyLoss(label_smoothing=alpha, reduction='mean')
loss_label_smoothing_torch = criterion_label_smoothing(logits, targets)

# 根据公式实现标签平滑的交叉熵损失
def custom_label_smoothing_cross_entropy(logits, targets, alpha):
    N, C = logits.size()
    log_probs = log_softmax(logits)
    one_hot = torch.zeros_like(log_probs).scatter(1, targets.view(-1, 1), 1)
    smooth_targets = (1 - alpha) * one_hot + alpha / C
    loss = - (smooth_targets * log_probs).sum(dim=1).mean()
    return loss

loss_label_smoothing_custom = custom_label_smoothing_cross_entropy(logits, targets, alpha)

# 打印结果
print("PyTorch 计算的标签平滑的交叉熵损失:", loss_label_smoothing_torch.item())
print("根据公式实现的标签平滑的交叉熵损失:", loss_label_smoothing_custom.item())

# 验证结果是否相等
assert torch.isclose(loss_label_smoothing_torch, loss_label_smoothing_custom, atol=1e-6), "标签平滑的数学公式验证失败"

# 同时带权重和标签平滑的交叉熵损失
criterion_both = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=alpha, reduction='mean')
loss_both_torch = criterion_both(logits, targets)

# 根据公式实现同时带权重和标签平滑的交叉熵损失
def custom_both_cross_entropy(logits, targets, weights, alpha):
    N, C = logits.size()
    log_probs = log_softmax(logits)
    
    # 创建目标的 one-hot 编码
    one_hot = torch.zeros_like(log_probs).scatter(1, targets.view(-1, 1), 1)
    
    # 应用标签平滑
    smooth_targets = (1 - alpha) * one_hot + alpha / C
    
    # 将类别权重应用到平滑后的目标上
    # weights 的形状为 (C,)
    weighted_smooth_targets = smooth_targets * weights  # 形状为 (N, C)
    
    # 计算加权的损失
    weighted_loss = - (weighted_smooth_targets * log_probs).sum(dim=1)  # 形状为 (N,)
    
    # 计算平均损失
    return weighted_loss.sum() / weights[targets].sum()

loss_both_custom = custom_both_cross_entropy(logits, targets, weights, alpha)

# 打印结果
print("PyTorch 计算的同时带权重和标签平滑的交叉熵损失:", loss_both_torch.item())
print("根据公式实现的同时带权重和标签平滑的交叉熵损失:", loss_both_custom.item())

# 验证结果是否相等
assert torch.isclose(loss_both_torch, loss_both_custom, atol=1e-6), "同时带权重和标签平滑的数学公式验证失败"

```

**输出**：

```python
PyTorch 计算的交叉熵损失: 0.45524317026138306
根据公式实现的交叉熵损失: 0.4552431106567383
PyTorch 计算的带权重的交叉熵损失: 0.5048722624778748
根据公式实现的带权重的交叉熵损失: 0.50487220287323
PyTorch 计算的标签平滑的交叉熵损失: 0.5469098091125488
根据公式实现的标签平滑的交叉熵损失: 0.5469098091125488
PyTorch 计算的同时带权重和标签平滑的交叉熵损失: 0.7722168564796448
根据公式实现的同时带权重和标签平滑的交叉熵损失: 0.772216796875
```

输出没有抛出 AssertionError，验证通过。

# 参考链接

[CrossEntropyLoss - Docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
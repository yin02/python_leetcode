
## 二分查找的伪代码
```python
红蓝边界
def binary（）：
    l＝-1
    r = N
    while l +1 != r:
        m= (l+r)//2
        if ISBLUE(m):
            l = m
        else:
            r = m
    return l or r
```

### 注意点

- **红蓝边界初始化**:
  - 初始边界设置在数组范围外，即 `l = -1` 和 `r = N`。这样可以避免整个数组都是红色或蓝色时出现错误。
  - 这样可以确保算法在各种情况下都能正确执行。
  - m 的min。一定是l = -1 和r=1 如果能进入循环体的话。
  - 要不要写成l = m+1 不要！！！细节容易出错，不是红蓝边界可以


- **循环条件**:
  - 循环体的条件为 `l + 1 != r`，当 `l + 1` 等于 `r` 时，循环结束。
  - 这种方式能确保边界逐渐收缩并最终定位正确，避免死循环。sd
 - 根据实际情况来反回l或者r
 -  **返回值选择**:
  - 根据问题的实际需求，选择返回 `l` 或 `r`。通常，查找第一个满足条件的元素时返回 `r`，查找最后一个满足条件的元素时返回 `l`。



### 示例思路

- **左蓝右红的边界收缩**:
  - 设想红蓝边界的收缩过程。寻找第一个大于等于5的元素时，`ISBLUE(<5)` 返回 `r`。
  - 主要是想象 满足条件的应该是哪半部分，左半部分还是有半部分

- **查找示例**:
  - 第一个大于等于5的元素: `ISBLUE(<5)` 返回 `r`。
  - 最后一个小于5的元素: `ISBLUE(<5)` 返回 `l`。
  - 第一个大于5的元素: `ISBLUE(<=5)` 返回 `r`。
  - 最后一个小于等于5的元素: `ISBLUE(<=5)` 返回 `l`。



### 设计思路

1. **上下限设定**:
   - 通常可以为问题自己假设定上限和下限，从而在一定条件下找到最佳解。
2. **最小化操作次数**:
   - 对于需要最小化操作次数的问题，可以设置合理的边界，优化搜索过程。求最小操作次数，某些条件加减，但是**最大的右边**，和**左边**都可以设出来）
   
3. **最大吞吐量问题**:
   - 对于求最大吞吐量或最小值最大化的问题，可以使用类似的二分查找策略
4. **模板记忆**:
   - 处理最大值问题时，考虑**等于**条件，如果**中间值**等于目标，向**右收缩优**先考虑**右边界**。
   - 处理最小值问题时，等于时候，向左收缩优先考虑左边界。
  
5. **返回值选择**:
   - 在不确定返回值的情况下，如果问题是求满足某个条件的值，选择返回满足等于条件的那个值通常是正确的。


### Code

```python
红蓝边界


def binary_search():
    l = -1  # 初始左边界
    r = N   # 初始右边界
    while l + 1 != r:  # 循环直到左边界和右边界相遇
        m = (l + r) // 2  # 计算中间值
        if ISBLUE(m):     # 判断当前值是否满足蓝色条件
            l = m         # 如果是蓝色，将左边界更新为中间值
        else:
            r = m         # 如果是红色，将右边界更新为中间值
    return l  # 或者返回 r，根据实际情况选择

``` 


```py
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = -1, len(nums)

        while l+1 != r:
            m = l + ((r - l) // 2)  # (l + r) // 2 can lead to overflow
            if nums[m] > target:
                r = m
            elif nums[m] < target:
                l = m
            else:
                return m
        return -1

```

 
# 满足条件内的，二分查找最大最小值套路

## 向上取整

向上取整是将一个数值向上调整到最接近的整数。以下是两种实现向上取整的方法：

## 向上取整

向上取整是将一个数值向上调整到最接近的整数。以下是两种实现向上取整的方法：
⌈ a / b ⌉ = (a + b - 1) // b =  a/b+ (b-1)//b

### 1. 公式

向上取整的公式用于计算两个整数的商，并确保结果向上取整。公式如下：


### 2. Python 内置函数

在 Python 中，可以使用 `math.ceil()` 函数来实现向上取整。这个函数是 Python 标准库 `math` 模块的一部分，用于将浮点数向上取整到最接近的整数。

**示例**:
```python
import math

# 示例值
value = 7 / 3

# 使用 math.ceil() 向上取整
result = math.ceil(value)

print(result)  # 输出: 3
```
## 1. Maximum Throughput

### 问题描述
给定一个整数数组 `throughput` 和一个整数数组 `scaling_cost`，以及一个预算值 `budget`。你需要找到一个最大的 `throughput` 值，使得在这个 `throughput` 下，所有的缩放成本之和不超过给定的预算 `budget`。

### 输入
- 一个整数数组 `throughput`，表示每种资源的吞吐量。
- 一个整数数组 `scaling_cost`，表示每种资源的扩张的成本。
- 一个整数 `budget`，表示可用的总预算。

### 输出
- 一个整数，表示在给定预算下能达到的最大吞吐量。

### Solution
条件下寻找最大值，是两个部分！！！！要分解来： **1.满足条件** **2.寻找最大值**

1. **初始化边界**: 设置二分查找的左右边界，`left` 初始为 `throughput` 数组中的最小值，`right` 设置为一个较大的值（如 \(10^9 + 1\)）。
2. **二分查找**: 在 `left` 和 `right` 之间进行二分查找。计算中值 `mid` 并尝试用 `mid` 作为目标吞吐量。
3. **计算成本**: 对于每种资源，计算将其吞吐量增加到 `mid` 所需的成本。如果总成本超过了预算 `budget`，则更新右边界。
4. **更新结果**: 根据计算的成本更新 `left` 和 `right`，直到找到满足预算的**最大吞吐量**。**成本是条件**哦

### Code

```python

#满足条件下l = mid 向右寻找大的 最后return l，根据满足条件=的情况返回最后的 l 
def getMaximumThroughput(throughput, scaling_cost, budget):
    left = min(throughput)-1 #这样才符合红蓝边界
    right = 10 ** 9 + 1
    while left + 1 < right:
        #目标值！！！！吞吐量，想要最大
        mid = (left + right) // 2
        #目前的budget
        tmp = budget
        #遍历每一个服务
        for i in range(len(scaling_cost)):
            #需要的个数
            cnt = (mid + throughput[i] - 1) // throughput[i] #向上取整在英文中通常被称为 "ceiling" 或 "rounding up""切り上げ"（きりあげ, kiriage），或 "天井"（てんじょう, tenjō）
            #因为是扩张，所以要减掉自身
            tmp -= (cnt - 1) * scaling_cost[i]
            if tmp < 0:
                break
        #如果还在预算之类，找 mid =更大吞吐量 ，记住mid 寻找值的含义超级重要！！！！！！
        if tmp >= 0:#条件满足的情况tmp 》=0，找更大的吞吐量ｌ＝mid，用语言来帮助你
            left = mid
        else:
            right = mid
    return left

#如果和第二题一样 将tmp改成mid也是可以的
def getMaximumThroughput(throughput, scaling_cost, budget):
    left = min(throughput)
    right = 10**9 + 1
    while left + 1 < right:
        mid = (left + right) // 2
        tmp = mid  # 将 tmp 设置为 mid
        total_cost = 0  # 初始化总成本为0
        for i in range(len(scaling_cost)):
            cnt = (mid + throughput[i] - 1) // throughput[i]  # 向上取整
            cost = (cnt - 1) * scaling_cost[i]  # 计算需要的扩展成本
            total_cost += cost  # 累加总成本
            if total_cost > budget:  # 如果总成本超过预算
                break
        if total_cost <= budget:  # 如果总成本在预算以内
            left = mid  # 可以支持这个吞吐量，尝试更大的
        else:
            right = mid  # 不行，尝试更小的
    return left


```

## 2. Minimum Operation to Achieve Maximum Value

## 最小操作次数问题

### 问题描述

给定一个整数数组 `executionTime` 和两个整数 `x` 和 `y`，你的任务是找到最小的操作次数，使得所有的工作在数组 `executionTime` 长度的时间内被完成。这里的每个主要工作需要 `x` 秒执行，而其他工作需要 `y` 秒执行。你需要计算在指定时间内完成所有工作的最小操作次数。

### 输入

- 一个整数数组 `executionTime`，表示每个工作的执行时间。
- 两个整数 `x` 和 `y`，分别表示主要工作和其他工作的执行时间。

### 输出

- 一个整数，表示最小的操作次数。

### 解决方案

1. **初始化变量**: 定义左右指针 `left` 和 `right`，用来表示操作次数的范围。设置初始的 `left` 为 0，`right` 为 `sum(executionTime) + 1`（确保有足够的范围来包含所有可能的操作次数）。
2. **二分查找**: 使用二分查找来确定最小操作次数。
   - 计算中间值 `mid` 作为候选的操作次数。
   - 计算在 `mid` 次操作内是否能够完成所有工作。
3. **检查操作次数**: 对于每个工作，检查它是否可以在 `mid` 次操作内完成。
   - 如果主要工作所需的时间超出 `mid` 次操作的总时间，则需要调整操作次数。
4. **调整范围**: 如果当前操作次数不足以完成所有工作，则增加 `left`，否则减少 `right`。
5. **返回结果**: 最终的 `right` 值即为最小的操作次数。

### Code

```python
#因为要找最小操作次数，so 可以立即得出 满足条件下 r = m 想左寻找，并且 return 最后的r
def getMinimumOperation(executionTime, x, y):
    left = 0#因为至少1次，所以设置为0
    right = sum(executionTime) + 1
    while left + 1 != right:
        mid = (left + right) // 2
        tmp = mid #直接就是次数，当前次数
        for num in executionTime:
            #如果当前操作次数下已经满足
            if num <= y * mid:
                continue#跳过这个情况，进入下一个循环。
            else:
                val = num - y * mid #剩余的量
                cnt = (val + x - y - 1) // (x - y) #向上取整需要的次数
                tmp -= cnt
            if tmp < 0:
                break
        if tmp < 0:
            left = mid
        else:
            right = mid
    return right
```

<font color="red" size="4">l+1 != r 常错，不是l =r 是l+1 != r。</font>

当找到最小的满足条件的操作次数时，返回 right。
返回 `right` 是因为在二分查找的过程中，`right` 最终会收敛到**最小满足条件的操作次数**。让我们深入分析一下为什么要返回 `right` 而不是 `left`。

### 1. 二分查找的基本逻辑

在这段代码中，我们通过二分查找不断缩小可能的操作次数范围。`left` 和 `right` 是表示操作次数的两个边界：

- `left` 表示我们已经确认**不能**完成任务的操作次数。
- `right` 表示我们已经确认**可以**完成任务的操作次数。

**目标**是找到最小的 `mid`，使得在 `mid` 次操作内能够完成所有任务。

### 2. 如何调整 `left` 和 `right`

在每次循环中，我们通过取 `mid = (left + right) // 2` 来假设操作次数为 `mid`，然后检查在 `mid` 次操作下，能否完成所有任务：

- 如果 **不能**在 `mid` 次操作内完成任务（`tmp < 0`），我们增加 `left`，即 `left = mid`，表示 `mid` 次操作不够，下一轮需要更多的操作次数。
  
- 如果 **能够**在 `mid` 次操作内完成任务（`tmp >= 0`），我们减少 `right`，即 `right = mid`，表示 `mid` 次操作是可行的，下一轮可能会找到更小的满足条件的次数。

### 3. 为什么返回 `right` 而不是 `left`

在二分查找结束时，`left + 1 == right` 是循环的终止条件。这时候：

- **`left`** 是我们最后一次尝试的不可行的操作次数（即操作次数不够）。
- **`right`** 是我们最后一次确认可以完成任务的操作次数，也是最小的可行次数。

### 总结：
- 我们返回 `right` 是因为 `right` 收敛到了最小的满足条件的操作次数，它是最小的能够完成任务的操作次数。
- `left` 表示的是不满足条件的最大值，而 `right` 是满足条件的最小值，当二分查找结束时，`right` 就是最终的答案。

如果你有其他问题或需要进一步解释，欢迎继续讨论！


### 插一个two pointer

根据图片中的问题描述，你的任务是设计一个算法，帮助 **TikRouter** 最小化传输数据包的次数。每次传输的总大小不能超过 **3.00 字节**。

### 问题分析
1. **输入**：一个浮点数数组 `packet_sizes`，表示每个数据包的大小。
2. **目标**：计算最少需要几次传输，确保每次传输的数据包总大小不超过 **3.00 字节**。
### 题目是 1.01-3 一个packet！！

### 改进思路
为了提高效率，可以采取以下的优化思路：

1. **将数据包按大小从大到小排序**：
   - 如果你先处理大的数据包，可以更容易地把剩下的小数据包组合到一起，避免浪费空间。
   
2. **双指针方法**：
   - 用两个指针分别指向排序后数组的开头（最大数据包）和结尾（最小数据包）。
   - 尝试将最大的包和最小的包组合在一起，看是否能够使它们的大小和小于等于 **3.00 字节**。
   - 如果能放在一起，则计为一次传输并同时移动两个指针（分别向内移动），否则单独传输最大的包并只移动左指针。

3. **贪心算法**：
   - 通过这种方法，可以确保每次传输都尽量装满数据包，从而减少总的传输次数。
   
### 优化算法伪代码

```python
def findMinimumTripsByTikRouter(packet_sizes):
    # 1. 将数据包按大小从大到小排序
    packet_sizes.sort(reverse=True)  
    left = 0
    right = len(packet_sizes) - 1
    trips = 0

    # 2. 使用双指针法
    while left <= right:
        if packet_sizes[left] + packet_sizes[right] <= 3.00:
            # 如果当前最大和最小的数据包可以放在一起
            right -= 1  # 移动右指针
        left += 1  # 移动左指针，单独传输最大的数据包
        trips += 1  # 计数传输次数

    return trips  # 返回最少传输次数
```

虽然很傻但是可以用二分法
```py
def findMinimumTripsByTikRouter(packet_sizes):
    packet_sizes.sort(reverse=True)  # 从大到小排序

    left = 0  # 不可能完成任务的边界
    right = len(packet_sizes) + 1  # 最多的传输次数加1，蓝色边界

    # 二分查找
    while left + 1 < right:
        mid = (left + right) // 2
        trips = 0
        l = 0
        r = len(packet_sizes) - 1
        
        # 检查是否可以在 mid 次传输内完成任务
        while l <= r:
            if packet_sizes[l] + packet_sizes[r] <= 3.00:
                r -= 1  # 尝试将最大和最小的数据包组合
            l += 1  # 每次传输至少需要传一个包
            trips += 1
            
            if trips > mid:  # 如果超过了 mid 次传输，无法完成
                break
        
        if trips > mid:  # 当前 mid 次传输不行，需要更多次
            left = mid
        else:  # 当前 mid 次传输可以完成，尝试减少次数
            right = mid

    return right  # 返回最少的传输次数
```

### 解释：
1. **排序步骤**：先将数据包按大小从大到小排序，方便后续每次都优先选择最大的包处理。
2. **双指针法**：用 `left` 指向最大的包，`right` 指向最小的包，检查能否组合在一起。如果可以，说明在一个传输中可以传送两个包，否则仅传输一个大包。每次传输后调整指针。
3. **传输计数**：每次传输都增加一次计数，直到所有数据包都处理完。

### 为什么这个方法有效？
- **时间复杂度降低**：通过排序和双指针技术，你只需要对每个数据包检查一次，这比逐个尝试组合不同包的暴力方法快得多。
- **贪心策略**：贪心地尽量组合大小合适的数据包，减少浪费的空间，从而减少传输次数。




# 当问题翻译成找到什么的时候都可以用二分法或者two pointer来试试

## Minimum Eating Speed

### 问题描述

给定一个整数数组 `piles`，每个元素代表一堆香蕉的数量。你需要在 `h` 小时内吃完所有的香蕉。你每小时可以选择一种固定的速度 `k`（即每小时吃 `k` 根香蕉）。请计算并返回能够在 `h` 小时内吃完所有香蕉的最小速度 `k`。

### 输入

- 一个整数数组 `piles`，表示每堆香蕉的数量。
- 一个整数 `h`，表示你有 `h` 小时来吃完所有的香蕉。

### 输出

- 一个整数，表示能在 `h` 小时内吃完所有香蕉的最小速度 `k`。

### 解决方案

1. **初始化变量**:
   - 定义左右指针 `l` 和 `r`，用于表示吃香蕉速度的范围。
   - 设置初始的 `l` 为 0，`r` 为 `max(piles) + 1`，即吃香蕉的最小速度从0开始，最大速度是最大一堆香蕉的数量加1。

2. **二分查找**:
   - 使用二分查找来确定最小的吃香蕉速度。
   - 计算中间值 `m` 作为候选的吃香蕉速度。
   - 计算在速度 `m` 下吃完所有香蕉堆所需的总时间 `totalTime`。

3. **检查速度**:
   - 对于每堆香蕉，检查在速度 `m` 下是否能在 `h` 小时内吃完所有香蕉。
   - 如果 `totalTime` 小于等于 `h`，则尝试减少速度 `r`。
   - 否则，增加速度 `l`。

4. **调整范围**:
   - 如果 `totalTime` 不超过 `h` 小时，则说明速度 `m` 可能太大，需要缩小右边界 `r`。
   - 如果 `totalTime` 超过了 `h` 小时，则速度 `m` 太慢，需要增加左边界 `l`。

5. **返回结果**:
   - 最终的 `r` 值即为能够在 `h` 小时内吃完所有香蕉的最小速度 `k`。

### Code

```python
class Solution:
    
    def minEatingSpeed(self, piles, h):
        l, r = 0, max(piles) + 1  # 初始化左右边界，l 为 0，r 为香蕉堆的最大值加 1
        while l + 1 != r:  # 当 l 和 r 之间只剩下一个值时停止循环
            m = (l + r) // 2  # 计算中间值 m
            totalTime = 0  # 初始化总时间
            for p in piles:
                totalTime += math.ceil(float(p) / m)  # 计算速度为 m 时吃完当前香蕉堆所需的时间
            if totalTime <= h:  # 如果在 h 小时内能吃完所有香蕉
                r = m  #
            else:
                l = m  
        return r  # 返回最小的吃香蕉的速度
```


## Search Insert Position

### 问题描述

给定一个排序数组 `nums` 和一个目标值 `target`，请找到目标值在数组中的插入位置。如果目标值存在于数组中，返回其索引。如果目标值不存在于数组中，返回它应该被插入的位置的索引。

### 输入

- 一个整数数组 `nums`，该数组已经按升序排列。
- 一个整数 `target`，表示要查找的目标值。

### 输出

- 一个整数，表示目标值的插入位置索引。

### 解决方案

1. **初始化变量**:
   - 定义左右指针 `l` 和 `r`，分别表示搜索范围的左边界和右边界。
   - 初始时，`l` 设置为 -1，`r` 设置为数组的长度 `len(nums)`。

2. **二分查找**:
   - 使用二分查找来确定目标值的位置或插入点。
   - 计算中间值 `m` 作为候选位置，并检查 `nums[m]` 是否等于目标值。

3. **检查条件**:
   - 如果 `nums[m] == target`，则直接返回索引 `m`。
   - 如果 `nums[m] > target`，则说明目标值应该在左半部分，更新右边界 `r = m`。
   - 如果 `nums[m] < target`，则说明目标值在右半部分或需要插入到右边，更新左边界 `l = m`。

4. **调整范围**:
   - 不断缩小搜索范围，直到 `l + 1 == r`。

5. **返回结果**:
   - 当循环结束时，`r` 即为目标值的插入位置索引。

### Code

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l, r = -1, len(nums)  # 右边的边界是len（num） 不要加1 不然就出错了， 这个是0 based
        while l + 1 != r:  
            m = (l + r) // 2  
            if nums[m] == target:
                return m 
            elif nums[m] > target:
                r = m  # 目标值在左半部分，更新右边界 r
            else:
                l = m  # 目标值在右半部分，更新左边界 l
        return r  # 返回目标值的插入位置索引

```

## Guess Number Higher or Lower

### 问题描述

你正在玩一个猜数字的游戏。游戏规则如下：

有一个秘密数字 `n`，这个数字位于 1 和 `n` 之间。你的目标是通过调用 `guess(num)` 函数来猜测这个数字。`guess(num)` 会返回以下三个值之一：

- `-1`：表示你猜的数字比秘密数字大。
- `1`：表示你猜的数字比秘密数字小。
- `0`：表示你猜的数字恰好是秘密数字。

请实现一个函数 `guessNumber(int n)` 来猜测秘密数字，并返回该数字。

### 输入

- 一个整数 `n`，表示数字的上限，猜测的数字在 `1` 到 `n` 之间。

### 输出

- 返回猜中的数字。

### 解决方案

1. **初始化变量**:
   - 定义左右边界 `l` 和 `r`，分别表示搜索范围的左边界和右边界。
   - 初始时，`l` 设置为 `0`，`r` 设置为 `n + 1`，这样可以确保二分查找时的范围完全覆盖 `1` 到 `n`。

2. **二分查找**:
   - 通过二分查找逐步缩小范围，直到找到秘密数字。
   - 计算中间值 `mid`，并调用 `guess(mid)` 函数来确定下一步动作。

3. **检查条件**:
   - 如果 `guess(mid)` 返回 `-1`，表示猜的数字太大，将右边界缩小到 `mid`。
   - 如果 `guess(mid)` 返回 `1`，表示猜的数字太小，将左边界增大到 `mid`。
   - 如果 `guess(mid)` 返回 `0`，表示猜中了，直接返回 `mid`。

4. **返回结果**:
   - 当 `guess(mid)` 返回 `0` 时，返回 `mid` 作为猜测的结果。

### Code

```python
class Solution:
    def guessNumber(self, n: int) -> int:
        l, r = 0, n + 1  # 初始化边界，l 从 0 开始，r 为 n+1
        while True:  # 循环直到找到秘密数字
            mid = l + (r - l) // 2  # 计算中间值
            myGuess = guess(mid)
            if myGuess == -1:
                r = mid  # 如果猜测大于秘密数字，缩小右边界
            elif myGuess == 1:
                l = mid  # 如果猜测小于秘密数字，增大左边界
            else:
                return mid  # 如果猜中了，返回中间值
```





## Problem: Arranging Coins

You have `n` coins and you want to build a staircase with these coins. The staircase consists of `k` rows, where the `i-th` row has exactly `i` coins. The last row of the staircase may be incomplete.

Given the integer `n`, return the number of complete rows of the staircase you will be able to form.

### Example:

- **Input**: `n = 5`
- **Output**: `2`
  - **Explanation**: The coins can form the following rows:
    - Row 1: 1 coin
    - Row 2: 2 coins
    - Row 3: 2 coins (incomplete, so it's not counted)
  
- **Input**: `n = 8`
- **Output**: `3`
  - **Explanation**: The coins can form the following rows:
    - Row 1: 1 coin
    - Row 2: 2 coins
    - Row 3: 3 coins
    - Row 4: 2 coins (incomplete, so it's not counted)

## Solution Code:

```python
class Solution:
    def arrangeCoins(self, n: int) -> int:
        l, r = 0, n + 1
        while l + 1 != r:
            # mid is the target number of complete rows
            mid = (l + r) // 2
            coins = mid * (mid + 1) // 2
            # If the coins needed is greater than n, move to the left
            if coins > n:
                r = mid
            else:
                l = mid
        return l
```



## Problem: Valid Perfect Square

Given a positive integer `num`, write a function that returns `True` if `num` is a perfect square, otherwise return `False`.

### Example:

- **Input**: `num = 16`
- **Output**: `True`
  - **Explanation**: \( 4^2 = 16 \), so 16 is a perfect square.

- **Input**: `num = 14`
- **Output**: `False`
  - **Explanation**: There is no integer \( x \) such that \( x^2 = 14 \), so 14 is not a perfect square.

## Solution Code:

```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        l = 0
        r = num + 1
        while l + 1 != r:
            mid = (l + r) // 2
            if mid ** 2 == num:
                return True
            elif mid ** 2 > num:
                r = mid
            else:
                l = mid
        return False
```

## 题目: 单一非重复元素

给定一个排序数组，其中每个元素都恰好出现两次，只有一个元素出现一次。找出这个唯一的非重复元素。

### 输入:
- 一个整数数组 `nums`，其中数组长度为奇数 `2n + 1`，并且除了一个元素外，所有元素都恰好出现两次。

### 输出:
- 返回数组中唯一出现一次的元素。

### 示例:

# 示例 1
输入: nums = [1, 1, 2, 3, 3, 4, 4]
输出: 2

# 示例 2
输入: nums = [1, 1, 2, 2, 3]
输出: 3

```py
from typing import List

class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        l, r = -1, len(nums)
        
        while l + 1 != r:
            mid = l + (r - l) // 2
            
            if mid % 2 == 0:
                if mid + 1 < len(nums) and nums[mid] == nums[mid + 1]:
                    l = mid
                else:
                    r = mid
            else:
                if mid - 1 >= 0 and nums[mid] == nums[mid - 1]:
                    l = mid
                else:
                    r = mid
        
        return nums[r]

from typing import List

class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        # 初始化左右边界，l = -1, r = 数组长度，这个数组一定是个奇数2n+1
        l, r = -1, len(nums)
        
        # 当左右边界没有相邻时，继续二分查找
        while l + 1 != r:
            # 计算中间位置，使用(l + (r - l) // 2)防止溢出
            mid = l + (r - l) // 2
            
            #是代表mid左边的是偶数对，因为是0 based 
            #偶数对不代表左边一定是成对的
            #右边是奇数对，但是按照道理讲,左边成对，mid 和mid+1 也应该是成对的，不然左边就有问题
            if mid % 2 == 0:
                # 如果mid是偶数且mid和mid+1位置的元素相等,偶数说明左边的肯定是一对的
                if mid + 1 < len(nums) and nums[mid] == nums[mid + 1]:
                    l = mid  # 向右边寻找
                else:
                    r = mid  # 否则向左边寻找
            else:
                # 如果mid是奇数且mid和mid-1位置的元素相等
                if mid - 1 >= 0 and nums[mid] == nums[mid - 1]:
                    l = mid  #向右边边寻找
                else:
                    r = mid  # 向左边寻找
        #在二分查找的过程中，l 被更新为已经确定成对出现的元素，r 被更新为可能包含非重复元素的区间，而 
        # 返回单个元素所在的位置
        return nums[r]

```
### 解法2
```py
def singleNonDuplicate(self, nums: List[int]) -> int:
        return nums[bisect_left(range(len(nums) - 1), True, key=lambda x: nums[x] != nums[x ^ 1])]
```


[看懂这一行代码](../vo面试问题/有用的function或者易错.md#位运算-x--1-的详细解释和示例)

## 有lower bound，就可以看看有没有 upperbound 然后想到binary search



# 题目描述

给定一个包裹的重量列表 `weights` 和一个指定的天数 `days`，要求在 `days` 天内将所有包裹运输完毕。需要确定一个最小的运载能力，使得可以在 `days` 天内完成所有包裹的运输。

运输的规则是：
- 每一天你只能从左到右按顺序运输包裹，不能调换包裹的顺序。
- 在一天内，你可以选择运输一部分包裹，但不能运输超过船的最大运载能力。

你需要计算出一个最小的船的运载能力，使得可以在规定天数内完成所有包裹的运输。

## 输入

- `weights`：一个整数列表，表示每个包裹的重量。
- `days`：一个整数，表示规定的天数。

## 输出

- 返回一个整数，表示最小的船的运载能力。

## 示例

**输入：**

```python
class Solution:

    def shipWithinDays(self, weights: List[int], days: int) -> int:
        l,r = max(weights)-1, sum(weights)+1
        min_cap  = r
        def canship(cap):
            day, curcap = 1, cap
            for w in weights:
                if curcap - w < 0: 
                    day += 1
                    curcap = cap
                curcap -= w
            return day <= days
        while l+1 != r:
            cap = (l+r)//2
            if canship(cap):
                r = cap
            else:
                l = cap
        return r
```

<h1 style="color:red; font-size: 36px;">忘记 `l + 1` 的错误</h1> l+1 ！=r


# 162. Find Peak Element

给定一个整数数组 `nums`，找到其中的一个峰值元素，并返回其索引。数组中的元素满足以下条件：

1. `nums[i]` ≠ `nums[i + 1]`，对于所有有效的 `i`。
2. 一个峰值元素是指其值严格大于左右相邻值的元素。

你可以假设 `nums[-1] = nums[n] = -∞`，即数组在边界以外的元素都视为负无穷。

你的解法应该在 O(log n) 时间内完成。

## 输入

- `nums`：一个整数列表，表示要查找峰值的数组。

## 输出

- 返回一个整数，表示峰值元素的索引。

### 解法描述2

1. **如何向左向右边搜寻**：
   - 左边要找下降的，右边的话找右边比较大的向右向上升的 


```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        l,r = -1, len(nums)
        for i in range(len(nums)):
            mid = (l+r)//2
            if mid < len(nums) -1 and nums[mid] < nums[mid+1]:
                l= mid
            elif mid > 0 and nums[mid] < nums[mid-1]:
                r = mid
            # 如果以上两种情况都不满足，说明 mid 可能是峰值，跳出循环
            else:
                break
        return mid
```

## 题目：成功配对的数量

给定两个整数数组 `spells` 和 `potions`，分别表示法术的力量和药水的效力。同时给定一个整数 `success`，表示成功的最低阈值。一个法术与药水的配对被认为是成功的，当且仅当 `spell * potion >= success`。你需要返回一个整数数组，表示每个法术能形成的成功配对的数量。

### 示例：


输入:
spells = [10, 20, 30]
potions = [1, 2, 3, 4, 5]
success = 100

输出:
[3, 4, 5]

解释:
- 对于第一个法术 `10`，它可以与 `potions` 中 `3`，`4`，`5` 形成成功的配对。
- 对于第二个法术 `20`，它可以与 `potions` 中 `2`，`3`，`4`，`5` 形成成功的配对。
- 对于第三个法术 `30`，它可以与 `potions` 中 `1`，`2`，`3`，`4`，`5` 形成成功的配对。




```py
from typing import List

class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        potions.sort()  # 对potions进行排序
        pairs = []
        n = len(potions)
        
        for spell in spells:
            l = -1  # 红边界，表示不满足条件的部分
            r = n   # 蓝边界，表示满足条件的部分
            while l + 1 != r:
                m = (l + r) // 2
                if potions[m] * spell >= success:
                    r = m  # 缩小蓝边界
                else: 
                    l = m  # 扩大红边界
            pairs.append(n - r)  # r 指向第一个满足条件的位置
        
        return pairs



```


### 题目：最小化最大差异

给定一个整数数组 `nums` 和一个整数 `p`，任务是从 `nums` 中选择 `p` 对数对，使得每一对数对的差值的最大值尽可能的小。返回这个最小化后的最大差值。

### 示例：

```python
输入:
nums = [1, 3, 6, 19, 20]
p = 2

输出:
2

解释:
- 选择数对 (1, 3) 和 (19, 20)，它们的差值分别是 2 和 1。最大差值为 2，这是可能的最小值。
``` 

```py

class Solution:
    def minimizeMax(self, nums: List[int], p: int) -> int:
        
        # 辅助函数，用于判断在给定阈值 threshold 下是否可以找到 p 对数对，
        # 使得每一对数对的差值不超过 threshold
        def isValid(threshold):
            i, cnt = 0, 0  # i 是当前遍历的索引，cnt 记录符合条件的数对数量
            while i < len(nums) - 1:
                # 如果当前数对的差值小于或等于阈值
                if abs(nums[i] - nums[i + 1]) <= threshold:
                    cnt += 1  # 满足条件的数对数量加一
                    i += 2  # 跳过下一个元素，因为已经配对
                else:
                    i += 1  # 如果不满足条件，继续检查下一个元素
                if cnt == p:  # 如果已经找到了 p 对数对，返回 True
                    return True
            return False  # 如果遍历完所有元素后没有找到足够的数对，返回 False
        
        # 如果 p 为 0，则不需要任何数对，返回 0
        if p == 0: 
            return 0

        nums.sort()  # 先对数组进行排序，以便后续的处理
        l, r = -1, 10**9 + 1  # 初始化二分查找的左右边界
        res = 10**9  # 初始化结果为一个较大的值
        
        # 使用二分查找来寻找最小的最大差值
        while l + 1 != r:
            m = l + (r - l) // 2  # 计算中间值
            if isValid(m):  # 如果当前阈值 m 可以找到 p 对数对
                res = m  # 更新结果为当前的 m
                r = m  # 缩小搜索范围，尝试更小的阈值
            else:
                l = m  # 如果不能找到足够的数对，增大最小可能阈值
        return res  # 返回最终找到的最小的最大差值
```


### 题目：寻找旋转排序数组中的最小值

给定一个升序排序但旋转过的数组 `nums`，请你编写一个函数，返回数组中的最小值。数组中不存在重复元素。

数组被旋转的意思是将数组的某一部分移到数组的另一端。例如，数组 `[0,1,2,4,5,6,7]` 可能变成 `[4,5,6,7,0,1,2]`，即将前面的 `[4,5,6,7]` 移动到了数组的末尾。

### 输入描述

- `nums`: 一个旋转排序的整数数组，长度范围 `[1, 5000]`，数组中的元素均为唯一的整数。

### 输出描述

- 返回数组中的最小值。

### 示例

#### 示例 1:

```python
输入: nums = [3,4,5,1,2]
输出: 1
解释: 原数组为 [1,2,3,4,5]，旋转后为 [3,4,5,1,2]。最小值为 1。
```


```py
from typing import List

class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = -1, len(nums)  # 初始化左边区域和右边区域

        while l + 1 != r:
            mid = l + (r - l) // 2
            # 判断中间值是否在左边区域
            if nums[mid] <= nums[l]:
                r = mid  # 看右边区域
            else:
                l = mid  # 看左边区域

        return nums[r]  # r 即为最小元素所在的位置
```

## Search In Rotated Sorted Array ??

   



### 题目：搜索旋转排序数组 II

已知存在一个按非降序排列的整数数组 `nums` ，数组中的值可以是重复的。将数组旋转了 `k` 次（`k` 是非负整数），使得数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`，例如，数组 `[0,1,2,4,4,4,5,6,6,7]` 可能变为 `[4,5,6,6,7,0,1,2,4,4]`。

给你这个旋转后的数组 `nums` 和一个整数 `target`，请你判断 `target` 是否存在于数组中。如果存在，返回 `true`；否则，返回 `false`。

### 输入描述

- `nums`: 一个经过旋转的整数数组，长度范围为 `[1, 5000]`，其中的元素可以是重复的整数。
- `target`: 一个整数，表示需要查找的目标值。

### 输出描述

- 如果在数组 `nums` 中找到目标值 `target`，返回 `true`；否则返回 `false`。

### 示例

#### 示例 1:

```python
输入: nums = [2,5,6,0,0,1,2], target = 0
输出: true
```
### 思路

这段代码通过检查 `target` 的值与数组第一个元素 `nums[0]` 之间的关系，将问题分为两种情况来处理：

1. **目标值大于等于数组的第一个元素**：
   - 如果 `target >= nums[0]`，那么目标值可能出现在数组的左半部分（从数组的开始到旋转点之间）。代码通过从左到右遍历数组来查找目标值。
   - 当遍历到第一个大于 `target` 的元素时，可以立即停止，因为目标值不可能出现在后续元素中。

2. **目标值小于数组的第一个元素**：
   - 如果 `target < nums[0]`，那么目标值可能出现在数组的右半部分（从旋转点到数组的末尾之间）。代码通过从右到左遍历数组来查找目标值。
   - 当遍历到第一个小于 `target` 的元素时，可以立即停止，因为目标值不可能出现在前面的元素中。

### 代码实现

```python
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        if target >= nums[0]:
            # 如果目标值大于等于数组的第一个元素，可能在数组的左半部分
            for i in range(len(nums)):
                if nums[i] == target:
                    return True
                # 如果当前元素大于目标值，目标值不可能出现在后面
                if nums[i] > target:
                    return False
        else:
            # 如果目标值小于数组的第一个元素，可能在数组的右半部分
            for i in range(len(nums) - 1, -1, -1):
                if nums[i] == target:
                    return True
                # 如果当前元素小于目标值，目标值不可能出现在前面
                if nums[i] < target:
                    return False
        return False
```

``` py
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        # 如果数组的首尾元素相等且数组长度大于1，则删除数组开头的元素
        # 目的是为了去掉重复的元素，避免干扰二分查找的逻辑
        while nums[0] == nums[-1] and len(nums) > 1:
            del nums[0]
        
        # 初始化红蓝边界
        left = -1  # 红色区域的右边界，表示到目前为止未找到目标值的区域
        right = len(nums)  # 蓝色区域的左边界，表示可能包含目标值的区域
        
        # 开始二分查找的过程，循环直到红蓝边界相遇
        while left + 1 != right:
            mid = (left + right) // 2  # 计算中间位置
            
            # 如果中间值等于目标值，直接返回 True
            if nums[mid] == target:
                return True
            
            # 如果 mid 处的值大于或等于 nums[0]，说明 mid 处于左半部分
            if nums[mid] >= nums[0]:
                # 如果目标值在左半部分范围内，缩小蓝色区域,向左边搜索
                if nums[0] <= target < nums[mid]:
                    right = mid
                else:
                    # 否则，目标值在右半部分，扩大红色区域，向右边搜索
                    left = mid
            else:
                # 如果 mid 处的值小于 nums[0]，说明 mid 处于右半部分
                # 如果目标值在右半部分范围内，扩大红色区域，向右边搜索
                if nums[mid] < target <= nums[-1]:
                    left = mid
                else:
                    # 否则，目标值在左半部分，缩小蓝色区域
                    right = mid
        
        # 如果二分查找结束后仍未找到目标值，返回 False
        return False
```

## 题目：在排序数组中查找元素的第一个和最后一个位置

给定一个按照升序排列的整数数组 `nums` 和一个目标值 `target`，找出给定目标值在数组中的第一个和最后一个位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

### 输入描述

- `nums`: 一个按升序排列的整数数组，长度范围为 `[0, 10^5]`。
- `target`: 一个整数，表示需要查找的目标值。

### 输出描述

- 返回一个长度为 2 的列表 `[leftIdx, rightIdx]`，表示目标值在数组中的第一个和最后一个位置。
- 如果目标值不存在于数组中，返回 `[-1, -1]`。

### 示例

#### 示例 1:

```python
输入: nums = [5,7,7,8,8,10], target = 8
输出: [3, 4]
```
```py
from typing import List

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # 查找目标值在数组中的第一个位置
        def binarysearchLeft(nums, target):
            l = -1
            r = len(nums)
            while l + 1 != r:
                mid = l + (r - l) // 2
                if nums[mid] >= target:
                    r = mid  # 将蓝色区域收缩到左侧
                else:
                    l = mid  # 将红色区域扩大到右侧
            return r
        
        # 查找目标值在数组中的最后一个位置
        def binarysearchright(nums, target):
            l = -1
            r = len(nums)
            while l + 1 != r:
                mid = l + (r - l) // 2
                if nums[mid] <= target:
                    l = mid  # 将红色区域扩大到左侧
                else:
                    r = mid  # 将蓝色区域收缩到右侧
            return l
        
        leftIdx = binarysearchLeft(nums, target)
        rightIdx = binarysearchright(nums, target)

        # 判断找到的索引是否在有效范围内，并且值是否与目标值相等
        if leftIdx <= rightIdx and rightIdx < len(nums) and nums[leftIdx] == target and nums[rightIdx] == target:
            return [leftIdx, rightIdx]
        
        return [-1, -1]
```


## 题目：最大化移除字符使得目标子序列仍然存在

### 描述

给定一个字符串 `s` 和一个字符串 `p`，再给定一个整数数组 `removable`，其中 `removable[i]` 表示可以从 `s` 中移除的字符索引。

在不改变字符相对顺序的前提下，请你找出可以从 `s` 中移除的最大字符数 `k`，使得 `p` 仍然是 `s` 的子序列。返回 `k`。

### 输入

- `s`: 一个由小写字母组成的字符串。
- `p`: 一个由小写字母组成的字符串，是 `s` 的潜在子序列。
- `removable`: 一个整数数组，表示 `s` 中可移除字符的索引。

### 输出

- 返回 `k`，即可以移除的最大字符数，使得 `p` 仍然是 `s` 的子序列。

### 示例

#### 示例 1:

```python
输入:
s = "abcacb"
p = "ab"
removable = [3, 1, 0]

输出:
2

解释:
移除索引 [3,1] 的字符后，"acb" 仍然包含 "ab" 作为子序列。


removable[i] 是一个索引，表示我们要修改 s_list 中的某个字符的位置。
s_list 是字符串 s 的列表表示（因为字符串在 Python 中是不可变的，所以需要转换为列表以便修改）。
s_list[removable[i]] 就是我们需要修改的具体字符。
```

```py
from typing import List

class Solution:
    def check(self, s: str, p: str, removable: List[int], k: int) -> bool:
        # 将字符串转换为列表以便修改
        s_list = list(s)
        # 将可移除的字符标记为非字母（在这里模拟成大写，使用较大的 ASCII 值）
        for i in range(k):
            s_list[removable[i]] = chr(ord(s_list[removable[i]]) - ord("a"))  # 将字符转换为“非字母”状态
        
        j = 0
        # 遍历修改后的字符串，检查是否能匹配 p
        for i in range(len(s_list)):
            if s_list[i].islower() and s_list[i] == p[j]:
                j += 1
                if j == len(p):
                    return True
        return False

    def maximumRemovals(self, s: str, p: str, removable: List[int]) -> int:
        n = len(removable)  # 获取 removable 的长度
        l, r = -1, n + 1
        while l + 1 != r:
            mid = (l + r) // 2
            if self.check(s, p, removable, mid):
                l = mid
            else:
                r = mid
        return l
```


[binary继续刷题单](https://leetcode.cn/circle/discuss/9oZFK9/)
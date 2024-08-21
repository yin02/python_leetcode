

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
  - 这种方式能确保边界逐渐收缩并最终定位正确，避免死循环。
 - 根据实际情况来反回l或者r
 -  **返回值选择**:
  - 根据问题的实际需求，选择返回 `l` 或 `r`。通常，查找第一个满足条件的元素时返回 `r`，查找最后一个满足条件的元素时返回 `l`。



### 示例思路

- **左蓝右红的边界收缩**:
  - 设想红蓝边界的收缩过程。寻找第一个大于等于5的元素时，`ISBLUE(<5)` 返回 `r`。

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








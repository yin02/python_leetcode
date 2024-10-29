## 恰好装满 01 题型

# [2915. Length of the Longest Subsequence That Sums to Target](https://leetcode.com/problems/length-of-the-longest-subsequence-that-sums-to-target/description/)

You are given a **0-indexed**  array of integers `nums`, and an integer `target`.

Return the **length of the longest subsequence**  of `nums` that sums up to `target`. If no such subsequence exists, return `-1`.

A **subsequence**  is an array that can be derived from another array by deleting some or no elements without changing the order of the remaining elements.

**Example 1:** 

```
Input: nums = [1,2,3,4,5], target = 9
Output: 3
Explanation: There are 3 subsequences with a sum equal to 9: [4,5], [1,3,5], and [2,3,4]. The longest subsequences are [1,3,5], and [2,3,4]. Hence, the answer is 3.
```

**Example 2:** 

```
Input: nums = [4,1,3,2,1,5], target = 7
Output: 4
Explanation: There are 5 subsequences with a sum equal to 7: [4,3], [4,1,2], [4,2,1], [1,1,5], and [1,3,2,1]. The longest subsequence is [1,3,2,1]. Hence, the answer is 4.
```

**Example 3:** 

```
Input: nums = [1,1,5,4,5], target = 3
Output: -1
Explanation: It can be shown that nums has no subsequence that sums up to 3.
```

**Constraints:** 

- `1 <= nums.length <= 1000`
- `1 <= nums[i] <= 1000`
- `1 <= target <= 1000`
### 如何判断01背包
#### 目标: 找到一个子序列，使得它们的和不超过给定的 target。这实际上是一个容量限制问题，类似于在背包中放入物品以达到特定的和。


辨别一个问题是否属于 0/1 背包问题，可以根据以下几个准则进行判断：

#### 逐条准则分析：
1. **固定容量**: 存在最大容量 `6`。
2. **物品选择**: 存在多个物品，每个物品有对应的重量和价值。
3. **选择的二元性**: 每个物品只能选择一次，选择与不选择。
4. **优化目标**: 目标是最大化在容量限制下的总价值。
5. **状态转移**: 可以通过动态规划建立状态转移方程。



```py
class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # f[j]represents max value of substring given target
        # x is the current value
        # f[j] = max(f[j],f[j-x]+1) for state transfer
        f = [0] + [-inf] * target
        s = 0 # s is the possible maxium number which smaller than target
        for x in nums:
                s = min(target,s+x)
                for j in range(s,nums-1,-1):#所以j的话，你目的是更新状态，是要么更新所有包括target，要么是比target小的所有和的状态
                    f[j] = max(f[j],f[j-x]+1)
        return f[target] if f[target] >0 else -1


```


```py
from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # 初始化动态规划数组 f
        # f[j] 表示总和为 j 时，最长子序列的长度
        # 初始化 f[0] 为 0，表示总和为 0 时，子序列长度为 0
        # 其余元素初始化为负无穷，表示这些总和不可达
        f = [0] + [-float('inf')] * target

        # 用于追踪当前最大可以考虑的总和
        current_sum = 0

        # 遍历输入的 nums 列表
        for num in nums:
            # 更新当前可以考虑的最大总和，确保不超过 target，
# 使用 min(current_sum + num, target) 是为了限制 current_sum 的值，使其不超过 target。这样可以确保我们在计算动态规划数组时，只处理那些和在 target 范围内的状态
            current_sum = min(current_sum + num, target)

            # 从 current_sum 开始向下遍历到 num
            # 这里的 j！ 表示当前正在处理的总和！
            for j in range(current_sum, num - 1, -1):
                # 更新 f[j] 的值
                # f[j] 的值取决于 f[j - num] + 1（表示添加当前 num 后的长度）和 f[j] 之间的较大值
                # 如果 f[j - num] + 1 表示在加上当前 num 后的子序列长度更长，则更新 f[j]
                f[j] = max(f[j], f[j - num] + 1)
                # 手写 max 效率更高
                # if f[j] < f[j - x] + 1:
                #     f[j] = f[j - x] + 1



        # 检查 f[target] 的值
        # 如果 f[target] 大于 0，表示找到了符合条件的子序列，返回其长度
        # 否则返回 -1，表示没有找到符合条件的子序列
        return f[target] if f[target] > 0 else -1

```
```py
        # 检查 f[0] 的值
        if f[0] > 0:
            return f[0]  # 如果 f[0] 的值大于 0，返回其长度
        else:
            return -1  # 否则返回 -1，表示没有找到符合条件的子序列

```

# [416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/description/)

Given an integer array `nums`, return `true` if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or `false` otherwise.

**Example 1:** 

```
Input: nums = [1,5,11,5]
Output: true
Explanation: The array can be partitioned as [1, 5, 5] and [11].
```

**Example 2:** 

```
Input: nums = [1,2,3,5]
Output: false
Explanation: The array cannot be partitioned into equal sum subsets.
```

**Constraints:** 

- `1 <= nums.length <= 200`
- `1 <= nums[i] <= 100`

01背包的原因
1. **固定容量**: 问题要求判断是否可以将数组分割成两个子集，目标是达到数组总和的一半，形成固定的容量限制。

2. **物品选择**: 每个数组元素可以被视为一个物品，具有相应的“重量”（值）和“价值”（选择的意义）。

3. **选择的二元性**: 每个物品（数组元素）只能选择一次，符合 0/1 背包问题中物品选择的基本规则。

4. **优化目标**: 目标是最大化选取的物品总和，使其恰好等于目标和（即总和的一半），满足优化条件。

5. **状态转移**: 可以构建动态规划状态转移方程，根据选择与不选择当前元素来更新状态，从而验证是否能达到目标和，符合 0/1 背包的状态转移特征。

```py
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # sub1 = sum//2
        # 01 bc bound is sum//2
        # choose one number or not
        # could or not,恰好装满 的01
        #j is accumlator? == sum//2
        @cache
        def dfs(i,j):
            if i <0:
                return j ==0
            # legal j still got space, previous choose, not choose
            return j >= nums[i] and dfs(i-1,j-nums[i]) or dfs(i-1,j)

        s = sum(nums)
        # s is even
        return s %2 == 0 and dfs(len(nums)-1,s//2)


```





### 在递归函数中使用 `i < 0` 作为条件而不是 `i == 0`，

### 1. **递归的过程**

在这个问题中，递归是通过选择或不选择当前物品来进行的。`i` 代表当前考虑的物品的索引，递归的过程如下：

- 当 `i` 从 `len(nums) - 1` 开始递减，表示我们在考虑从最后一个物品到第一个物品。

### 2. **终止条件的意义**

- **`i < 0`**:
  - 这个条件表示我们已经考虑了所有物品，即没有更多的物品可以选择。
  - 在这种情况下，递归函数会检查当前目标和 `j` 是否为 0。如果 `j` 为 0，说明我们成功找到了一个组合，使得物品的总和恰好等于目标和。返回 `True`。
  - 例如，假设我们已经检查完所有的物品并且目标和仍未满足，如果此时 `j` 不为 0，说明没有可能的组合使得目标和成立。

- **`i == 0`**:
  - 如果我们只使用 `i == 0` 作为条件，可能会导致不完整的判断。例如，`i == 0` 只意味着我们已检查了第一个物品。
  - 当 `i` 递减到 `0` 时，***仍然需要考虑当前物品***是否能帮助达到目标和。因此，检查是否可以通过第一个物品来满足目标和的条件是必要的。

### 3. **代码逻辑的清晰性**

使用 `i < 0` 作为结束条件，可以更加清晰地表明何时完成了所有可能的选择，特别是在处理基于物品索引的递归中。

You are given an integer array `nums` and an integer `target`.

You want to build an **expression**  out of nums by adding one of the symbols `'+'` and `'-'` before each integer in nums and then concatenate all the integers.

- For example, if `nums = [2, 1]`, you can add a `'+'` before `2` and a `'-'` before `1` and concatenate them to build the expression `"+2-1"`.

Return the number of different **expressions**  that you can build, which evaluates to `target`.

**Example 1:** 

```
Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
```








```python
# 定义函数 solution
def solution(nums, target):
    # 将 target 加上所有数的和，使 target 转化为一个正数
    # 因为正负号可以互相抵消，因此增加总和不会影响最终结果
    # p是要变成target的正数，sum-p就是负数
        # P - (S - P) = Target
        # 2P = S + T
        # P = (S + T) / 2,这些正数必须是 2的倍数，转换成找到这些正数
 #这边把target 当成p，有点confusing，也可以命名是postive，找到p（另一个target的意思）
    target += sum(nums)
    
    # 如果 target 小于 0 或 target 不是偶数，直接返回 0
    # 这是因为如果 target 为奇数，无法用正负号相加得到
    if target < 0 or target % 2:
        return 0
    
    # 将 target 减半，转换为子集和问题，即我们需要找到一个子集的和为 target/2，就是找到题目要的target
    target //= 2

    # 如果 target == 0 并且有 nums == [] 的特殊情况
    if target == 0:
        return 1 if sum(nums) == 0 else 0
    # 获取 nums 数组的长度
    n = len(nums)
    
    # 初始化一个大小为 target + 1 的数组 f，表示和为 i 时的组合数
    # f[c] 表示和为 c 的子集数目
    f = [0] * (target + 1)
    
    # 设置初始状态，当和为 0 时，有且只有一种组合，即不选择任何元素
    #上面处理过如果target一开始就是
    f[0] = 1

    # 遍历 nums 数组中的每个数 x
    for x in nums:
        # 从目标值 target 倒序遍历到 x，避免重复使用相同元素
        for c in range(target, x - 1, -1):
            # 状态转移方程：f[c] 是原来的 f[c] 加上 f[c - x]
            # 意思是：和为 c 的组合数 = 不包含 x 的组合数 + 包含 x 的组合数
            f[c] = f[c] + f[c - x]
    
    # 返回和为 target 的组合数
    return f[target]
```


```py
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # 通过转换方程得到所需和为 (sum(nums) + target) / 2
        # P - (S - P) = T
        # 2P = S + T
        # P = (S + T) / 2
        # 这边是 把target 变成2p了，把数字都放在了target了
        target += sum(nums)
        if target < 0 or target % 2:  # 如果 target 为负或不能被2整除，则返回 0
            return 0
        target //= 2

        n = len(nums)

        @cache  # 使用缓存来加速递归
        def dfs(i, c):
            if i < 0:  # 如果没有更多元素
                return 1 if c == 0 else 0  # 如果计数为0则找到一种组合，否则没有
            if c < nums[i]:  # 如果当前计数小于当前数字
                return dfs(i - 1, c)
            return dfs(i - 1, c) + dfs(i - 1, c - nums[i])  # 不包含和包含当前数字的两种情况

        return dfs(n - 1, target)  # 从最后一个元素和目标 target 开始递归，从上往下


```
改成递归

```py
from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        
        # 如果 target 过大或无法分配成两个子集和，则返回 0
        if total_sum < target or (total_sum + target) % 2 != 0:
            return 0
        
        # 转换为子集和问题
        subset_sum = (total_sum + target) // 2
        n = len(nums)
        
        # 初始化 DP 表，f[i][c] 表示前 i 个数能构成和为 c 的组合数
        f = [[0] * (subset_sum + 1) for _ in range(n + 1)]
        f[0][0] = 1  # 只有一种方式得到和为 0，即什么都不选

        # 动态规划填充表格
        for i, x in enumerate(nums):
            for c in range(subset_sum + 1):
                if c < x:
                    f[i + 1][c] = f[i][c]  # 如果当前和不足以包括 x，则只继承上一个状态
                else:
                    f[i + 1][c] = f[i][c] + f[i][c - x]  # 包含和不包含当前元素 x 的两种情况

        return f[n][subset_sum]



```

```python
f = [[0] * (target + 1) for _ in range(2)]
f[0][0] = 1  # 初始状态，容量为0时的方式数为1

for i, x in enumerate(nums):
    for c in range(target + 1):
        if c < x:
            f[(i + 1) % 2][c] = f[i % 2][c]
        else:
            f[(i + 1) % 2][c] = f[i % 2][c] + f[i % 2][c - x]

return f[n % 2][target]
```
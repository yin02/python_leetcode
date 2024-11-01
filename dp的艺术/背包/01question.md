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
### 如何判断01背包，01背包就是，容量和重量合法后决定价值

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
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:#如果把 target 看作“背包”的容量，那么 x 相当于每次添加到“背包”里的物品的“重量”
        # f[j]represents max value of substring given target
        # x is the current value
        # f[j] = max(f[j],f[j-x]+1) for state transfer
        f = [0] + [-inf] * target
        s = 0 # s is the possible maxium number which smaller than target
        for x in nums:
            s = min(s + x, target)
            for j in range(s, x - 1, -1):# 剩余空间
                # f[j] = max(f[j], f[j - x] + 1)
                # 手写 max 效率更高
                if f[j] < f[j - x] + 1:
                    f[j] = f[j - x] + 1
        return f[-1] if f[-1] > 0 else -1

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
        def dfs(i,j):# 空间index i， accumlator j
            if i <0:#如果所有结束
                return j ==0 #是不是正好和为0，因为要被每个数字-
            # legal j still got space, i-1 state 到i state choose, not choose
            return j >= nums[i] and dfs(i-1,j-nums[i]) or dfs(i-1,j)

        s = sum(nums)
        # s is even，不然不可以分成两堆是必然，是否合法能跑到最后
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
你要写的标准御用答案
```py
def solution(nums, target):
    target += sum
    target//=2
    # Early return if target is 0
    if target == 0:
        return 1 if sum(nums) ==0 else 0
    n = len(nums)

    f = [0] * (target+1)

    f[0] = 1#要收缩到0
    for x in nums:
        for j in range(target,x-1,-1):
            f[j] = f[j]+f[j-x]#等价于 f[j] += f[j - x],不选或者选上
            #j-x是表示和的核心
    return f[target]

```


```py
    total_sum = sum(nums)
    # Adjust target based on the sum
    target += total_sum
    # If target is not even, it's not possible to split into two equal sums
    if target % 2 != 0:
        return 0
    target //= 2

    # Early return if target is 0
    if target == 0:
        return 1 if total_sum == 0 else 0
    
    n = len(nums)
    # Initialize the dp array with zeros
    f = [0] * (target + 1)
    f[0] = 1

    # Iterate over each number in nums
    for x in nums:
        for j in range(target, x - 1, -1):
            f[j] += f[j - x]
    
    return f[target]

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



# [2787. Ways to Express an Integer as Sum of Powers](https://leetcode.com/problems/ways-to-express-an-integer-as-sum-of-powers/description/)

Given two **positive**  integers `n` and `x`.

Return the number of ways `n` can be expressed as the sum of the `x^th` power of **unique**  positive integers, in other words, the number of sets of unique integers `[n<sub>1</sub>, n<sub>2</sub>, ..., n<sub>k</sub>]` where `n = n<sub>1</sub>^x + n<sub>2</sub>^x + ... + n<sub>k</sub>^x`.

Since the result can be very large, return it modulo `10^9 + 7`.

For example, if `n = 160` and `x = 3`, one way to express `n` is `n = 2^3 + 3^3 + 5^3`.

**Example 1:** 

```
Input: n = 10, x = 2
Output: 1
Explanation: We can express n as the following: n = 3^2 + 1^2 = 10.
It can be shown that it is the only way to express 10 as the sum of the 2^nd power of unique integers.
```

**Example 2:** 

```
Input: n = 4, x = 1
Output: 2
Explanation: We can express n in the following ways:
- n = 4^1 = 4.
- n = 3^1 + 1^1 = 4.
```

**Constraints:** 

- `1 <= n <= 300`
- `1 <= x <= 5`

```py
MX_N, MX_X = 300, 5
#f[x][0] = 1，表示用任何 x 次方的数来表示和为 0 的方式只有 1 种（即什么都不选）
f = [[1] + [0] * MX_N for _ in range(MX_X)]#[1] + [0] * MX_N 是一个整体，它创建了一个长度为 MX_N + 1 的列表，前面是横，然后纵range
for x in range(MX_X):
    for i in count(1):
        v = i ** (x + 1)
        if v > MX_N:
            break
        for s in range(MX_N, v - 1, -1):
            f[x][s] += f[x][s - v]#其中 f[x][s] 表示使用 x+1 次方的数字表示 s 的方式数量，而 v 是某个数字的 x + 1 次方

class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        return f[x - 1][n] % 1_000_000_007

```






# [3181. Maximum Total Reward Using Operations II](https://leetcode.com/problems/maximum-total-reward-using-operations-ii/description/)

You are given an integer array `rewardValues` of length `n`, representing the values of rewards.

Initially, your total reward `x` is 0, and all indices are **unmarked** . You are allowed to perform the following operation **any**  number of times:

- Choose an **unmarked**  index `i` from the range `[0, n - 1]`.
- If `rewardValues[i]` is **greater**  than your current total reward `x`, then add `rewardValues[i]` to `x` (i.e., `x = x + rewardValues[i]`), and **mark**  the index `i`.

Return an integer denoting the **maximum ** total reward you can collect by performing the operations optimally.

**Example 1:** 

<div class="example-block">
Input: rewardValues = [1,1,3,3]

Output: 4

Explanation:

During the operations, we can choose to mark the indices 0 and 2 in order, and the total reward will be 4, which is the maximum.

**Example 2:** 

<div class="example-block">
Input: rewardValues = [1,6,4,3,2]

Output: 11

Explanation:

Mark the indices 0, 2, and 1 in order. The total reward will then be 11, which is the maximum.

**Constraints:** 

- `1 <= rewardValues.length <= 5 * 10^4`
- `1 <= rewardValues[i] <= 5 * 10^4`

### 方法一：动态规划

记 `rewardValues` 的最大值为 $m$，因为最后一次操作前的总奖励一定小于等于 $m-1$，所以可获得的最大总奖励小于等于 $2m-1$。假设上一次操作选择的奖励值为 $x_1$，那么执行操作后的总奖励 $x \geq x_1$。根据题意，后面任一操作选择的奖励值 $x_2$ 一定都大于 $x$，从而有 $x_2 > x_1$，因此执行的操作是按照奖励值单调递增的。

根据以上推断，首先将 `rewardValues` 从小到大进行排序，使用 `dp[k]` 表示总奖励 $k$ 是否可获得，初始时 `dp[0] = 1`，表示不执行任何操作获得总奖励 0。然后我们对 `rewardValues` 进行遍历，令当前值为 $x$，那么对于 $k \in [x, 2x-1]$（将 $k$ 倒序枚举），将 `dp[k]` 更新为 `dp[k - x] | dp[k]`（符号 `|` 表示或操作），表示先前的操作可以获得总奖励 $k - x$，那么加上 $x$ 后，就可以获取总奖励 $k$。最后返回 `dp` 中可以获得的最大总奖励。




先会这个吧
```python
class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        rewardValues.sort()
        m = rewardValues[-1]
        dp = [0] * (2 * m)
        dp[0] = 1
        for x in rewardValues:
            for k in range(2 * x - 1, x - 1, -1):
                if dp[k - x] == 1:
                    dp[k] = 1
        res = 0
        for i in range(len(dp)):
            if dp[i] == 1:
                res = i
        return res

```

第二个

```py
class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        #表示当前奖励value是 x 时，能够获得的最大总奖励
        @cache
        def dfs(x: int) -> int:
            i = bisect_right(rewardValues, x)
            ans = 0
            #比 x大的奖励
            for v in rewardValues[i:]:
                # v+更新的dfs
                ans = max(ans, v + dfs(x + v))
            return ans

        rewardValues.sort()
        return dfs(0)

```






更快时间
```py
f[i][j] 表示前i个数中，是否能够和为jd

v= rewardValues[i]选或者不选
不选：f[i][j] = f[i-1][j]
选：f[i][j] = f[i-1][j-v]
or
二进制的优化
0 | 0 = 0
0 | 1 = 1
1 | 0 = 1
1 | 1 = 1


整数类型：
byte: 8 bit
int：通常为 4 字节（32 位）
long：通常为 8 字节（64 位）
short：通常为 2 字节（16 位）
byte：1 字节（8 位）

一般来说 常用的应该是64 bit 用for循环
如果是位运算的话就是 1 bit ， 64倍提速
1 << v，将 1 左移 v 位，1 << 3 的结果是 0000 1000

(1 << v) - 1:

这个表达式会生成一个低于 v 的所有位都是 1 的二进制数。
例如，(1 << 3) - 1 会变成 0000 0111。

f & ((1 << v) - 1):

将 f 和上述结果进行按位与运算。这会保留 f 中低于 v 的位，其余位将被置为 0
计算 (1 << v) - 1:
1 << 3 变成 0000 1000
0000 1000 - 1 得到 
算 f & ((1 << v) - 1):
                  0000 0111
f = 13 的二进制为 0000 1101
                0000 0101 (结果)


bitset优化，不能选重复的 ，因为后面的x要大于和才行那不可能

问：为什么 2m−1 是答案的上界？
仔细读这句话5遍，所以这题最后一步m必选，那么有m情况下 2m-1也必然是最大的结果

答：如果最后一步选的数是 x，而 x<m，那么把 x 替换成 m 也符合要求，矛盾，所以最后一步选的一定是 m。在选 m 之前，元素和至多是 m−1，选了 m 之后，元素和至多是 2m−1。我们无法得到比 2m−1 更大的元素和。

mask = 1<<V-1
f &mask

```
```py
class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        f = 1
        for v in sorted(set(rewardValues)):
          # 首先得到低位的T和F ，1和0， 移动-1 然后andf，得到之前状态的1，0
          #再次移动
            f |= (f & ((1 << v) - 1)) << v
        return f.bit_length() - 1#变成index 才是这个j

```
再次优化
设 m=max(rewardValues)，如果数组中包含 m−1，则答案为 2m−1，无需计算 DP。
```py
class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        m = max(rewardValues)
        if m - 1 in rewardValues:
            return m * 2 - 1

        f = 1
        for v in sorted(set(rewardValues)):
            f |= (f & ((1 << v) - 1)) << v
        return f.bit_length() - 1 #从有1的那个长度 -1


```

优化三
如果有两个不同元素之和等于 m−1，也可以直接返回 2m−1。

```py

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        m = max(rewardValues)
        s = set()
        for v in rewardValues:
            if v in s:
                continue
            if v == m - 1 or m - 1 - v in s:
                return m * 2 - 1
            s.add(v)

        f = 1
        for v in sorted(s):
            f |= (f & ((1 << v) - 1)) << v
        return f.bit_length() - 1
```
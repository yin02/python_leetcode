
# 1.Best Time to Buy And Sell Stock

## 问题描述
给定一个数组 `prices`，其中 `prices[i]` 表示某支股票第 `i` 天的价格。你只能选择某一天买入这支股票，并选择在未来的某一天卖出。求你能获得的最大利润。

- **输入**: 一个整数数组 `prices`，表示每天的股票价格。
- **输出**: 一个整数，表示可以获得的最大利润。如果无法获得利润，则返回 `0`。

## solution
1. **初始化变量**: 
2. **遍历价格数组**: 

## code

```python

    def maxProfit(self,prices):
        res = 0
        lowest = prices[0]
        for price in prices:
            if price < lowest:
                lowest = price
            res = min(res,price- lowest)
        return res
```


---

---

## 2. Contains Duplicate II

### 问题描述
给定一个整数数组 `nums` 和一个整数 `k`，判断数组中是否存在两个不同的索引 `i` 和 `j`，使得 `nums[i] == nums[j]`，且 `i` 和 `j` 的差的绝对值最大为 `k`。

### 输入
- 一个整数数组 `nums`，表示数字序列。
- 一个整数 `k`，表示允许的索引差值。

### 输出
- 布尔值 `true` 或 `false`，表示是否存在满足条件的元素对。

### Solution

1. **初始化集合**: 创建一个空的集合 `window`，用来记录当前窗口内的元素，并初始化左指针 `L`。
2. **遍历数组**: 使用右指针 `R` 遍历 `nums`，如果当前窗口大小超过 `k`，移除 `L` 指针指向的元素并移动 `L`。
3. **检查重复元素**: 如果 `nums[R]` 在集合中，返回 `true`；否则，将 `nums[R]` 添加到集合中。
4. **返回结果**: 如果遍历完数组没有发现满足条件的元素对，返回 `false`。

### Code

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        window = set()
        L = 0

        for R in range(len(nums)):
            if R - L > k:
                window.remove(nums[L])
                L += 1
            if nums[R] in window:
                return True
            window.add(nums[R])
        return False

```
---

## 3.Number of Subarrays of Size K and Avg Greater than or Equal to Threshold

### 问题描述
给定一个整数数组 `arr` 和两个整数 `k` 和 `threshold`，求大小为 `k` 的子数组中，平均值大于或等于 `threshold` 的子数组数量。

### 输入
- 一个整数数组 `arr`，表示数字序列。
- 一个整数 `k`，表示子数组的大小。
- 一个整数 `threshold`，表示平均值阈值。

### 输出
- 一个整数，表示满足条件的子数组数量。

### Solution

1. **初始化变量**: 创建 `curSum` 来保存当前窗口的和，并初始化 `res` 为 0，表示满足条件的子数组数量。
2. **计算初始窗口和**: 计算前 `k` 个元素的和，并将其赋值给 `curSum`。
3. **滑动窗口遍历数组**: 从第一个元素开始，滑动窗口，每次将窗口右侧的元素加入 `curSum` 中，并判断平均值是否大于等于 `threshold`，如果是则 `res` 增加。
4. **更新窗口**: 每次滑动窗口时，将窗口左侧的元素从 `curSum` 中减去。

### Code

```python
class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        res = 0
        curSum = sum(arr[:k-1]) #exlusive arr[:k-1] 实际上是从索引 0 开始到 k-2 结束的元素的集合。
#以下是为什么要这样设置的原因：
# 初始设置：在进入循环之前，算法先计算数组前 k-1 个元素的和。因为循环会从 k 个元素开始，将这个元素加到 curSum 中，从而得到第一个长度为 k 的子数组的和。
        for L in range(len(arr) - k + 1):
            curSum += arr[L + k - 1]
            if (curSum / k) >= threshold:
                res += 1
            curSum -= arr[L]
        return res

```

---

## 4.Longest Substring Without Repeating Characters

### 问题描述
给定一个字符串 `s`，找出其中不含有重复字符的最长子串的长度。

### 输入
- 一个字符串 `s`，表示要处理的字符串。

### 输出
- 一个整数，表示最长无重复字符子串的长度。

### Solution

1. **初始化变量**: 创建一个集合 `charSet` 用于存储当前子串中的字符，初始化左指针 `l` 和结果变量 `res` 为 0。
2. **遍历字符串**: 使用右指针 `r` 遍历字符串 `s`。
3. **处理重复字符**: 如果当前字符 `s[r]` 已经在 `charSet` 中，移动左指针 `l`，并从 `charSet` 中移除左指针指向的字符，直到 `s[r]` 不在 `charSet` 中。
4. **更新结果**: 将 `s[r]` 添加到 `charSet` 中，更新最长子串的长度 `res`，取 `r - l + 1` 的最大值。
5. **返回结果**: 遍历完成后，返回结果 `res`。

### Code

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        charSet = set()
        l = 0
        res = 0

        for r in range(len(s)):
            while s[r] in charSet:
                charSet.remove(s[l])
                l += 1
            charSet.add(s[r])
            res = max(res, r - l + 1)
        return res
```


---

## 5.Longest Repeating Character Replacement

### 问题描述
给定一个字符串 `s` 和一个整数 `k`，你可以将字符串中的 `k` 个字符替换成任意字符，求在执行替换操作后，字符串中最长的重复字符子串的长度。

### 输入
- 一个字符串 `s`，表示要处理的字符串。
- 一个整数 `k`，表示可以替换的字符个数。

### 输出
- 一个整数，表示替换后最长重复字符子串的长度。

### Solution

1. **初始化变量**: 创建一个哈希表 `count` 用于存储当前窗口内每个字符的出现次数，初始化左指针 `l` 和最大频率字符的频率 `maxf`。
2. **遍历字符串**: 使用右指针 `r` 遍历字符串 `s`。
3. **更新字符计数**: 每次移动右指针时，增加当前字符的计数，并更新 `maxf`，即当前窗口内出现频率最高的字符的次数。
4. **调整窗口**: 如果当前窗口长度减去 `maxf` 大于 `k`，说明替换次数超过了 `k`，需要通过移动左指针来缩小窗口。
5. **返回结果**: 最后返回最长的有效窗口的长度 `(r - l + 1)`。

### Code

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = {}
        
        l = 0
        maxf = 0
        for r in range(len(s)):
            count[s[r]] = 1 + count.get(s[r], 0)
            maxf = max(maxf, count[s[r]])

            if (r - l + 1) - maxf > k:
                count[s[l]] -= 1
                l += 1

        return (r - l + 1)
```


---

## 6.Permutation In String

### 问题描述
给定两个字符串 `s1` 和 `s2`，判断 `s2` 中是否包含 `s1` 的一个排列。换句话说，判断是否存在一个子串的排列与 `s1` 相同。

### 输入
- 一个字符串 `s1`，表示要查找的排列。
- 一个字符串 `s2`，表示要搜索的字符串。

### 输出
- 布尔值 `true` 或 `false`，表示 `s2` 中是否包含 `s1` 的排列。

### Solution

1. **特殊情况处理**: 如果 `s1` 的长度大于 `s2`，直接返回 `False`，因为 `s2` 不可能包含 `s1` 的排列。
2. **初始化字符计数**: 创建两个长度为 26 的数组 `s1Count` 和 `s2Count` 来存储 `s1` 和 `s2` 中对应字符的频率。初始化时，将 `s1` 和 `s2` 前 `len(s1)` 个字符的频率填入 `s1Count` 和 `s2Count`。
3. **计算初始匹配数**: 遍历 `s1Count` 和 `s2Count`，计算出初始状态下完全匹配的字符数 `matches`。
4. **滑动窗口遍历 `s2`**: 使用滑动窗口在 `s2` 中遍历，窗口大小为 `len(s1)`，在每次移动窗口时更新 `s2Count` 并调整 `matches` 值。如果在某次遍历中 `matches` 达到 26，说明找到了 `s1` 的一个排列，返回 `True`。
5. **返回结果**: 如果遍历结束仍未找到，返回 `False`。

### Code

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False

        s1Count, s2Count = [0] * 26, [0] * 26
        for i in range(len(s1)):
            s1Count[ord(s1[i]) - ord("a")] += 1
            s2Count[ord(s2[i]) - ord("a")] += 1

        matches = 0
        for i in range(26):
            matches += 1 if s1Count[i] == s2Count[i] else 0

        l = 0
        for r in range(len(s1), len(s2)):
            if matches == 26:
                return True

            index = ord(s2[r]) - ord("a")
            s2Count[index] += 1
            if s1Count[index] == s2Count[index]:
                matches += 1
            elif s1Count[index] + 1 == s2Count[index]:
                matches -= 1

            index = ord(s2[l]) - ord("a")
            s2Count[index] -= 1
            if s1Count[index] == s2Count[index]:
                matches += 1
            elif s1Count[index] - 1 == s2Count[index]:
                matches -= 1
            l += 1
            
        return matches == 26
```
### 代码解释：
1. **字符频率计数**: 使用两个长度为 26 的数组 `s1Count` 和 `s2Count` 来存储字母 'a' 到 'z' 的频率，以此判断两个字符串是否包含相同字符及频率。
2. **滑动窗口**: 利用滑动窗口技术遍历 `s2`，每次调整窗口时更新 `s2Count` 并判断当前窗口中的子串是否为 `s1` 的排列。
3. **匹配计数 `matches`**: 记录 `s1Count` 和 `s2Count` 中完全匹配的字符数。当 `matches` 达到 26 时，表示 `s2` 当前窗口中的子串是 `s1` 的一个排列。

这个代码高效地判断了 `s2` 中是否存在 `s1` 的一个排列，时间复杂度为 O(n)，空间复杂度为 O(1)。



---

## 7. Frequency of The Most Frequent Element

### 问题描述
给定一个整数数组 `nums` 和一个整数 `k`，你可以将数组中的 `k` 个元素增大任意值。求数组中出现频率最高的元素的最大频率。

### 输入
- 一个整数数组 `nums`，表示数字序列。
- 一个整数 `k`，表示可以增加的总数值。

### 输出
- 一个整数，表示频率最高元素的最大频率。

### Solution

1. **对数组进行排序**: 首先将 `nums` 排序，以便更容易处理增大元素的操作。
2. **初始化变量**: 定义左右指针 `l` 和 `r`，用来维护一个滑动窗口，同时定义结果变量 `res` 和当前窗口的总和 `total`。
3. **滑动窗口遍历**: 使用右指针 `r` 遍历数组，逐步将 `nums[r]` 添加到窗口的总和 `total` 中。
4. **调整窗口**: 如果当前窗口中的最大频率元素 `nums[r]` 乘以窗口大小大于 `total + k`（即不能通过增加最多 `k` 次来让所有元素等于 `nums[r]`），则移动左指针 `l` 来缩小窗口，并从 `total` 中减去 `nums[l]`。
5. **更新结果**: 每次调整完窗口后，更新最大频率 `res`，即 `r - l + 1` 的最大值。
6. **返回结果**: 最终返回结果 `res`，即最大频率。

### Code

```python
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        nums.sort()
        l, r = 0, 0
        res, total = 0, 0
        while r < len(nums):
            total += nums[r]
            while nums[r] * (r - l + 1) > total + k:
                total -= nums[l]
                l += 1
            res = max(res, r - l + 1)
            r += 1
        return res
```


### 代码解释：
1. **排序**: 通过对 `nums` 进行排序，使得我们可以在窗口内更容易地调整元素，使它们相等。
2. **滑动窗口**: 使用滑动窗口技术，右指针 `r` 向右扩展窗口，左指针 `l` 根据需要收缩窗口，保证窗口内的所有元素通过最多 `k` 次操作可以变得相等。
3. **最大频率更新**: 每次更新最大频率 `res`，它等于窗口内元素的数量，即 `r - l + 1`。

该代码在 O(n log n) 的时间复杂度内解决了问题，其中排序的时间复杂度为 O(n log n)，滑动窗口部分为 O(n)。



---

## 8. Fruits into Baskets

### 问题描述
在一条线上种植了多棵树，每棵树上都有不同数量的水果。你有两个篮子，每个篮子只能装一种类型的水果。你需要找出一个连续的子数组，使得可以使用这两个篮子装满尽可能多的水果。

### 输入
- 一个整数数组 `fruits`，表示每棵树上的水果种类。

### 输出
- 一个整数，表示可以采摘的最大水果数量。

### Solution

1. **使用哈希表记录水果种类**: 使用 `collections.defaultdict(int)` 来记录当前窗口内每种水果的数量。
2. **初始化变量**: 左指针 `l`，当前窗口中的水果总数 `total`，以及结果 `res` 都初始化为 0。
3. **滑动窗口遍历**: 使用右指针 `r` 遍历水果数组 `fruits`，每次将当前水果加入窗口并更新总数 `total`。
4. **收缩窗口**: 当窗口内的不同水果种类超过 2 种时，移动左指针 `l`，减少窗口中的水果数量，直到窗口内只剩下两种不同的水果。
5. **更新结果**: 在每次调整窗口后，更新最大水果数量 `res`，即当前窗口内的水果总数。
6. **返回结果**: 最后返回 `res`，即最多包含两种不同水果的最长子数组长度。

### Code

```python
from collections import defaultdict

class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        # 使用 defaultdict 来存储每种水果的数量
        count = defaultdict(int)
        
        # 初始化左指针 l，当前窗口中的水果总数 total，和结果 res
        l, total, res = 0, 0, 0

        # 遍历整个水果列表，r 是右指针
        for r in range(len(fruits)):
            # 将右指针指向的水果计数加1
            count[fruits[r]] += 1
            # 更新当前窗口中的水果总数
            total += 1

            # 当窗口内的不同水果种类超过 2 种时，收缩窗口
            while len(count) > 2:
                f = fruits[l]  # 取得左指针指向的水果
                count[f] -= 1  # 将该水果的计数减1
                total -= 1     # 更新当前窗口中的水果总数
                l += 1         # 移动左指针
                # 如果某种水果的计数变为 0，将其从字典中移除
                if not count[f]:
                    count.pop(f)

            # 记录当前最大窗口大小
            res = max(res, total)

        # 返回最大窗口大小，即最多包含两种不同水果的最长子数组长度
        return res
```

### 代码解释：
1. **哈希表 `count`**: 用于记录当前窗口内每种水果的数量。
2. **滑动窗口**: 使用双指针（`l` 和 `r`）维护一个窗口，确保窗口内最多包含两种不同类型的水果。
3. **窗口调整**: 当窗口内水果种类超过两种时，左指针 `l` 向右移动，减少窗口中的水果数量，直到窗口内只有两种水果。
4. **最大窗口大小**: 通过比较 `res` 和 `total` 的值，记录窗口内最多包含两种不同水果的最大长度。

这个算法在 O(n) 时间内找到满足条件的最长子数组，适用于需要高效处理的场景。

---

## 9. Maximum Number of Vowels in a Substring of Given Length

### 问题描述
给定一个字符串 `s` 和一个整数 `k`，求长度为 `k` 的子字符串中元音字母的最大数量。

### 输入
- 一个字符串 `s`，表示要处理的字符串。
- 一个整数 `k`，表示子字符串的长度。

### 输出
- 一个整数，表示长度为 `k` 的子字符串中元音字母的最大数量。

### Solution

1. **初始化变量**: 使用左指针 `l`，记录当前最大元音数量的变量 `res`，以及当前窗口内元音总数 `total`，并初始化为 0。
2. **定义元音集合**: 使用字符串 `vowels = "aeiou"` 来定义元音字符。
3. **滑动窗口遍历**: 使用右指针 `r` 遍历字符串 `s`，每次遇到元音字符时，将其计入 `total`。
4. **调整窗口**: 如果当前窗口的大小超过了 `k`，移动左指针 `l`，并将 `total` 中左端移出的字符数减去。
5. **更新结果**: 每次更新 `res` 为当前窗口内元音字母的最大数量。
6. **返回结果**: 最终返回 `res`，即长度为 `k` 的子字符串中包含的最多元音字母的数量。

### Code

```python
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        l, res, total = 0, 0, 0
        vowels = "aeiou"
        for r in range(len(s)):
            if s[r] in vowels:
                total += 1
            if (r - l + 1) > k:
                if s[l] in vowels:
                    total -= 1
                l += 1
            res = max(res, total)
        return res
```

### 代码解释：
1. **滑动窗口**: 通过滑动窗口技术，右指针 `r` 向右移动，扩大窗口范围，左指针 `l` 根据需要调整窗口大小，保持窗口长度为 `k`。
2. **元音计数**: `total` 用于记录当前窗口内的元音字母数量，并随着窗口的调整动态更新。
3. **最大值更新**: 每次更新 `res`，记录当前窗口内最多元音字母的数量。

这个算法在 O(n) 时间内完成，适用于需要找到固定长度子串中某些特定字符最大数量的问题。



---

## 10.Minimum Number of Flips to Make the Binary String Alternating

### 问题描述
给定一个二进制字符串 `s`，你可以将字符串中的字符翻转。求使得字符串变为交替模式（如 `"010101"` 或 `"101010"`）所需的最小翻转次数。

### 输入
- 一个二进制字符串 `s`，表示要处理的字符串。

### 输出
- 一个整数，表示使字符串成为交替模式所需的最小翻转次数。

### Solution

1. **扩展字符串**: 将字符串 `s` 复制一次，形成一个新字符串 `s + s`，这样可以在滑动窗口中模拟环状的字符串处理。
2. **构造交替字符串**: 构造两个交替模式的字符串 `alt1`（从 `"0"` 开始）和 `alt2`（从 `"1"` 开始）。
3. **初始化变量**: 定义结果变量 `res` 为正无穷大，以及两个变量 `diff1` 和 `diff2` 来记录当前窗口与两个交替字符串的不同之处。
4. **滑动窗口遍历**: 使用右指针 `r` 遍历扩展后的字符串 `s`，计算当前窗口中与 `alt1` 和 `alt2` 不同的字符数，并更新 `diff1` 和 `diff2`。
5. **调整窗口**: 当窗口大小大于原始字符串长度 `n` 时，左指针 `l` 需要右移，并相应地更新 `diff1` 和 `diff2`。
6. **更新结果**: 在窗口大小达到 `n` 时，更新结果 `res` 为当前最小的翻转次数。
7. **返回结果**: 最终返回结果 `res`，即最小的翻转次数。

### Code

```python
class Solution:
    def minFlips(self, s: str) -> int:
        n = len(s)
        s = s + s
        # 构造两种可能的交替字符串
        alt1, alt2 = "", ""
        for i in range(len(s)):
            alt1 += "0" if i % 2 == 0 else "1"
            alt2 += "1" if i % 2 == 0 else "0"
        
        res = float('inf')
        diff1, diff2 = 0, 0  # 记录与 alt1 和 alt2 的差异
        l = 0

        for r in range(len(s)):
            # 计算当前字符与 alt1 和 alt2 的差异
            if s[r] != alt1[r]:
                diff1 += 1
            if s[r] != alt2[r]:
                diff2 += 1
            
            # 当窗口大小超过 n 时，调整左指针
            if (r - l + 1) > n:
                if s[l] != alt1[l]:
                    diff1 -= 1
                if s[l] != alt2[l]:
                    diff2 -= 1
                l += 1
            
            # 当窗口大小正好为 n 时，更新结果
            if (r - l + 1) == n:
                res = min(res, diff1, diff2)
        
        return res
```

### 代码解释：
1. **字符串扩展**: 通过将字符串 `s` 复制一次，我们可以用一个窗口滑动的方法处理环状问题，而无需显式地处理环状结构。
2. **交替模式构造**: `alt1` 和 `alt2` 是两种可能的交替二进制字符串，用来与输入字符串进行比较。
3. **滑动窗口**: 通过滑动窗口技术，我们可以高效地计算每个可能窗口中的翻转次数，并实时更新结果。
4. **时间复杂度**: 这个算法的时间复杂度为 O(n)，其中 n 是字符串的长度，非常适合处理大规模数据。

这个代码非常高效地解决了如何最小化翻转次数使得二进制字符串变为交替模式的问题。


---

## 11 Minimum Size Subarray Sum

### 问题描述
给定一个含有正整数的数组 `nums` 和一个正整数 `target`，找出该数组中满足其和大于或等于 `target` 的长度最小的连续子数组，并返回其长度。如果不存在符合条件的子数组，返回 `0`。

### 输入
- 一个整数 `target`，表示目标和。
- 一个整数数组 `nums`，表示数字序列。

### 输出
- 一个整数，表示满足条件的最小子数组长度。如果不存在这样的子数组，则返回 `0`。

### Solution

1. **初始化变量**: 将结果变量 `res` 初始化为正无穷大，用来存储最小子数组长度。定义左指针 `l` 和当前窗口的总和 `total`，均初始化为 0。
2. **滑动窗口遍历**: 使用右指针 `r` 遍历数组 `nums`，每次将 `nums[r]` 加到当前窗口的总和 `total` 中。
3. **调整窗口**: 当 `total` 大于或等于 `target` 时，移动左指针 `l`，缩小窗口，并在每次缩小窗口时更新最小子数组长度 `res`。
4. **返回结果**: 最终返回 `res`，如果 `res` 仍然是正无穷大，说明不存在满足条件的子数组，返回 `0`。

### Code

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        res = float("inf")
        l, total = 0, 0
        for r in range(len(nums)):
            total += nums[r]
            while total >= target:
                res = min(res, r - l + 1)
                total -= nums[l]
                l += 1
        return res if res != float('inf') else 0
```


### 代码解释：
1. **滑动窗口**: 使用双指针（`l` 和 `r`）维护一个窗口，当窗口内的元素和 `total` 大于或等于 `target` 时，缩小窗口，试图找到更短的子数组。
2. **最小长度更新**: 每次找到满足条件的子数组时，更新最小长度 `res`。
3. **返回结果**: 最后，如果 `res` 未被更新，则返回 `0`，否则返回 `res`。

这个算法在 O(n) 的时间复杂度内解决了问题，是求解最小长度子数组的高效方法。


---

## 12 Find K Closest Elements

### 问题描述
给定一个排序好的数组 `arr` 和两个整数 `k` 和 `x`，从数组中找到 `k` 个最接近 `x` 的元素。结果应该按升序排列。如果有两个数与 `x` 的差值相同，则选择较小的数。

### 输入
- 一个排序好的整数数组 `arr`。
- 一个整数 `k`，表示要找到的元素数量。
- 一个整数 `x`，表示目标值。

### 输出
- 一个整数数组，表示按升序排列的 `k` 个最接近 `x` 的元素。

### Solution

1. **初始化左右指针**: 左指针 `l` 指向数组的起始位置，右指针 `r` 指向数组的末尾位置。
2. **滑动窗口**: 在 `r - l + 1 > k` 的条件下，调整左右指针，以保持窗口大小为 `k`。每次比较 `arr[l]` 和 `arr[r]` 与 `x` 的距离：
   - 如果 `arr[l]` 更接近 `x` 或者距离相等时，则收缩右边界 `r`。
   - 否则，收缩左边界 `l`。
3. **返回结果**: 当窗口大小等于 `k` 时，返回窗口内的子数组，即 `arr[l : r + 1]`。

### Code

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        l = 0
        r = len(arr) - 1
        
        # 当窗口大小大于 k 时调整窗口
        while r - l + 1 > k:
            # 如果左边的距离小于等于右边的距离，收缩右边界
            if abs(arr[l] - x) <= abs(arr[r] - x):
                r -= 1
            # 否则收缩左边界
            else:
                l += 1
            
        # 返回窗口内的元素
        return arr[l : r + 1]



class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        l, r = -1, len(arr) - k + 1  # 初始化红蓝边界
        while l != r - 1:
            m = (l + r) // 2
            # 判断条件，调整边界，注意检查索引 m + k 是否超出范围
            if m + k < len(arr) and x - arr[m] > arr[m + k] - x:
                l = m  #找右边
            else:
                r = m  # 找左边，相等的时候，尽量找的是左边的
        return arr[r:r + k]  # 相等的是从r开始的 

```

### 代码解释：
1. **滑动窗口**: 使用双指针技术来维持一个大小为 `k` 的窗口，确保窗口内的元素是最接近目标值 `x` 的 `k` 个元素。
2. **指针调整**: 通过比较左右两端元素与目标值 `x` 的距离，决定收缩哪一边，以确保最终得到的 `k` 个元素是最接近 `x` 的。
3. **返回结果**: 最终返回窗口内的子数组，这个子数组是按升序排列的最接近 `x` 的 `k` 个元素。

这个算法在 O(n) 的时间复杂度内有效地解决了问题，适合处理大规模数据。




### 代码解释：
1. **二分查找**: 通过二分查找的方法，快速确定窗口的起始位置。这种方法的时间复杂度为 O(log(n - k))，其中 n 是数组的长度。
2. **左右边界调整**: 在二分过程中，根据与目标值 `x` 的距离，动态调整左右边界，确保找到的 `k` 个元素是最接近 `x` 的。
3. **返回结果**: 返回从位置 `r` 开始的连续 `k` 个元素，这些元素是距离 `x` 最近的，并且已经按升序排列。

这个算法在效率上非常优越，特别适合需要在大数组中快速找到最接近目标值的元素时使用。


---

## Minimum Operations to Reduce X to Zero

### 问题描述
给定一个整数数组 `nums` 和一个整数 `x`，你需要从数组的开头或末尾移除最少的元素，使得这些元素的和等于 `x`。请计算并返回移除的最小操作数。如果无法得到这样的操作数，则返回 `-1`。

### 输入
- 一个整数数组 `nums`。
- 一个整数 `x`。

### 输出
- 一个整数，表示最少的操作数。如果不存在这样的操作数，则返回 `-1`。

### Solution

1. **初始化变量**:
   - 计算数组的总和 `total`，以及数组的长度 `n`。
   - 初始化前缀和 `lsum` 和后缀和 `rsum`。
   - 使用两个指针 `left` 和 `right` 分别表示前缀和后缀的范围。
   - 初始答案 `ans` 设为 `n + 1`，用于记录最少的操作数。
   
2. **处理特殊情况**: 如果 `total` 小于 `x`，直接返回 `-1`，因为无论如何移除都不可能使和达到 `x`。

3. **滑动窗口遍历**:
   - 使用 `left` 指针遍历数组，从左到右逐步计算前缀和 `lsum`。
   - 使用 `right` 指针遍历后缀和 `rsum`，当 `lsum + rsum` 大于 `x` 时，右指针 `right` 向右移动，减少后缀和 `rsum`。
   - 当 `lsum + rsum` 等于 `x` 时，更新 `ans` 为当前的最小操作数。

4. **返回结果**: 最后，如果 `ans` 仍然大于 `n`，返回 `-1`，否则返回 `ans`。

### Code

```python
class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        n = len(nums)
        total = sum(nums)
        if total < x:
            return -1
        
        right = 0
        lsum, rsum = 0, total
        ans = n + 1
        
        for left in range(-1, n - 1):
            if left != -1:
                lsum += nums[left]
            
            while right < n and lsum + rsum > x:
                rsum -= nums[right]
                right += 1
            
            if lsum + rsum == x:
                ans = min(ans, (left + 1) + (n - right))
        
        return -1 if ans > n else ans
```

### 代码解释：
1. **双指针和滑动窗口**: 使用双指针的方法，左指针 `left` 和右指针 `right` 分别负责前缀和和后缀和的移动，以寻找最优解。
2. **更新操作**: 当 `lsum + rsum` 等于 `x` 时，更新最少的操作数 `ans`。
3. **复杂度分析**: 该算法的时间复杂度为 O(n)，空间复杂度为 O(1)，非常适合处理大规模数据。

这个算法通过高效的滑动窗口技术解决了从数组两端移除元素使和达到目标值的问题。

## Equal Substring Within Budget

### 问题描述
给定两个字符串 `s` 和 `t`，以及一个整数 `maxCost`。我们需要计算在不超过 `maxCost` 的条件下，可以将 `s` 中的一段连续子字符串转换为 `t` 中对应子字符串的最长长度。转换的代价是将 `s` 中一个字符转换为 `t` 中对应字符的 ASCII 码的差值的绝对值。

### 输入
- 一个字符串 `s`。
- 一个字符串 `t`，与 `s` 长度相同。
- 一个整数 `maxCost`。

### 输出
- 一个整数，表示可以在 `maxCost` 内将 `s` 转换为 `t` 的最长子字符串的长度。

### Solution

1. **初始化变量**:
   - `curCost` 用于跟踪当前子字符串转换的代价总和。
   - `l` 和 `r` 分别为左右指针，用于表示当前考察的子字符串的范围。
   - `res` 用于存储满足条件的最长子字符串长度。

2. **遍历字符串**:
   - 使用右指针 `r` 遍历字符串 `s`，计算 `s[r]` 转换为 `t[r]` 的代价，并加到 `curCost` 上。
   - 如果 `curCost` 超过了 `maxCost`，则通过移动左指针 `l` 来减少窗口的代价，直到 `curCost` 小于或等于 `maxCost`。

3. **更新结果**:
   - 在每次迭代中，如果当前窗口的代价在预算内，更新最大长度 `res`。

### Code

```python
class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        curCost = 0
        l = 0
        res = 0
        for r in range(len(s)):
            curCost += abs(ord(s[r]) - ord(t[r]))
            while curCost > maxCost:
                curCost -= abs(ord(s[l]) - ord(t[l]))
                l += 1
            res = max(res, r - l + 1)
        return res
```


## Number of Subarrays with Sum Equal to Goal

### 问题描述
给定一个二进制数组 `nums` 和一个整数 `goal`，我们需要找到数组中和等于 `goal` 的连续子数组的个数。

### 输入
- 一个二进制数组 `nums`，其中每个元素是 `0` 或 `1`。
- 一个整数 `goal`。

### 输出
- 一个整数，表示和等于 `goal` 的连续子数组的个数。

### Solution

1. **定义辅助函数**:
   - 定义一个辅助函数 `helper(x)`，该函数计算和小于等于 `x` 的连续子数组的数量。
   - 初始化变量 `res`（用于存储结果）、`l`（左指针）、`cur`（当前子数组的和）。

2. **滑动窗口遍历数组**:
   - 使用右指针 `r` 遍历数组，将 `nums[r]` 加到当前和 `cur` 中。
   - 如果当前和 `cur` 超过 `x`，则通过移动左指针 `l` 来缩小窗口，直到 `cur` 小于或等于 `x`。
   - 通过 `res += (r - l + 1)` 累加满足条件的子数组的数量。

3. **计算目标结果**:
   - 通过 `helper(goal)` 计算和小于等于 `goal` 的子数组数量。
   - 通过 `helper(goal - 1)` 计算和小于等于 `goal-1` 的子数组数量。
   - 两者相减即可得到和恰好等于 `goal` 的子数组数量。

### Code

```python

#想要计算 =2 ，就用＜＝2 来减去< = 1
class Solution:
    def numSubarraysWithSum(self, nums, goal):
        def helper(x):
            if x < 0: return 0
            res, l, cur = 0, 0, 0
            for r in range(len(nums)):
                cur += nums[r]
                while cur > x:
                    cur -= nums[l]
                    l += 1
                res += (r - l + 1)
            return res
        
        return helper(goal) - helper(goal - 1)
```

## Subarray Product Less Than K

### 问题描述
给定一个正整数数组 `nums` 和一个整数 `k`，你需要找到数组中乘积小于 `k` 的连续子数组的个数。

### 输入
- 一个正整数数组 `nums`。
- 一个整数 `k`。

### 输出
- 一个整数，表示乘积小于 `k` 的连续子数组的个数。

### Solution

1. **初始化变量**:
   - `res` 用于存储满足条件的子数组数量。
   - `l` 为左指针，初始化为 `0`。
   - `product` 用于存储当前子数组的乘积，初始化为 `1`。

2. **滑动窗口遍历数组**:
   - 使用右指针 `r` 遍历数组，将 `nums[r]` 乘到当前乘积 `product` 中。
   - 如果当前乘积 `product` 大于等于 `k`，通过移动左指针 `l` 来缩小窗口，并将 `nums[l]` 从 `product` 中除去，直到 `product` 小于 `k`。
   - 计算当前窗口内满足条件的子数组数量，`res += (r - l + 1)`。

3. **返回结果**:
   - 最后，`res` 即为满足乘积小于 `k` 的所有子数组的数量。

### Code

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums, k):
        res, l, product = 0, 0, 1
        for r in range(len(nums)):
            product *= nums[r]
            while l <= r and product >= k:
                product //= nums[l]
                l += 1
            res += (r - l + 1)
        return res
```

## Maximum Subarray Length with At Most K Occurrences of Each Element

### 问题描述
给定一个整数数组 `nums` 和一个整数 `k`，找出满足数组中每个元素最多出现 `k` 次的最长连续子数组的长度。

### 输入
- 一个整数数组 `nums`。
- 一个整数 `k`。

### 输出
- 一个整数，表示满足条件的最长连续子数组的长度。

### Solution

1. **初始化变量**:
   - `res` 用于存储满足条件的最长子数组长度。
   - `count` 是一个 `defaultdict`，用于记录当前窗口中每个元素的出现次数。
   - `l` 为左指针，初始化为 `0`。

2. **滑动窗口遍历数组**:
   - 使用右指针 `r` 遍历数组，增加 `nums[r]` 的出现次数。
   - 如果某个元素的出现次数超过 `k`，通过移动左指针 `l` 来缩小窗口，并减少 `nums[l]` 的出现次数，直到该元素的出现次数不超过 `k`。
   - 在每次迭代中，更新满足条件的最长子数组长度 `res`。

3. **返回结果**:
   - 最后，`res` 即为满足条件的最长连续子数组的长度。

### Code

```python
from collections import defaultdict

class Solution:
    def maxSubarrayLength(self, nums, k):
        res = 0
        count = defaultdict(int)
        l = 0
        for r in range(len(nums)):
            count[nums[r]] += 1
            while count[nums[r]] > k:
                count[nums[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
        
        return res
```

## Count Subarrays with Exactly K Maximum Elements

### 问题描述
给定一个整数数组 `nums` 和一个整数 `k`，你需要找出满足数组中最大元素恰好出现 `k` 次的子数组的个数。

### 输入
- 一个整数数组 `nums`。
- 一个整数 `k`。

### 输出
- 一个整数，表示满足条件的子数组个数。

### Solution

1. **初始化变量**:
   - `max_n` 存储数组中的最大元素。
   - `max_cnt` 用于记录当前窗口中最大元素的出现次数。
   - `l` 为左指针，初始化为 `0`。
   - `res` 用于存储满足条件的子数组数量。

2. **滑动窗口遍历数组**:
   - 使用右指针 `r` 遍历数组，如果 `nums[r]` 等于最大元素 `max_n`，则增加 `max_cnt`。
   - 如果 `max_cnt` 大于 `k`，或者 `max_cnt` 等于 `k` 但左指针 `l` 指向的元素不是最大元素 `max_n`，则通过移动左指针 `l` 来缩小窗口，并根据条件减少 `max_cnt`。
   - 如果当前窗口内最大元素的出现次数恰好等于 `k`，则更新结果 `res`。

3. **延迟更新结果**:
   - 一个关键的技巧是，我们不立即更新结果，而是延迟到 `l` 到达最边缘，即左指针到达窗口左侧时，才更新 `res`。
   - 这样可以确保在计算满足条件的子数组时，考虑到所有可能的窗口位置。

4. **返回结果**:
   - 最终的 `res` 即为满足条件的子数组数量。

### Code

```python
class Solution:
    def countSubarray(self, nums, k):
        # 找到数组中的最大元素
        max_n, max_cnt = max(nums), 0
        l = 0
        res = 0
        for r in range(len(nums)):
            # 如果当前元素是最大元素，增加计数
            if nums[r] == max_n:
                max_cnt += 1
            # 如果最大元素的计数超过k，或当前窗口内满足条件的最大元素数量等于k但左指针指向的元素不是最大元素
            while max_cnt > k or (l <= r and max_cnt == k and nums[l] != max_n):
                # 如果左指针指向的元素是最大元素，减少计数
                if nums[l] == max_n:
                    max_cnt -= 1
                # 移动左指针
                l += 1
            # 如果当前窗口内最大元素的出现次数恰好等于k，更新结果
            if max_cnt == k:
                res = l + 1
        # 返回满足条件的子数组数量
        return res
```

## Minimum Window Substring

### 问题描述
给定两个字符串 `s` 和 `t`，你需要在 `s` 中找到一个最小的子串，使得该子串包含 `t` 中的所有字符（包括重复字符）。如果这样的子串存在，则返回这个子串，如果不存在，则返回空字符串。

### 输入
- 一个字符串 `s`。
- 一个字符串 `t`。

### 输出
- 一个字符串，表示 `s` 中最小的包含 `t` 中所有字符的子串。如果不存在，则返回空字符串。

### Solution

1. **初始化变量**:
   - `countT` 字典用于存储 `t` 中每个字符的出现次数。
   - `window` 字典用于记录当前窗口中各个字符的出现次数。
   - `have` 记录当前窗口中已经满足 `t` 中字符需求的数量。
   - `need` 记录需要满足的不同字符的总数，即 `countT` 的长度。
   - `res` 存储结果子串的起始和结束位置，初始为 `[-1, -1]`。
   - `resLen` 存储当前最小子串的长度，初始为无穷大。

2. **构建目标字符计数**:
   - 遍历字符串 `t`，构建 `countT` 字典，记录 `t` 中每个字符的出现次数。

3. **滑动窗口遍历字符串**:
   - 使用右指针 `r` 遍历字符串 `s`，将字符 `s[r]` 加入 `window` 中。
   - 如果 `s[r]` 在 `countT` 中，并且当前窗口中的该字符数量与 `countT` 中相等，增加 `have` 计数。
   - 当 `have` 等于 `need` 时，说明当前窗口已经包含了 `t` 中的所有字符，尝试收缩窗口：
     - 如果当前窗口的长度小于 `resLen`，更新结果 `res` 和 `resLen`。
     - 然后，通过移动左指针 `l`，从窗口中移除最左侧的字符，并检查是否影响 `have` 计数。

4. **返回结果**:
   - 如果找到了满足条件的子串，则返回该子串，否则返回空字符串。

### Code

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if t == "":
            return ""

        countT, window = {}, {}
        for c in t:
            countT[c] = 1 + countT.get(c, 0)

        have, need = 0, len(countT)
        res, resLen = [-1, -1], float("infinity")
        l = 0
        for r in range(len(s)):
            c = s[r]
            window[c] = 1 + window.get(c, 0)

            if c in countT and window[c] == countT[c]:
                have += 1

            while have == need:
                # 更新结果
                if (r - l + 1) < resLen:
                    res = [l, r]
                    resLen = r - l + 1
                # 从窗口左侧移除字符
                window[s[l]] -= 1
                if s[l] in countT and window[s[l]] < countT[s[l]]:
                    have -= 1
                l += 1

        l, r = res
        return s[l : r + 1] if resLen != float("infinity") else ""
```


## Sliding Window Maximum

### 问题描述
给定一个整数数组 `nums` 和一个整数 `k`，你需要找到数组中每个长度为 `k` 的滑动窗口的最大值。返回包含这些最大值的数组。

### 输入
- 一个整数数组 `nums`。
- 一个整数 `k`，表示滑动窗口的大小。

### 输出
- 一个整数数组，包含每个滑动窗口的最大值。

### Solution

1. **初始化变量**:
   - `output` 用于存储每个滑动窗口的最大值。
   - `q` 是一个双端队列（`deque`），用于存储当前窗口中有可能成为最大值的元素的索引。
   - `l` 和 `r` 分别为左指针和右指针，初始化为 `0`。

2. **滑动窗口遍历数组**:
   - 使用右指针 `r` 遍历数组。
   - 如果队列中最后一个元素对应的值小于当前元素 `nums[r]`，则将其弹出，因为它不可能再成为窗口内的最大值。
   - 将当前元素的索引 `r` 添加到队列 `q` 中。

3. **维护窗口的大小**:
   - 如果左指针 `l` 超过了队列中第一个元素的索引，说明队列中的这个元素已经不在当前窗口范围内，需要将其移除。
   - 当窗口的大小达到 `k` 时，窗口的最大值即为队列中第一个元素对应的值，将其添加到 `output` 中。
   - 移动左指针 `l`，以缩小窗口的范围。

4. **返回结果**:
   - 最终的 `output` 包含每个滑动窗口的最大值，返回这个数组。

### Code

```python
from collections import deque
from typing import List

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        output = []
        q = deque()  # 双端队列，用于存储当前窗口中的可能最大值的索引
        l = r = 0
        
        while r < len(nums):
            # 移除队列中所有小于当前元素的值，因为它们不可能成为最大值
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            q.append(r)

            # 如果队列中最左边的值已经不在窗口内，移除它
            if l > q[0]:
                q.popleft()

            # 当窗口大小达到k时，将当前窗口的最大值加入结果
            if (r + 1) >= k:
                output.append(nums[q[0]])
                l += 1  # 缩小窗口
            r += 1  # 扩大窗口
        
        return output
```
## Contain Duplicate
Create a set, check if already exists.

## isAnagram
Check the word count for every character. Pay attention to the length: if the lengths are different, there's no need to check further. Compare the result for equality. The key is the words, and the value is the counts stored in a dictionary.

## Get Concatenation
**Input:** `nums = [1,2,1]`  
**Output:** `[1,2,1,1,2,1]`

Iterate through the array twice.

## Replace Element
**Input:** `[17,18,5,4,6,1]`  
**Output:** `[18,6,6,6,1,-1]`

Since the last element needs to be replaced with -1:
- Record the max value.
- Update it.
- Renew the value.
- If the current value has no greater element on its right, replace it with -1.

## isSubsequence
**Input:** `s = "abc"`, `t = "ahbgdc"`  
**Output:** `True`

Use two pointers for the smaller string.

## Length of the Last Element
**Input:** `s = "Hello World"`  
**Output:** `5`

Reverse iterate, exclude the empty spaces at the start, return the count if it's greater than 1, otherwise keep adding, and return at the end.

## Two Sum
Use `prevMap` to return the index. Use a hashmap for `value:index`. Enumerate through, `diff = target - n`, and if not found, keep updating `value:index`.

## Group Anagram
`ans => dic(list)`

Get count for 26 characters:
- `#word #C`
- `ord(c - "a") += 1`

Word count goes into `answer`. `count:key, v: word`, where one key can have multiple values. Convert the list to a tuple since it cannot be used as a key. Return `ans.values()`.

## Generate
Initialize the result, make `[0]` on both sides. Get the height, calculate every width by the previous one. Since the first is already obtained, calculate each number.

## Remove Element
Given a value, `target`, return the count not equal to the target and modify the non-existent numbers to `_`.

Get a pointer count with an index. If not satisfied, increase the pointer and replace the current number with the pointer count from the start.

## Num of Unique Email
**Input:** `emails = ["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]`

Create a set. Split the email into local and domain parts. The local part is the first split `[0]`.

## isIsomorphic
Use two hash sets to check if the characters exist inside each other and keep adding.

## CanFlower
Check if both sides have zero, directly add to it. Check if three empty spaces are the only option. Iterate through the entire array.

## Major Element
Use a counter. Whatever is left is the major element.

## Next Greater Element
Some element `x` in an array is the first greater element to the right of `x` in the same array.

**Input:** `nums1 = [4,1,2]`, `nums2 = [1,3,4,2]`  
**Output:** `[-1,3,-1]`

Store `v:i` for `num1`. Initialize all to `-1`, which could solve the details. Use a stack, iterate through `num2`, get the current value, and if it exists in `num1`, add it to the stack. Only pop the value if it's greater than the stack. Last in, first out. Get the index of the current `num1` by the value popped, then add the value to the current index.

## Pivot Index
**Problem:** The pivot index is where the sum of all the numbers to the left equals the sum of all the numbers to the right.

**Input:** `nums = [1, 7, 3, 6, 5, 6]`  
**Output:** `3`

- Get the total sum.
- Increment `leftsum += num[i]`.
- Calculate `right sum = total - num[i] - leftsum`.

## Num Array
求一定范围内的和。

Create a prefix sum, then sum the numbers from 0 to the prefix. `left` and `right` define the range. Subtract the right sum from the left sum. The left sum is `self.prefix[left-1] if left > 0 else 0`.

## Find Disappeared Numbers
Map `[0,n-1]` to `[1,n]`. Some numbers may not exist, so they wouldn't be indexed.

Iterate through all the numbers. This index is positive, so `abs(n)-1`. Turn each appropriate index negative. Create a `res`, and append each index. If there's a positive number, it means the number is missing, so add it directly.

## Maximum Number of Balloons
**Problem:** Count the maximum number of times the word "balloon" can be formed from the given string.

**Solution:** Count the frequency of each letter in the word "balloon" and use the minimum count considering the repetition of 'l' and 'o'.

## Word Pattern
**Problem:** Determine if a pattern matches a string following a one-to-one mapping.

**Solution:** Use two hash maps to ensure bijective mapping between pattern characters and words.

## Design HashSet
**Problem:** Implement a hash set without using built-in hash set libraries.

**Solution:** Use a list of buckets, each bucket being a list, to handle collisions.

## Design HashMap
**Problem:** Implement a hash map without using built-in hash map libraries.

**Solution:** Similar to HashSet, use a list of buckets where each bucket is a list of key-value pairs.

## Monotonic Array
**Solution:** Check if the entire array is sorted in non-decreasing order or non-increasing order.

## Number of Good Pairs
**Problem:** Count the number of good pairs `(i, j)` such that `nums[i] == nums[j]` and `i < j`.

**Solution:** Use a hash map to count occurrences of each number and use combinations to calculate pairs.

## Pascal's Triangle II
**Problem:** Return the k-th row of Pascal's triangle.

- Return the `rowIndex-th` (0-indexed) row of Pascal's triangle.
- Initialize the first row as `[1]`.
- Initialize the next entire array.
- Each element in the next row equals `res[j]`.

## Find Words That Can Be Formed by Characters
**Problem:** Count words that can be formed using the given characters.

**Solution:** Use a counter to compare the frequency of characters in each word with the given characters.

## Largest Number
**Problem:** Find the largest number formed by three identical digits in a string.

**Input:** `s = "6777133339"`  
**Output:** `"777"`

**Solution:** Iterate through the string and check for triplets of the same digit. The row always has the same width and the gap is the same length. The key is the position, which is the length; the value is the count of the gap. Find the key with the most gaps.

```python
class Solution:
    def largestGoodInteger(self, num):
        res = "0"
        for i in range(len(num) - 2):
            if num[i] == num[i + 1] == num[i + 2]:
                return max(res, num[i:i + 3])
        return "" if res == "0" else res
```




## 1608.    Special Array With X Elements Greater Than or Equal X
Given an array `arr` of integers, find the largest integer `x` such that there are at least `x` elements in the array that are greater than or equal to `x`.

### Example
#### Input:
`arr = [3, 5, 6, 7, 8]`

#### Steps:
- **Sort the array**: `arr = [3, 5, 6, 7, 8]`
- **Iterate through the array**:
  - For `x = 3`, there are 5 elements ≥ 3 (condition not satisfied).
  - For `x = 5`, there are 4 elements ≥ 5 (condition not satisfied).
  - For `x = 6`, there are 3 elements ≥ 6 (condition satisfied).
  - For `x = 7`, there are 2 elements ≥ 7 (condition not satisfied).
  - For `x = 8`, there is 1 element ≥ 8 (condition not satisfied).
- **Result**: The largest `x` that satisfies the condition is `6`.

#### Output:
`6`

### Implementation
This problem can be approached by first sorting the array and then checking each element to see if it satisfies the condition of being the largest integer such that there are at least `x` elements greater than or equal to `x`.

```python
class Solution:
    def specialArray(self, nums):
        nums.sort()  # 将数组排序
        prev = -1  # 初始化 prev 为 -1
        total_right = len(nums)  # total_right 初始值为数组的长度
        i = 0  # 初始化索引 i

        while i < len(nums):
            # 检查当前数是否等于 total_right 或者 total_right 是否在 prev 和 nums[i] 之间
            if nums[i] == total_right or (prev < total_right < nums[i]):
                return total_right
            
            # 跳过相同的数字，确保 i 指向下一个不同的元素
            while i + 1 < len(nums) and nums[i] == nums[i + 1]:
                i += 1
            
            # 更新 prev 为当前元素 nums[i]
            prev = nums[i]
            # 移动索引 i 指向下一个元素
            i += 1
            # 更新 total_right 为数组右边剩余的元素数量
            total_right = len(nums) - i

        return -1  # 如果没有找到符合条件的值，返回 -1
```


### 解法描述2

1. **频率表想象**：
   - 想象一个频率表，从上到下按大小排序。
   - 由于数组中的元素都是整数，可以将最后一个元素作为分界点，把所有大于等于该值的元素都放在一起。

2. **反向检查**：
   - 由于数组中的元素是整数，我们可以反向逐个检查，累计大于等于当前值的元素个数。
   - 通过这种方式，我们可以找到一个值，使得数组中大于等于这个值的元素个数恰好等于该值。然后加起来，就知道大于等于的有多少个

### 代码实现

```python
def specialArray(nums):
    nums.sort()  # 将数组排序
    prev = -1  # 初始化 prev 为 -1
    total_right = len(nums)  # total_right 初始值为数组的长度，表示右边剩余元素的数量
    i = 0  # 初始化索引 i

    while i < len(nums):
        # 检查当前数是否等于 total_right 或者 total_right 是否在 prev 和 nums[i] 之间
        if nums[i] == total_right or (prev < total_right < nums[i]):
            return total_right
        
        # 跳过相同的数字，确保 i 指向下一个不同的元素,并且保证在范围内
        while i + 1 < len(nums) and nums[i] == nums[i + 1]:
            i += 1
        
        # 更新 prev 为当前元素 nums[i]
        prev = nums[i]
        # 移动索引 i 指向下一个元素
        i += 1
        # 更新 total_right 为数组右边剩余的元素数量
        total_right = len(nums) - i

    return -1  # 如果没有找到符合条件的值，返回 -1
```

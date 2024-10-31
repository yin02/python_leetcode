

# 位运算 `x ^ 1` 的详细解释和示例
**<span style="color:red; font-size:1.2em;">计算 x 的邻近 index 应该出现的位置</span>**

`x ^ 1` 是一个位运算表达式，通常用于找到数组中索引 `x` 的“相邻索引”。这个操作翻转了 `x` 的二进制表示中的最低位（即 `0` 变为 `1` 或 `1` 变为 `0`），从而有效地找到了与 `x` 形成成对关系的索引。

- **如果 `x` 是偶数**：`x ^ 1` 的结果为 `x + 1`。
- **如果 `x` 是奇数**：`x ^ 1` 的结果为 `x - 1`。

这意味着，`x ^ 1` 可以用来找到与 `x` 相邻的、成对的元素索引。

### 示例 1：处理成对的元素

假设我们有一个数组，其中每个元素都成对出现，除了一个元素之外：

```python
# 示例数组
nums = [1, 1, 2, 2, 3, 4, 4]

# 假设 x = 4
x = 4

# 计算 x ^ 1 ,计算x的邻近index 应该出现的位置
adjacent_index = x ^ 1  # 结果是 5

# 打印结果


x = 4, nums[x] = 3
adjacent_index = 5, nums[adjacent_index] = 4
```





在这个例子中：

x = 4 对应的元素是 3。
x ^ 1 = 5 对应的元素是 4，它与 x 处的元素不同，表示这不是一个成对的索引关系


## 示例数组
nums = [1, 1, 2, 2, 3, 3, 4, 4]

## 假设 x = 2
x = 2

## 计算 x ^ 1
adjacent_index = x ^ 1  # 结果是 3

## 打印结果

```

x = 2, nums[x] = 2
adjacent_index = 3, nums[adjacent_index] = 2

```

### 总结

- **偶数索引**：如果 `x` 是偶数，那么 `x ^ 1` 会给出 `x + 1`，即下一个索引，它们应当是成对的。
- **奇数索引**：如果 `x` 是奇数，那么 `x ^ 1` 会给出 `x - 1`，即上一个索引，它们应当是成对的。

例如，假设 `x = 4`，那么 `x ^ 1` 会得到 `5`。这意味着索引 `4` 和 `5` 的元素应当是成对出现的 **（4 和 5 应该成对出现）**。


### `bisect_left` 基本用法

`bisect_left` 是 Python 标准库 `bisect` 模块中的一个函数，用于在有序序列中找到指定值的插入点。与 `bisect_right` 类似，`bisect_left` 返回的是在保持序列有序的情况下，可以插入值的位置，并且插入点位于所有等于该值的元素的左侧。

#### 基本示例

```python
import bisect

# 定义一个有序列表
sorted_list = [1, 2, 4, 4, 5, 6]

# 使用 bisect_left 查找 4 的插入点
index = bisect.bisect_left(sorted_list, 4)

print(index)  # 输出 2
```
### `bisect_right` 基本用法

`bisect_right` 是 Python 标准库 `bisect` 模块中的一个函数，用于在有序序列中找到指定值的插入点。与 `bisect_left` 类似，`bisect_right` 返回的是在保持序列有序的情况下，可以插入值的位置，但插入点位于所有等于该值的元素的右侧。

#### 基本示例

```python
import bisect

# 定义一个有序列表
sorted_list = [1, 2, 4, 4, 5, 6]

# 使用 bisect_right 查找 4 的插入点
index = bisect.bisect_right(sorted_list, 4)

print(index)  # 输出 4
```
```py
import bisect

# 定义一个有序列表
sorted_list = [1, 2, 4, 4, 5, 6]

# 查找插入点
left_index = bisect.bisect_left(sorted_list, 4)
right_index = bisect.bisect_right(sorted_list, 4)

print(left_index)  # 输出 2, 4 应插入在索引 2 位置（现有的 4 之前）
print(right_index)  # 输出 4, 4 应插入在索引 4 位置（现有的 4 之后）
```




### 参数解释

- **a（范围 `range(len(nums) - 1)`）**：
  - 这是一个 `range` 对象，生成从 `0` 到 `len(nums) - 2` 的一系列索引。这些索引将用于遍历 `nums` 数组的元素，以找到唯一的非重复元素。

- **x（值 `True`）**：
  - 在此场景中，`bisect_left` 使用 `True` 作为要查找的插入点的目标值。`bisect_left` 会在 `a` 中找到第一个使 `key(x)` 返回 `True` 的位置。

- **key（lambda 表达式 `lambda x: nums[x] != nums[x ^ 1]`）**：
  - `key` 是一个函数，用于计算 `a` 中每个元素的值以进行比较。在这个例子中，它通过 `lambda` 表达式 `lambda x: nums[x] != nums[x ^ 1]` 来判断索引 `x` 处的元素是否与其相邻的索引（`x ^ 1`）处的元素不相等。 
  - `x ^ 1` 用于找到与 `x` 成对的索引，即：
    - 如果 `x` 是偶数，则 `x ^ 1` 为 `x + 1`。
    - 如果 `x` 是奇数，则 `x ^ 1` 为 `x - 1`。
  - 该 `lambda` 表达式返回 `True` 或 `False`，用于告诉 `bisect_left` 是否找到了唯一的非重复元素的位置。

### 总结

- `bisect_left` 使用 `range(len(nums) - 1)` 生成一系列索引，并通过 `lambda` 表达式来判断每个索引 `x` 和其相邻索引处的元素是否不相等。
- `bisect_left` 将返回第一个使 `key` 函数返回 `True` 的位置，从而确定唯一的非重复元素的索引。


## 题目: 单一非重复元素

给定一个排序数组，其中每个元素都恰好出现两次，只有一个元素出现一次。找出这个唯一的非重复元素。

### 输入:
- 一个整数数组 `nums`，其中数组长度为奇数 `2n + 1`，并且除了一个元素外，所有元素都恰好出现两次。

### 输出:
- 返回数组中唯一出现一次的元素。



```py
def singleNonDuplicate(self, nums: List[int]) -> int:
        return nums[bisect_left(range(len(nums) - 1), True, key=lambda x: nums[x] != nums[x ^ 1])]
```

## 一句话解释
- **返回第一个是 `True` 的值，也就是 `nums[x] != nums[x ^ 1]`，原本所在的索引（通过 `bisect_left` 查找）对应的值（`nums[index]`）。**

- **这段代码通过 `bisect_left` 查找第一个使 `nums[x] != nums[x ^ 1]` 为 `True` 的索引，然后返回该索引在原数组 `nums` 中对应的值（即 `nums[index]`）。**

## 在给定的数组 `[1,1,2,3,3,4,4,8,8]` 中，我们通过 `key` 函数 `lambda x: nums[x] != nums[x ^ 1]` 来判断数组中的元素是否与其相邻元素相等，从而生成一个布尔数组。

### 原数组：

`[1, 1, 2, 3, 3, 4, 4, 8, 8]`

### 应用 `key` 函数后的布尔数组：

通过 `lambda x: nums[x] != nums[x ^ 1]` 计算，我们得到以下布尔数组：

`[False, False, True, True, True, True, True, True, True]`

- `False` 表示索引 `x` 处的元素与 `x ^ 1` 处的元素相等（即成对出现）。
- `True` 表示索引 `x` 处的元素与 `x ^ 1` 处的元素不相等（即不成对）。

### 目标：

我们要找到布尔数组中第一个 `True` 的位置，这相当于在数组中找到第一个非成对元素的位置。

通过 `bisect_left`，我们可以有效地找到这个位置，即找到第一个 `True` 的位置（左插一个 `True`），从而确定唯一的非重复元素的索引。



## right写法
```py
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        # bisect_right和bisect_left等价写法
        return nums[bisect_right(range(len(nums) - 1), False, key=lambda x: nums[x] != nums[x ^ 1])]
```


## dict.get() 方法的语法
``` py
dic = {}

value = dict.get(key, default_value)
```

key：需要查找的键。
default_value（可选）：如果键不存在时返回的默认值。默认情况下，这个参数是 None

``` py
my_dict = {'name': 'Alice', 'age': 25, 'city': 'New York'}

# 存在的键
name = my_dict.get('name')
print(name)  # 输出: Alice

# 不存在的键
country = my_dict.get('country')
print(country)  # 输出: None

my_dict = {'name': 'Alice', 'age': 25, 'city': 'New York'}

# 存在的键
name = my_dict.get('name', 'Unknown')
print(name)  # 输出: Alice

# 不存在的键，返回默认值
country = my_dict.get('country', 'USA')
print(country)  # 输出: USA

my_dict = {'name': 'Alice', 'age': 25, 'city': 'New York'}

# 安全地获取键的值
age = my_dict.get('age', 0)  # 如果 'age' 不存在，则返回 0
print(age)  # 输出: 25

# 如果键不存在时提供默认值
salary = my_dict.get('salary', 0)  # 如果 'salary' 不存在，则返回 0
print(salary)  # 输出: 0


```


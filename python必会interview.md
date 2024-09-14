# Sorting 
## example
```python
# Sorting a List of Strings,先比较第一个字母，再比较第二个字母
elements = ["grape", "apple", "banana", "orange"]

elements.sort()  # Sorts the list in place

print(elements)  # Output: ['apple', 'banana', 'grape', 'orange']
```
## Sorting a List of Integers in Descending Order
``` py
elements = [5, 3, 6, 2, 1]

elements.sort(key=None, reverse=True)  # Sorts the list in place, in descending order

print(elements)  # Output: [6, 5, 3, 2, 1]
```

# Sorting Words by Length Using a Custom Function
``` py
def get_word_length(word: str) -> int:
    return len(word)

words = ["apple", "banana", "kiwi", "pear", "watermelon", "blueberry", "cherry"]

words.sort(key=get_word_length)  # Sorts the list in place by word length

print(words)  # Output: ['kiwi', 'pear', 'apple', 'banana', 'cherry', 'blueberry', 'watermelon']
```
## Sorting Words by Length Using a Lambda Function
``` py
words = ["apple", "banana", "kiwi", "pear", "watermelon", "blueberry", "cherry"]

words.sort(key=lambda word: len(word))  # Sorts the list in place by word length

print(words)  # Output: ['kiwi', 'pear', 'apple', 'banana', 'cherry', 'blueberry', 'watermelon']
```
The lambda function `lambda word: len(word)` is equivalent to the function `get_word_length` we defined in the previous example. It takes a word as input and returns the length of the word.

The syntax includes:

- **The keyword** `lambda`.
- **The input variable** `word`. We can use any variable name here.
- **The colon** `:`, after which we define the function body.
- **The expression** `len(word)`, which is the return value of the function.


## Sorted Copy

There is another way to sort a list in Python, using the sorted() function. The sorted() function returns a new list with the elements sorted in the specified order. The original list remains unchanged

``` py
words = ["kiwi", "pear", "apple", "banana", "cherry", "blueberry"]

sorted_words = sorted(words)

print(sorted_words) # ['apple', 'banana', 'blueberry', 'cherry', 'kiwi', 'pear']

numbers = [5, -3, 2, -4, 6, -2, 4]


# reverse

sorted_numbers = sorted(numbers, reverse=True)

print(sorted_numbers) # [6, 5, 4, 2, -2, -3, -4]


# abs

numbers = [5, -3, 2, -4, 6, -2, 4]

sorted_numbers = sorted(numbers, key=abs)

print(sorted_numbers) # [2, -2, -3, 4, -4, 5, 6]


```


# Unpacking
The biggest advantage of using Python for coding interviews is its simplicity and readability. In this section we will learn some of the shortcuts Python provides to make our code easy to read and write.

One of these shortcuts is unpacking
``` py
point1 = [0, 0]
point2 = [2, 4]

x1, y1 = point1 # x1 = 0, y1 = 0
x2, y2 = point2 # x2 = 2, y2 = 4

slope = (y2 - y1) / (x2 - x1)

print(slope) # Output: 2.0



# The below code accomplishes the same without unpacking but with slightly more code:
x1, y1 = point1[0], point1[1]
x2, y2 = point2[0], point2[1]


If we attempt unpacking with too many variables on the left-side, we will get a ValueError.
x, y = [0, 0, 0] # ValueError: too many values to unpack (expected 2)

Unpacking also works with tuples and sets with the same syntax.
```


## loop unpacking
We can also use unpacking in loops to iterate over a list of lists. This is useful when we know the size of the inner lists and want to unpack them into variables.
```py
points = [[0, 0], [2, 4], [3, 6], [5, 10]]

for x, y in points:
    print(f"x: {x}, y: {y}")

# We could accomplish this without packing with slightly more code:
for point in points:
    x, y = point[0], point[1]
    print(f"x: {x}, y: {y}")

```


## Enumerate
Suppose we wanted to loop over an array and we needed to access both the index and the element at that index. This is simple to accomplish:
```py
nums = [2, 7, 9, 2]

for i in range(len(nums)):
    n = nums[i]
    print(i, n)
```

But the Pythonic way to do this is to use the enumerate() function:
``` py
nums = [2, 7, 9, 2]

for i, n in enumerate(nums):
    print(i, n)
```

The enumerate() function returns a tuple of the index and the element at that index. We can unpack this tuple into two variables in the for loop.

``` py

names = ['Alice', 'Bob', 'Charlie']

# This is more readable than using range()
for name in names:
    print(name)


# This is more readable than using range() 
# and still allows us to access the index 
# of each element
for i, name in enumerate(names):
    print(i, name)
``` 

## Zip
Python provides an easy way to iterate over multiple lists at the same time using the zip() function. The zip() function takes multiple lists as arguments and returns an iterator of tuples. Each tuple contains an element from each list.

This is useful when we have multiple lists of the **<span style="color:red">same length and want to iterate over them together
</span>**


```py
names = ['Alice', 'Bob', 'Charlie', 'David']
scores = [90, 85, 88, 92]

for name, score in zip(names, scores):
    print(f"{name} scored {score}")
```

## inequality
``` py
Python allows us to take a small shortcut when making multiple comparisons.

The following code:

x = 5

if 0 < x and x < 10:
    print('x is between 0 and 10')
#is equivalent to:

x = 5

if 0 < x < 10:
    print('x is between 0 and 10')
```


## Min Max Shortcut

```py
transactions = -2

if transactions < 0:
    transactions = 0
```
into
``` py
transactions = -2

transactions = max(0, transactions)

```

``` py

transactions = 101

# This will ensure transactions is never greater than 100
if transactions > 100:
    transactions = 100

# This is equivalent to the above code
transactions = min(100, transactions)

```
## Resizable List
`append()`: Adds an element to the end of the list.
`pop()`: Removes and returns the last element of the list
`insert()`: Inserts an element at a specified index in the list.
```py
my_list = [1, 2, 3]

my_list.append(4) # [1, 2, 3, 4]
my_list.append(5) # [1, 2, 3, 4, 5]

my_list.pop() # [1, 2, 3, 4]

my_list.insert(1, 3) # [1, 3, 2, 3, 4]
```

### small pratice oop

`append_elements(arr1: List[int], arr2: List[int]) -> List[int]. `
It should append the elements of arr2 to the end of arr1 and return the resulting list.

`pop_n(arr: List[int], n: int) -> List[int].`
 It should remove the last n elements from the list and return the resulting list. If n is greater than the length of the list, it should return an empty list.

`insert_at(arr: List[int], index: int, element: int) -> List[int]. `
It should insert the element at the specified index in the list and return the resu
```py
from typing import List


def append_elements(arr1: List[int], arr2: List[int]) -> List[int]:
    for elements in arr2:
        arr1.append(arr2)
    return arr1



def pop_n(arr: List[int], n: int) -> List[int]:
    while n >0 and arr:
        arr.pop()
        n-=1
    return arr


def insert_at(arr: List[int], index: int, element: int) -> List[int]:
    if index < 0 or index >= len(arr):
        arr.append(element)
    else:
        arr.insert(index,element)
    return arr
```

### Time Complexity of Common List Operations

**append()**
- Time complexity: O(1)

**pop()**
- Time complexity: O(1)

**insert()**
- Time complexity: O(n)
  - Where `n` is the number of elements in the list.



`index()`: Returns the index of the first occurrence of a specified element in the list.
If the element is not in the list, we will get an ValueError.
`remove()`: Removes the first occurrence of a specified element from the list.
`extend()`: Adds the elements of another list to the end of the list.
```py
my_list = [1, 3, 2, 3]

my_list.index(3) # 1

my_list.remove(3) # [1, 2, 3]

my_list.extend([4, 5]) # [1, 2, 3, 4, 5]
```
If we want to check if an element is in a list, we can use the in operator:

```py
my_list = [1, 2, 3]

if 3 in my_list:
    print("3 is in the list")
```
`append_elements(arr1: List[int], arr2: List[int]) -> List[int]`. It should append the elements of arr2 to the end of arr1 and return the resulting list. Yes, this is the same function from the previous lesson.
`remove_elements(arr1: List[int], arr2: List[int]) -> List[int]`. It should remove all elements of arr2 from arr1 and return the resulting list.

```py

def append_elements(arr1: List[int], arr2: List[int]) -> List[int]:
    arr1.extend(arr2)
    return arr1
  

def remove_elements(arr1: List[int], arr2: List[int]) -> List[int]:
    for number in arr2:
        if number in arr1:
            arr1.remove(number)
    return arr1


```

### list contact
```py
list1 = [1, 2, 3]
list2 = [4, 5, 6]

result = list1 + list2 # [1, 2, 3, 4, 5, 6]
```


### List Initialization

f we want to create an empty list of a specific size, this is the easiest way to do it in Python
```py
my_list = [0] * 5
```

The above code will create a list of 5 zeros. It might seem strange to multiply a list by a number, but this is the standard way to create a list of a specific size in Python. We could have replaced the 0 with any other value.
```py
my_list = [1] * 3
```
`create_list_with_value(size: int, index: int, value: int) -> List[int]` which should create and return a list of length size. All values in the list should be 0, except for the value at index index, which should be the parameter value.
``` py
def create_list_with_value(size: int, index: int, value: int) -> List[int]:
    arr = [0] * size
    arr[index] = value
    return arr
```

It's common to need to make a copy of a list in many algorithms. Python provides multiple ways to clone a list. Here are a few ways to do it:

1.Using the copy() method:
```py
original_list = [1, 2, 3]
cloned_list = original_list.copy()
```
2. Using the slicing syntax:
```py
original_list = [1, 2, 3]
cloned_list = original_list[:]
```
3.Using the list() constructor:
```
original_list = [1, 2, 3]
cloned_list = list(original_list)
```

Keep in mind, that if you have a list of objects, the above methods will not create **deep copies** of the elements within the list. In this case we have a list of integers, which are primitive types, so we don't need to worry about that. But if you had a **list of lists**, for example, you would need to use the copy.deepcopy() method to create a deep copy.

4.Using copy.deepcopy() for deep copies:
```py
import copy

original_list = [[1, 2], [3, 4]]

cloned_list = copy.deepcopy(original_list)
```

### List Comprehension
A very powerful feature in Python is comprehension. It applies to lists and other data types in Python that we will cover later. It allows us to **create lists in a concise way**. Here's an example:
```py
my_list = [i for i in range(5)]

print(my_list) # [0, 1, 2, 3, 4]
```


we have two lists arr1 and arr2. We use the `zip()` function to iterate over `both lists` at the same time. We then `add the elements of both lists` together and create a new list. The expression i + j is the value we want to add to the list.
```py
arr1 = [1, 2, 3]
arr2 = [4, 5, 6]

result = [i + j for i, j in zip(arr1, arr2)]

```

condition to the comprehension:
we have a list `arr`. We iterate over the list and add the element to the `new list` only if it's even. This might seem like overkill since we can accomplish the same thing with a simple loop, but it demonstrates the flexibility of list comprehension.
```py
arr = [1, 2, 3, 4, 5]

result = [i for i in arr if i % 2 == 0]
```

## stack

`append()` is used to push an element onto the stack.
```python
stack = []
stack.append(1)
stack.append(2)
```
`pop()` is used to remove and return the top element from the stack.

```py
stack = [1, 2]

top_element = stack.pop() # 2
```

[-1] can be used to access the top element of the stack `without removing it`. This assumes that the stack is not empty.

```py
stack = [1, 2]

top_element = stack[-1] # 2

```

len() can be used to check if the stack is empty.
```py
stack = [1, 2]

while len(stack) > 0:
    print(stack.pop())
```

### Time Complexity of List Operations

- **append()**: O(1)
- **pop()**: O(1)
- **[-1]** (accessing the last element): O(1)
- **len()**: O(1)

### Queue Enqueue and Dequeue
    append()` is used to enqueue an element to the right side of the queue.
```py
from collections import deque

queue = deque()

queue.append(1) # [1]
queue.append(2) # [1, 2]
```
`popleft()` is used to remove and return the leftmost element from the queue.

```py
queue = deque([1, 2]) # pass a list to initialize the queue

queue.popleft() # [2]
queue.popleft(2) # []
```

[0] and [-1] can be used to access the leftmost and rightmost elements of the queue respectively. This assumes that the queue is not empty.

```py
queue = deque([1, 2, 3, 4])

leftmost_element = queue[0] # 1
rightmost_element = queue[-1] # 4
```

len() can be used to check if the queue is empty.
```py
queue = deque([1, 2])

while len(queue) > 0:   
    print(queue.popleft())
```
### Time Complexity

- **append()**: O(1)
- **popleft()**: O(1)
- **[0]**, **[-1]**, or **[i]** (where `i` is a valid index): O(1)
- **len()**: O(1)

### Double Ended Queue

The `deque` provided in Python is actually a double-ended queue. It allows you to push or pop from either end of the queue efficiently. Here are two more common operations supported by a deque:

1. appendleft() is used to enqueue an element to the left side of the queue.
```python
from collections import deque

queue = deque()

queue.appendleft(1) # [1]
queue.appendleft(2) # [2, 1]
```

2.`pop()` is used to remove and return the rightmost element from the queue.
```py
queue = deque([1, 2]) # pass a list to initialize the queue

queue.pop() # [1]
queue.pop() # []
```
### Time Complexity

- **appendleft()**: O(1)
- **pop()**: O(1)
Implement the following function using the queue operations described above:

`rotate_list(arr: List[int], k: int) -> Deque[int]` that takes a list of integers arr and an integer k. It should convert the list into a deque. And next, rotate the values in the list to the right by k steps and return the resulting deque.
Example: rotate_list([1, 2, 3, 4, 5], 2) should return deque([4, 5, 1, 2, 3]).

```python
def rotate_list(arr: List[int], k: int) -> Deque[int]:
    queue = deque(arr)
    for i in range(k):
        queue.appendleft(queue.pop())
    return queue

    # do not modify below this line
print(rotate_list([1, 2, 3, 4, 5], 0))
print(rotate_list([1, 2, 3, 4, 5], 1))
print(rotate_list([1, 2, 3, 4, 5], 2))
print(rotate_list([1, 2, 3, 4, 5], 3))
print(rotate_list([1, 2, 3, 4, 5], 4))
print(rotate_list([1, 2, 3, 4, 5], 5))
```


### Multi-Dimensional List

```py
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

nested_list = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

print(nested_list[0])  # [1, 2, 3]
print(nested_list[1])  # [4, 5, 6]
print(nested_list[2])  # [7, 8, 9]

print(nested_list[0][0])  # 1

print(nested_list[2][2])  # 9

print(nested_list[1][2])  # 6

```

```py

def find_max_in_each_list(nested_arr: List[int]) -> List[int]:
    output = []
    for arr in nested_arr:
        max_in_arr = arr[0]
        for num in arr:
            max_in_arr = max(max_in_arr,num)
        output.append(max_in_arr)
    return output
```

### 2D Grid

It's common to represent a 2D grid as a list of lists in Python. For example, a 2x3 grid can be represented as:
```py
grid = [
    [0, 0, 0],
    [0, 0, 0]
]

rows = len(grid)    # 2
cols = len(grid[0]) # 3
```
In this example, the variable grid is a list of lists. The variable rows is the number of rows in the grid, and the variable cols is the number of columns in the grid. We assume that each sub-list in the grid has the same length, which is usually the case for a 2D grid in algorithm problems.

### Nested List Comprehension
It's common to need to initialize a 2-D list of a given size, especially a 2-D grid. Suppose we wanted to declare a 2x3 grid of zeros. You might be tempted to try this:
```py
grid = [[0] * 3] * 2

print(grid) # [[0, 0, 0], [0, 0, 0]]
```


At first it seems correct. We create a list of size 3 with [0] * 3. We add it to a list, and multiply the result by 2. However, this code will not work as expected. The issue is that the **inner list is a reference to the same list object.** This means that if we change one of the inner lists, all the inner lists will change as shown below.
```py
grid = [[0] * 3] * 2

grid[0][0] = 1

print(grid) # [[1, 0, 0], [1, 0, 0]]
```
**A better way** to initialize a 2-D list is to use a nested list comprehension:
```py
grid = [[0 for i in range(3)] for j in range(2)] #rows and columns

grid[0][0] = 1

print(grid) # [[1, 0, 0], [0, 0, 0]]
```

We create the inner list of zeroes with list comprehension. Then we use that list for the outer list with list comprehension again. This works as expected because each inner list is a separate object.

But there's a more concise solution you may prefer:

```py
grid = [[0] * 3 for _ in range(2)]
```
This code is equivalent to the previous code but uses less characters. We create a list of zeroes with [0] * 3 and use it for the outer list with list comprehension. Since the variable in the for loop is not used, we use an **underscore _** to indicate that it is a throwaway variable. This is a common convention in Python.
#Time Complexity of Initializing a 2-D Grid

To initialize a 2-D grid of size `n x m` with a given value:

- **Time Complexity**: O(n * m)

Where:
- `n` is the number of rows.
- `m` is the number of columns.

### Hash Map Basics

**Insertion**: Insert a key-value pair into the hash map.

```python
my_dict = {}
my_dict['a'] = 1  # {'a': 1}
```
Access: Access the value associated with a key.
```python
my_dict = {'a': 1}

print(my_dict['a']) # 1
```

Deletion: Delete a key-value pair from the hash map.
```py
my_dict = {'a': 1, 'b': 2}

del my_dict['a'] # {}

my_dict.pop('b') # {}
```
As shown above, there are two ways to delete a key-value pair from a hash map: using the del keyword or the pop() method. The only difference is that the pop() method returns the value associated with the key, and if the key does not exist, pop() will raise a KeyError, while the del keyword will not.

Lookup: Check if a key exists in the hash map.
```py
my_dict = {'a': 1}
does_a_exist = 'a' in my_dict # True
does_b_exist = 'b' in my_dict # False
```
For lookup operations, you can also use the in operator, similar to how you would check if an element is in a list.

### Default Dict
In algorithms, it's very common to count the frequency of elements in a list or a string. The straight forward way to do this is with the following code:
```py
nums = [1, 2, 4, 3, 2, 1]
freq = {}

for num in nums:
    if num in freq:
        freq[num] += 1
    else:
        freq[num] = 1

print(freq)  # {1: 2, 2: 2, 4: 1, 3: 1}
```

The above code iterates through nums, if a number is not in the dictionary, it adds it with a value of 1. If the number is already in the dictionary, it increments the value by 1. This is perfectly fine, but Python provides a more `elegant` way to do this using the collections module.

The collections module provides a class called `defaultdict` that is a subclass of the built-in dict class. It allows you to set a `default value` for a key that doesn't exist in the dictionary. This can be very useful when counting the frequency of elements in a list. Consider the following code:
```py
from collections import defaultdict

nums = [1, 2, 4, 3, 2, 1]
freq = defaultdict(int)

for num in nums:
    freq[num] += 1

print(freq)  # {1: 2, 2: 2, 4: 1, 3: 1}
```
We removed the need for conditional statements within the loop, but how? Well we created a `defaultdict`, and passed in `int` as the default value. This means that `if a key doesn't exist in the dictionary`, it will be created with a `default value of the integer 0`. This allows us to increment the value without checking if the key exists.

This pattern is also used with other data types, such as lists and sets. For example, if you wanted to create a dictionary where the default value is an empty list, you would use:
```py
nums = [1, 2, 4, 3, 2, 1]
d = defaultdict(list)

for num in nums:
    d[num].append(num)

print(d)  # {1: [1, 1], 2: [2, 2], 4: [4], 3: [3]}
```

### 6 HASHMAP Counter
If all we want to do is count the occurrences of elements in a list, an even better solution exists than the defaultdict. We can use the collections.Counter class:
```py
from collections import Counter

nums = [1, 2, 4, 3, 2, 1]

counter = Counter(nums)

print(counter)  # Counter({1: 2, 2: 2, 4: 1, 3: 1})

counter[100] += 1 # No error, even though 100 is not a key

print(counter)  # Counter({1: 2, 2: 2, 4: 1, 3: 1, 100: 1})
```
The `Counter` class is a subclass of the `dict` class, and it provides a more concise way to count the occurrences of elements in a list. It returns a dictionary where the keys are the elements in the list and the values are the number of times each element appears. It behaves similarly to a defaultdict with an integer default value, meaning if a key doesn't exist, it will return 0.

Sometimes using the Counter class can feel like cheating, so in a real interview it might be worth checking with you're interviewer if they're fine with you using it. Usually if the problem you are given is a challenging one, they should be fine with it. Worst case, you can use a defaultdict or implement the counting logic yourself with a regular dictionary.

If we want to count the occurences of multiple lists or strings, we can use the update() method:

```py
nums1 = [1, 2, 4, 3, 2, 1]
nums2 = [1, 2, 3, 4, 5]

counter = Counter(nums1)
counter.update(nums2)

print(counter)  # Counter({1: 3, 2: 3, 4: 2, 3: 2, 5: 1})
```
## Dict Comprehension


```py
nums = [1, 2, 3, 4, 5]

squared = {num: num * num for num in nums}

print(squared)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
```
In the above code, we created a dictionary where the keys are the numbers from the list nums and the values are the square of each number. Notice that the syntax is similar to list comprehension, but we use curly braces instead of square brackets. Inside the curly braces, we have a key-value pair separated by a colon before the loop.

If we wanted all the keys from a dictionary, we could loop over the dictionary and append the keys to a list. But Python provides a more concise way to get all the keys from a dictionary using the keys() method

```py
my_dict = {"a": 1, "b": 2, "c": 3}

keys = my_dict.keys()

print(keys)  # dict_keys(['a', 'b', 'c'])

keys_list = list(keys)

print(keys_list)  # ['a', 'b', 'c']
```
The keys() method returns a view object that displays a list of all the keys in the dictionary. We can convert this view object to a list using the list() method.

Similarly, we can get all the values from a dictionary using the values() method.
```py
my_dict = {"a": 1, "b": 2, "c": 3}

values = my_dict.values()

print(values)  # dict_values([1, 2, 3])

values_list = list(values)

print(values_list)  # [1, 2, 3]
```
```py
my_dict = {"a": 1, "b": 2, "c": 3}

for key, value in my_dict.items():
    print(key, value)
a 1
b 2
c 3


```

## Hash Set Basics
Insertion: Insert a key into the hash set.
```py
my_set = set()

my_set.add('a') # {'a'}
```
Deletion: Delete a key from the hash set.

```py
my_set = {'a'}

my_set.remove('a') # {}
my_set.remove('a') # KeyError

my_set.add('b') # {'b'}
my_set.discard('b') # {}
my_set.discard('b') # {} (no error)
```

As shown above, you can use the remove() method to delete an element from a hash set. If the element does not exist, the method will raise a KeyError. Alternatively, you can use the discard() method, which will not raise an error if the element does not exist.


Lookup: Check if a key exists in the hash set.

```py
my_set = {'a'}

'a' in my_set # True
'b' in my_set # False
```
- For lookup operations, you can also use the in operator, similar to how you would check if an element is in a list.

Just like with lists and dictionaries, Python also supports set comprehension. We can use set comprehension to initialize sets in a more concise way.

```py
nums = [1, 3, 5]

squared = {num * num for num in nums}

print(squared)  # {1, 9, 25}
```
In the above code, we created a set where the elements are the square of each number in the list nums. Notice that the syntax is similar to list comprehension, but we use curly braces instead of square brackets. Inside the curly braces, we have the expression we want to add to the set.
Both dictionaries and sets use curly braces. The difference between them is that dictionaries have key-value pairs separated by a colon, while sets only have the elements themselves

## Tuple Keys
It's common to store pairs of values in a dictionary or set. For example, we might store the row, column pair of a cell in a grid. While we **cannot store a list as a key in a dictionary**, we can use a **tuple instead**.
```py
dict_of_pairs = {}

dict_of_pairs[(0, 0)] = 1
dict_of_pairs[(0, 1)] = 2

print(dict_of_pairs)  # {(0, 0): 1, (0, 1): 2}

set_of_pairs = set()

set_of_pairs.add((0, 0))
set_of_pairs.add((0, 1))

print(set_of_pairs)  # {(0, 0), (0, 1)}
```
## Heap Push
Heaps (or priority queues) are a data structure that allow you to insert (push) and remove (pop) elements based on some priority associated with each element. In Python, a heap is a minimum heap by default, meaning that the element with the smallest priority is always at the top of the heap.
```py
import heapq

heap = [] # min heap

heapq.heappush(heap, 3)
heapq.heappush(heap, 1)

print(heap[0])  # 1

heapq.heappush(heap, 0)

print(heap[0])  # 0
```
We first imported the heapq module, which contains functions for working with heaps.
We created an empty list called heap. Heaps are implemented as lists in Python.
We pushed the elements 3 and 1 onto the heap.
We accessed the element with the smallest priority, which is 1. The element with the smallest priority is always at index 0. This is the same as calling .top() in other languages.
We pushed the element 0 onto the heap.
We accessed the element with the smallest priority, which is now 0

## heap pop
```py
import heapq

heap = []

heapq.heappush(heap, "banana")
heapq.heappush(heap, "apple")
heapq.heappush(heap, "kiwi")

print(heapq.heappop(heap))  # apple
print(heapq.heappop(heap))  # banana
print(heapq.heappop(heap))  # kiwi
```
We pushed the strings "banana", "apple", and "kiwi" onto the heap.
We popped the element with the smallest priority, which is "apple". By default, the priority of strings is determined by their lexicographical order, with smaller lexicographical strings having higher priority.
We popped the next element with the smallest priority, which is now "banana".
We popped the last element with the smallest priority, which is now "kiwi".
The heap is now empty. If we try to pop an element from an empty heap, we will get an IndexError

The time complexity of `heapq.heappop()` is \( O(\log(n)) \), where \( n \) is the number of elements in the heap.



## Heapify
If we are given a list of elements up front, we can convert them into a heap using the heapq.heapify() function. This function rearranges the elements in the list so that they form a valid heap. The heap is a min heap by default, meaning that the element with the smallest priority is at index 0.
```py
import heapq

heap = [4, 2, 5, 3, 1]

heapq.heapify(heap)

while heap:
    print(heapq.heappop(heap))
1
2
3
4
5

```
The time complexity of `heapq.heapify()` is \( O(n) \), where \( n \) is the number of elements in the heap. This means it's more efficient than pushing elements onto the heap one by one, which would take \( O(n \log(n)) \) time.


## Max Heap

Suppose we had a set of numbers [4, 2, 3, 5]. We can insert these numbers into a max heap by negating them and inserting them into a min heap. The negated numbers would be [-4, -2, -3, -5]. When we pop elements from the min heap, we would negate them again to get the original numbers.

```py
import heapq

nums = [4, 2, 3, 5]
max_heap = []

for num in nums:
    heapq.heappush(max_heap, -num) # Negate the number

while max_heap:
    top = -heapq.heappop(max_heap) # Negate the number back
    print(top)


5
4
3
2

```

## Custom Heap

Unfortunately, Python does not have a custom `key` parameter for the heapq module. This means that we `cannot directly create a heap with custom priorities`. However, we can `simulate` a custom heap by using a `tuple as the element` in the heap.
With tuples, Python will use the `first element of the tuple` as the priority. If `two tuples` have the same first element, Python will compare the `second element` of the tuples, and so on.
```py
import heapq

nums = [4, -2, 3, -5]
heap = []

for num in nums:
    pair = (abs(num), num)
    heapq.heappush(heap, pair)

while heap:
    pair = heapq.heappop(heap)
    original_num = pair[1]
    print(original_num)
```
```
-2
3
4
-5
```

We pushed tuples onto the heap where the first element was the absolute value of the number and the second element was the original number. `[(4, 4), (2, -2), (3, 3), (5, -5)].`
The heap was a min heap based on the first element of each tuple.
We popped the tuples and printed the second element from each, which was the original number.

## Heap N Smallest
Heaps provide a very convenient way to find the smallest elements in a collection. For this we can use `heapq.nsmallest()`:
```py
import heapq

my_array = [1, 6, 3, 5, 7, 9, 8, 10, 2, 12]

heapq.nsmallest(3, my_array)  # returns [1, 2, 3]
heapq.nsmallest(5, my_array)  # returns [1, 2, 3, 5, 6]
heapq.nsmallest(1, my_array)  # returns [1]
```

We initialized an unsorted array `my_array` with some integers.
We called heapq.nsmallest(3, my_array). This returns the `3 smallest elements` in my_array. The elements are returned in sorted order.
We also called heapq.nsmallest(5, my_array) which returns the `5 smallest elements `in my_array.
We also called heapq.nsmallest(1, my_array) which returns the `smallest element` in my_array.

### Time and Space Complexity

The time complexity of `heapq.nsmallest()` is \( O(m \log(n)) \), where \( n \) is the number of elements to return and \( m \) is the size of the input.

One way to implement `nsmallest()` is to iterate over the input and push each element onto a heap. We ensure the size of the heap is at most \( n \) by popping the largest element if the heap size exceeds \( n \). Thus, we will use a max heap.

## Heap N Largest
We also have heapq.nlargest() to get the n largest elements in a collection.
```py
import heapq

my_array = [1, 6, 3, 5, 7, 9, 8, 10, 2, 12]

heapq.nlargest(3, my_array)  # returns [12, 10, 9]
heapq.nlargest(5, my_array)  # returns [12, 10, 9, 8, 7]
heapq.nlargest(1, my_array)  # returns [12]
```
We initialized an unsorted array my_array with some integers.
We called heapq.nlargest(3, my_array). This returns the 3 largest elements in my_array. The elements are returned in decreasing order.
We also called heapq.nlargest(5, my_array) which returns the 5 largest elements in my_array in decreasing order.
We also called heapq.nlargest(1, my_array) which returns the largest element in my_array

### Time and Space Complexity

The time complexity of `heapq.nlargest()` is \( O(m \log(n)) \), where \( n \) is the number of elements to return and \( m \) is the size of the input.

One way to implement `nlargest()` is to iterate over the input and push each element onto a heap. We ensure the size of the heap is at most \( n \) by popping the smallest element if the heap size exceeds \( n \). Thus, we will use a min heap.

## Sorted Dict Basics
Python does not have a built-in sorted dictionary data structure. However, we can use the sorted containers library to create a sorted dictionary in Python. It supports the same operations as a regular dictionary, but the keys are always sorted. A sorted dictionary may not contain duplicate keys.
`Insertion`: Insert a key-value pair into the sorted dict.

```py
from sortedcontainers import SortedDict

sorted_dict = SortedDict()

sorted_dict['C'] = 90

sorted_dict['B'] = 80

sorted_dict['A'] = 70

print(sorted_dict)  # SortedDict({'A': 70, 'B': 80, 'C': 90})
```

`Access`: Access the value associated with a key.
```py
sorted_dict = SortedDict({'a': 1})

print(sorted_dict['a']) # 1
```
`Deletion`: Delete a key-value pair from the sorted dict.
```py
sorted_dict = SortedDict({'a': 1, 'b': 2, 'c': 3, 'd': 4})

# removes & return the first key-value pair in sorted order
sorted_dict.popitem() # ('a', 1)

# removes the last key-value pair in sorted order
sorted_dict.popitem(last=True) # ('d', 4) 

# remove & return the value associated with the key
sorted_dict.pop('b') # 2

del sorted_dict['c'] # {}
```

As shown above, there are several ways to delete a key-value pair from a sorted dictionary. You can use the popitem() method to remove and return the first or last key-value pair in sorted order. You can also use the pop() method to remove and return the value associated with a specific key, or the del keyword to delete a key-value pair.
The popitem() method will raise a KeyError if the dictionary is empty.
The pop() method will raise a KeyError if the key does not exist.
The del keyword will also raise a KeyError if the key does not exist.

`Lookup`: Check if a key exists in the sorted dict.
```py
sorted_dict = SortedDict({'a': 1})

does_a_exist = 'a' in sorted_dict # True
does_b_exist = 'b' in sorted_dict # False
```
For lookup operations, you can also use the in operator, similar to how you would check if an element is in a list.

`Iterating`: Loop through the sorted dict.
```py
sorted_dict = SortedDict({'a': 1, 'b': 2, 'c': 3})

for key, value in sorted_dict.items():
    print(key, value)
```
`Notice` that we loop through the sorted dictionary using the items() method, which returns a list of key-value pairs. We will iterate over the key-value pairs in sorted order based on the keys. If we only iterated over the keys, but we needed the values as well, we would have to do a lookup for each key, which would be less efficient. (See the time complexity section below.)
### Time Complexity

- **Insertion:** \( O(\log n) \)
- **Access:** \( O(\log n) \)
- **Deletion:** \( O(\log n) \)
- **Lookup:** \( O(\log n) \)

## Sorted Set Basics
Sorted sets are very similar to hash sets, but they store keys in sorted order. Sorted sets may not contain duplicate elements. The common operations on a sorted set include:
`Insertion` : Insert a key into the sorted set.
```py
from sortedcontainers import SortedSet

my_set = SortedSet()

my_set.add(90)
my_set.add(80)
my_set.add(85)

print(my_set) # SortedSet([80, 85, 90])
```

Deletion: Delete a key from the sorted set.
```py
my_set = SortedSet([90, 80, 85, 95])

my_set.remove(90) # SortedSet([80, 85, 95])

my_set.discard(100) # SortedSet([80, 85, 95]) 

my_set.pop() # 95

my_set.pop(0) # 80

print(my_set) # SortedSet([85])

my_set.clear() # SortedSet([])
```
The remove() method will raise a KeyError if the element does not exist, while the discard() method will not.
The pop() method will remove and return the largest element in the sorted set.
The pop(0) method will remove and return the smallest element in the sorted set.
The clear() method will remove all elements from the sorted set.
`Lookup`: Check if a key exists in the sorted set.
```py
my_dict = {'a': 1}

does_a_exist = 'a' in my_dict # True
does_b_exist = 'b' in my_dict # False
```
Iterating: Loop through the sorted set.
```py
sorted_set = SortedSet([4, 3, 5, 2, 1])

for num in sorted_set:
    print(num)  # 1, 2, 3, 4, 5
```

### Time Complexity

- **Insertion:** \( O(\log n) \)
- **Deletion:** \( O(\log n) \)
- **Lookup:** \( O(\log n) \)
- **Note:** `clear()` is \( O(n) \)



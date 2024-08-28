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


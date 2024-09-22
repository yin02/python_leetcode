## validate Parenthese
Validate Parentheses
Solved 
You are given a string s consisting of the following characters: '(', ')', '{', '}', '[' and ']'.

The input string s is valid if and only if:

Every open bracket is closed by the same type of close bracket.
Open brackets are closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
Return true if s is a valid string, and false otherwise

```py

class Solution:
    def isValid(self, s: str) -> bool:
        Map = {"}":"{","]":"[",")":"("}
        stack = []
        for c in s:
        # open 
            if c not in Map:
                stack.append(c)
                continue
        #closed
            # no open parenthese correspoding,
            if not stack or stack[-1] != Map[c]:
                return False
            # it has opened parentheses
            stack.pop() 
        return not stack
```


## Baseball Game
You are keeping the scores for a baseball game with strange rules. At the beginning of the game, you start with an empty record.

You are given a list of strings operations, where operations[i] is the ith operation you must apply to the record and is one of the following:

An integer x.
Record a new score of x.
'+'.
Record a new score that is the sum of the previous two scores.
'D'.
Record a new score that is the double of the previous score.
'C'.
Invalidate the previous score, removing it from the record.
Return the sum of all the scores on the record after applying all the operations.

```py
class Solution:
            
    def calPoints(self,operations):
        stack = []
        for op in operations:
            if op == "+":
                stack.append(stack[-1]+stack[-2])
            elif op == "D":
                stack.append(stack[-1]*2)
            elif op == "C":
                stack.pop()
            else:
                stack.append(int(op))
        return sum(stack)
```

## Minimum Stack

### question: minimum stack
Design a stack class that supports the push, pop, top, and getMin operations.

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack

```py
class MinStack:
# 最重要的是知道minstack的作用只有记录当前最小值，而不是和stack 一样储存所有的，虽然数量一样
    def __init__(self):
        self.stack = []
        # this store 当前栈的最小值，不用全部一样
        self.minstack = []
        

    def push(self, val: int) -> None:
        self.stack.append(val)
        val = min(val,self.minstack[-1] if self.minstack else val)
        self.minstack.append(val)
        

    def pop(self) -> None:
        self.stack.pop()
        self.minstack.pop()
        

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minstack[-1]
        
```
## 225. Implement Stack using Queues
implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (push, top, pop, and empty).

Implement the MyStack class:

void push(int x) Pushes element x to the top of the stack.
int pop() Removes the element on the top of the stack and returns it.
int top() Returns the element on the top of the stack.
boolean empty() Returns true if the stack is empty, false otherwise.
Notes:

You must use only standard operations of a queue, which means that only push to back, peek/pop from front, size and is empty operations are valid.
Depending on your language, the queue may not be supported natively. You may simulate a queue using a list or deque (double-ended queue) as long as you use only a queue's standard operations.


当然，下面是 `pop` 和 `top` 方法的中文解释：

### `pop` 方法
`pop` 方法用于移除并返回栈顶元素。具体步骤如下：

1. **移动所有元素，除了最后一个**:
   ```python
   for i in range(len(self.q) - 1):
       self.q.append(self.q.popleft())
   ```
   这个循环将队列中的所有元素（除了最后一个）从前面移动到队列的末尾。这样，队列的最后一个元素就变成了队列的第一个元素。

2. **移除并返回最后一个元素**:
   ```python
   return self.q.popleft()
   ```
   循环结束后，原队列的最后一个元素（现在在队列的前面）被移除并返回。这模拟了栈的 `pop` 操作，即移除栈顶元素。

### `top` 方法
`top` 方法用于获取（但不移除）栈顶元素。具体步骤如下：

1. **移动所有元素，除了最后一个**:
   ```python
   for i in range(len(self.q) - 1):
       self.q.append(self.q.popleft())
   ```
   这个循环与 `pop` 方法中的循环相同，将队列中的所有元素（除了最后一个）从前面移动到队列的末尾。这样，队列的最后一个元素就变成了队列的第一个元素。

2. **获取栈顶元素**:
   ```python
   res = self.q[0]
   ```
   现在，队列的第一个元素（之前是队列的最后一个元素）就是栈顶元素。我们将这个元素保存到 `res` 变量中。

3. **重新将元素放回队列**:
   ```python
   self.q.append(self.q.popleft())
   ```
   为了保持队列的状态，以确保后续操作（如 `pop`）能够正确工作，我们将刚刚访问的元素重新移动到队列的末尾。

4. **返回栈顶元素**:
   ```python
   return res
   ```
   最后，返回保存的栈顶元素 `res`。

### 总结
- **`pop`**: 将队列中的所有元素（除了最后一个）移动到末尾，然后移除并返回最后一个元素（即栈顶元素）。
- **`top`**: 将队列中的所有元素（除了最后一个）移动到末尾，然后获取并返回最后一个元素（即栈顶元素），但不移除它。

这种通过队列实现栈的方式利用了队列的 FIFO（先进先出）特性，通过旋转元素来模拟栈的 LIFO（后进先出）行为。
```py
from collections import deque

class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x: int) -> None:
        self.q.append(x)

    def pop(self) -> int:
        # Moving all elements except the last one to the end of the queue
        for i in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
        # Removing and returning the last element (which is the top of the stack)
        return self.q.popleft()

    def top(self) -> int:
        # Similar to pop, move all elements except the last one to the end of the queue
        for i in range(len(self.q) - 1):
            self.q.append(self.q.popleft())
        # Get the last element (top of the stack) without removing it
        res = self.q[0]
        self.q.append(self.q.popleft())  # Re-append the element to maintain stack order
        return res

    def empty(self) -> bool:
        # Check if the queue is empty (stack is empty)
        return len(self.q) == 0
        
```

## 通过栈来实现队列

```py
class MyQueue:

    def __init__(self):
        # 使用两个栈 s1 和 s2 实现队列
        self.s1 = []
        self.s2 = []

    def push(self, x: int) -> None:
        # 将元素 x 压入栈 s1
        self.s1.append(x)

    def pop(self) -> int:
        # 如果栈 s2 为空，将 s1 中的所有元素倒入 s2
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        # 弹出并返回 s2 顶部元素，即队列的最前元素
        return self.s2.pop()

    def peek(self) -> int:
        # 如果 s2 为空，将 s1 中的所有元素倒入 s2
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        # 返回 s2 顶部元素，但不弹出，即查看队列的最前元素
        return self.s2[-1]

    def empty(self) -> bool:
        # 判断两个栈是否都为空，如果都为空则队列为空
        return not self.s1 and not self.s2
```



## 735.Asteroid collision

We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.

 

Example 1:

Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.
Example 2:

Input: asteroids = [8,-8]
Output: []
Explanation: The 8 and -8 collide exploding each other.
Example 3:

Input: asteroids = [10,2,-5]
Output: [10]
Explanation: The 2 and -5 collide resulting in -5. The 10 and -5 collide resulting in 10.

### 为什么只用先考虑正的
Here’s the logic behind it:

Asteroid Directions:

- A positive value in the asteroid list (asteroids) represents an asteroid moving to the right.
A negative value represents an asteroid moving to the left.
How collisions happen:

- A collision will only happen if a right-moving asteroid (positive) meets a left-moving asteroid (negative).
  
- If both asteroids are moving in the same direction (both positive or both negative), they will not collide.


```py
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for elemenet in asteroids:
            # assume stack is postive stack
            while stack and stack[-1] >0 and elemenet <0:
                if abs(stack[-1]) > abs(elemenet):
                    break
                elif abs(stack[-1]) == abs(elemenet):
                    stack.pop()
                    break
                else:
                    stack.pop()
                    # 没有stack append，因为他会继续和下面对撞不会共存的
            else:
                stack.append(elemenet)
        return stack

```
## Evaluate Reverse Polish Notation
You are given an array of strings tokens that represents a valid arithmetic expression in Reverse Polish Notation.

Return the integer that represents the evaluation of the expression.

The operands may be integers or the results of other operations.
The operators include '+', '-', '*', and '/'.
Assume that division between integers always truncates toward zero.
Example 1:

Input: tokens = ["1","2","+","3","*","4","-"]

Output: 5

Explanation: ((1 + 2) * 3) - 4 = 5
```py
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for c in tokens:
            if c == "+":
                stack.append(stack.pop() + stack.pop())
            elif c == "-":
                # last in first out, 所以先被弹出的要被剪掉
                a, b = stack.pop(), stack.pop()
                stack.append(b - a)
            elif c == "*":
                stack.append(stack.pop() * stack.pop())
            elif c == "/":
                a, b = stack.pop(), stack.pop()
                stack.append(int(float(b) / a))
            else:
                stack.append(int(c))
        return stack[0]



```
# 496. Next Greater Element I

**Difficulty:** Easy

The next greater element of some element `x` in an array is the first greater element that is to the right of `x` in the same array.

You are given two distinct 0-indexed integer arrays `nums1` and `nums2`, where `nums1` is a subset of `nums2`.

For each `0 <= i < nums1.length`, find the index `j` such that `nums1[i] == nums2[j]` and determine the next greater element of `nums2[j]` in `nums2`. If there is no next greater element, the answer is `-1`.

Return an array `ans` of length `nums1.length` such that `ans[i]` is the next greater element as described above.

## Example 1:

**Input:**
nums1 = [4,1,2]
nums2 = [1,3,4,2]
**output:**
[-1, 3, -1]

```py
# iterative way
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1Idx = {n:i for i,n in enumerate(nums1)}
        res = [-1]* len(nums1)

        for i in range(len(nums2)):
            #if num２　number never exist
            if nums2[i] not in nums1:
                continue
            for j in range(i+1,len(nums2)):
                if nums2[j]> nums2[i]:
                    # current position
                    idx = nums1Idx[nums2[i]]
                    # value for current postion to be updated
                    res[idx] = nums2[j]
                    break
        return res
# stack way

class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1Idx = {n:i for i,n in enumerate(nums1)}
        res = [-1]* len(nums1)

        #monotoic stack
        stack = []
        # put nums2 into stack if it exist in stack1
        for i in range(len(nums2)):
            # want cur
            cur = nums2[i]
            while stack and cur > stack[-1]:
                #get value
                val = stack.pop()
                # get index by val
                idx = nums1Idx[val]
                #update the val with current
                res[idx] = cur

            if cur in nums1:
                stack.append(cur)
        return res
            


```
### 1544. Make The String Great
Attempted

Given a string s of lower and upper case English letters.

A good string is a string which doesn't have two adjacent characters s[i] and s[i + 1] where:
```
0 <= i <= s.length - 2
```
s[i] is a lower-case letter and s[i + 1] is the same letter but in upper-case or vice-versa.
To make the string good, you can choose two adjacent characters that make the string bad and remove them. You can keep doing this until the string becomes good.

Return the string after making it good. The answer is guaranteed to be unique under the given constraints.

Notice that an empty string is also good.

Example 1:

Input: s = "leEeetcode"
Output: "leetcode"
Explanation: In the first step, either you choose i = 1 or i = 2, both will result "leEeetcode" to be reduced to "leetcode".

```py
class Solution:
    # 因为是检查相邻的两个
    def makeGood(self, s: str) -> str:
        stack  = []
        #每一个char
        for char in s:
            #如果两个char相同
            if(
                stack
                and stack[-1] != char
                and stack[-1].lower() == char.lower()
            ):
                stack.pop()
            # 如果不同就加入
            else:
                stack.append(char)
        return "".join(stack)

```
### 2390. Removing Stars From a String
You are given a string s, which contains stars *.

In one operation, you can:

Choose a star in s.
Remove the closest non-star character to its left, as well as remove the star itself.
Return the string after all stars have been removed.

Note:

The input will be generated such that the operation is always possible.
It can be shown that the resulting string will always be unique.

```python
def removeStars(self,s):
    stack = []
    for c in s:
        if c == "*":
            stack and stack.pop()
        else:
            stack.append(c)
    return "".join(stack)
        
```
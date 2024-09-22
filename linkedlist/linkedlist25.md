# Linked List and Design Problem 

## 1. Reverse Linked List
**Description**: Reverse a singly linked list.
- **Input**: A singly linked list.
- **Output**: The reversed linked list.
指向下一个的指针
 指向前面
这个过程中
就反转
recursive是回溯
"备下一"
"反指头"
进：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# iterative way
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev,cur = none,head
        while cur:
            temp = curr.next
            cur.next = prev
            prev = cur
            cur = temp
        return prev
# recursive way
def reverseList(self, head: Optional[ListNode]) -> Optional[
    if not head:
        return None
    newHead = head
    if head.next:
        newHead = self.reverseList(head.next)
        head.next.next = head


        

 那后面的newHead = self.reverseList(head.next)开始回溯的时候都是从下面开始的
recursive call的执行都是7 包扩上面的执行


"备下一" (Prepare the next): temp = curr.next - Store the next node.
"反指头" (Reverse pointer): curr.next = prev - Point the current node back to the previous node.
进：      prev = curr
            curr = temp
curr.next是指针
```

## 2. Merge Two Sorted Lists
**Description**: Merge two sorted linked lists into one sorted linked list.
- **Input**: Two sorted linked lists.
- **Output**: A merged sorted linked list.
```py
class Solution:
    def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
        dummy = node = listNode()
        while list1 and list2:
            if list1.val < list2.val:
                node.next = list1
                list1 = list1.next
            else:
                node.next = list2
                list2 = list2.next
            node = node.next
        return dummy.next
```

## 3. Palindrome Linked List
**Description**: Check if a linked list is a palindrome.
- **Input**: A singly linked list.
- **Output**: Boolean indicating if the list is a palindrome.
```py
def isPalindrome(self,head):
    # find the middle
    # reverse the second half
    # check palindrome
    fast = head
    slow = head
    while head.next and head:
        slow = slow.next
        fast = fast.next.next
    prev = None
    while slow:
        tmp = slow.next# store for next
        slow.next = slow
        prev = slow
        slow = tmp
    # check the palindrome
    
    left,right = head, prev
    while right:
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
        
    return True

```

## 4. Remove Linked List Elements
**Description**: Remove all elements from a linked list that match a given value.
- **Input**: A linked list and an integer `val`.
- **Output**: The linked list with all occurrences of `val` removed.
```py
class Solution:
  def deleteDuplicates(self, head):
    cur = head
    while cur:
      while cur.next and cur.next.val == cur.val:
        cur.next = cur.next.next
      cur = cur.next
    return head
```
## 5. Remove Duplicates From Sorted List
**Description**: Remove duplicates from a sorted linked list.
- **Input**: A sorted linked list.
- **Output**: A sorted linked list without duplicates.

## 6. Middle of the Linked List
**Description**: Find the middle node of a linked list.
- **Input**: A singly linked list.
- **Output**: The middle node of the linked list.

## 7. Intersection of Two Linked Lists
**Description**: Find the node where two linked lists intersect.
- **Input**: Two singly linked lists.
- **Output**: The intersecting node or `null` if no intersection exists.

## 8. Merge in Between Linked Lists
**Description**: Merge one linked list into another between two given positions.
- **Input**: Two linked lists, and two integers `a` and `b` representing the range in the first list.
- **Output**: The modified linked list after merging.

## 9. Remove Nodes From Linked List
**Description**: Remove nodes based on specific criteria (e.g., value threshold).
- **Input**: A singly linked list and a condition.
- **Output**: The modified linked list with nodes removed.

## 10. Reorder List
**Description**: Rearrange the linked list by alternating between the first and last nodes.
- **Input**: A singly linked list.
- **Output**: The reordered linked list.

## 11. Maximum Twin Sum of a Linked List
**Description**: Find the maximum sum of paired nodes in a linked list where twin nodes are mirrored from the start and end.
- **Input**: A singly linked list.
- **Output**: The maximum twin sum.

## 12. Remove Nth Node From End of List
**Description**: Remove the nth node from the end of a linked list.
- **Input**: A singly linked list and an integer `n`.
- **Output**: The linked list with the nth node from the end removed.

## 13. Swapping Nodes in a Linked List
**Description**: Swap the nodes at two given positions in a linked list.
- **Input**: A linked list and two positions.
- **Output**: The linked list with the nodes swapped.

## 14. LFU Cache
**Description**: Design a Least Frequently Used (LFU) cache.
- **Input**: Cache capacity and `get`/`put` operations.
- **Output**: The state of the cache after operations.

## 15. Copy List With Random Pointer
**Description**: Create a deep copy of a linked list with random pointers.
- **Input**: A linked list where each node has a `random` pointer.
- **Output**: A deep copy of the linked list.

## 16. Design Linked List
**Description**: Design a linked list that supports operations like add, delete, and access.
- **Input**: A series of operations (add, delete, get).
- **Output**: The result of operations on the linked list.

## 17. Design Browser History
**Description**: Design a system to simulate browser history with back, forward, and visit operations.
- **Input**: A series of operations like visit, back, and forward.
- **Output**: The state of browser history after the operations.

## 18. Add Two Numbers
**Description**: Add two numbers represented by linked lists, where each node contains a single digit.
- **Input**: Two non-empty linked lists.
- **Output**: A linked list representing the sum of the two numbers.

## 19. Linked List Cycle
**Description**: Detect if a linked list has a cycle.
- **Input**: A singly linked list.
- **Output**: The node where the cycle begins or `null` if no cycle exists.

## 20. Find the Duplicate Number
**Description**: Find the duplicate number in an array of integers.
- **Input**: An array of integers where one number repeats.
- **Output**: The duplicate number.

## 21. Swap Nodes in Pairs
**Description**: Swap every two adjacent nodes in a linked list.
- **Input**: A singly linked list.
- **Output**: The linked list with adjacent nodes swapped.

## 22. Sort List
**Description**: Sort a linked list in ascending order.
- **Input**: A singly linked list.
- **Output**: The sorted linked list.

## 23. Partition List
**Description**: Partition a linked list such that all nodes less than `x` come before nodes greater than or equal to `x`.
- **Input**: A linked list and an integer `x`.
- **Output**: The partitioned linked list.

## 24. Rotate List
**Description**: Rotate a linked list to the right by `k` places.
- **Input**: A singly linked list and an integer `k`.
- **Output**: The rotated linked list.

## 25. Reverse Linked List II
**Description**: Reverse the nodes of a linked list from position `m` to `n`.
- **Input**: A singly linked list and two integers `m` and `n`.
- **Output**: The linked list with the segment between `m` and `n` reversed.

## 26. Design Circular Queue
**Description**: Design a circular queue with operations like enqueue, dequeue, isEmpty, and isFull.
- **Input**: Queue capacity and operations.
- **Output**: The state of the queue after operations.

## 27. Insertion Sort List
**Description**: Sort a linked list using insertion sort.
- **Input**: A singly linked list.
- **Output**: The sorted linked list.

## 28. Split Linked List in Parts
**Description**: Split a linked list into `k` parts as equally as possible.
- **Input**: A singly linked list and an integer `k`.
- **Output**: An array of `k` linked list parts.

## 29. LRU Cache
**Description**: Design a Least Recently Used (LRU) cache.
- **Input**: Cache capacity and `get`/`put` operations.
- **Output**: The state of the cache after operations.

## 30. Merge K Sorted Lists
**Description**: Merge `k` sorted linked lists into one sorted linked list.
- **Input**: An array of `k` sorted linked lists.
- **Output**: A single merged sorted linked list.

## 31. Reverse Nodes in K Group
**Description**: Reverse the nodes of a linked list in groups of `k`.
- **Input**: A singly linked list and an integer `k`.
- **Output**: The linked list with nodes reversed in groups of `k`.

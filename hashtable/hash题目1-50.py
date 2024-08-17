class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        hashset = set()

        for n in nums:
            if n in hashset:
                return True
            hashset.add(n)
        return False



class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        countS, countT = {}, {}

        for i in range(len(s)):
            countS[s[i]] = 1 + countS.get(s[i], 0)
            countT[t[i]] = 1 + countT.get(t[i], 0)
        return countS == countT


class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        ans = []
        for i in range(2):
            for n in nums:
                ans.append(n)
        return ans
    
class Solution:
    def replaceElements(self, arr: List[int]) -> List[int]:
        rightMax = -1
        for i in range(len(arr) -1, -1, -1):
            newMax = max(rightMax, arr[i])
            arr[i] = rightMax
            rightMax = newMax
        return arr

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i, j = 0, 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == len(s)
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        """
	one shortcut
	"""
	#	return len(s.split()[-1])
        c = 0
        for i in s[::-1]:
            if i == " ":
                if c >= 1:
                    return c
            else:
                c += 1
        return c

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        prevMap = {}  # val -> index

        for i, n in enumerate(nums):
            diff = target - n
            if diff in prevMap:
                return [prevMap[diff], i]
            prevMap[n] = i

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        res = ""
        for i in range(len(strs[0])):
            for s in strs:
                if i == len(s) or s[i] != strs[0][i]:
                    return res
            res += strs[0][i]
        return res


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = collections.defaultdict(list)

        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord("a")] += 1
            ans[tuple(count)].append(s)
        return ans.values()
#pascals triangle
class Solution:
    def generate(self, rowIndex) -> List[List[int]]:
        res = [[1]]
        for i in range(numRows-1):
            temp = [0] + res[-1] +[0]
            row = []
            for j in range(len(res[-1])+1):
                row.append(temp[j]+ temp[j+1])
            res.append(row)
        return res

def removeElement(nums,val):
    k = 0
    for i in range(len(nums)):
        if nums[i] != val:
            nums[k] = nums[i]
            k +=1
    return k

class Solution:
    def numUniqueEmails(self, emails: list[str]) -> int:
        unique_emails: set[str] = set()
        for email in emails:
            local_name, domain_name = email.split('@')
            local_name = local_name.split('+')[0]
            local_name = local_name.replace('.', '')
            email = local_name + '@' + domain_name
            unique_emails.add(email)
        return len(unique_emails)

class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        mapST, mapTS = {}, {}

        for c1, c2 in zip(s, t):
            if (c1 in mapST and mapST[c1] != c2) or (c2 in mapTS and mapTS[c2] != c1):
                return False
            mapST[c1] = c2
            mapTS[c2] = c1

        return True

    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
       # Solution with O(n) space complexity
       f = [0] + flowerbed + [0]
       
       for i in range(1, len(f) - 1):  # skip first & last
           if f[i - 1] == 0 and f[i] == 0 and f[i + 1] == 0:
               f[i] = 1
               n -= 1
       return n <= 0


    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
       # Another solution with O(1) space complexity
       for i in range(len(flowerbed)):
            if n == 0:
                return True
            if ((i == 0 or flowerbed[i - 1] == 0)   # If at the first element or the previous element equals to 0
                and (flowerbed[i] == 0)             # If current element equals to 0
                and (i == len(flowerbed) - 1 or flowerbed[i + 1] == 0)): # If at the last element or the next element equals to 0
                # Place flower at the current position
                flowerbed[i] = 1
                n -= 1

       return n == 0
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        res, count = 0, 0

        for n in nums:
            if count == 0:
                res = n
            count += (1 if n == res else -1)
            
        return res
    
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:

        # O (n + m)
        nums1Idx = { n:i for i, n in enumerate(nums1) }
        res = [-1] * len(nums1)

        stack = []
        for i in range(len(nums2)):
            cur = nums2[i]

            # while stack exists and current is greater than the top of the stack
            while stack and cur > stack[-1]:
                val = stack.pop() # take top val
                idx = nums1Idx[val]
                res[idx] = cur

            if cur in nums1Idx:
                stack.append(cur)
        
        return res

class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        total = sum(nums)  # O(n)

        leftSum = 0
        for i in range(len(nums)):
            rightSum = total - nums[i] - leftSum
            if leftSum == rightSum:
                return i
            leftSum += nums[i]
        return -1


class  NumArray:
    def __init__(self,num):
        self.prefix = []
        cur = 0
        for n in num:
            cur +=n
            self.prefix.append(cur)
    def sumRange(self,left,right):
        rightSum = self.prefix[right]
        leftSum = self.prefix[left-1] if left >0 else 0
        return rightSum- leftSum

class NumArray:
    def __init__(self,nums):
        prefix = []
        cur = 0
        for n in nums:
            cur += cur
            self.prefix.append(cur)
    def sumRange(self,left,right):
        rightSum = self.prefix[right]
        leftSum = self.prefix[left-1] if left >0 else 0
        return rightsum - leftsum
    
def findDisappearedNumbers(nums):
    for n in nums:
        # map [0,n-1] to [1,n], some number may not exist, so it wouldn't be indexed
        #i 有可能会重复，因为i = abs（n）-1
        i = abs(n)-1
        # change the number that could be mapped
        nums[i] = -1 * abs(nums[i])
        res = []
        # i+1 ==n
        for i, n in enumerate(nums):
            if n >0:
                res.append(i+1)
        return res


from collections import Counter


class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        countText = Counter(text)
        balloon = Counter("balloon")

        res = len(text)  # or float("inf")
        for c in balloon:
            res = min(res, countText[c] // balloon[c])
        return res




# the word maxium, you have to go through the entire word,so it is fine to just use hash

class Solution:
    def max(text):
        counterText = Counter(text)
        ballon = Counter("balloon")
        # not possible
        res = len(text)
        for c in balloon:
            res = min(res,counterText[c]//ballon[c])
        return res



    def wordPattern(self,pattern,s):
        words = s.split(" ")
        if len(pattern) != len(word):
            return False
        charToWord = {}
        wordToChar = {}
        for c,w in zip(pattern,words):
            if c in charToWord and charToWord[c] != w:
                return False
            if w in wordToChar and wordToChar[w] != c:
                return False
            
            charToWord[c] = w
            wordToChar[w] = c
        return True

# hashset-use linkedlist to avoid hash collision 
# key %100 = 100

#linked ilist distinguis 1 and 101

class MyHashset:
    def __init__(self) -> None:
        self.hash = []
    def add(self,key):
        if not self.contains(key):
            self.hashset.append(key)
    def remove(self,key):
        if self.contains(key):
            self.hashset.remove(key)
    def contains(self,key):
        return True if key in MyHashset else False

# init the size and data 
# add: check if it is already inside or not, get the idx ,append to list
# remove: check if it is already inside, not  return false, get the index ,remove from the list
# hash: get the index of the key with modulo of 1000
# contains: check if the key in the hashset 
class MyHashset:
    def __init__(self):
        self.size = 1000
        self.data = [[]for i in range(self.size)]
    def add(self,key):
        if self.contain(key):
            return
        idx = self.hash(key)
        self.data[idx].append(key)
    def remove(self,key):
        if not self.contain(key):
            return
        idx = self.hash(key)
        self.data[idx].remove(key)
    def contains(self,key):
        idx = self.hash(key)
        return any(v == key for v in self.data[idx])
    
    def hash(self,key):
        return key %self.size
    
 # design a hashmap
class ListNode:
    def __init__(self, key=-1, val=-1, next=None):
        self.key = key  # Initialize the key for the node
        self.val = val  # Initialize the value for the node
        self.next = next  # Pointer to the next node

class MyHashMap:
    def __init__(self):
        self.map = [ListNode() for _ in range(1000)]  # Create a list of 1000 dummy nodes
    
    def hashcode(self, key):
        return key % len(self.map)  # Hash function to map a key to an index

    def put(self, key: int, value: int) -> None:
        cur = self.map[self.hashcode(key)]  # Get the corresponding bucket
        #检查有没有下一个节点
        while cur.next:
            if cur.next.key == key:
                cur.next.val = value  # If the key already exists, update the value
                return
            cur = cur.next
        #没有的话直接加
        cur.next = ListNode(key, value)  # Otherwise, add a new node at the end

    def get(self, key: int) -> int:
        cur = self.map[self.hashcode(key)].next  # Get the head of the corresponding bucket
        # 当存在当前的节点，和key不是一样的key
        while cur and cur.key != key:
            cur = cur.next  # Traverse the linked list to find the key
        # if find
        if cur:
            return cur.val  # Return the value if key is found
        return -1  # Return -1 if the key is not found

    def remove(self, key: int) -> None:
        cur = self.map[self.hashcode(key)]  # Get the corresponding bucket
        while cur.next and cur.next.key != key:
            cur = cur.next  # Traverse the linked list to find the key
        if cur and cur.next:
            cur.next = cur.next.next  # Remove the node by updating the pointers

# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key, value)
# param_2 = obj.get(key)
# obj.remove(key)

# monotonic Array

def isMonotonic(self,nums):
    increase, decrease = True , True
    for i in range(len(nums)-1):
        if not (nums[i] <= nums[i+1]):
            increase = False
        if not (nums[i] >= nums[i+1]):
            decrease = False
    return increase or decrease

def isMonotonic(self,nums):
    if nums[-1] - nums[0] <0:
        nums.reverse()
    for i in range(len(nums-1)):
        if not (nums[i] < nums[i+1]):
            return False
    return True

#you know number of good pair
#get hashset and count the number
#acutually this question is kind of different
# the _ _  i<j, so the first gonna pick 1 second pick the left 1 if total is 4 
# the first pos we have the 4 opition the second one we have 3 option, and divide the repeat is A22 is  2!
# so we get the formula c * (c-1)//2
def numIdenticalPairs(self,nums):
    count = Counter(nums)
    res = 0
    for n,c in count.items():
        res += c* (c-1)//2
    return res

#second option
# think about it , 11 1 第三个1 其实加了 2个pair 第四个1 加了3个pair，每一个都可以和前面的组
def numIdenticalPair(self,nums):
    res = 0
    count = {}
    for n in nums:
        if n in count:
            #count是1的时候先加上1 ，再更新成2， 2 的时候先加上2 再变成3
            res += count[n]
            count[n] += 1
        else:
            count[n] = 1

# 对于下一行的 same index的值+1
def getRow(self,rowIndex):
    res = [1]
    for i in range(rowIndex):
        # intialze the next whole array
        next_row = [0] *(len(res)+1)
        #上一行只有 res的长度
        for j in range(len(res)):
            #上一行的左孩子
            next_row[j] += res[j]
            #上一行的右孩子
            next_row[j+1] += res[j]
        res = next_row
    return res


# setting the good to be true condtion cause every 
#chars 对于每个单词的次数是有限的
from collections import Counter, defaultdict

def countCharacters(self, words, chars):
    count = Counter(chars)
    res = 0
    for w in words:
        cur_word = defaultdict(int)
        good = True
        for c in w:
            cur_word[c] += 1
            if c not in count or cur_word[c] > count[c]:
                good = False
                break
        if good:
            res += len(w)
    return res

#row always have the same widith
#gap is the same length
#key is the position which is the length the value is the count of gap
#so basically we goona find the key that has most amount of gaps
class Solution:
    def leastBricks(self,wall):
        countGap = {0:0}
        for r in wall:
            total = 0
            for b in r[:-1]:
                total +=b
                countGap[total] =1 +countGap.get(total,0)
        return len(wall) - max(countGap.values())

class Solution:
    def largestGoodInteger(self,num):
        res = "0"
        for i in range(len(num)-2):
            if num[i] == num[i+1] == num[i+2]:
                return max(res,num[i:i+3])
        return "" if res == "0" else res
 
# if not in the second , which is means it dont have the outgoing edeges
# or linkedlist
class Solution:
    def destCity(paths):
        s = set()
        for p in paths:
            s.add(p[0])
        for p in paths:
            if p[1] not in s:
                return p[1]


#找到两个max 和两个min
# sort it is another way
class Solution:
    def maxProductDifference(self,nums):
        max1 = max2 = 0
        min1 = min2 = float("inf")
        for n in nums:
            if n >max2:
                if n> max1:
                    max1,max2 = n ,max1
                else:
                    max2 = n
            if n < min2:
                if n < min1:
                    min1, min2 = n, min1
                else:
                    min2 = n
        return (max1*max2)-(min1*min2)
    

#找到之前的 曾经的 地点，你想的是dfs
#或者hashset记录所有的
class Solution:
    def isPathCrossing(self,path):
        dir = {
            "N":[0,1],
            "S":[0,-1],
            "E": [1,0],
            "W":[-1,0]
        }
        visit = set()
        x,y = 0,0
        for c in path:
            visit.add((x,y))
            dx,dy = dir[c]
            x,y = x+dx, y+dy
            if (x,y) in visit:
                return True
        return False

#只要考虑多少个变化能变成少的那个就行
# only two options, one is the count start with 0, another one is len(s)-count start with 1
class Solution:
    def minOperations(self,s):
        count = 0# start with 0, for even we hope 0,odd we hope 1
        for i in range(len(s)):
            #odd ,because i%2 ==1
            if i%2:
                count+=1 if s[i] == "0" else 0
            else:
                count+=1 if s[i] == "1" else 0
        return min(count,len(s)-count)
    

#1897 redistribute characters to make all string equal
# make sure % len(array) and divide the whole array, equal distribute
class Solution:
    def makeEqual(self,word):
        char_cnt = defaultdict(int)
        for w in word:
            for c in w:
                char_cnt +=1
        for c in char_cnt:
            if char_cnt % len(word):
                return False
        return True


def maxLengthBetweenEqualCharacters(s):
    char_index = {}
    res = -1
    for i,c in enumerate(s):
        if c in char_index:
            res = max(res,i-char_index[c]-1)
        else:
            char_index[c] = i
    return res


#set mismatch, 1. hash ,0,2
def findErrorNums(nums):
    res = [0,0]
    count = Counter(nums)
    # change to 1 based
    for i in range(1,len(nums)+1):
        if count[i] ==0:
            res [1] == i
        if count[i] == 2:
            res[0] == i 

    return res

# use mapping,bc 1,N

#        nums[n-1] = -nums[n-1]
# 处理第一个数字 4:

# 绝对值 4 -> 对应索引 4-1 = 3
# nums[3] = 7（正数），将其变成负数：nums = [4, 3, 2, -7, 8, 2, 6, 1]
# 处理第二个数字 3:

# 绝对值 3 -> 对应索引 3-1 = 2
# nums[2] = 2（正数），将其变成负数：nums = [4, 3, -2, -7, 8, 2, 6, 1]
# 处理第三个数字 -2（绝对值是 2）:

# 绝对值 2 -> 对应索引 2-1 = 1
# nums[1] = 3（正数），将其变成负数：nums = [4, -3, -2, -7, 8, 2, 6, 1]

def findErrorNums(nums):
    res = [0,0]
    #对于每一个数值做处理，不论位置在哪里，都应该是1，N
    #位置是n-1
    for n in nums:
        n = abs(n)
        nums[n-1] = -nums[n-1]
        #找到处理过两次的
        if nums[i-1] >0:
            res[0] = n
    #再次遍历
    for i,n in enumerate(nums):
        # 找到同样处理过，并且不是刚刚数字的
        if  n >0 and i +1!= res[0]:
            res[1] = i +1
            return res
        
def findErrorNums(nums):
    N = len(nums)
    x = 0#duplicate -missing
    y = 0#duplicate**2 -missing**2
    for i in range(1,N+1):
        x += nums[i-1]-i#相同的不管是平方还是什么会被抵消掉
        y += nums[i-1] **2 - i **2 #do math
    missing = (y-x**2)//(2*x)
    duplicate = missing +x
    return [duplicate,missing]



def firstUniqueChar(s):
    count = defaultdict(int)#char ==> count
    for c in s:
        count[c] +=1
    for i,c in enumerate(s):
        if count[c] ==1:
            return i
    return -1

#把一个array当作hashset
def intersection(nums1,nums2):
    seen = set(nums1)
    res =[]
    for n in nums2:
        if n in seen:
            res.append(n)
            seen.remove(n)
    return res


#这题，只需要看mismatch就行
from collections import Counter

class Soutlion:
    def countStudent(self, student, sandwiches):
        # 初始化剩余学生数为学生列表的长度
        res = len(student)
        # 统计每种类型学生的数量
        cnt = Counter(student)
        
        # 遍历每个三明治
        for s in sandwiches:
            # 如果还有学生想吃这种类型的三明治
            if cnt[s] > 0:
                # 供给一个三明治，剩余学生数减少1
                res -= 1
                # 该类型学生数量减少1
                cnt[s] -= 1
            else:
                # 如果没有学生想吃这种类型的三明治，返回剩余学生数
                return res
        
        # 如果所有三明治都被满足，返回剩余学生数
        return res
def countStudent(student,sandwiches):
    res = len(student)
    cnt = Counter(student)
    for s in sandwiches:
        if cnt[s] >0:
            res -= 1
            cnt -=1
        else:
            return res
    return res

#find the pattern
# 2 situation: 1 i< k 要么是ticket时间，要么是 前面小的ticket需要的时间
    # 2. i>k， 要么是k-1次，要么就是有的次数
def timeRrequiredToBuy(tickets):
    res = 0
    for i in range(len(tickets)):
        if i <k:
            res += min(tickets[i],tickets[k])
        else:
            res += min(tickets[i],tickets[k]-1)
    return res


#这边x

#q：找到一个值，这个数组大于等于这个值的正好这么多，
# 上界就是len
def speicalArray(nums):
    nums.sort()
    prev = -1
    # total_right 初始值为数组的长度，表示右边剩余元素的数量。
    total_right = len(nums)
    while i < len(nums):
         # 检查当前数是否等于 total_right 或者 total_right 是否在 prev 和 nums[i] 之间
        if nums[i] == total_right or (prev < total_right<nums[i]):
            return total_right
      # nums 跳过相同的数字，确保i指向下一个不同的元素
      #条件 i + 1 < len(nums)：确保 i + 1 在数组范围内，这样访问 nums[i + 1] 是安全的。
        while i +1 <len(nums) and nums[i] ==nums[i+1]:
            i +=1
#         更新 prev 为当前元素 nums[i]。
# 移动索引 i 指向下一个元素。
# 更新 total_right 为数组右边剩余的元素数量。
        prev = nums[i]
        #到下面一个
        i+=1
        total_right = len(nums)-i
    return -1

# caculate the increase amount get the postion like counting sort
#counting sort、
def specialArray(nums):
    count = [0] * (len(nums)+1)
    for n in nums:
        # 最后一个前面的index就是自己，最后后面的都算作最后一个上面
        index = n if n < len(nums) else len(nums)        #min(n,len(nums))
        count[index] +=1
    total_right = 0
    #反过来 一个一个增加如果和当前数字相等
    #后面的计数是从小往大的，因为右边大的越来越多，往左走时候
    for i in reversed(range(len(nums)+1)):
        total_right += count[i]
        if i == total_right:
            return total_right
    return -1



class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def merge(arr, L, M, R):
            left, right = arr[L:M+1], arr[M+1:R+1]
            i, j, k = L, 0, 0
            while j < len(left) and k < len(right):
                if left[j] <= right[k]:
                    arr[i] = left[j]
                    j += 1
                else:
                    arr[i] = right[k]
                    k += 1
                i += 1
            while j < len(left):
                nums[i] = left[j]
                j += 1
                i += 1
            while k < len(right):
                nums[i] = right[k]
                k += 1
                i += 1

        def mergeSort(arr, l, r):
            if l == r:
                return arr
            m = (l + r) // 2
            mergeSort(arr, l, m)
            mergeSort(arr, m + 1, r)
            merge(arr, l, m, r)
            return arr
        
        return mergeSort(nums, 0, len(nums) - 1)



class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        freq = [[] for i in range(len(nums) + 1)]

        for n in nums:
            count[n] = 1 + count.get(n, 0)
        for n, c in count.items():
            freq[c].append(n)

        res = []
        for i in range(len(freq) - 1, 0, -1):
            for n in freq[i]:
                res.append(n)
                if len(res) == k:
                    return res

        # O(n)


class Solution:
    def encode(self, strs):
        res = ""
        for s in strs:
            res += str(len(s)) + "#" + s
        return res

    def decode(self, s):
        res = []
        i = 0
        
        while i < len(s):
            j = i
            while s[j] != '#':
                j += 1
            length = int(s[i:j])
            i = j + 1
            j = i + length
            res.append(s[i:j])
            i = j
            
        return res
    
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res = [1] * (len(nums))

        for i in range(1, len(nums)):
            res[i] = res[i-1] * nums[i-1]
        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
        return res

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        cols = collections.defaultdict(set)
        rows = collections.defaultdict(set)
        squares = collections.defaultdict(set)  # key = (r /3, c /3)

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if (
                    board[r][c] in rows[r]
                    or board[r][c] in cols[c]
                    or board[r][c] in squares[(r // 3, c // 3)]
                ):
                    return False
                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                squares[(r // 3, c // 3)].add(board[r][c])

        return True

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        numSet = set(nums)
        longest = 0

        for n in numSet:
            # check if its the start of a sequence
            if (n - 1) not in numSet:
                length = 1
                while (n + length) in numSet:
                    length += 1
                longest = max(length, longest)
        return longest


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        low = 0
        high = len(nums) - 1
        mid = 0

        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid +=1
            else:
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1
        return nums

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        low = 0
        high = len(nums) - 1
        mid = 0

        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid +=1
            else:
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1
        return nums


class Codec:
    def __init__(self):
        self.encodeMap = {}
        self.decodeMap = {}
        self.base = "http://tinyurl.com/"

    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL.
        """
        if longUrl not in self.encodeMap: 
            shortUrl = self.base + str(len(self.encodeMap) + 1)
            self.encodeMap[longUrl] = shortUrl
            self.decodeMap[shortUrl] = longUrl
        return self.encodeMap[longUrl]

    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL.
        """
        return self.decodeMap[shortUrl]


class Solution:
    def leastBricks(self, wall: List[List[int]]) -> int:
        countGap = { 0 : 0 }    # { Position : Gap count }

        for r in wall:
            total = 0   # Position
            for b in r[:-1]:
                total += b
                countGap[total] = 1 + countGap.get(total, 0)

        return len(wall) - max(countGap.values())    # Total number of rows - Max gap

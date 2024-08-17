def isPalindrome(s):
    newStr = ""
    for c in s:
        if c.isalnum():
            newStr += c.lower()
    return newStr == newStr[::-1]



class solution:
    def isPalindrome(self,s):
        l,r = 0,len(s)-1
        while l <r:
            while l < r and not alphaNum(s[l]):
                l +=1
            while r >l and not alphaNum(s[r]):
                r -=1
            if s[1].lower() != s[r].lower():
                return False
            l,r = l+1,r-1
        return True

    def alphaNum(self,c):
        return (ord('A')<= ord(c)<= ord("Z") or
                ord('a')<= ord(c)<= ord("z") or
                ord('0')<= ord(c)<= ord("9"))
    

    def alphaNum(self,c):
        return (ord('A') <= ord(c) <= ord('C') or
                ord('a') <= ord(c) <= ord('z') or
                ord('0') <= ord(c) <= ord('9')
                )
    def isPalinDrome(s):
        l,r = 0, len(s)-1
        while l < r:
            while l < r and not alphaNum(s[l]):
                l +=1
            while l < r and not alphaNum(s[r]):
                r -=1
            if s[l].lower() != s[r].lower():
                return False
            l,r = l+1,r-1
        return True
    

    def validPalindrome(self,s):
        l,r =  0, len(s)-1
        while l <r:
            #if l != r 要么delete left 要么delete right，
            # 记住slice 在python 右边是not inclusive
            if s[l] != s[r]:
                #skip represnet the skip and del,python r not inclusive
                skipl,skipR = s[l+1:r+1],s[l:r]
            l,r = l+1,r-1
        return True



class Solution:
    #首先sort，固定一个窗口，就是差值的极大值
    def miniumDifference(self,nums):
        nums.sort()
        l,r = 0 , k-1
        res = float("inf")
        while r < len(nums):
            res = min(res,nums[r]-nums[l])
            l, r = l+1,r+1
        return res
 
    def mergeAlternately(word1,word2):
        # 理解alternately 就是交换着加上去
        i = j = 0
        res = []
        while i < len(word1) and j < len(word2):
            res.append(word1[i])
            res.append(word2[j])
            i += 1
            j += 1
        res.append(word1[i:])
        res.append(word2[j:])
        return ''.join(res)
    
    #这题是modify ，也可以用stack来解
    def reverseString(self,s):
        l = 0 
        r = len(s)-1
        while l < r:
            s[l],s[r] = s[r],s[l]
            l,r = l+1, r-1
    #因为后面，有空间，所以从后面往前，我们是加到1 上面的，
    def merge(slef,nums1, m,nums2,n):
        while m >0 and n>0:
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -=1
        else:
            nums1[m+n-1] = nums2[n-1]
            n-=1
        #剩下的是1的话无所wei,因为都是sorted
        if n >0:
            nums1[:n] = nums2[:n ]

    #two pointer 还有swap position 两个条件的nums哦！！！快慢指针,对了只要modify
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        slow = 0
        for fast in range(len(nums)):
            
            if nums[fast] != 0 and nums[slow] == 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
            #也要更新slow
            if nums[slow] != 0:
                slow += 1


#和上一题一样，左边的作为uniqiue number，每遇到一个unique 就和右边的指针替换，
#右边是iterate 所有一遍，l就是指向unique的，都不会回头
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        L = 1
        
        for R in range(1, len(nums)):
            if nums[R] != nums[R - 1]:
                nums[L] = nums[R]
                L += 1
        return L


#这个 的l 是位置当前 要被替换掉的， l 前面的是之前有效的
#可以往下走的时候，count从第一个开始，数到两个就是一个条件，min（2，count） 循环这个是要一个一个往前放置
#while 外循环遍历数组，顺便更新r指针，
# 内while是查看重复的次数，
# 然后for是把重复的按照当前的最多保存两个，一个一个的把右边的搬过来

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        l, r = 0, 0  # 初始化左右指针
        while r < len(nums):  
            count = 1
            # 计算当前元素的重复次数
            while r + 1 < len(nums) and nums[r] == nums[r + 1]:
                r += 1
                count += 1
            # 保留当前元素最多两次
            for i in range(min(2, count)):
                nums[l] = nums[r]
                l += 1
            r += 1  # 移动右指针到下一个新元素
        return l  # 返回新数组的长度



    # 2 sum input is sorted, l,r beiging r is the end, big shift the right
    # if too small shift to left
    def twosum(self,number,target):
        l,r = 0 ,len(number)-1
        while l <r :
            cursum = number[l] + number[r]
            if cursum > target:
                r -=1
            elif cursum < target:
                l +=1
            else:
                return [l,r]
        return []


#固定一个一点点向后，但是要给最后两个留个作为
    def threesum(self,num):
        num.sort()
        ans = []
        n =len(num)
        for i in range(n-2):
            x = num[i]
            if i >0 and x == num[i-1]:
                continue
            #opitmize,if the first three bigger than 0
            if x +num[i+1] + num[i+2] >0:
                break
            #if the last smaller than 0
            if x +num[i+1] + num[i+2] <0:
                break

            l = i+1
            r = n-1
            while l < r:
                s = x + num[l] + num[r]
                if s >0:
                    r -=1
                elif s <0:
                    l+=1
                else:
                    ans.append([x,num[l]],num[r])
                    l +=1
                    # skip the repeat
                    while l < r and num[l] == num[l-1]:
                        l +=1
                    r -=1
                    while l < r and num[r] ==num[r-1]:
                        r-=1
            return ans  

class Solution:
    def foursum(self,nums,target):
        def findNsum(l,r,target,N,result,results):
            # conditions 
            # 1. not enough # N  2. N <2 3. target < 区间内最小数的 N个数和 4.大于最大的可能
            if r-l+1 <N or N <2 or target < nums[l]*N or target > nums[r] * N:
                return
            if N ==2:
                while l < r:
                    s = nums[l] + nums[r]
                    if s == target:
                        results.append(result +[nums[l],nums[r]])
                        l+=1
                        while l < r and nums[l] == nums[l-1]:
                            l +=1
                    elif s < target:
                        l +=1
                    else:
                        r -=1
            else:
                #区间从左到右遍历
                for i in range(1,r+1):
                    # 如果遇到一样的到下一个，继续遍历
                    if i == l or (i >l and nums[i-1] != nums[i]):
                    #     i + 1: 将新的左边界设为 i + 1，以避免重复使用当前元素。
                    # r: 右边界保持不变。
                    # target - nums[i]: 调整目标和，减去当前选定的元素值。
                    # N - 1: 减少需要寻找的数字个数。
                    # result + [nums[i]]: 将当前选定的元素加入到结果列表中。
                    # results: 最终结果列表。
                findNsum(i+1,r,target-nums[i],N-1,result + [nums[i]],results)
        nums.sort()
        results = []
        findNsum(0,len(nums)-1,target,4,[],results)
        return results
    
    def containMostWater(height):
        l,r = 0, len(height)-1
        res = 0
        while l < r:
            res = max(res,min(height[l],height[r])*(r-l))
            if height[l] < height[r]:
                l +=1
            else:
                r -=1
        return res
    

#这题有最大最小值，所以可以sort，因为最大最小值是固定的还有subsquence 不一定要是连续的
class Solution:
    def numSubsqe(self,nums,target):
        nums.sort()
        res = 0
        mod = (10**9+7)
        r = len(nums)-1
        for i, left in enumerate(nums):
            while (left+nums[r]) > target and i <=r:
                r -=1
            if i <= r:
                res +=(2**(r-i))
        return res
    


class Solution:
    #1 solution is to +k the postion is gonna be k%len but the thing is the solution time complexity is O(n)
    # 2. solution is to reverse and reverse the first k half and rever the remaining
    def rotate(self,nums):
        k = k % len(nums)
        l,r = 0 , len(nums)-1
        while  l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l,r = l+1,r-1
        l,r = 0 , k-1
        while  l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l,r = l+1,r-1
        l,r = k,len(nums)-1
        while  l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l,r = l+1,r-1
    
#if  we  can guarntee both its neighbour bigger than itself, or less it
#very interesting, 前面的数字之间空着一个格子，然后让后面的填， 前面应该是half，后面就是half-1啦，然后左右两边要么全部都是大于
# 要么全部都是小小于 
class Solution:
    def rearrangeArray(self,nums):
        nums.sort()
        res = []
        l,r = 0, len(nums)-1
        while len(res) != len(nums):
            res.append(nums[l])
            l +=1
            if l <= r:
                res.append(nums[r])
                r-=1
        return res
    
    def rearrangeArray(self, nums: List[int]) -> List[int]:
        nums.sort()

        i, j, n = 0, 0, len(nums)
        ans = [0]*n

        while i < n  and j < n:
            ans[i] = nums[j]kkk
            i = i + 2
            j = j + 1

        i = 1
        while i < n and j < n:
            ans[i] = nums[j]
            i = i + 2
            j = j + 1

        return ans
    
class Solution(object):
    def numRescueBoats(self, people: list[int], limit: int) -> int:
        people.sort()
        right = len(people) - 1
        left = res = 0
        while left <= right:
            if people[left] + people[right] <= limit:
                left += 1
            right -= 1
            res += 1
        return res





    











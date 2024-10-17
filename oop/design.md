### 6. **Design Twitter**

**Difficulty**: Medium

#### Description:
Design a Twitter-like system that allows users to post tweets, follow others, and retrieve the 10 most recent tweets from their feed.

#### Example:
```python
Input:
["Twitter", "postTweet", "getNewsFeed", "follow", "postTweet", "getNewsFeed"]
[[], [1, 5], [1], [1, 2], [2, 6], [1]]
Output: [null, null, [5], null, null, [6, 5]]
```


```py
from collections import defaultdict
import heapq
from typing import List

class Twitter:
    def __init__(self):
        # 用于计数每条推文的顺序，确保推文按时间顺序排列
        self.count = 0
        # 存储每个用户的推文列表，userId -> list of [count, tweetId]
        self.tweetMap = defaultdict(list)
        # 存储每个用户关注的用户集合，userId -> set of followeeId
        self.followMap = defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        # 用户发布一条推文，将推文添加到用户的推文列表中
        # 使用负的 self.count 确保最新的推文在最小堆中具有最高优先级
        self.tweetMap[userId].append([self.count, tweetId])
        self.count -= 1

    def getNewsFeed(self, userId: int) -> List[int]:
        # 初始化返回结果列表，用于存储最新的推文 ID
        res = []
        # 初始化最小堆，用于存储用户及其关注者的推文，最多保留 10 条最新推文
        minHeap = []
        
        # 用户关注自己，这样用户可以看到自己的推文
        # 这是一种常见的社交媒体逻辑，用户自然应该看到自己发布的内容
        self.followMap[userId].add(userId)  # 确保用户自己也在关注列表中
        
        # 遍历用户关注的所有用户（包括自己），获取每个用户的推文
        for followeeId in self.followMap[userId]:
            if followeeId in self.tweetMap:
                # 获取该关注者的最新推文的索引（即推文列表的最后一个元素）
                index = len(self.tweetMap[followeeId]) - 1
                
                # 获取推文的计数和推文 ID
                # 计数用于按照时间顺序排序推文，推文 ID 用于标识具体的推文
                count, tweetId = self.tweetMap[followeeId][index]
                
                # 将该推文的信息加入最小堆中，包含计数、推文 ID、用户 ID 和前一个推文的索引
                # 使用最小堆来确保我们可以按顺序获取最新的推文
                heapq.heappush(minHeap, [count, tweetId, followeeId, index - 1])
        
        # 获取最多 10 条最新推文
        while minHeap and len(res) < 10:
            # 从堆中弹出计数最小（即时间最晚）的推文
            count, tweetId, followeeId, index = heapq.heappop(minHeap)
            
            # 将该推文的 ID 添加到结果列表中
            res.append(tweetId)
            
            # 如果该用户还有前面的推文，继续获取前一条推文的信息并加入堆中
            # 这样可以继续获取该用户的更早的推文，直到堆为空或者获取到 10 条推文
            if index >= 0:
                count, tweetId = self.tweetMap[followeeId][index]
                heapq.heappush(minHeap, [count, tweetId, followeeId, index - 1])
        
        # 返回新闻推送的推文 ID 列表，包含用户和关注者的最多 10 条最新推文
        return res

    def follow(self, followerId: int, followeeId: int) -> None:
        # 用户关注另一个用户，将该用户添加到关注列表中
        self.followMap[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        # 用户取消关注另一个用户，如果该用户在关注列表中则将其移除
        # 确保用户不能取消关注自己
        if followeeId in self.followMap[followerId] and followeeId != followerId:
            self.followMap[followerId].remove(followeeId)

```
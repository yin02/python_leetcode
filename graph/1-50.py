
def dfs(graph,start):
    visited = set()
    stack = [start]#1,2
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)

class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        visit = set()
        def dfs(i, j):
            if i >= len(grid) or j >= len(grid[0]) or i < 0 or j < 0 or grid[i][j] == 0:
                return 1
            if (i, j) in visit:
                return 0
            visit.add((i, j))
            perim = dfs(i, j + 1)
            perim += dfs(i + 1, j)
            perim += dfs(i, j - 1)
            perim += dfs(i - 1, j)
            return perim
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]:
                    return dfs(i, j)

class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        orderInd = { c : i for i, c in enumerate(order)}
        
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            
            for j in range(len(w1))
                if j == len(w2):
                    return False
                
                if w1[j] != w2[j]:
                    if orderInd[w2[j]] < orderInd[w1[j]]:
                        return False
                    break
        return True

    def isAlienSorted(self, words: List[str], order: str) -> bool:
        index = {c: i for i, c in enumerate(order)}
        return all(s <= t for s, t in pairwise([index[c] for c in word] for word in words))



class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        if "0000" in deadends:
            return -1
        def children(wheel):
            res = []
            for i in range(4):
                digit = str((int(wheel[i]) + 1) % 10)
                res.append(wheel[:i] + digit + wheel[i + 1 :])
                digit = str((int(wheel[i]) + 10 - 1) % 10)
                res.append(wheel[:i] + digit + wheel[i + 1 :])
            return res
        q = deque()
        visit = set(deadends)
        q.append(["0000", 0])  # [wheel, turns]
        while q:
            wheel, turns = q.popleft()
            if wheel == target:
                return turns
            for child in children(wheel):
                if child not in visit:
                    visit.add(child)
                    q.append([child, turns + 1])
        return -1

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]:
            return 0
        islands = 0
        visit = set()
        rows, cols = len(grid), len(grid[0])
        def dfs(r, c):
            if (
                r not in range(rows)
                or c not in range(cols)
                or grid[r][c] == "0"
                or (r, c) in visit
            ):
                return
#前面已经检查过是否在visit，所以后面才加visit
            visit.add((r, c))
            directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            for dr, dc in directions:
                dfs(r + dr, c + dc)

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visit:
                    islands += 1
                    dfs(r, c)
        return islands

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        rows, cols = len(grid), len(grid[0])
        visited=set()
        islands=0
        def bfs(r,c):
            q = deque()
            visited.add((r,c))
            q.append((r,c))
        
            while q:
                row,col = q.pop()# 这边变成了dfs
                directions= [[1,0],[-1,0],[0,1],[0,-1]]
            
                for dr,dc in directions:
                    r,c = row + dr, col + dc
                    if (r) in range(rows) and (c) in range(cols) and grid[r][c] == '1' and (r ,c) not in visited:shi
                    
                        q.append((r , c ))
                        visited.add((r, c ))
        for r in range(rows):
            for c in range(cols):
            
                if grid[r][c] == "1" and (r,c) not in visited:
                    bfs(r,c)
                    islands +=1 
        return islands

1.Stack to dfs（Graph）	
	def dfs(graph,start):
	    visited = set()
	    stack = [start]#1,2
	    while stack:
	        vertex = stack.pop()
	        if vertex not in visited:
	            visited.add(vertex)
	            stack.extend(neighbor for neighbor in graph[vertex] if neighbor not in visited)
2.Island primeter	class Solution:
	    def islandPerimeter(self, grid: List[List[int]]) -> int:
	        visit = set()
	        def dfs(i, j):
	            if i >= len(grid) or j >= len(grid[0]) or i < 0 or j < 0 or grid[i][j] == 0:
	                return 1
	            if (i, j) in visit:
	                return 0
	            visit.add((i, j))
	            perim = dfs(i, j + 1)
	            perim += dfs(i + 1, j)
	            perim += dfs(i, j - 1)
	            perim += dfs(i - 1, j)
	            return perim
	        for i in range(len(grid)):
	            for j in range(len(grid[0])):
	                if grid[i][j]:
	                    return dfs(i, j)
3.Verifying an Alien Dictionary	class Solution:
	    def isAlienSorted(self, words: List[str], order: str) -> bool:
	        orderInd = { c : i for i, c in enumerate(order)}
	        
	        for i in range(len(words) - 1):
	            w1, w2 = words[i], words[i + 1]
	            
	            for j in range(len(w1))
	                if j == len(w2):
	                    return False
	                
	                if w1[j] != w2[j]:
	                    if orderInd[w2[j]] < orderInd[w1[j]]:
	                        return False
	                    break
	        return True
	
	    def isAlienSorted(self, words: List[str], order: str) -> bool:
	        index = {c: i for i, c in enumerate(order)}
	        return all(s <= t for s, t in pairwise([index[c] for c in word] for word in words))
	
	
4.openLock	class Solution:
	    def openLock(self, deadends: List[str], target: str) -> int:
	        if "0000" in deadends:
	            return -1
	        def children(wheel):
	            res = []
	            for i in range(4):
	                digit = str((int(wheel[i]) + 1) % 10)
	                res.append(wheel[:i] + digit + wheel[i + 1 :])
	                digit = str((int(wheel[i]) + 10 - 1) % 10)
	                res.append(wheel[:i] + digit + wheel[i + 1 :])
	            return res
	        q = deque()
	        visit = set(deadends)
	        q.append(["0000", 0])  # [wheel, turns]
	        while q:
	            wheel, turns = q.popleft()
	            if wheel == target:
	                return turns
	            for child in children(wheel):
	                if child not in visit:
	                    visit.add(child)
	                    q.append([child, turns + 1])
	        return -1
5.numIslands	class Solution:
	    def numIslands(self, grid: List[List[str]]) -> int:
	        if not grid or not grid[0]:
	            return 0
	        islands = 0
	        visit = set()
	        rows, cols = len(grid), len(grid[0])
	        def dfs(r, c):
	            if (
	                r not in range(rows)
	                or c not in range(cols)
	                or grid[r][c] == "0"
	                or (r, c) in visit
	            ):
	                return
	#前面已经检查过是否在visit，所以后面才加visit
	            visit.add((r, c))
	            directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
	            for dr, dc in directions:
	                dfs(r + dr, c + dc)
	
	        for r in range(rows):
	            for c in range(cols):
	                if grid[r][c] == "1" and (r, c) not in visit:
	                    islands += 1
	                    dfs(r, c)
	        return islands
	
	class Solution:
	    def numIslands(self, grid: List[List[str]]) -> int:
	        if not grid:
	            return 0
	        rows, cols = len(grid), len(grid[0])
	        visited=set()
	        islands=0
	        def bfs(r,c):
	            q = deque()
	            visited.add((r,c))
	            q.append((r,c))
	        
	            while q:
	                row,col = q.pop()# 这边变成了dfs
	                directions= [[1,0],[-1,0],[0,1],[0,-1]]
	            
	                for dr,dc in directions:
	                    r,c = row + dr, col + dc
	                    if (r) in range(rows) and (c) in range(cols) and grid[r][c] == '1' and (r ,c) not in visited:shi
	                    
	                        q.append((r , c ))
	                        visited.add((r, c ))
	        for r in range(rows):
	            for c in range(cols):
	            
	                if grid[r][c] == "1" and (r,c) not in visited:
	                    bfs(r,c)
	                    islands +=1 
	        return islands
	
6.Clone graph 	可以用bfs 或者dfs，bfs不需要check if node in oldToNew
	class Solution:
	    def cloneGraph(self, node: "Node") -> "Node":
	#创一个hash
	        oldToNew = {}
	        def dfs(node):
	            if node in oldToNew:
	                return oldToNew[node]
	
	            copy = Node(node.val)
	            oldToNew[node] = copy
	
	            for nei in node.neighbors:
	                copy.neighbors.append(dfs(nei))
	            return copy
	        return dfs(node) if node else None
	
	bfs
	class Solution:
	    def cloneGraph(self, node: 'Node') -> 'Node':
	        if not node: return node
	        
	        q, clones = deque([node]), {node.val: Node(node.val, [])}
	        while q:
	            cur = q.popleft() 
	            cur_clone = clones[cur.val]            
	            for ngbr in cur.neighbors:
	                if ngbr.val not in clones:
	                    clones[ngbr.val] = Node(ngbr.val, [])
	                    q.append(ngbr)
	                    
	                cur_clone.neighbors.append(clones[ngbr.val])
	                
	        return clones[node.val]
	
7.maxAreaOfIsland	    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
	        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
	        rows, cols = len(grid), len(grid[0])
	        ans = 0
	        for i in range(rows):
	            for j in range(cols):
	                if grid[i][j] == 1:
	                    grid[i][j] = 0  # 标记为已访问
	                    temp_ans = 1
	                    queue = collections.deque()
	                    queue.append((i, j))  # 将起始坐标加入队列
	                    while queue:
	                        r, c = queue.popleft()
	                        for rd, cd in directions:
	                            v_i, v_j = r + rd, c + cd
	                            if v_i < 0 or v_j < 0 or v_i >= rows or v_j >= cols or grid[v_i][v_j] == 0:
	                                continue
	                            grid[v_i][v_j] = 0  # 标记为已访问
	                            queue.append((v_i, v_j))
	                            temp_ans += 1
	                    ans = max(ans, temp_ans)
	        return ans
8.Topological Sorting order	
	import collections
	class Solution:
	    def topologicalSortingKahn(self, graph: dict):
	        indegrees = {u: 0 for u in graph}
	        for u in graph:
	            for v in graph[u]:
	                indegrees[v] += 1
	        
	        S = collections.deque([u for u in indegrees if indegrees[u] == 0])
	        order = []
	        
	        while S:
	            u = S.pop()
	            order.append(u)
	            for v in graph[u]:
	                indegrees[v] -= 1
	                if indegrees[v] == 0:
	                    S.append(v)
	        
	        if len(indegrees) != len(order):
	            return []
	        return order
	    
	    def findOrder(self, n: int, edges):
	        graph = dict()
	        for i in range(n):
	            graph[i] = []
	            
	        for u, v in edges:
	            graph[u].append(v)
	            
	        return self.topologicalSortingKahn(graph)
9.dfs topo	
	import collections
https://www.cs.usfca.edu/~galles/visualization/TopoSortDFS.html	class Solution:
Graph Topological Sort Using Depth-First Search	    def topologicalSortingDFS(self, graph: dict):
	        visited = set()
	        onStack = set()
	        order = []
	        hasCycle = False
	        
	        def dfs(u):
	            nonlocal hasCycle
	            if u in onStack:
	                hasCycle = True
	            if u in visited or hasCycle:
	                return
	            
	            visited.add(u)
	            onStack.add(u)
	    
	            for v in graph[u]:
	                dfs(v)
	                    
	            order.append(u)
	            onStack.remove(u)
	        
	        for u in graph:
	            if u not in visited:
	                dfs(u)
	        
	        if hasCycle:
	            return []
	        order.reverse()
	        return order
	    
	    def findOrder(self, n: int, edges):
	        graph = dict()
	        for i in range(n):
	            graph[i] = []
	        for v, u in edges:
	            graph[u].append(v)
	        
	        return self.topologicalSortingDFS(graph)
10. 课程表 course schedule	import collections
	class Solution:
	    def topoSort(self,numCourses,graph):
	    #hash get indegree
	        indegree = {u: 0 for u in range(numCourses)}
	        for u in graph:
	            for v in graph[u]:
	                indegree[v] += 1
	        S = collections.deque([u for u in indegree if indegree[u]== 0])
	        while S:
	            u = S.pop()
	            numCourses -= 1
	            for v in graph[u]:
	                indegree[v]-= 1
	                if indegree[v] == 0:
	                    S.append(v)
	        if numCourses == 0:
	            return True
	        return False
	    def canFinish(self,numCourses,prerequisites):
	        graph = collections.defaultdict(list)
	        for v,u in prerequisites:
	            graph[u].append(v)
	        return self.topoSort(numCourses,graph)
11. Course ScheduleII	from collections import defaultdict, deque
	class Solution:
	    def findOrder(self, numCourses, prerequisites):
	        indegrees = [0] * numCourses
	        graph = defaultdict(list)
	        
	        for course, pre in prerequisites:
	            graph[pre].append(course)
	            indegrees[course] += 1
	        
	        queue = deque([u for u in range(numCourses) if indegrees[u] == 0])
	        order = []
	        while queue:
	            u = queue.popleft()
	            order.append(u)
	            for v in graph[u]:
	                indegrees[v] -= 1
	                if indegrees[v] == 0:
	                    queue.append(v)
	        
	        if len(order) != numCourses:
	            return []
	        return order
	
	import collections
	class Solution:
	    def topologicalSortingKahn(self, graph, numCourses):
	        # Initialize indegrees for all courses to ensure no KeyError
	        indegrees = {u: 0 for u in range(numCourses)}
	        
	        # Then, update indegrees based on the actual graph edges
	        for u in graph:
	            for v in graph[u]:
	                indegrees[v] += 1
	        S = collections.deque([u for u in indegrees if indegrees[u] == 0])
	        order = []
	        while S:
	            u = S.pop()
	            order.append(u)
	            for v in graph[u]:
	                indegrees[v] -= 1
	                if indegrees[v] == 0:
	                    S.append(v)
	        if len(order) != numCourses:  # Check if all courses are covered
	            return []
	        return order
	    def findOrder(self, numCourses, prerequisites):
	        graph = collections.defaultdict(list)
	        for v, u in prerequisites:
	            graph[u].append(v)
	        return self.topologicalSortingKahn(graph, numCourses)
	
	
12 Find Eventual Safe States	理解拓扑只有，非环才能被加入长度
	class Solution:
	    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
	        reversed_graph = [[] for _ in graph]  # 反向图的邻接表
	        for u, vs in enumerate(graph):# graph也可键值对
	            for v in vs:
	                reversed_graph[v].append(u)  # 将原图的边反向加入反向图
	        in_degrees = [len(u) for u in graph]  # 每个节点的出度(其实是入度，因为后面reverse了我们反着看，也可以用reversed来造)，u是key ，len（u）是值，这样造键值对！！！！！
	        queue = deque([u for u, degree in enumerate(in_degrees) if degree == 0])  
	        while queue:
	            for u in reversed_graph[queue.popleft()]:  
	                in_degrees[u] -= 1  
	                if in_degrees[u] == 0: 
	                    queue.append(u)  
	        return [u for u, degree in enumerate(in_degrees) if degree == 0]
	
	class Solution:
	    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
	        n = len(graph)  # 获取图中节点的数量
	        color = [0] * n  # 用于记录节点的颜色状态，0表示未访问，1表示正在访问，2表示安全
	        def safe(x: int) -> bool:
	            if color[x] > 0:  # 如果节点已经被访问过
	                return color[x] == 2  # 如果节点已经标记为安全，则返回True
	            color[x] = 1  # 将当前节点标记为正在访问
	            for y in graph[x]:  # 遍历当前节点的邻居节点
	                if not safe(y):  # 如果邻居节点不安全
	                    return False  # 返回False
	            color[x] = 2  # 如果当前节点的所有邻居节点都安全，则将当前节点标记为安全
	            return True
	        return [i for i in range(n) if safe(i)]  # 返回所有安全节点的列表
	
13. Number of connected component	dfs
	from collections import defaultdict
	class Solution:
	    def countComponents(self, n: int, edges: List[List[int]]) -> int:
	        visit = [False] * n
	        graph = [[] for _ in range(n)]
	        for i, j in edges:
	            graph[i].append(j)
	            graph[j].append(i)
	        
	        def dfs(i):
	            visit[i] = True
	            for neighbor in graph[i]:
	                if not visit[neighbor]:
	                    dfs(neighbor)
	        
	        count = 0
	        for i in range(n):
	            if not visit[i]:
	                dfs(i)
	                count += 1
	        return count
	
	Union
	
	class UnionFind:
	    def __init__(self, size):
	        self.parent = list(range(size))
	        self.count = size
	    
	    def find(self, x):
	        if self.parent[x] != x:
	            self.parent[x] = self.find(self.parent[x]) # 路径压缩
	        return self.parent[x]
	    
	    def union(self, x, y):
	        rootX = self.find(x)
	        rootY = self.find(y)
	        if rootX != rootY:
	            self.parent[rootX] = rootY
	            self.count -= 1
	class Solution:
	    def countComponents(self, n: int, edges: List[List[int]]) -> int:
	        uf = UnionFind(n)
	        for i, j in edges:
	            uf.union(i, j)
	        return uf.count
	
	
	class UnionFind:
	    def __init__(self):
	        self.f = {}
	        
	    def findParent(self, x):
	        y = self.f.get(x, x)
	        if x != y:
	            y = self.f[x] = self.findParent(y)
	        return y
	    
	    def union(self, x, y):
	        self.f[self.findParent(x)] = self.findParent(y)
	class Solution:
	    def countComponents(self, n: int, edges: List[List[int]]) -> int:
	        dsu = UnionFind()
	        for a, b in edges:
	            dsu.union(a, b)
	        return len(set(dsu.findParent(x) for x in range(n)))
	
14.Detect cycle in an undirected graph | GeeksforGeeks	def detect_cycle(graph):
	    def dfs(vertex, visited, recStack):
dfs相对于当时的自身节点visited只有可能是父亲	        visited.add(vertex)
	        recStack.add(vertex)
	        
	        for neighbor in graph.get(vertex, []):
	            if neighbor not in visited:
	                if dfs(neighbor, visited, recStack):#如果邻居有true
	                    return True
	            elif neighbor in recStack:
	                return True
	        
	        recStack.remove(vertex)
	        return False
	    
	    visited = set()
	    recStack = set()
	    
	    for vertex in graph:
	        if vertex not in visited:
	            if dfs(vertex, visited, recStack):
	                return True
	    
	    return False
15 . UnionFind	import heapq
	class Unionfind:
	    def __init__(self,n):
	        self.par = {}
	        self.rank = {}
	        for i in range(1,n+1):
	            self.par[i] = i
	            self.rank[i] = 0
	    def find(self,n):
	        p = self.par[n]
	        while p != self.par[p]:
	            self.par[p] = self.par[self.par[p]]
	            p = self.par[p]
	        return p
	    def union(self,n1,n2):
	        p1 ,p2  = self.find(n1),self.find(n2)
	        if self.rank(p1) > self.rank(p2):
	            self.par[p2] = p1
	        elif self.rank(p2) > self.rank(p1):
	            self.par[p1] = p2
	        else:
	            self.par[p1] = p2
	            self.rank[p2] +=1
	        return True
 self.par[p] = self.par[self.par[p]]	
	
第二种压缩	
    # 第二种路径压缩的 find 方法deffind(self,x:int)->int:	
        if self.parent[x]!=x:
           self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
	
	class UF:
	    count: int
	    parent: List[int]
	    
	    def __init__(self, n: int):
	        self.count = n
	        self.parent = [i for i in range(n)]
	    
	    def union(self, p: int, q: int):
	        rootP = self.find(p)
	        rootQ = self.find(q)
	        if rootP == rootQ:
	            return
	        self.parent[rootQ] = rootP
	        self.count -= 1
	    
	    def connected(self, p: int, q: int) -> bool:
	        rootP = self.find(p)
	        rootQ = self.find(q)
	        return rootP == rootQ
	    
	    def find(self, x: int) -> int:
	        if self.parent[x] != x:
	            self.parent[x] = self.find(self.parent[x])
	        return self.parent[x]
	    
	    def count(self) -> int:
	        return self.count
16Graph Valid Tree	
	def validTree(self, n, edges):
	    if not n:
	        return True
	    adj = {i: [] for i in range(n)}
	    for n1, n2 in edges:
	        adj[n1].append(n2)
	        adj[n2].append(n1)
	    visit = set()
	    def dfs(i, prev):
	        if i in visit:
	            return False
	        visit.add(i)
	        for j in adj[i]:
	            if j == prev:
	                continue
	            if not dfs(j, i):
	                return False
	        return True
	    return dfs(0, -1) and n == len(visit)
17.Dijkstar's shortest path	
	import heapq
	def shortestPath(edges, n, src):
	    adj = {}  
	    for i in range(1, n + 1):
	        adj[i] = []  
	    for s, e, w in edges:
	        adj[s].append([e, w])
	    shortest = {}
	    minHeap = [[0, src]]
	    while minHeap:
	        w1, c1 = heapq.heappop(minHeap)
	        if c1 in shortest:
	            continue
	        shortest[c1] = w1
	        for n2, w2 in adj[c1]:
	            if n2 not in shortest:
	                heapq.heappush(minHeap, [w1 + w2, n2])
	    return shortest
如果只关心起点 start 到某一个终点 end 的最短路径，是否可以修改代码提升算法效率。	  while pq:
	        curState = heapq.heappop(pq)
	        curNodeID, curDistFromStart = curState.id, curState.distFromStart
	        # 在这里加一个判断就行了，其他代码不用改，因为是minHeap
	        if curNodeID == end:
	            return curDistFromStart
	        if curDistFromStart > distTo[curNodeID]:
	            continue
18.Network Delay Time	class Solution:
	    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
	        edges = collections.defaultdict(list)
	        for u, v, w in times:
	            edges[u].append((v, w))
	        minHeap = [(0, k)]
	        visit = set()
	        t = 0
	        while minHeap:
	            w1, n1 = heapq.heappop(minHeap)
	            if n1 in visit:
	                continue
	            visit.add(n1)
	            t = w1
	            for n2, w2 in edges[n1]:
	                if n2 not in visit:
	                    heapq.heappush(minHeap, (w1 + w2, n2))
	        return t if len(visit) == n else -1
	        # O(E * logV)
19.Swim in Rising Water	class Solution:
	    def swimInWater(self, grid: List[List[int]]) -> int:
	        N = len(grid)
	        visit = set()
	        minH = [[grid[0][0], 0, 0]]  # (time/max-height, r, c)
	        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
	        visit.add((0, 0))
	        while minH:
	            t, r, c = heapq.heappop(minH)
	            if r == N - 1 and c == N - 1:
	                return t
	            for dr, dc in directions:
	                neiR, neiC = r + dr, c + dc
	                if (
	                    neiR < 0
	                    or neiC < 0
	                    or neiR == N
	                    or neiC == N
	                    or (neiR, neiC) in visit
	                ):
	                    continue
	                visit.add((neiR, neiC))
	                heapq.heappush(minH, [max(t, grid[neiR][neiC]), neiR, neiC])
20.Path with Maximum Probability	
	import collections
	import heapq
	class Solution:
	    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
	        adj = collections.defaultdict(list)
	        for i in range(len(edges)):
	            src, dst = edges[i]
	            adj[src].append([dst, succProb[i]])
	            adj[dst].append([src, succProb[i]])
	        
	        pq = [(-1, start)]
	        visit = set()  
	        while pq:
	            prob, cur = heapq.heappop(pq)
	            visit.add(cur)  
	            if cur == end:
	                return prob * -1
	            
	            for nei, edgeProb in adj[cur]:
	                if nei not in visit:
	                    heapq.heappush(pq, (prob * edgeProb, nei))
	        return 0
21.Bellman-Ford   	可以处理负数
	function BellmanFord(G, S):
	    distance[] = Infinity for each vertex in G
	    distance[S] = 0
	# check all edges
	    for 1 to V-1:
	        for each edge (u, v) in G:
	            if distance[u] + weight(u, v) < distance[v]:
	                distance[v] = distance[u] + weight(u, v)
	
	# negative cycle
	    for each edge (u, v) in G:
	        if distance[u] + weight(u, v) < distance[v]:
	            return "Graph contains a negative-weight cycle"
	    
	    return distance
	
22.Primes‘algorthim mst	
Spanning tree is basically used to find a minimum path to connect all nodes in a graph	import heapq
	def minimumSpanningTree(edges, n):
	    adj = {}
	    for i in range(1, n + 1):
	        adj[i] = []
	    for n1, n2, weight in edges:
	        adj[n1].append([n2, weight])
	        adj[n2].append([n1, weight])
	    
	    minHeap = []
	    for neighbor, weight in adj[1]:
	        heapq.heappush(minHeap, [weight, 1, neighbor])
	    
	    mst = []
	    visit = set()
	    visit.add(1)
	    
	    while len(visit) < n:
	        weight, n1, n2 = heapq.heappop(minHeap)
	        if n2 in visit:
	            continue
	        mst.append([n1, n2])
	        visit.add(n2)
	        for neighbor, weight in adj[n2]:
	            if neighbor not in visit:
	                heapq.heappush(minHeap, [weight, n2, neighbor])
	    
	    return mst
23.Kruskal's algorithms(mst)	import heapq
这个是把weight 先过来排序，用边来考虑	class Unionfind:
然后prim更像是用node去考虑一切的	    def __init__(self,n):
	        self.par = {}
	        self.rank = {}
	        for i in range(1,n+1):
	            self.par[i] = i
	            self.rank[i] = 0
	    def find(self,n):
	        p = self.par[n]
	        while p != self.par[p]:
	            self.par[p] = self.par[self.par[p]]
	            p = self.par[p]
	        return p
	    def union(self,n1,n2):
	        p1 ,p2  = self.find(n1),self.find(n2)
	        if self.rank(p1) > self.rank(p2):
	            self.par[p2] = p1
	        elif self.rank(p2) > self.rank(p1):
	            self.par[p1] = p2
	        else:
	            self.par[p1] = p2
	            self.rank[p2] +=1
	        return True
	def minimumSpanningTree(edges,n):
	    minHeap = []
	    for n1,n2 , weight in edges:
	        heapq.heappush(minHeap,[weight,n1,n2])
	    Unionfind = Unionfind(n)
	    mst =[]
	    while len(mst)<n-1:
	        weight,n1,n2 = heapq.heappop(minHeap)
	        #如果不能用
	        if not Unionfind.union(n1,n2):
	            continue
	        mst.append([n1,n2])
	    return mst   
24.Mahanton distance	class Solution:
	    def minCostConnectPoints(self,points):
	        N = len(points)
	        adj = {i:[] for i in range(N)}
	        for i in range(N):
	            x1,y1 = points[i]
	            for j in range(i+1,N):
	                x2,y2 = points[j]
	                dist = abs(x1-x2)+abs(y1-y2)
	                adj[i].append([dist,j])
	                adj[j].append([dist,i])
	        res =0
	        visit = set()
	        minH = [[0,0]]
	        while len(visit) < N:
	            cost, i = heapq.heappop(minH)
	            if i in visit:
	                continue
	            res += cost
	            visit.add(i)
	            for neiCost,nei in adj[i]:
	                if nei not in visit:
	                    heapq.heappush(minH,[neiCost,nei])
	        return res
	
25.计算给定图中每个节点的连通分量大小	from collections import deque
	def getVisibleProfilesCount(connection_nodes, connection_from, connection_to, queries):
	    graph = {i: set() for i in range(1, connection_nodes + 1)}
	    connected_component_sizes = {}
	    
	    for f, t in zip(connection_from, connection_to):
	        graph[f].add(t)
	        graph[t].add(f)
	    
	    def bfs(start_node):
	        if start_node in connected_component_sizes:
	            return connected_component_sizes[start_node]
	        
	        visited = set([start_node])
	        queue = deque([start_node])
	        while queue:
	            node = queue.popleft()
	            for neighbour in graph[node]:
	                if neighbour not in visited:
	                    visited.add(neighbour)
	                    queue.append(neighbour)
	        
	        component_size = len(visited)
	        for node in visited:
	            connected_component_sizes[node] = component_size
	        
	        return component_size
	    
	    result = []
	    for query in queries:
	        result.append(bfs(query))
	    
	    return result
26.Longest Consecutive Sequence	class UnionFind:
 return the length of the longest	    def __init__(self, elements):
consecutive elements sequence. You must use the Union Find algorithm	        self.parent = {element: element for element in elements}
	        self.rank = {element: 1 for element in elements}
	    def find(self, element):
	        if self.parent[element] != element:
	            self.parent[element] = self.find(self.parent[element])
	        return self.parent[element]
	    def union(self, u, v):
	        root_u = self.find(u)
	        root_v = self.find(v)
	        if root_u != root_v:
	            # Union by rank
	            if self.rank[root_u] > self.rank[root_v]:
	                self.parent[root_v] = root_u
	            elif self.rank[root_u] < self.rank[root_v]:
	                self.parent[root_u] = root_v
	            else:
	                self.parent[root_v] = root_u
	                self.rank[root_u] += 1
	def longest_consecutive(nums):
	    if not nums:
	        return 0
	    
	    uf = UnionFind(nums)
	    num_set = set(nums)  # To check the existence of neighbors in O(1)
	    for num in nums:
	        if num + 1 in num_set:
	            uf.union(num, num + 1)
	    
	    max_length = 0
	    for num in nums:
	        root = uf.find(num)
	        max_length = max(max_length, uf.rank[root])
	    return max_length
	# Example usage
	print(longest_consecutive([100, 4, 200, 1, 3, 2]))  # Output: 4
27.矩阵中从左上角到右下角的最短路径所需的最小时间	
包括矩阵的大小 n、已访问过的单元格集合 visited 以及优先队列 pq	import heapq
	def min_time_to_travel_to_bottom(grid):
	    n = len(grid)
	    visited = set()
	    pq = [(0, 0, 0, grid[0][0])]
	    
	    while pq:
	        time, row, column, ladder_height = heapq.heappop(pq)
	        if (row, column) == (n-1, n-1):
	            return time
	        if (row, column, ladder_height) in visited:
	            continue
	        visited.add((row, column, ladder_height))
	        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
	            r, c = row + dr, column + dc
	            if 0 <= r < n and 0 <= c < n:
	                next_ladder_height = max(ladder_height, grid[r][c])
	                additional_time = max(0, grid[r][c] - ladder_height)
	                if (r, c, next_ladder_height) not in visited:
	                    heapq.heappush(pq, (time + additional_time, r, c, next_ladder_height))
	    
	    return -1
	
 res = tt * (tt - 1) * (tt - 2) // 6  # 计算棋盘上选3个方格的组合数	
择 3 个方格的排列再除以 3!。所以，我们除以 3! 来去除重复计算	def knightPlacement(n, m):
从 tt 个元素中选择 3 个元素的组合数	    ins = 0  # 初始化无效放置的数量
需要考虑到组合数不考虑元素的顺序	    tt = n * m  # 计算棋盘的总方格数
	    res = tt * (tt - 1) * (tt - 2) // 6  # 计算棋盘上选3个方格的组合数
• 28	    # 骑士的移动方向
	    dirs = [(1, 2), (2, 1), (1, -2), (2, -1), (-1, -2), (-1, 2), (-2, -1), (-2, 1)]
	#总的-遇到的=没有遇到的
	    # 遍历棋盘上的每一个方格，第一个骑士可以下的地方，任意一个点
	    for i in range(1, n+1):
	        for j in range(1, m+1):
	            # 第二个骑士可以去的方向，可遇到的
	            for k in range(8):#第二步
	                tx = i + dirs[k][0]
	                ty = j + dirs[k][1]
	                # 对于有效的第二个骑士
	                if 1 <= tx <= n and 1 <= ty <= m:
	                    ins += n * m - 2  # 第三个棋子可以遇到总数量（虚假的） ，第一，二个 已经确定了，剩下mn-2
	                    # 对于第三个棋子，因为有了第二个棋子走所以k+1的循环下一个方向
	                    for l in range(k + 1, 8):#第三步
	                        t1 = i + dirs[l][0]
	                        t2 = j + dirs[l][1]
	                        if 1 <= t1 <= n and 1 <= t2 <= m: #第三个真正可以遇到的，应该是 虚假的可以遇到总量- 不能遇到的
	                            ins -= 2  # 对于第三个valid有效的位置，总是有两个是不可以选的
	        #总的-遇到的=没有遇到的
	return res - ins // 2
	
	能遇到的是相互的 所以//2
	第三个棋子因为我们可能会遇到第一个或者第二个所以要用排除法，对于每个有效的第三个排除所有不可能的就是第三个有可能的点
29，286. Walls and Gates	class Solution:
	    def wallsAndGates(self, rooms: List[List[int]]) -> None:
	        """
	        Do not return anything, modify rooms in-place instead.
	        """
	        rows,cols = len(rooms),len(rooms[0])
	        visit = set()
思路：	        q = deque()
这题不是从空的开始bfs， 而是从 门bfs，然后搞空门条件就行 只要考虑遍历一遍就行
queue一层一层扫，所以一层结束这个dist 才会+1	        def addRoom(r,c):
	            if (
然后排除这个东西也记得，图的排除条件	                min(r, c) < 0
	                or r == rows
用or 排除 1高宽 2visit 3障碍	                or c == cols
	                or (r, c) in visit
	                or rooms[r][c] == -1
	            ):
	                return
	            visit.add((r,c))
	            q.append([r,c])
	        for r in range(rows):
	            for c in range(cols):
	                if rooms[r][c] ==0:
	                    q.append([r,c])
	                    visit.add((r,c))
	
	        dist  = 0
	        while q:
	            for i in range(len(q)):
	                r,c = q.popleft()
	                rooms[r][c] = dist
	                addRoom(r+1,c)
	                addRoom(r-1,c)
	                addRoom(r,c+1)
	                addRoom(r,c-1)
	            dist +=1
31.994. Rotting Oranges	
这题和上面一题不一样的是：我们要track 这个fresh的记录，因为 只有在没有fresh orange的时候我们才需要返回	class Solution:
这边不需要visit； 一旦一个橘子变成腐烂状态，它将永远保持腐烂，所以不需要额外的 visited 集合	    def orangesRotting(self, grid: List[List[int]]) -> int:
为什么要fresh > 0？：fresh > 0 这个条件确保了循环只有在还有新鲜的橘子存在且队列中仍然有腐烂的橘子时才会继续运行。这是因为如果没有新鲜的橘子了，即 fresh == 0，那么所有的橘子都已经被腐烂了，而且无论队列中是否还有橘子，循环都没有必要再继续执行。所以 fresh > 0 这个条件保证了算法在必要时终止，而不是无谓地继续运行下去。	        q = collections.deque()
	        fresh =0
	        time = 0
记住	        for r in range(len(grid)):
q.append([r,c])或者q.append((r,c))你完全可以使用小括号 () 来创建元组，而不使用方括号 [],这样做并不会影响代码的功能，只是改变了数据的表示方式。	            for c in range(len(grid[0])):
	                if grid[r][c] == 1:
	                    fresh +=1
	                if grid[r][c] == 2:
	                    q.append([r,c]) # 如果没办法直接bfs，就加入queue
	        directions = [(0,1),(0,-1),(1,0),(-1,0)]
	        while fresh >0 and q:
	            length = len(q)
	            for i in range(length):
	                r,c = q.popleft()
	                for dr,dc in directions:
	                    row,col = r + dr, c +dc
	                    if(
	                        row in range(len(grid))
	                        and col in range(len(grid[0]))
	                        and grid[row][col] ==1
	                    ):
	                        grid[row][col] =2#改变状态
	                        q.append([row,col])
	                        fresh -= 1
	            time +=1
	        return time if fresh ==0 else -1
32.1905. Count Sub Island	
	class Solution:
	    #，同时dfs的方法
	    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
	        ROWS, COLS = len(grid1), len(grid1[0])
	        visit = set()
	        def dfs(r, c):
	            if (
	                r < 0
	                or c < 0
	                or r == ROWS
	                or c == COLS
	                or grid2[r][c] == 0
	                or (r, c) in visit
	            ):
	                #  这里返回True有一个特定的逻辑意义：在递归的过程中，这表示当前分支的探索是合法的或者说不影响当前岛屿为子岛屿的判定
	                return True
	            visit.add((r, c))
	            res = True
	            if grid1[r][c] == 0:
	                res = False
	#  如果上下左右，有一个是False，也就是说有一个res 是false，上下左右 and res可以检测
	            res = dfs(r - 1, c) and res
	            res = dfs(r + 1, c) and res
	            res = dfs(r, c - 1) and res
	            res = dfs(r, c + 1) and res
	            return res
	        count = 0
	        for r in range(ROWS):
	            for c in range(COLS):
	                # 如果2 有陆地， 而且没有visit，可以recursively call dfs
	                if grid2[r][c] and (r, c) not in visit and dfs(r, c):
	                    count += 1
	        return count
33.Pacific Atlantic Water Flow	
	class Solution:
	    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
	        ROWS, COLS = len(heights), len(heights[0])
	        pac, atl = set(), set()
	        def dfs(r, c, visit, prevHeight):
	            if (
	                (r, c) in visit
	                or r < 0
	                or c < 0
	                or r == ROWS
	                or c == COLS
	                #prevHeight来控制水流的方向
	                or heights[r][c] < prevHeight
	            ):
	                return
	            visit.add((r, c))
	            dfs(r + 1, c, visit, heights[r][c])
	            dfs(r - 1, c, visit, heights[r][c])
	            dfs(r, c + 1, visit, heights[r][c])
	            dfs(r, c - 1, visit, heights[r][c])
	        for c in range(COLS):
	            # pac的visit row
	            dfs(0, c, pac, heights[0][c])
	            #altantic visit
	            dfs(ROWS - 1, c, atl, heights[ROWS - 1][c])
	        for r in range(ROWS):
	            #col
	            dfs(r, 0, pac, heights[r][0])
	            dfs(r, COLS - 1, atl, heights[r][COLS - 1])
	        res = []
	        for r in range(ROWS):
	            for c in range(COLS):
	                if (r, c) in pac and (r, c) in atl:
	                    res.append([r, c])
	        return res
34   Surrounded Regions	
	
这题重点是理解题目的意思是：任何 不在边界或者边界相邻的 O都会变成 X，所以	
	class Solution:
对于每一个边界上的 O，我们以它为起点，	    def solve(self, board: List[List[str]]) -> None:
	        if not board:
标记所有与它直接或间接相连的字母 O；	            return
最后我们遍历这个矩阵，对于每一个字母：	        
如果该字母没有被标记过，则该字母为被字母 X 包围的字母 O，我们将其修改为字母 X。	        n, m = len(board), len(board[0])
如果该字母被标记过，则该字母为没有被字母 X 包围的字母 O，我们将其还原为字母 O；	        def dfs(x, y):
	            if not 0 <= x < n or not 0 <= y < m or board[x][y] != 'O':
	                return
	            
	            board[x][y] = "A"
	            dfs(x + 1, y)
	            dfs(x - 1, y)
	            dfs(x, y + 1)
	            dfs(x, y - 1)
	        
	        for i in range(n):
	            dfs(i, 0)
	            dfs(i, m - 1)
	        
	        for i in range(m - 1):
	            dfs(0, i)
	            dfs(n - 1, i)
	        
	        for i in range(n):
	            for j in range(m):
	                if board[i][j] == "A":
	                    board[i][j] = "O"
	                elif board[i][j] == "O":
	                    board[i][j] = "X"
	
	
	
	
	
	
	
	
	
	
	class Solution:
	    def solve(self, board: List[List[str]]) -> None:
	        ROWS, COLS = len(board), len(board[0])
	        def capture(r, c):
	            if r < 0 or c < 0 or r == ROWS or c == COLS or board[r][c] != "O":
	                return
	            board[r][c] = "T"
	            capture(r + 1, c)
	            capture(r - 1, c)
	            capture(r, c + 1)
	            capture(r, c - 1)
	        # 1. (DFS) Capture unsurrounded regions (O -> T) 
	        for r in range(ROWS):
	            for c in range(COLS):# 两个选一个 
	                if board[r][c] == "O" and (r in [0, ROWS - 1] or c in [0, COLS - 1]):
	                    capture(r, c)
	        # 2. Capture surrounded regions (O -> X)
	        for r in range(ROWS):
	            for c in range(COLS):
	                if board[r][c] == "O":
	                    board[r][c] = "X"
	        # 3. Uncapture unsurrounded regions (T -> O)
	        for r in range(ROWS):
	            for c in range(COLS):
	                if board[r][c] == "T":
	                    board[r][c] = "O"
1466. Reorder Routes to Make All Paths Lead to the City Zero	class Solution:
	    def minReorder(self, n: int, connections: List[List[int]]) -> int:
	        def dfs(a: int, fa: int) -> int:
	            return sum(c + dfs(b, a) for b, c in g[a] if b != fa)
	        g = [[] for _ in range(n)]
	        for a, b in connections:
	            g[a].append((b, 1))
	            g[b].append((a, 0))
	        return dfs(0, -1)
909. Snakes and Ladders	
已解答	class Solution:
中等	    def snakesAndLadders(self, board: List[List[int]]) -> int:
相关标签	        length = len(board)
相关企业	        board.reverse()
You are given an n x n integer matrix board where the cells are labeled from 1 to n2 in a Boustrophedon style starting from the bottom left of the board (i.e. board[n - 1][0]) and alternating direction each row.	        def intoSquare(square):
You start on square 1 of the board. In each move, starting from square curr, do the following:	            r = (square-1) // length
        • Choose a destination square next with a label in the range [curr + 1, min(curr + 6, n2)].	            c = (square -1) % length
                ○ This choice simulates the result of a standard 6-sided die roll: i.e., there are always at most 6 destinations, regardless of the size of the board.	            if r%2:
        • If next has a snake or ladder, you must move to the destination of that snake or ladder. Otherwise, you move to next.	                c = length-1 -c
	            return [r,c]
	        q = deque()
	        q.append([1,0])# square,moves
	        visit = set()
	        while q:
	            square, moves = q.popleft()
	            for i in range(1,7):
	                nextSquare = square +i
	                r,c = intoSquare(nextSquare)
	                if board[r][c] != -1:
	                    nextSquare = board[r][c]
	                if nextSquare == length * length:
	                    return moves + 1
	                if nextSquare not in visit:
	                    visit.add(nextSquare)
	                    q.append([nextSquare,moves+1])
	        return -1
check if it the  prerequisite	
	class Solution:
	    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
	        adj = defaultdict(list)
	        for prereq, crs in prerequisites:
	            adj[crs].append(prereq)
	        
	        def dfs(crs):
	            if crs not in prereqMap:
	                prereqMap[crs] = set()
	                for pre in adj[crs]:
	                    prereqMap[crs] |= dfs(pre)
	            prereqMap[crs].add(crs)
	            return prereqMap[crs]
	        prereqMap = {} # map course -> set indirect prereqs
	        for crs in range(numCourses):
	            dfs(crs)
	        res = []
	        for u, v in queries:
	            res.append(u in prereqMap[v])
	        return res
1958. Check if Move is Legal	
	#首先八个方向，直到这个长高
	#把这个变成颜色
	#定义一个legal函数是否长度大于等于3 并且颜色相等，8个方向
	# 只要有个一个正确这个就是正确的
	class Solution:
	    def checkMove(self, board: List[List[str]], rMove: int, cMove: int, color: str) -> bool:
	        ROWS, COLS = len(board), len(board[0])
	        direction = [[1, 0], [-1, 0], [0, 1], [0, -1],
	                     [1, 1], [-1, -1], [1, -1], [-1, 1]]
	        board[rMove][cMove] = color
	        
	        def legal(row, col, color, direc):
	            dr, dc = direc
	            row, col = row + dr, col + dc
	            length = 1
	            
	            while(0 <= row < ROWS and 0 <= col < COLS):
	                length += 1
	                if board[row][col] == '.': return False
	                if board[row][col] == color:
	                    return length >= 3
	                row, col = row + dr, col + dc
	            return False
	        
	        for d in direction:
	            if legal(rMove, cMove, color, d): return True
	        return False
Shortest Bridge	
	def shortestBridge(grid: List[List[int]]) -> int:
	    N = len(grid)  # 网格的大小
	    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
	    # 检查坐标是否无效（越界）
	    def invalid(r, c):
	        return r < 0 or c < 0 or r >= N or c >= N
	    visit = set()
	    # 深度优先搜索（DFS）函数，用于找到并标记第一个岛屿
	    def dfs(r, c):
	        if invalid(r, c) or not grid[r][c] or (r, c) in visit:
	            return
	        visit.add((r, c))  # 将坐标加入访问
	        for dr, dc in directions:
	            dfs(r + dr, c + dc)  
	    # 广度优先搜索（BFS）函数，用于寻找最短桥
	    def bfs():
	        res, q = 0, deque(visit)  # 初始化步数和队列，队列中包含第一个岛屿的所有坐标
	        while q:
	            for _ in range(len(q)):
	                r, c = q.popleft()
	                for dr, dc in directions:
	                    curR, curC = r + dr, c + dc
	                    if invalid(curR, curC) or (curR, curC) in visit:
	                        continue
	                    if grid[curR][curC]:  # 如果找到第二个岛屿，返回步数
	                        return res
	                    q.append((curR, curC))  # 将当前坐标加入队列
	                    visit.add((curR, curC))  # 标记当前坐标已访问
	            res += 1  # 每层遍历结束，步数加一
	    # 找到第一个岛屿并用 DFS 标记
	    for r in range(N):
	        for c in range(N):
	            if grid[r][c]:
	                dfs(r, c)  # 标记第一个岛屿
	                return bfs()  # 扩展岛屿，寻找最短桥
Shortest Path in Binary Matrix	
	def shortestPathBinaryMatrix(grid):
	    N = len(grid)
	    q  = deque([(0,0,1)])
	    visit = set((0,0))
	    dir = [[0,1],[1,0],[-1,0],[0,-1]
	    [1,1],[1,-1],[-1,1],[-1,1]]
	    while q:
	        r,c ,length = q.popleft()
	        # r,c 不在表格内或者visited 有值
	        if (min(r,c)< 0 or max(r,c) >=N or
	            grid[r][c]):
	            continue
	        #结束条件
	        if  r == N-1 and c == N-1:
	            return length
	        for dr,dc in direct:
	            if (r+dr,c+dc) not in visit:
	                q.append((r+dr,c+dc,length+1))
	                visit.add((r+dr,c+dc))
	    return -1
684. Redundant Connection	
	class Solution:
	    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
	        par = [i for i in range(len(edges) + 1)]
	        rank = [1] * (len(edges) + 1)
	        def find(n):
	            p = par[n]
	            while p != par[p]:
	                par[p] = par[par[p]]
	                p = par[p]
	            return p
	        # return False if already unioned
	        def union(n1, n2):
	            p1, p2 = find(n1), find(n2)
	            if p1 == p2:
	                return False
	            if rank[p1] > rank[p2]:
	                par[p2] = p1
	                rank[p1] += rank[p2]
	            else:
	                par[p1] = p2
	                rank[p2] += rank[p1]
	            return True
	        for n1, n2 in edges:
	            if not union(n1, n2):
	                return [n1, n2]
721. Accounts Merge	
	class UnionFind:
	    def __init__(self, n):
	        self.par = [i for i in range(n)]
	        self.rank = [1] * n
	    
	    def find(self, x):
	        while x != self.par[x]:
	            self.par[x] = self.par[self.par[x]]
	            x = self.par[x]
	        return x
	    
	    def union(self, x1, x2):
	        p1, p2 = self.find(x1), self.find(x2)
	        if p1 == p2:
	            return False
	        if self.rank[p1] > self.rank[p2]:
	            self.par[p2] = p1
	            self.rank[p1] += self.rank[p2]
	        else:
	            self.par[p1] = p2
	            self.rank[p2] += self.rank[p1]
	        return True
	class Solution:
	    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
	        uf = UnionFind(len(accounts))
	        emailToAcc = {} # email -> index of acc
	        for i, a in enumerate(accounts):
	            for e in a[1:]:
	                if e in emailToAcc:
	                    uf.union(i, emailToAcc[e])
	                else:
	                    emailToAcc[e] = i
	        emailGroup = defaultdict(list) # index of acc -> list of emails
	        for e, i in emailToAcc.items():
	            leader = uf.find(i)
	            emailGroup[leader].append(e)
	        res = []
	        for i, emails in emailGroup.items():
	            name = accounts[i][0]
	            res.append([name] + sorted(emailGroup[i])) # array concat
	        return res
	      
2359. Find Closest Node to Given Two Nodes	class Solution:
	    def closestMeetingNode(self, edges: List[int], node1: int, node2: int) -> int:
	        n, min_dis, ans = len(edges), len(edges), -1
	        def calc_dis(x: int) -> List[int]:
manhattan distance 就是vertical 或者horizontal	            dis = [n] * n
	            d = 0
	            while x >= 0 and dis[x] == n:
	                dis[x] = d
	                d += 1
	                x = edges[x]
	            return dis
	        for i, d in enumerate(map(max, zip(calc_dis(node1), calc_dis(node2)))):
	            if d < min_dis:
	                min_dis, ans = d, i
	        return ans
	
	
	
	def closetMeetingNode(edges,node):
	    adj = collections.defaultdict(list)
	    for i,dist in edges:
	        adj[i].append(dist)
	    def bfs(src,distMap):
	        q = deque()
	        q.append([src,0])
	        distMap[src] = 0
	        while q:
	            node,dist = q.popleft()
	            for nei in adj[node]:
	                if nei not in distMap:
	                    q.append([nei,dist+1])
	                    distMap[nei] = dist+1
	    node1Dist = {}# map node--->distance1
	    node2Dist = {}# map the node--->distance2
	    bfs(node1,node1dist)
	    bfs(node2,node2dist)
	    res = -1
	    resDist = float('inf')
	    for i in range(len(edges)):
	        if i in node1Dist and i in node2Dist:
	            dist = max(node1Dist[i],node2Dist[i])
	            if dist < resDist:
	                res = i
	                resDist = dist
	    return res
	
1162. As Far from Land as Possible	
	#题目自己定义的陆地的初始值是1
	class Solution:
	    def maxdistance(self,grid):
	        N= len(grid)
	        q = deque()
	        for r in range(grid):
	            for c in range(grid[0]):
	                if grid[r][c]:
	                    q.append([r,c])
	        res = -1
	        direct = [[0,1],[1,0],[-1,0],[0,-1]]
	        while q:
	            r,c = q.popleft()
	            res = grid[r][c]#res赋1，最后是要剪掉的
	            for dr,dc in direct:
	                newR,newC = r+dr,c + dc
	                if (min(newR,newC)>=0 and
	                    max(ner,newC) <= N and
	                    grid[newR][newC] ==0):
	                    q.append([newR,newC])
	                    grid[newR][newC] = grid[r][c]+1
	        return res-1 if res >1 else -1
45    1129. Shortest Path with Alternating Colors	
	from collections import deque, defaultdict
	class Solution:
	    def shortestAlternatingPaths(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
	        # 构建红色和蓝色边的邻接表
	        red = defaultdict(list)
	        blue = defaultdict(list)
	        for src, dist in redEdges:
	            red[src].append(dist)  # 将红色边加入到红色邻接表中
	        for src, dist in blueEdges:
	            blue[src].append(dist)  # 将蓝色边加入到蓝色邻接表中
	        
	        # 初始化答案数组，所有节点的初始距离设置为-1（表示未访问过）
	        ans = [-1 for i in range(n)]
	        
	        # 初始化队列，存储当前节点、路径长度、边的颜色（起始节点0，长度0，颜色None）
	        q = deque()
	        q.append((0, 0, None))
	        
	        # 访问集，用于记录已经访问过的节点和边的颜色
	        visit = set()
	        visit.add((0, None))
	        
	        # 广度优先搜索（BFS）
	        while q:
	            node, length, edgecolor = q.popleft()  # 从队列中取出当前节点、路径长度和边的颜色
	            
	            # 如果当前节点还未被访问过，更新其最短路径长度
	            if ans[node] == -1:
	                ans[node] = length
	            
	            # 如果当前边的颜色不是红色，我们可以通过红色边前进
	            if edgecolor != "RED":
	                for nei in red[node]:  # 遍历红色边的邻居节点
	                    if (nei, "RED") not in visit:  # 如果邻居节点通过红色边访问且未访问过
	                        visit.add((nei, "RED"))  # 将该节点和边的颜色加入访问集
	                        q.append((nei, length + 1, "RED"))  # 将该节点加入队列，路径长度加1，颜色为红色
	            
	            # 如果当前边的颜色不是蓝色，我们可以通过蓝色边前进
	            if edgecolor != "BLUE":
	                for nei in blue[node]:  # 遍历蓝色边的邻居节点
	                    if (nei, "BLUE") not in visit:  # 如果邻居节点通过蓝色边访问且未访问过
	                        visit.add((nei, "BLUE"))  # 将该节点和边的颜色加入访问集
	                        q.append((nei, length + 1, "BLUE"))  # 将该节点加入队列，路径长度加1，颜色为蓝色
	        
	        # 返回最终的答案数组
	        return ans
	
	
	
	class Solution:
	    def shortestAlternatingPaths(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
	        # 构建图，g[x]存储从节点x出发的边，边的格式为 (y, color)
	        g = [[] for _ in range(n)]
	        for x, y in redEdges:
	            g[x].append((y, 0))  # 0 表示红色边
	        for x, y in blueEdges:
	            g[x].append((y, 1))  # 1 表示蓝色边
	        # 初始化距离数组，-1表示未访问过
	        dis = [-1] * n
	        # 访问集，记录访问过的节点和边颜色 (节点, 边颜色)
	        vis = {(0, 0), (0, 1)}
	        # 队列，初始状态加入节点0，通过红色边和蓝色边分别访问
	        q = [(0, 0), (0, 1)]
	        # 当前层级的深度
	        level = 0
	        
	        while q:
	            # 将当前队列中的元素取出，清空队列准备加入下一层的元素
	            tmp = q
	            q = []
	            for x, color in tmp:
	                # 如果当前节点 x 的距离还未被更新，则更新为当前层级的深度
	                if dis[x] == -1:
	                    dis[x] = level
	                # 遍历从当前节点 x 出发的所有边
	                for p in g[x]:
	                    # p[1] 是边的颜色，如果和当前边颜色不同且未访问过，则可以继续前进
	                    if p[1] != color and p not in vis:
	                        vis.add(p)
	                        q.append(p)
	            # 当前层级遍历完后，层级加一
	            level += 1
	        
	        return dis
	
	
	
	
	
	
	
	


	
	
二分法的精美	
	
	
图的规律总结！必须	常见的DFS用来解决什么问题？(1) 图中（有向无向皆可）的符合某种特征（比如最长）的路径以及长度（2）排列组合（3） 遍历一个图（或者树）（4）找出图或者树中符合题目要求的全部方案
	
	
	常见的BFS用来解决什么问题？(1) 简单图（有向无向皆可）的最短路径长度，注意是长度而不是具体的路径（2）拓扑排序 （3） 遍历一个图（或者树）
	
	
	
	
	
	
1.字典hash	import sys
	def func():
	  #一行一行的读取，
	    input = sys.stdin.read().strip().split('\n')
	    #第一个
	    n = int(input[0])
	    #提取line（for），然后每一个line 分隔提取每一个元素，并且map int到每一个元素转换，并且变成列表
	    lists = [list(map(int, line.split())) for line in input[1:n*2+1]]
	    #初始化字典
	    combine = {}
	    #根据第一个元素的n，知道一半的长度
	    for i in range(n):
	      #id
	        id_list = lists[2*i]
	      #对应的value
	        filter_list = lists[2*i + 1]
	        #对于每一次的i：v， 通过zip将两个list合并起来
	        total = {id_u: filter_u for id_u, filter_u in zip(id_list, filter_list)}
	        #合并字典的方式，要记住哦
	        combine.update(total)
	    # 字典根据value的值来排大小， sorted后面是一个整体来看
	    sorted_keys = [k for k, v in sorted(combine.items(), key=lambda x: x[1])]
	    # 转换为字符串并输出
	    result = ' '.join(map(str, sorted_keys))
	    print(result)
	if __name__ == "__main__":
	    func()
	
	
	#good 简单的方法
	import sys
	def func():
	    n = input()
	    n = int(n)
	    combine = {}
	    for i in range(n):
	        list1=list(map(int, input().split()))
	        list2 = list(map(int, input().split()))
	        total = {i:value for i,value in zip(list1,list2)}
	        combine.update(total)
	    sorted_keys = [k for k,v in sorted(combine.items(),key = lambda x:x[1])]
	    result = ' '.join(map(str, sorted_keys))
	    print(result)
	    
	if __name__ == "__main__":
	    func()
	
Contains Duplicate 	
	class Solution:
	    def containsDuplicate(self, nums: List[int]) -> bool:
	        hashset = set()
	        for n in nums:
	            if n in hashset:
	                return True
	            hashset.add(n)
	        return False
242. Valid Anagram	count_s[s[i]] 使用的是字符 s[i] 作为键，这是因为我们需要记录每个字符出现的次数。
	
	class Solution:
	    def isAnagram(self,s,t):
	        if len(s) != len(t):
	            return False
	        count_s, count_t = {}, {}
	        for i in range(len(s)):
	            count_s[s[i]] = count_s.get(s[i],0)+1
	            count_t[t[i]] = count_t.get(t[i],0)+1
	        return count_s == count_t
	        
	
	
二分法解决什么问题	
	
1.	单调性问题：当问题的解空间具有单调性（即解空间是单调递增或单调递减）时，可以使用二分查找。例如，当问题的解随输入变量的增加或减少呈现单调变化时。


2.	范围确定的优化问题：当需要在一个确定范围内找到最优解，并且可以根据某些条件判断某个解是否满足要求时，可以使用二分查找。例如，最大化或最小化某个值，并在每次迭代中根据条件调整查找范围。


3.	决策性问题：当问题可以通过是/否决策来缩小解空间时，可以使用二分查找。例如，给定一个预算，判断某个配置是否在预算范围内，如果是则尝试更高配置，否则尝试更低配置


•	二分查找法适用于解决具有单调性、范围确定以及需要决策性的优化问题。
        •	在这些问题中，通过每次迭代缩小解空间，可以高效地找到最优解。





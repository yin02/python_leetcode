### 放进 python tutor 走一下  https://pythontutor.com/render.html#mode=display


dfs(row+1) 比如dfs(2) 然后dfs（3）不行 再最后一个情况下就会跳过 backtrack（3）这个（因为执行完了）
所以就是remove 这个node，并且 回到了dfs（2）的状态


# 注意step 81 -85的变化，backtrack里面的值所以就是回到上一行，每一列都不行的情况下，等于这个当前的backtrack（r+1）就执行完了，然后执行下一行，移除 backtrack（r）的node位置，然后继续backtrack的 col的下一列选择，（移除当前node）【row状态】移动下一列，因为之前的for 循环


```py

class Solution:
    def solveNQueens(self, n: int):
        res = []  # To store all the solutions
        board = [["." for _ in range(n)] for _ in range(n)]  # Initialize an empty board
        col = set()  # To track columns where queens are placed
        posDiag = set()  # To track positive diagonals (r + c)
        negDiag = set()  # To track negative diagonals (r - c)
        
        def backtrack(r):
            # If all queens are placed, add the current board configuration to the result
            if r == n:
                copy = ["".join(row) for row in board]
                res.append(copy)
                return
            
            # Try placing a queen in each column of the current row
            for c in range(n):
                if c in col or (r + c) in posDiag or (r - c) in negDiag:
                    continue  # Skip if the position is under attack
                
                # Place the queen
                col.add(c)
                posDiag.add(r + c)
                negDiag.add(r - c)
                board[r][c] = "Q"
                
                # Recur to place queens in the next row
                backtrack(r + 1)
                
                # Backtrack: Remove the queen and unmark the positions
                col.remove(c)
                posDiag.remove(r + c)
                negDiag.remove(r - c)
                board[r][c] = "."
        
        # Start the backtracking process from the first row
        backtrack(0)
        
        return res

# Test the solution
sol = Solution()
print(sol.solveNQueens(4))
```
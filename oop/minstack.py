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
        

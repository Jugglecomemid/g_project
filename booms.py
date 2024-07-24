# 思路 使用图结构和BFS
# 炸弹a爆炸能引爆炸弹b，意味着炸弹a到炸弹b具有一条有向边。
# 可以先建立炸弹之间的图结构，再用广度优先搜索找出最大爆炸数量
import math
from collections import defaultdict, deque

def explode(b1, b2):
    '''
    验证 b1 到 b2 能否引爆
    '''
    x1, y1, r1 = b1
    x2, y2, r2 = b2
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance <= r1

def build_boom_graph(booms_lst):
    '''
    遍历炸弹 建立能炸的范围图
    '''
    graph = defaultdict(list)
    n = len(booms_lst)
    for i in range(n):
        for j in range(n):
            if i != j and explode(booms_lst[i], booms_lst[j]):
                graph[i].append(j)
    return graph

def bfs(start, graph):
    '''
    广度优先搜索
    '''
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    return len(visited)

def main():
    booms = [[1,2,3],[2,3,1],[3,4,2],[4,5,3],[5,6,4]]
    booms_graph = build_boom_graph(booms)
    max_exploded = 0
    for i in range(len(booms)):
        max_exploded = max(max_exploded, bfs(i, booms_graph))
    return max_exploded


if __name__ == '__main__':
    print(main())




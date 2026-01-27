"""
1971. Find if Path Exists in Graph
"""
from collections import deque
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        if source == destination:
            return True

        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        seen = set()
        seen.add(source)

        d = deque()
        d.append(source)

        while d:
            node = d.popleft()

            if node == destination:
                return True

            for nei in graph[node]:
                if nei not in seen:
                    seen.add(nei)
                    d.append(nei)

        return False
        
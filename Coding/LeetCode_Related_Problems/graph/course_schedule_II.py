"""
210. Course Schedule II
"""
# We are detecting if there is cycle or not in the graph
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = defaultdict(list)

        for u, v in prerequisites:
            graph[u].append(v)
        
        not_visited = 0
        visiting = 1
        visited = 2
        
        states = [not_visited] * numCourses
        finish = []

        def dfs(node):
            state = states[node]

            if state == visited:
                return True
            elif state == visiting:
                return False
            states[node] = visiting

            for nei in graph[node]:
                if not dfs(nei):
                    return False
            
            states[node] = visited
            finish.append(node)
            return True  

        for i in range(numCourses):
            if not dfs(i):
                return []
        
        return finish

                
# Time complexity: O(n + E)       
# Space: O(V + E)       




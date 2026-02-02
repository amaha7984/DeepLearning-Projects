"""
207. Course Schedule
"""
# We are detecting if there is cycle or not in the graph
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:

        graph = defaultdict(list)
        
        for i, j in prerequisites:
            graph[i].append(j)
        
        unvisited = 0
        visiting = 1
        visited = 2
        
        states = [unvisited] * numCourses
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
            return True
        
        for i in range(numCourses):
            if not dfs(i):
                return False
        return True

#----------------------------------------Iterative DFS (stack)-----------------------------------------#
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:

        graph = defaultdict(list)
        courses = prerequisites

        for u, v in courses:
            graph[u].append(v)
        
        not_visited = 0
        visiting = 1
        visited = 2

        states = [0] * numCourses

        stack = []

        for node in range(numCourses):
            if states[node] != not_visited: #skipping nodes that were already processed
                continue

            stack.append((node, 0))

            while stack:
                node, done = stack.pop()
                if done == 1:
                    states[node] = visited
                    continue

                state = states[node]
                if state == visited:
                    continue
                elif state == visiting:
                    return False
                states[node] = visiting
                stack.append((node, 1))
    
                for nei in graph[node]:
                    state = states[nei]
                    if state == visiting:
                        return False
                    if state == not_visited:
                        stack.append((nei, 0))

        return True
                
# Time complexity: O(n * (E + V))              




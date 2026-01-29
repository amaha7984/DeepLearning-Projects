"""
721. Accounts Merge
"""

from typing import Optional
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:

        # 
        graph = defaultdict(set)
        email_to_name = {}

        for account in accounts:
            name = account[0]

            for email in account[1:]:
                graph[email].add(account[1])
                graph[account[1]].add(email)

                email_to_name[email] = name
        
        res = [] #final listoflist

        visited = set()

        for email in graph:
            if email not in visited:
                stack = [email]
                visited.add(email)
                local_res = [] #store all the visited neighbor emails for given email

                while stack:
                    node = stack.pop()

                    local_res.append(node)

                    for edge in graph[node]:
                        if edge not in visited:
                            stack.append(edge)
                            visited.add(edge)
                
                res.append([email_to_name[email]] + sorted(local_res))
        return res
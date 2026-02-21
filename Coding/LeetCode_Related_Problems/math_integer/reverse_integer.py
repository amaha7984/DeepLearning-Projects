# 7. Reverse Integer

class Solution:
    def reverse(self, x: int) -> int:
        max_val = 2**31 - 1

        if x < 0:
            sign = -1
        else:
            sign = 1
        
        x = abs(x)
        new_x = 0
        while x:
            val = x % 10  # In math, we can access last element of integer, for 123, we can get 3
            x = x // 10 # we remove the last element from the integer

            if new_x > (max_val - val) // 10:
                return 0
            new_x = new_x * 10 + val
        return sign * new_x


#explanation on why we have such if statement in line 18
"""
According to constraint, we shouldn't have new_x greater than max_val
Our next operation is always new_x = new_x * 10 + val, and we should make sure than it should go beyond max_val
But we should check before that, not after the operation because we do not want to store more than 32-bit
So, the condition should be (also we will be doing mathematical operation):
1. new_x <= max_val 
2. new_x * 10 + val <= max_val (Note: new_x = new_x * 10 + val)
3. new_x * 10 <= max_val - val
4. new_x <= (max_val - val) // 10
# This means if new_x > (max_val - val) // 10 is True, then the next operation will lead to new_val > max_val. \
# So, we are looking into future operation and avoiding any value greater than max_val (32-bit) will not processed at all
"""
        

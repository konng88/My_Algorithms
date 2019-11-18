def maxArea(height):
    V = min(height[0],height[-1]) * (len(height) - 1)
    left,right = 0,len(height)-1
    def compute_max(left,right,height,V):
        while left != right:
            if height[left] == height[right]:
                return max(compute_max(left,right-1,height,V),compute_max(left+1,right,height,V))

            elif height[left] > height[right]:
                right -= 1
            else:
                left += 1
            v = min(height[left],height[right]) * (right - left)
            if v > V:
                V = v
        return V
    return compute_max(left,right,height,V)
print(maxArea([2,3,10,5,7,8,9]))

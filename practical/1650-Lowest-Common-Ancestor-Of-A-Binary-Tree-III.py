# https://github.com/doocs/leetcode/blob/8380f3faa3965229fed73d62de85994aa690d41d/solution/1600-1699/1650.Lowest%20Common%20Ancestor%20of%20a%20Binary%20Tree%20III/README_EN.md



# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None



def lowestCommonAncestor(self, p: Node, q: Node) -> Node:
    # https://www.youtube.com/watch?v=vZxxksAP8yk
    p_copy = p
    q_copy = q

    while p_copy != q_ccopy:
        q_copy = q_copy.parent if q_copy else p 
        p_copy = p_copy.parent if p_copy else q

        return p_copy


 res = lowestCommonAncestor(p: Node, q: Node)
print(res.val)]

# T: O(n)
# S: O(1)

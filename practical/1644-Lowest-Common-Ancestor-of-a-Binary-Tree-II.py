# https://algo.monster/liteproblems/1644

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        self.ancestor = None

        def dfs(current_node):
            if current_node is None:
                return False
            
            left = dfs(current_node.left)
            right = dfs(current_node.right)
            mid = current_node == p or current_node == q

            if mid + left + right >= 2:
                self.ancestor = current_node

            return mid or left or right

        dfs(root)
        return self.ancestor

def build_tree(tree_list, index=0):
    """
    Build a binary tree from a list representation (like LeetCode's format).
    None values in the list represent null nodes.
    """
    if index >= len(tree_list) or tree_list[index] is None:
        return None
    
    node = TreeNode(tree_list[index])
    node.left = build_tree(tree_list, 2 * index + 1)
    node.right = build_tree(tree_list, 2 * index + 2)
    return node

def find_node(root, val):
    """
    Find a node with the given value in the tree (for test case setup).
    Returns None if not found.
    """
    if root is None:
        return None
    if root.val == val:
        return root
    return find_node(root.left, val) or find_node(root.right, val)

def run_test_case(tree_values, p_val, q_val, expected_val):
    """
    Run a single test case and print the result.
    """
    print(f"\nRunning test case: tree = {tree_values}, p = {p_val}, q = {q_val}")
    
    # Build the tree
    root = build_tree(tree_values)
    if root is None:
        print("Empty tree provided")
        return
    
    # Find the nodes p and q
    p_node = find_node(root, p_val)
    q_node = find_node(root, q_val)
    
    if p_node is None or q_node is None:
        print(f"Error: Could not find nodes {p_val} and/or {q_val} in the tree")
        return
    
    # Find the LCA
    solution = Solution()
    lca = solution.lowestCommonAncestor(root, p_node, q_node)
    
    # Verify the result
    result = lca.val if lca else None
    print(f"Expected LCA: {expected_val}, Actual LCA: {result}")
    print("Test", "PASSED" if result == expected_val else "FAILED")

def main():
    # Define test cases
    test_cases = [
        # Format: (tree_values, p_val, q_val, expected_lca_val)
        ([3,5,1,6,2,0,8,None,None,7,4], 5, 1, 3),
        ([3,5,1,6,2,0,8,None,None,7,4], 5, 4, 5),
        ([1,2], 1, 2, 1),
        ([6,2,8,0,4,7,9,None,None,3,5], 2, 8, 6),
        ([6,2,8,0,4,7,9,None,None,3,5], 3, 5, 4),
        ([2,None,3,None,4,None,5,None,6], 3, 6, 3),
    ]
    
    # Run all test cases
    for i, (tree_values, p_val, q_val, expected_val) in enumerate(test_cases, 1):
        print(f"\n{'='*40}")
        print(f"Test Case {i}:")
        run_test_case(tree_values, p_val, q_val, expected_val)

if __name__ == "__main__":
    main()
# The provided solution can be used to find the lowest common ancestor (LCA) of two nodes in a binary tree.

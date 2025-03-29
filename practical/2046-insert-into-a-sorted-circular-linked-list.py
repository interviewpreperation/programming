# https://github.com/doocs/leetcode/blob/main/solution/0700-0799/0708.Insert%20into%20a%20Sorted%20Circular%20Linked%20List/README_EN.md


# https://algo.monster/liteproblems/708



class Node:
    def __init__(self, value=None, next_node=None):
        self.value = value
        self.next_node = next_node


class Solution:
    def insert(self, head: 'Optional[Node]', insert_value: int) -> 'Node':
        # Create a new node with the insert_value
        new_node = Node(insert_value)

        # If the linked list is empty, initialize it with the new node
        if head is None:
            new_node.next_node = new_node  # Point the new_node to itself
            return new_node
      
        # Initialize two pointers for iterating the linked list
        previous, current = head, head.next_node

        # Traverse the linked list
        while current != head:
            # Check if the insert_value should be inserted between previous and current
            # The first condition checks for normal ordered insertion
            # The second condition checks for insertion at the boundary of the largest and smallest values
            if (
                previous.value <= insert_value <= current.value or
                (previous.value > current.value and (insert_value >= previous.value or insert_value <= current.value))
            ):
                break  # Correct insertion spot is found

            # Move to the next pair of nodes
            previous, current = current, current.next_node

        # Insert new_node between previous and current
        previous.next_node = new_node
        new_node.next_node = current

        # Return the head of the modified linked list
        return head

###################################################################
# cracking faang: https://www.youtube.com/watch?v=XN9OsmP2YTk
# 1. head is null
# 2. insert into LL
# 3. edge insert
# 4. unival LL

class Solution:
    def insert(self, head: 'Optional[Node]', insert_value: int) -> 'Node':
        # case 1
        if not head:
            new_head = Node(insertVal)
            new_head.next = new_head
            return new_head

        cur = head

        while cur.next != head
            # case 2
            if cur.val <= insertVal <= cur.next.Val:
                new_node = Node(insertVal, cur.next)
                cur.next = new_node
                
                return head

            # case 3
            elif cur.val > cur.next.val:
                if insertVal >= cur.val or insertVal <= cur.next.val:
                    new_node = Node(insertVal, cur.next)
                    cur.next = new_node

                    return head
            cur = cur.next
        
        # case 4
        new_node = Node(insertVal, cur.next)
        cur.next = new_node

        return head

# T: O(n) ... n is size of LL
# S: O(1)

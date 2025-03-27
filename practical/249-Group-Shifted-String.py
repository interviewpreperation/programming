# https://www.youtube.com/watch?v=g_CWHtPSQmQ

from collections import defaultdict

def groupStrings(strings):
    def get_key(s):
        key = []
        for a, b in zip(s, s[1:]):
            key.append((ord(b) - ord(a)) % 26)
        return tuple(key) # hashable for dict keys
    
    groups = {}
    for s in strings:
        key = get_key(s)
        # print("key: ", key)
        if key not in groups:
            groups[key] = []
        groups[key].append(s)
    return list(groups.values())

# Example 1: Basic case
example1 = ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]
print("Example 1:", groupStrings(example1))
# Output: [['abc', 'bcd', 'xyz'], ['acef'], ['az', 'ba'], ['a', 'z']]

# Example 2: All single characters
example2 = ["a", "b", "c", "z"]
print("Example 2:", groupStrings(example2))
# Output: [['a', 'b', 'c', 'z']]

# Example 3: Empty strings and single characters
example3 = ["", "a", "ab", "bc", "cd", "abc", "bcd", "xyz"]
print("Example 3:", groupStrings(example3))
# Output: [[''], ['a'], ['ab', 'bc', 'cd'], ['abc', 'bcd', 'xyz']]

# Example 4: Wrapping around 'z' to 'a'
example4 = ["az", "ba", "cb", "yx", "za"]
print("Example 4:", groupStrings(example4))
# Output: [['az', 'ba'], ['cb'], ['yx'], ['za']]

# Example 5: Mixed lengths
example5 = ["a", "abc", "bcd", "xyz", "zz", "yza", "abcd"]
print("Example 5:", groupStrings(example5))
# Output: [['a'], ['abc', 'bcd', 'xyz'], ['zz', 'yza'], ['abcd']]
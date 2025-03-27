# https://www.youtube.com/watch?v=g_CWHtPSQmQ

from collections import defaultdict

class Solution:
    def groupStrings(strings):
        def get_key(s):
            key = []
            for a, b in zip(s, s[1:]):
                key.append((ord(b) - ord(a)) % 26)
            return tuple(key) # hashable for dict keys
        
        groups = {}
        for s in strings:
            key = get_key(s)
            if key not in groups:
                groups[key] = []
            groups[key].append(s)
        return list(groups.values())
    

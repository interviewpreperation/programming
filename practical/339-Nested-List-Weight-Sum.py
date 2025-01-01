import collections

# https://www.lintcode.com/problem/551

def depthSum(nesteList):
    depth = 1
    res = 0

    q = collections.deque(nesteList)
    while q:
        for _ in range(len(q)):
            cur = q.popleft()
            print("cur: ", cur, "isinstance: ", isinstance(cur, int), "res: ", res, "depth: ", depth)
            if isinstance(cur, int):
                res += cur * depth
            else:
                q.extend(cur)
                print("q: ", q, cur)
        depth += 1
    return res



print("res: ", depthSum([[1, 1], 2, [1, 1]]), "gt: 10")
print("res: ", depthSum([[1, 1, 1], 2, [1, 1]]), "gt: 12")


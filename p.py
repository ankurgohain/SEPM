from bisect import bisect_left
from collections import defaultdict
from typing import List


class Solution:
    def minMirrorPairDistance(self, nums: List[int]) -> int:
        # value -> sorted list of indices where value appears
        pos = defaultdict(list)
        for i, x in enumerate(nums):
            pos[x].append(i)

        best = 10**6

        for i, x in enumerate(nums):
            rev = int(str(x)[::-1])
            if rev not in pos:
                continue

            idxs = pos[rev]
            k = bisect_left(idxs, i)

            # candidate on the right
            if k < len(idxs) and idxs[k] != i:
                best = min(best, abs(idxs[k] - i))

            # candidate on the left
            if k > 0:
                left_idx = idxs[k - 1]
                if left_idx != i:
                    best = min(best, abs(left_idx - i))

            if best == 1:
                return 1

        return -1 if best == 10**6 else best
    
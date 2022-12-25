import os
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)


l1 = [9,1,1,2,3,4,5,1,1,2]
l2 = ["def",2,3,4,"abc", "abc", "deg", "xyz"]

s1 = set(l1)
s2 = set(l2)

print(f"Set 1 - {set(s1)}")
print(f"Set 2 - {set(s2)}")
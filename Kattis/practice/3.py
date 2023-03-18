from sys import stdin

num, divisor = [int(x) for x in stdin.readline().split()]
nums = [int(x) for x in stdin.readline().split()]
possibles = {}

for i in nums:
    possibles[int(i // divisor)] = possibles.get(int(i // divisor), 0) + 1

print(sum((x * (x - 1)) // 2 for x in possibles.values()))
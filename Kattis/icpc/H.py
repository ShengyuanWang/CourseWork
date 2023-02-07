def main():
    n, s, k = map(int, input().split())
    lst = []
    for _ in range(n):
        lst.append(int(input()))
    lst = sorted(lst)
    gaps = []

    for i in range(1, n):
        gaps.append(lst[i]-lst[i-1])
    gaps = [gaps[0]] + gaps + [gaps[-1]]
    tot = sum(gaps)

    s, k = s/2, k/2
    for i in range(1, n+1):
        if s > gaps[i-1] or s > gaps[i]:
            print("-1")
            return
        dis = max(s, min(gaps[i-1], gaps[i]-s, k))

        gaps[i-1] -= dis
        gaps[i] -= dis
    print(int(tot-sum(gaps)))



main()
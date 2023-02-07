def main():
    n, m = map(int, input().split())
    lst = []
    res = [0 for _ in range(n+1)]
    for _ in range(m):
        a = int(input())
        lst.append([a, a+1])
    pos = 1
    for i in range(n):
        i += 1
        for v in lst:
            s, t = v[0], v[1]
            if i == s:
                i = t
            elif i == t:
                i = s
        res[i] = pos
        pos += 1
    for n in res[1:]:
        print(n)
    return


if __name__ == '__main__':
    main()
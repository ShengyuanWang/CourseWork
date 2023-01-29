def main():
    n = int(input())
    for _ in range(n):
        lst = list(input().split(','))
        m = len(lst)
        res = 0
        for i in range(m):
            if lst[i] == "":
                continue
            res += int(lst[i])*(60 **(m-i-1))
        print(res)
    return


if __name__ == '__main__':
    main()
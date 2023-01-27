def main():
    n, q = map(int, input().split())
    lst = list(map(int, input().split()))
    lst = [0] + lst
    for _ in range(q):
        t, a, b = map(int, input().split())
        if t == 1:
            lst = changeLocation(lst, a, b)
        else:
            print(distance(lst, a, b))
    return


def changeLocation(lst, c, x):
    lst[c] = x
    return lst


def distance(lst, a, b):
    return abs(lst[a] - lst[b])


if __name__ == '__main__':
    main()
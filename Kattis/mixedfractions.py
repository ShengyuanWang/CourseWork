def main():
    while (True):
        a, b = map(int, input().split())
        if a == 0 and b == 0:
            return
        res1 = a // b
        res2 = a - b * res1
        print(res1, res2, "/", b)

if __name__ == '__main__':
    main()
def main():
    n = int(input())
    lst = []
    for _ in range(n):
        lst.append(int(input()))
    lst = sorted(lst)
    r = lst[-1]
    total = sum(lst)
    l = 0
    if total < 2*r:
        print("1")
        return
    while total >= 2*r:
        total -= lst[l]
        l += 1
    print(n-l+1)






if __name__ == "__main__":
    main()
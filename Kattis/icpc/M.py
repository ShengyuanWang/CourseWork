def main():
    n = int(input())
    rec = [0 for _ in range(n+1)]
    for i in range(1,n+1):
        rec[i] = int(input())
    invite = set()
    for i in range(1,n+1):
        tmp = []
        s, t = i, rec[i]
        tmp.append(s)
        while t not in tmp:
            tmp.append(t)
            s = t
            t = rec[s]
        invite.add(str(sorted(tmp)))
    fo









if __name__ == "__main__":
    main()
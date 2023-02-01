def main():
    n, k = map(int, input().split())
    tmp_0 = [0 for _ in range(n+1)]
    pos_0 = [0 for _ in range(n)]
    pos_1 = [0 for _ in range(n)]
    neg_0 = [0 for _ in range(n)]
    neg_1 = [0 for _ in range(n)]
    poo_0 = [0 for _ in range(n)]
    poo_1 = [0 for _ in range(n)]
    nee_0 = [0 for _ in range(n)]
    nee_1 = [0 for _ in range(n)]
    ans = [0 for _ in range(n)]

    for _ in range(k):
        c, l, r = input().split()
        l, r = int(l), int(r)
        l -= 1
        r -= 1

        if c == 'R':
            tmp_0[l] += 1
            tmp_0[r+1] -= 1
        elif c == 'D':
            tmp_0[l] -= 1
            tmp_0[r+1] += 1
        elif c == 'H':
            if l == r:
                tmp_0[l] += 1
                tmp_0[r+1] -= 1
            elif (r-l+1) % 2 == 0:
                pos_0[(r + l) // 2 + 1] += ((r - l + 1) // 2)
                if (r + l) // 2 + 2 < n:
                    poo_0[(r+l) // 2+2] += 1
                if r + 2 < n:
                    poo_0[r+2] -= 1
                pos_1[(r + l) // 2] += ((r - l + 1) // 2)
                if (r + l) // 2 - 1 >= 0:
                    poo_1[(r+l) // 2-1] += 1
                if l - 2 >= 0:
                    poo_1[l-2] -= 1
            else:
                pos_0[(r + l) // 2] += ((r - l) // 2 + 1)
                if (r + l) // 2 + 1 < n:
                    poo_0[(r+l) // 2+1] += 1
                if r + 2 < n:
                    poo_0[r+2] -= 1
                pos_1[(r + l) // 2 - 1] += ((r - l) // 2)
                if (r + l) // 2 - 2 >= 0:
                    poo_1[(r+l) // 2-2] += 1
                if l - 2 >= 0:
                    poo_1[l-2] -= 1
        elif c == 'V':
            if l == r:
                tmp_0[l] -= 1
                tmp_0[r+1] += 1
            elif (r-l+1) % 2 == 0:
                neg_0[(r+l) // 2+1] += ((r-l+1) // 2)
                if (r+l) // 2+2 < n:
                    nee_0[(r+l) // 2+2] += 1
                if r+2 < n:
                    nee_0[r+2] -= 1
                neg_1[(r+l) // 2] += ((r-l+1) // 2)
                if (r+l) // 2-1 >= 0:
                    nee_1[(r+l) // 2-1] += 1
                if l-2 >= 0:
                    nee_1[l-2] -= 1
            else:
                neg_0[(r + l) // 2] += ((r - l) // 2 + 1)
                if (r + l) // 2 + 1 < n:
                    nee_0[(r+l) // 2+1] += 1
                if r+2 < n:
                    nee_0[r+2] -= 1
                neg_1[(r+l) // 2-1] += ((r-l) // 2)
                if (r+l) // 2-2 >= 0:
                    nee_1[(r+l) // 2-2] += 1
                if l-2 >= 0:
                    nee_1[l-2] -= 1

    cur = 0
    for i in range(n):
        cur += tmp_0[i]
        ans[i] += cur
    cur_pos, cur_neg = 0, 0

    for i in range(n):
        if i:
            poo_0[i] += poo_0[i-1]
        if i:
            nee_0[i] += nee_0[i-1]
        cur_pos += pos_0[i] - poo_0[i]
        cur_neg += neg_0[i] - nee_0[i]
        ans[i] += cur_pos - cur_neg

    cur_pos, cur_neg = 0, 0

    for i in range(n-1, -1, -1):
        if i + 1 < n:
            poo_1[i] += poo_1[i+1]
        if i+1 < n:
            nee_1[i] += nee_1[i+1]
        cur_pos += pos_1[i] - poo_1[i]
        cur_neg += neg_1[i] - nee_1[i]
        ans[i] += (cur_pos - cur_neg)
    for i in range(n):
        print(ans[i])


if __name__ == '__main__':
    main()
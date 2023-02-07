def main():
    t, s = map(int, input().split())
    taxi = 2*t*t + 2*t + 1
    spi = getSpi(s)
    gcd = computeGCD(taxi, spi)
    taxi //= gcd
    spi //= gcd

    if taxi >= spi:
        print("1")
        return


    if spi == 1:
        print(taxi)
    else:
        print(str(taxi) + "/" + str(spi))
    return


def getSpi(n):
    cnt = 0
    for x in range(-n, n+1):
        for y in range(-n, n+1):
            x, y  = abs(x), abs(y)
            if x + y - min(x,y)/2 <= n:
                cnt += 1
    return cnt


def computeGCD(x, y):
    if x > y:
        small = y
    else:
        small = x
    for i in range(1, small + 1):
        if ((x % i == 0) and (y % i == 0)):
            gcd = i

    return gcd

if __name__ == '__main__':
    main()
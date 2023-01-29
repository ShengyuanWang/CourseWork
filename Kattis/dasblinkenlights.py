def main():
    p, q, s = map(int, input().split())
    if (lcm(p, q) <= s):
        return 'yes'
    else:
        return 'no'



def gcd(a, b):
    while b > 0:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a*b / gcd(a, b)

if __name__ == "__main__":
    res = main()
    print(res)
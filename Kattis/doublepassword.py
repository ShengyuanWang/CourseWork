def main():
    a = input()
    b = input()
    res = 1
    for i in range(4):
        if a[i] != b[i]:
            res *= 2
    return res
if __name__ == '__main__':
    res = main()
    print(res)
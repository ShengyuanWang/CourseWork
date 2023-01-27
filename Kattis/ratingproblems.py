def main():
    n, k = map(int, input().split())
    remain = n - k
    grade = 0
    for i in range(k):
        grade += int(input())
    max_grade = 3 * remain + grade
    min_grade = -3 * remain + grade
    print(min_grade/n, max_grade/n)


if __name__ == '__main__':
    main()

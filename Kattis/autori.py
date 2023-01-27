def main():
    inp = input()
    res  = ""
    for w in inp:
        if w.isupper():
            res += w
    print(res)
    return

if __name__ == '__main__':
    main()
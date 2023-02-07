def beast_bully(beast_info):
  #beast_info is a list with shape (n + 1, )
  n = len(beast_info)
  power = beast_info[1:n]
  sorted_power = sorted(power)
  print("sorted_power is", sorted_power)
  surv = 0
  for i in range(n-1):
    surv = surv + 1
    max = sorted_power[n-2-i]
    print("max is", max)
    s = sum(sorted_power[0:n-1-i])
    print("sum is", s)
    if 2*max > s:
      print(surv)
      return surv
    elif 2*max == s:
      print(surv)
      return surv
    else:
      sorted_power = sorted_power[1:n-1-i]
  print(surv)
  return surv

def main():
  beast_info = list(map(int, input().split()))
  beast_bully(beast_info)

  

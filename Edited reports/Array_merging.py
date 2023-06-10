
# Online Python - IDE, Editor, Compiler, Interpreter

def max_in_row(a,d):
    prev = a[0]
    count = 1
    d[prev] = count
    for i in range(1,len(a)):
        if a[i]==prev:
            count+=1
            if d[a[i]]<count:
                d[a[i]] = count
        else:
            count = 1
            if d[a[i]]<count:
                d[a[i]] = count
        prev = a[i]
       
def max_sum(a,b,da, db):
    max_in_row(a,da)
    max_in_row(b,db)
    mx = 0
    temp = 0
    for i in range(len(da)):
        temp = da[i]+db[i]
        if temp > mx:
            mx = temp
    return mx

t = int(input())
anses = []
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    da = [0]*(2*n+1)
    db = [0]*(2*n+1)
    ans = max_sum(a,b,da,db)
    anses.append(ans)
for ans in anses:
    print(ans)
    
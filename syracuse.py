def syracuse(n):  
    while n > 1:
        if n%2 == 0:
            n = n/2
        else:
            n = 3*n+1
        print(n)

print(syracuse(126567887654788876))

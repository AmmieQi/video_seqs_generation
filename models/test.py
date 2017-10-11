import time
import time


a = 'abcdcb'

n = len(a)
R = [0 for i in xrange(n)]
maxRi = -1
maxRight = -1
maxR = 0
maxI = 0

for i in xrange(n):
    tmpr = i
    #maxr = 0
    index = i+1
    flag = True
    if i > maxRight:
        while flag:
            invdex = 2*i - index
            if index < n and invdex >= 0 and a[index] == a[invdex]:
                R[i] += 1
                index += 1
            else:
                tmpr = index - 1
                maxRight = tmpr
                maxRi = i
                if R[i] > maxR:
                    maxI = i
                    maxR = R[i]
                flag = False
    else:
        invdex = 2*maxRi-index
        if R[invdex] <= maxRight-i:
            R[i] = R[invdex]
        else:
            index = maxRight + 1
            R[i] = R[invdex]
            while flag:
                invdex2 = 2*i - index
                if index < n and invdex2 >= 0 and a[index] == a[invdex]:
                    R[i] += 1
                    index += 1
                else:
                    tmpr = index - 1
                    maxRight = tmpr
                    maxRi = i
                    if R[i] > maxR:
                        maxI = i
                        maxR = R[i]
                    flag = False

print 'max r = ', maxR*2+1, 'max i = ', maxI
b = [a[i] for i in range(maxI)]

import math as m

def convolution (f, g):
    padding_size = (int)(len(g)/2)
    new_f = [0]*padding_size + f + [0]*padding_size
    
    result = []

    for i in range(len(f)):
        tmp = 0
        for j in range(len(g)):
            tmp += (new_f[i+j] *g[j])  
        # result.append(tmp)
        result.append(round(tmp))
    
    return result


f = [1, 2, 3, 4, 5, 6]
k = [1/3, 1/3, 1/3]
g = [1, 2, 3, 4, 5, 3]

result = convolution(f, k)
print(g)
print(result)
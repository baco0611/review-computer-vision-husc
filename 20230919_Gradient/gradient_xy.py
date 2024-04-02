import numpy as np

# Define a function
def calc_gradient(I, point=(1,1)):
    x, y = point

    Gx = I[x, y+1] - I[x, y-1]
    Gy = I[x-1, y] - I[x+1, y]

    magnitude = np.sqrt(Gy * Gy + Gx * Gx)
    orientation = np.arctan2(Gy, Gx)
    
    return orientation, magnitude

f = [[255,  255,    255], 
     [255,    100,    0], 
     [255,    0,      0]]
I = np.array(f)

print(I)
print(I[1][1])

theta, G = calc_gradient(I, (1,1))
degree_value = theta * (180 / np.pi)

print('[Theta_r] ', theta)
print('[Theta_d] ', degree_value)
print('[G] ', G)


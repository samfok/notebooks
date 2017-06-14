# demonstrate the alignment of the correlate function
import numpy as np

a = np.array([1,2,3,4,5])
b = np.array([-1,0,2])

print(a)
print(b)

print(np.correlate(a, b, 'valid'))
print(np.correlate(a, b, 'same'))
print(np.correlate(a, b, 'full'))

print(np.correlate(b, a, 'valid'))
print(np.correlate(b, a, 'same'))
print(np.correlate(b, a, 'full'))

# alignment of correlate function
# c = correlate(a,b) for varying length of b
# align, multiply, then sum
# c[0] = 
#       a0 a1 a2
#       b0 
#       b0 b1
#       b0 b1 b2
#    b0 b1 b2 b3
# b0 b1 b2 b3 b4

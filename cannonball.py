import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from sklearn.model_selection import train_test_split
def linear(x, m, b):
    return m * x + b
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c
data = np.loadtxt("cannon.txt", delimiter=",", skiprows=1)
t = data[:, 0].astype(np.float32)
x = data[:, 1].astype(np.float32)
h = data[:, 2].astype(np.float32)
print("t = ", t)
print("x = ", x)
print("h = ", h)
params, params_cov = scipy.optimize.curve_fit(linear, t, x)
slope = params[0]
intercept = params[1]
print('x = {:.3f} t + {:.3f}'.format(slope,intercept))
plt.figure()
plt.scatter(t, x, label='Data')
plt.plot(t, linear(t, slope, intercept),label='Linear Fit') #change this label if you have a non-linear fit
plt.legend(loc='best')
plt.xlabel("time (s)") #change the units as appropriate
plt.ylabel("distance in x direction (meters)")  #change the units as appropriate
plt.show()

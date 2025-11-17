import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from sklearn.model_selection import train_test_split
def linear(x, m, b):
    return m * x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c
filename = "cannon.txt"
plot_prefix = "cannon"
x_col, h_col = 1, 2
data = np.loadtxt(filename, delimiter=",", dtype=str, skiprows=1)
t = data[:, 0].astype(np.float32)
x = data[:, x_col].astype(np.float32)
h = data[:, h_col].astype(np.float32)
params_x, _ = scipy.optimize.curve_fit(linear, t, x)
slope, intercept = params_x
print('x = {:.3f} t + {:.3f}'.format(slope, intercept))
plt.figure(figsize=(8,5))
plt.scatter(t, x, label='Data')
plt.plot(t, linear(t, slope, intercept), color='red', label='Linear Fit')
plt.xlabel("Time (s)")
plt.ylabel("Distance x (m)")
plt.title(f"{plot_prefix} - x vs Time")
params_h, _ = scipy.optimize.curve_fit(quadratic, t, h)
a, b, c = params_h
print('h = {:.3f} t^2 + {:.3f} t + {:.3f}'.format(a, b, c))
t_fit = np.linspace(t.min(), t.max(), 100)
h_fit = quadratic(t_fit, a, b, c)
plt.figure(figsize=(8,5))
plt.scatter(t, h, color='blue', label='Data')
plt.plot(t_fit, h_fit, color='red', label='Quadratic Fit')
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title(f"{plot_prefix} - Height vs Time")
filename = "piano.txt"
plot_prefix = "piano"
data = np.loadtxt(filename, delimiter=",", dtype=str, skiprows=1)
t = data[:, 0].astype(np.float32)
x = data[:, x_col].astype(np.float32)
h = data[:, h_col].astype(np.float32)
params_x, _ = scipy.optimize.curve_fit(linear, t, x)
slope, intercept = params_x
print('x = {:.3f} t + {:.3f}'.format(slope, intercept))
plt.figure(figsize=(8,5))
plt.scatter(t, x, label='Data')
plt.plot(t, linear(t, slope, intercept), color='red', label='Linear Fit')
plt.xlabel("Time (s)")
plt.ylabel("Distance x (m)")
plt.title(f"{plot_prefix} - x vs Time")
plt.savefig(f"{plot_prefix}_x_fit.png")
params_h, _ = scipy.optimize.curve_fit(quadratic, t, h)
a, b, c = params_h
print('h = {:.3f} t^2 + {:.3f} t + {:.3f}'.format(a, b, c))
t_fit = np.linspace(t.min(), t.max(), 100)
h_fit = quadratic(t_fit, a, b, c)
plt.figure(figsize=(8,5))
plt.scatter(t, h, color='blue', label='Data')
plt.plot(t_fit, h_fit, color='red', label='Quadratic Fit')
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title(f"{plot_prefix} - Height vs Time")
plt.show()
#the lines fit the plotted points very well, in the linear one you
#can see its off a little bit, but in the quadratic there ins't as much
#of a notice, I think this is due to the amount of data points it uses
#comapred the linear. 

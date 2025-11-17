import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear(x, m, b):
    return m * x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

data = np.loadtxt("cannon.txt", delimiter=",",dtype=str, skiprows=1)
t = data[:, 0].astype(float)
x = data[:, 1].astype(float)
h = data[:, 2].astype(float)

params_x, _ = curve_fit(linear, t, x)
m, b = params_x
print(f"CANNON: x = {m:.3f} t + {b:.3f}")
plt.figure()
plt.scatter(t, x)
plt.plot(t, linear(t, m, b))
plt.xlabel("Time (s)")
plt.ylabel("Distance x (m)")
plt.title("Cannon – x vs Time")
params_h, _ = curve_fit(quadratic, t, h)
a, b2, c = params_h
print(f"CANNON: h = {a:.3f} t^2 + {b2:.3f} t + {c:.3f}")
t_fit = np.linspace(t.min(), t.max(), 200)
plt.figure()
plt.scatter(t, h)
plt.plot(t_fit, quadratic(t_fit, a, b2, c))
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Cannon – Height vs Time")
data = np.loadtxt("piano.txt", delimiter=",", dtype=str, skiprows=1)
t = data[:, 0].astype(float)
x = data[:, 1].astype(float)
h = data[:, 2].astype(float)
params_x, _ = curve_fit(linear, t, x)
m, b = params_x
print(f"PIANO: x = {m:.3f} t + {b:.3f}")
plt.figure()
plt.scatter(t, x)
plt.plot(t, linear(t, m, b))
plt.xlabel("Time (s)")
plt.ylabel("Distance x (m)")
plt.title("Piano – x vs Time")
params_h, _ = curve_fit(quadratic, t, h)
a, b2, c = params_h
print(f"PIANO: h = {a:.3f} t^2 + {b2:.3f} t + {c:.3f}")
t_fit = np.linspace(t.min(), t.max(), 200)
plt.figure()
plt.scatter(t, h)
plt.plot(t_fit, quadratic(t_fit, a, b2, c))
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Piano – Height vs Time")
plt.show()
#the lines fit the plotted points very well, in the linear one you
#can see its off a little bit, but in the quadratic there ins't as much
#of a notice, I think this is due to the amount of data points it uses
#comapred the linear. 

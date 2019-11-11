import numpy as np
import matplotlib.animation as animation
from scipy.integrate import odeint, ode
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def f(y, t):
    y1, y2, y3, y4 = y
    return [y2,
            -y1 / (y1 ** 2 + y3 ** 2) ** (3 / 2),
            y4,
            -y3 / (y1 ** 2 + y3 ** 2) ** (3 / 2)]


t = np.linspace(0, 20, 1001)
y0 = [1, 0, 0, 0.4]
[y1, y2, y3, y4] = odeint(f, y0, t, full_output=False).T

fig2 = plt.figure(facecolor='white')
ax = plt.axes(xlim=(-0.2, 1.2), ylim=(-0.4, 0.4))
line, = ax.plot([], [], lw=3)
ax.grid(True)

def redraw(i):
    x = y1[0:i + 1]
    y = y3[0:i + 1]
    line.set_data(x, y)

anim = animation.FuncAnimation(fig2, redraw, frames=126, interval=50)
plt.show()

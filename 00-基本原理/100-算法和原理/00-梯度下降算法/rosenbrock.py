# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation as amat
  
"this function: f(x,y) = (1-x)^2 + 100*(y - x^2)^2"
  
  
def Rosenbrock(x, y):
    return np.power(1 - x, 2) + np.power(100 * (y - np.power(x, 2)), 2)
 
 
def show(X, Y, func=Rosenbrock):
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(X, Y, sparse=True)
    Z = func(X, Y)
    plt.title("gradeAscent image")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', )
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')
    amat.FuncAnimation(fig, Rosenbrock, frames=200, interval=20, blit=True)
    plt.show()
 
if __name__ == '__main__':
    X = np.arange(-2, 2, 0.1)
    Y = np.arange(-2, 2, 0.1)
    Z = Rosenbrock(X, Y)
    show(X, Y, Rosenbrock)
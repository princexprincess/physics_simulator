import time
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
global g, dt
g = np.array([0,0,-9.81])
dt = 0.001

class Mass:
    def __init__(self, pos, mass = 0.1):
        self.m = mass
        self.pos = pos
        self.v = np.array([0,0,0])
        self.a = np.array([0,0,0])
        self.F = np.array([0,0,0]) + g * self.m
        self.springs = []
    def update_F(self):
        self.F = g * self.m
        for spring in self.springs:
            f = spring.apply_force(self)
            self.F += f
        if self.pos[2] < 0:
            n = np.array([0,0,np.square(self.pos[2])*1000])
            self.F += n
    def update_pos(self):
        self.a = self.F / self.m
        self.v = 0.999*(self.v + self.a * dt)
        self.pos = self.pos + self.v * dt

class Spring:
    def __init__(self, mass1, mass2, k = 1000):
        self.node1 = mass1
        self.node2 = mass2
        self.vec = np.array(self.node2.pos) - np.array(self.node1.pos)
        self.l0 = np.linalg.norm(self.vec)
        self.l = np.linalg.norm(self.vec)
        self.k = k
        self.f = 0
    def update_len(self):
        self.vec = np.array(self.node2.pos)-np.array(self.node1.pos)
        self.l = np.linalg.norm(self.vec)
    def apply_force(self, mass):
        self.f = self.vec/np.linalg.norm(self.vec) * ((self.l - self.l0) * self.k)
        if mass == self.node1:
            return self.f
        elif mass == self.node2:
            return -self.f

def get_line(spring):
    pos1 = spring.node1.pos
    pos2 = spring.node2.pos
    xline = np.linspace(pos1[0],pos2[0],100)
    yline = np.linspace(pos1[1],pos2[1],100)
    zline = np.linspace(pos1[2],pos2[2],100)
    line = [xline, yline, zline]
    return line

def plot_fig(fig, ax, springs):
    ax.clear()
    for spring in springs:
        xline = np.linspace(spring.node1.pos[0],spring.node2.pos[0],100)
        yline = np.linspace(spring.node1.pos[1],spring.node2.pos[1],100)
        zline = np.linspace(spring.node1.pos[2],spring.node2.pos[2],100)
        ax.plot(xline,yline,zline,'black')
    ax.auto_scale_xyz([-0.1,0.2],[-0.1,0.2],[0,0.5])
    plt.pause(0.001)

h = 0.3
positions = [[0,0,0+h],[0.1,0,0+h],[0,0.1,0+h],[0.1,0.1,0+h],[0,0,0.1+h],[0.1,0,0.1+h],[0,0.1,0.1+h],[0.1,0.1,0.1+h]]
cube = []
for pos in positions:
    cube.append(Mass(pos))
pairs = itertools.combinations(cube,2)
springs = []
for pair in pairs:
    spring = Spring(pair[0],pair[1])
    springs.append(spring)
    pair[0].springs.append(spring)
    pair[1].springs.append(spring)

# ------------------ iteration ---------------------
t = 0
k = 0
fig = plt.figure(figsize=(6,8))
ax = Axes3D(fig)
plt.ion()
while t < 10:
    for mass in cube:
        mass.update_F()
        mass.update_pos()
    for spring in springs:
        spring.update_len()
    if k % 10 == 0:
        plot_fig(fig, ax, springs)
    t += dt
    k += 1
plt.ioff()
plt.show()
# ------------------ plot ---------------------

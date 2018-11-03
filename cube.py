# usr/bin/env/ python3
import time
import threading
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
global g, dt, damp
g = np.array([0,0,-9.81])
dt = 0.001
damp = 0.998

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
            n = np.array([0,0,np.square(1000*self.pos[2])])
            self.F += n

    def update_pos(self):
        self.a = self.F / self.m
        self.v = damp * (self.v + self.a * dt)
        self.pos = self.pos + self.v * dt

class Spring:
    def __init__(self, mass1, mass2, k = 10000):
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
    def potential_energy(self):
        pe = 0.5 * self.k * np.square(self.l - self.l0)
        return pe

def get_line(spring):
    pos1 = spring.node1.pos
    pos2 = spring.node2.pos
    xline = np.linspace(pos1[0],pos2[0],100)
    yline = np.linspace(pos1[1],pos2[1],100)
    zline = np.linspace(pos1[2],pos2[2],100)
    line = [xline, yline, zline]
    return line

# -------------------- plot ------------------------
def plot_fig(t, fig, ax, springs):
    ax.clear()
    for spring in springs:
        xline = np.linspace(spring.node1.pos[0],spring.node2.pos[0],100)
        yline = np.linspace(spring.node1.pos[1],spring.node2.pos[1],100)
        zline = np.linspace(spring.node1.pos[2],spring.node2.pos[2],100)
        ax.plot(xline,yline,zline,'black')
    ax.auto_scale_xyz([-0.1,0.2],[-0.1,0.2],[0,0.5])
    plt.title('t = '+str(t))
    plt.pause(0.00001)

# ------------------ iteration ---------------------
def bounce(cube,springs):
    h = 0.3
    for mass in cube:
        mass.pos[2] += h
    t = 0
    k = 0
    fig = plt.figure(figsize=(6,8))
    ax = Axes3D(fig)
    plt.ion()
    plot_fig(0, fig, ax, springs)
    time.sleep(5)
    while t < 10:
        for mass in cube:
            mass.update_F()
            mass.update_pos()
        for spring in springs:
            spring.update_len()
        if k % 5 == 0:
            plot_fig(t, fig, ax, springs)
            print(cube[0].pos)
        t += dt
        k += 1
        print(t)
    plt.ioff()
    plt.show()

def bounce_energy(cube,springs):
    h = 0.3
    for mass in cube:
        mass.pos[2] += h
    t = 0
    k = 0
    pe = []
    ke = []
    te = []
    x = []
    evals = 0
    start = time.time()
    while t < 2:
        pe_t = 0
        ke_t = 0
        for mass in cube:
            mass.update_F()
            mass.update_pos()
            ke_t += 0.5 * mass.m * np.square(np.linalg.norm(mass.v))
            pe_t += mass.m * 9.81 * mass.pos[2]
        for spring in springs:
            spring.update_len()
            evals += 1
            pe_t += spring.potential_energy()
        if k % 2 == 0:
            te_t = pe_t + ke_t
            pe.append(pe_t)
            ke.append(ke_t)
            te.append(te_t)
            x.append(t)
        t += dt
        k += 1
        print(t)
    duration = time.time() - start
    print(evals/duration)
    fig = plt.figure(figsize=(10,6))
    plt.plot(x,pe,label='potential energy')
    plt.plot(x,ke,label='kinetic energy')
    plt.plot(x,te,label='total energy')
    plt.title('bouncing energy graph')
    plt.xlabel('t')
    plt.ylabel('energy')
    plt.legend(loc='upper right')
    plt.show()

def breath(cube,springs):
    t = 0
    k = 0
    l0s = []
    for spring in springs:
        l0s.append(spring.l0)
    fig = plt.figure(figsize=(6,8))
    ax = Axes3D(fig)
    plt.ion()
    plot_fig(0, fig, ax, springs)
    time.sleep(5)
    while t < 10:
        for i in range(len(springs)):
            springs[i].l0 = l0s[i] + 0.02*np.sin(10*t)
        for mass in cube:
            mass.update_F()
            mass.update_pos()
        for spring in springs:
            spring.update_len()
        if k % 5 == 0:
            plot_fig(t, fig, ax, springs)
        t += dt
        k += 1
        print(t)
    plt.ioff()
    plt.show()

def breath_energy(cube,springs):
    t = 0
    k = 0
    l0s = []
    for spring in springs:
        l0s.append(spring.l0)
    pe = []
    ke = []
    te = []
    x = []
    evals = 0
    start = time.time()
    while t < 2:
        pe_t = 0
        ke_t = 0
        for i in range(len(springs)):
            springs[i].l0 = l0s[i] + 0.02*np.sin(10*t)
        for mass in cube:
            mass.update_F()
            mass.update_pos()
            ke_t += 0.5 * mass.m * np.square(np.linalg.norm(mass.v))
            pe_t += mass.m * 9.81 * mass.pos[2]
        for spring in springs:
            spring.update_len()
            evals += 1
            pe_t += spring.potential_energy()
        if k % 5 == 0:
            te_t = pe_t + ke_t
            pe.append(pe_t)
            ke.append(ke_t)
            te.append(te_t)
            x.append(t)
        t += dt
        k += 1
        print(t)
    duration = time.time() - start
    print(evals/duration)
    fig = plt.figure(figsize=(10,6))
    plt.plot(x,pe,label='potential energy')
    plt.plot(x,ke,label='kinetic energy')
    plt.plot(x,te,label='total energy')
    plt.title('breathing energy graph')
    plt.xlabel('t')
    plt.ylabel('energy')
    plt.legend(loc='upper right')
    plt.show()

def complex_bounce(cube,springs):
    h = 0.3
    theta = 0.1
    R = rotation_matrix(theta,theta,theta)
    for mass in cube:
        mass.pos = np.dot(R,mass.pos)
        mass.pos[2] += h
    t = 0
    k = 0
    fig = plt.figure(figsize=(6,8))
    ax = Axes3D(fig)
    plt.ion()
    plot_fig(0, fig, ax, springs)
    time.sleep(5)
    while t < 10:
        for mass in cube:
            mass.update_F()
            mass.update_pos()
        for spring in springs:
            spring.update_len()
        if k % 5 == 0:
            plot_fig(t, fig, ax, springs)
        t += dt
        k += 1
        print(t)
    plt.ioff()
    plt.show()

def rotation_matrix(r,p,y):
    R_x = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    R_y = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    R_z = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    R = np.dot(np.dot(R_x,R_y),R_z)
    return R

def construct(nodes):
    cube = []
    for node in nodes:
        cube.append(Mass(node))
    pairs = itertools.combinations(cube,2)
    springs = []
    for pair in pairs:
        spring = Spring(pair[0],pair[1])
        springs.append(spring)
        pair[0].springs.append(spring)
        pair[1].springs.append(spring)
    return cube, springs

if __name__ == "__main__":
    h = 0.0
    nodes = [[0,0,0+h],[0.1,0,0+h],[0,0.1,0+h],[0.1,0.1,0+h],[0,0,0.1+h],[0.1,0,0.1+h],[0,0.1,0.1+h],[0.1,0.1,0.1+h]]
    tetra = [[0,0,0],[0.1,0,0],[0,0.1,0],[0.1,0.1,0],[0.05,0.05,0.1]]
    cube, springs = construct(nodes)
    tetrahedron, edges = construct(tetra)

    # bounce(cube,springs)
    # breath(cube,springs)
    # bounce_energy(cube,springs)
    # breath_energy(cube,springs)
    complex_bounce(cube,springs)
    # complex_bounce(tetrahedron,edges)
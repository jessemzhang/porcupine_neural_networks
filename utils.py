from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

# Make 3D plot
def plot_L_3D(all_wi,all_wj,L,w1_opt,w2_opt,L_opt):
    fig = plt.figure(figsize=(10,10))
    ax = plt.gca(projection='3d')

    wi,wj = np.meshgrid(all_wi,all_wj)
    surf = ax.plot_surface(wi, wj, L, cmap=cm.coolwarm,
                                rstride=10, cstride=10)
    ax.scatter(w1_opt,w2_opt,L_opt,c='r')
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel('loss')
    plt.show()

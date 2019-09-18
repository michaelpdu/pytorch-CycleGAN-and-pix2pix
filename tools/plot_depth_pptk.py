#!/usr/bin/env python
#coding:utf8

import os,sys
import numpy as np
import struct
import pptk

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image

#from stl import mesh
#from stl_tools import numpy2stl

def main():
    fig = plt.figure()
    try:
        img = Image.open(sys.argv[1])
        Z = np.asarray(img)
    except IOError as e:
        Z = np.fromfile(sys.argv[1], np.uint16)
        Z.shape = 400,640

    Z = Z.copy()

    stl_alpha = Z.copy()
    print Z.shape
    z_min = Z[0,0]
    z_max = Z[0,0]

    # 高低翻转，记录最值，设置α通道
    for i in range(len(Z)):
        for j in range(len(Z[i])):
            stl_alpha[i,j] = 255

            if Z[i,j]> z_max:
                z_max = Z[i,j]
            if Z[i,j]< z_min:
                z_min = Z[i,j]
    print(type(Z[0,0]),Z.shape)

    print(z_min,z_max)

    middle = z_min+(z_max-z_min)//2
    Z[10,:] = middle
    Z[0,:] = middle
    Z[:,10] = middle
    Z[:,0] = middle
    Z[Z.shape[0]-1,:] = middle
    Z[:,Z.shape[1]-1] = middle

    pptk_array = []
    # 去掉最值接近的值
    for i in range(len(Z)):
        for j in range(len(Z[i])):
            if Z[i][j]> z_max - 10 or Z[i][j]< z_min + 10:
                Z[i][j] = 0
            else:
                Z[i][j] = Z[i][j] * 640 / z_max
                Z[i,j] = 640 - Z[i,j] 
                pptk_array.append([i,j,Z[i][j]])

    v = pptk.viewer(pptk_array)
    v.set(point_size= 0.25, lookat = (200.0,320.0,320.0))
 
    Z[0,0]=640
    size = Z.shape
    Y = np.arange(0,size[0],1)
    X = np.arange(0,size[1],1)
    X,Y = np.meshgrid(X,Y)

    #stl = mesh.Mesh(np.zeros(Z.shape[0],dtype=mesh.Mesh.dtype),remove_empty_areas=False)
    #stl.x[:] = X[:]
    #stl.y[:] = Y[:]
    #stl.z[:] = Z[:]
    #save('test.stl')


    #stl_array = 10.0 *Z[:, : ] + 1.0*stl_alpha[:,:] # Compose RGBA channels to give depth
    
    #numpy2stl(stl_array, "test.stl", scale=0.1, solid=False)
    #numpy2stl(stl_array, "test.stl", scale=0.05, mask_val=5., solid=True)

    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111,projection='3d')

    #ax.plot_surface(X,Y,Z,cmap=cm.jet)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount = 100, ccount = 100, alpha=0.9, linewidth=0)#, antialiased=False)

    #ax.plot_wireframe(X, Y, Z, rcount=100, ccount=100, cmap=cm.jet)

    # 轮廓
    ax.contour(X,Y,Z, zdir='z',offset = 200,cmap=cm.jet)
    ax.contour(X,Y,Z, zdir='x',offset = 0,cmap=cm.coolwarm)
    ax.contour(X,Y,Z, zdir='y',offset = 0,cmap=cm.coolwarm)


    ## triangles
    #X2 = []
    #for each in range(640):
    #    X2+=[each,]*640

    #Y2 = []
    #for each in range(640):
    #    Y2 += list(range(640))
    #Z2 = ([0,]*640)*120
    #for each in z:
    #    Z2+=each
    #Z2 += ([0,]*640)*120
    ##print len(X2),len(Y2),len(Z2)
    #ax.plot_trisurf(X2, Y2, Z2, cmap=cm.coolwarm) #,linewidth=0, antialiased=False)

    #plt.show()

if '__main__' == __name__:
    main()
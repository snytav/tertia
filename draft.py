import numpy as np

a = np.loadtxt('plasma_values_after_MoveParticlesLayerSplit_0000000001.dat')
B = a[:,1]

B = B.reshape(int(B.shape[0]/200),200)

A = a[:,2]
A = A.reshape(int(A.shape[0]/200),200)

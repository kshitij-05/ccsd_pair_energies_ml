# cython: infer_types=True
import math
import numpy as np
cimport cython

DTYPE = np.double
@cython.boundscheck(False)
@cython.wraparound(False)


def gen_ao_int(double[::1] eri , int nbasis):
	twoe = np.zeros((nbasis,nbasis,nbasis,nbasis),dtype = DTYPE)
	cdef double[:,:,:,::1] twoe_view = twoe
	cdef int n,i,j,k,l
	cdef float ij,kl
	n=0
	for i in range(nbasis):
		for j in range(i+1):
			for k in range(nbasis):
				for l in range(k+1):
					ij = i * (i + 1) / 2 + j
					kl = k * (k + 1) / 2 + l
					if ij>=kl:
						twoe_view[i,j,k,l] = eri[n]
						twoe_view[i,j,l,k] = eri[n]
						twoe_view[j,i,k,l] = eri[n]
						twoe_view[j,i,l,k] = eri[n]
						twoe_view[k,l,i,j] = eri[n]
						twoe_view[l,k,i,j] = eri[n]
						twoe_view[k,l,j,i] = eri[n]
						twoe_view[l,k,j,i] = eri[n]
						n+=1
	return twoe

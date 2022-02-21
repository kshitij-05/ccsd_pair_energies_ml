import numpy as np
from pyscf import gto, scf ,dft ,cc, lo
from gen_ao_ints import gen_ao_int
import h5py
import time
import sys
DTYPE = np.double
from ccsd_helper import init_CC_amps,CC_iter
from integral_transformation import int_tranf,ginp
from scf_helper import make_density , make_fock
#############################################
###         READ INPUT
#############################################



input_file=open('geom.txt','r')
file_content=input_file.readlines()
input_file.close()
geoms = []
mol = []
mol_name = []



for i in range(len(file_content)):
	file_content[i] = file_content[i].split()

for i in range(len(file_content)):
	if len(file_content[i])==4:
		mol.append(file_content[i])
		if len(file_content[i+1])==0:
			geoms.append(mol)
			mol=[]

def no_of_electrons(atoms):
	symbol = ['H','He', 'Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca', 'Sc',
	 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn','Ga', 'Ge', 'As', 'Se', 'Br', 'Kr','Rb', 'Sr', 'Y', 'Zr',
	 'Nb', 'Mo', 'Tc', 'Ru','Rh', 'Pd', 'Ag', 'Cd','In', 'Sn', 'Sb', 'Te', 'I', 'Xe','Cs', 'Ba', 'La', 'Ce', 'Pr',
	 'Nd', 'Pm', 'Sm',  'Eu','Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu','Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
	 'Au', 'Hg','Tl','Pb','Bi','Po','At','Rn']
	return symbol.index(atoms)+1

tot_noc = 0
no_e = 0
no_e_mol = []
pyscf_input_geoms=[]
str_geom =''
for i in range(len(geoms)):
	for j in range(len(geoms[i])):
		str_geom +=geoms[i][j][0]+'  '+geoms[i][j][1]+'  '+geoms[i][j][2]+'  '+geoms[i][j][3]+';'
		no_e += no_of_electrons(geoms[i][j][0])
	pyscf_input_geoms.append(str_geom)
	str_geom = ''
	no_e_mol.append(no_e)
	if no_e%2 ==1:
		no_e+=1
	tot_noc += no_e/2
	no_e = 0

print('Total no. of data points :', tot_noc)

def ao2mo(twoe,coeff):
	temp1 = np.einsum('pqrs,sl->pqrl', twoe, coeff)
	moint = np.einsum('pqrl,rk->pqkl', temp1, coeff)
	temp1 = np.einsum('qj,pqkl->pjkl', coeff, moint)
	moint = np.einsum('pi,pjkl->ijkl', coeff, temp1)
	return moint


basis_sets = [ 'ccpvdz' ]

#'6311g**','321g','631g','6-31++G','6-31++G**','6-31++G*','6-31++G**','DZP','6-311++G' ,'cc-pvtz' ,'cc-pvdz','aug-cc-pvdz'

X = []
ymp2 = []
yccsd = []


for basis_set in basis_sets:
	for no_mol in range(len(pyscf_input_geoms)):

		#print(pyscf_input_geoms[i])
		###############################################
		###      SCF AND MP2
		###############################################

		t1 = time.time()
		if no_e_mol[no_mol]%2 ==1:
			mol = gto.M(atom=pyscf_input_geoms[no_mol], basis=basis_set,spin =1)   # gen molecular attributes
		else :
			mol = gto.M(atom=pyscf_input_geoms[no_mol], basis=basis_set) 
		mf = mol.RHF().run()										# Hartree Fock
		ovlp = scf.hf.get_ovlp(mol)									# Extracting overlap mat
		nbasis = ovlp.shape[0]
		#print(nbasis)

		eri = mol.intor('int2e', aosym='s8')						# Extracting twoe electron integrals
		twoe = gen_ao_int(eri,nbasis)								# tranforming into a 4D array
		coeff = mf.mo_coeff
		fock_ao = mf.get_fock()
		occupancy = mf.mo_occ
		fock  =mf.mo_energy
		coeff = lo.orth_ao(mf, 'nao')

		tmp1 = np.matmul(coeff.transpose(),fock)
		newfock = np.matmul(tmp1,coeff)
		fock = newfock

		moint = ao2mo(twoe,coeff)

		o = 0
		for i in range(nbasis):
			if occupancy[i] !=0:
				o+=1

		v = nbasis-o
		print("No. of occupied :" , o)
		print("No. of virtual  :" , v)
		#################################################
		###     TRANSFORMING THE MOLECULAR INTEGRALS 
		#################################################

		ovov= moint[:o,o:,:o,o:]
		oovv = ovov.transpose(0,2,1,3)
		fo = fock[:o]
		fv = fock[o:]
		fia = fo[:, None] - fv[None, :]
		fijab = fia[:, None, :, None] + fia[None, :, None, :]	# Demominator (ei + ej - ea - eb) array
		#################################################
		#   MAKING Ei and CALC Emp2
		#################################################

		act_vir = 30
		print(v)
		emp2 = 0.0
		emp2ij = np.zeros((o,o))

		if act_vir<= v:
			t2 = time.time()
			fock_d = np.zeros((nbasis,nbasis))
			for i in range(nbasis):
				fock_d[i,i] = fock[i]
			ginp_ = ginp(fock_d,fock,o,v,nbasis,coeff,twoe,None)
			eris = int_tranf(ginp_,"New_mol")
			eris.int_transform(twoe)
			eris.incore = 5
			emp2, t1,t2 = init_CC_amps(o,v,eris)
			t1,t2 = CC_iter(t1,t2,eris,1e-12)
			eccsdij = np.zeros((o,o))
			for i in range(o):
				for j in range(o):

					#-----------------------
					#   MP2 amplitudes 
					#-----------------------

					temp = np.einsum(',ab->ab', 2, ovov[i,:,j,:])
					temp -=ovov[i,:,j,:].T
					temp = np.einsum('ab,ab->ab', temp,ovov[i,:,j,:])
					temp = temp/fijab[i,j,:,:]
					emp2ij[i,j] = np.sum(temp)
					temp = np.ndarray.flatten(temp)
					temp = np.sort(temp)
					
					fia =  fock_d[:o, o:]

					#-----------------------
					#   CCSD pair energies 
					#-----------------------

					eccsdij[i,j] = 2 * np.einsum('a,a',fia[i,:], t1[i,:])
					tau = np.einsum('a,b->ab', t1[i,:], t1[j,:])
					tau += t2[i,j,:,:]
					eccsdij[i,j] += 2 * np.einsum('ab,ab', tau, ovov[i,:,j,:])
					eccsdij[i,j] += -np.einsum('ab,ba', tau, ovov[i,:,j,:])

					X.append(temp[:(act_vir*act_vir)])
					yccsd.append(eccsdij[i,j])
					ymp2.append(emp2ij[i,j])


			emp2 = np.sum(emp2ij)
			print('Mp2 energy new', emp2)
			print(np.sum(eccsdij))


'''

f = h5py.File('methanef2.hdf5' , 'w')
f.create_dataset('X' , data = np.array(X))
f.create_dataset('ymp2' , data = np.array(ymp2))
f.create_dataset('ccsd' , data = np.array(yccsd))


'''





import numpy as np
from pyscf import lib
from pyscf.ao2mo import _ao2mo
import time
import h5py

def AOTOMO(twoe,coeff):
    temp1 = np.einsum('pqrs,sl->pqrl', twoe, coeff)
    moint = np.einsum('pqrl,rk->pqkl', temp1, coeff)
    temp1 = np.einsum('qj,pqkl->pjkl', coeff, moint)
    moint = np.einsum('pi,pjkl->ijkl', coeff, temp1)
    return moint


class ginp:
    def __init__(self,fock,mo_energy,no,nv,nbasis,coeff,twoe,dipao):
      self.fock=fock
      self.mo_energy=mo_energy
      self.no= no
      self.nv=nv
      self.nbasis=nbasis
      self.coeff=coeff
      self.twoe=twoe
      self.dipao=dipao

class int_tranf:
    def __init__(self,ginp,inp_name):
      self.fock=ginp.fock
      self.mo_energy=ginp.mo_energy
      self.no= ginp.no
      self.nv=ginp.nv
      self.nbasis=ginp.nbasis
      self.coeff=ginp.coeff
      self.dipao=ginp.dipao
      self.dipmo=None
      self.INT_FILE=None
      self.inp_name=inp_name
      #self.int_pyscf=ginp.int_pyscf
      self.naux=0
      #cc_restart
      self.cc_restart=0
      self.printlevel = 0
      self.ovoo=None
      self.ovov=None
      self.oovv=None
      self.ovvo=None
      self.ovvv=None
      self.ooov=None
      self.oooo=None
      self.vvvv=None
      self.capmat=None
      self.Loo=None
      self.Lvv=None
      self.Lov=None
      self.incore=5
      self.df=False
      self.DumpEOM=True
      self.fc=False
      self.fc_no=0

    def set_fno(self,f_active,nvir_act,fno_coeff):
      self.fock=f_active
      self.mo_energy=f_active.diagonal()
      self.nv=nvir_act
      self.nbasis=self.no+self.nv
      self.coeff=fno_coeff
    def set_fno_fc(self,f_active,fc_no,nvir_act,fno_coeff):
      f_activet=f_active[fc_no:,fc_no:]
      self.fock=f_activet
      self.mo_energy=f_activet.diagonal()
      f_activet=None
      no=self.no
      self.no=no-fc_no
      self.nv=nvir_act
      self.nbasis=self.no+self.nv
      self.coeff=fno_coeff
      self.fc=True
      self.fc_no=fc_no
    def integral_sort(self,moint):
        no=self.no
        nv=self.nv
        opt = True
        if opt:
            #---------------------------------------------------------------
            #  Using cython version of integral sort function
            #---------------------------------------------------------------
            tm1 = time.time()
            #self.ovoo, self.ovov, self.oovv, self.ovvo, self.ovvv, self.ooov, self.oooo, self.vvvv = integral_sort_cy(no,nv,moint)
            nt = no+nv
            self.oooo = moint[:no,:no,:no,:no]
            self.ovoo = moint[:no,no:nt,:no,:no]
            self.ovov = moint[:no,no:nt,:no,no:nt]
            self.oovv = moint[:no,:no,no:nt,no:nt]
            self.ovvo = moint[:no,no:nt,no:nt,:no]
            self.ovvv = moint[:no,no:nt,no:nt,no:nt]
            self.ooov = moint[:no,:no,:no,no:nt]
            self.vvvv = moint[no:nt,no:nt,no:nt,no:nt]

        else:
            #---------------------------------------------------------------
            #  Original pure python version
            #---------------------------------------------------------------
            ovoo = np.zeros([no, nv, no, no])
            ovov = np.zeros([no, nv, no, nv])
            oovv = np.zeros([no, no, nv, nv])
            ovvo = np.zeros([no, nv, nv, no])
            ovvv = np.zeros([no, nv, nv, nv])
            ooov = np.zeros([no, no, no, nv])
            oooo = np.zeros([no, no, no, no])
            vvvv = np.zeros([nv, nv, nv, nv])

            for i in range(no):
                for j in range(nv):
                    for k in range(no):
                        for l in range(no):
                            ovoo[i, j, k, l] = moint[i, no + j, k, l]
            for i in range(no):
                for j in range(nv):
                    for k in range(no):
                        for l in range(nv):
                            ovov[i, j, k, l] = moint[i, no + j, k, no + l]

            for i in range(no):
                    for j in range(no):
                        for k in range(nv):
                            for l in range(nv):
                                oovv[i, j, k, l] = moint[i, j, no + k, no + l]

            for i in range(no):
                    for j in range(nv):
                        for k in range(nv):
                            for l in range(no):
                                ovvo[i, j, k, l] = moint[i, no + j, no + k, l]

            for i in range(no):
                    for j in range(nv):
                        for k in range(nv):
                            for l in range(nv):
                                ovvv[i, j, k, l] = moint[i, no + j, no + k, no + l]

            for i in range(no):
                    for j in range(no):
                        for k in range(no):
                            for l in range(nv):
                                ooov[i, j, k, l] = moint[i, j, k, no + l]
            for i in range(no):
                    for j in range(no):
                        for k in range(no):
                            for l in range(no):
                                oooo[i, j, k, l] = moint[i, j, k, l]

            for i in range(nv):
                    for j in range(nv):
                        for k in range(nv):
                            for l in range(nv):
                                vvvv[i, j, k, l] = moint[no + i, no + j, no + k, no + l]
            # with h5py.File('vvvv.hdf5', 'w') as f:
            #     for a in range (nv):
            #         arr1=vvvv[a,:,:,:]
            #         f.create_dataset(str(a), data=arr1)
            # with h5py.File('vvvv.hdf5', 'r') as f:
            #     for a in range (nv):
            #
            #         d1 = f[str(a)]
            #         print(d1)


            self.ovoo=ovoo
            self.ovov = ovov
            self.oovv = oovv
            self.ovvo=ovvo
            self.ovvv = ovvv
            self.ooov = ooov
            self.oooo = oooo
            self.vvvv = vvvv
     ##################
     #helper functions
    #####################
    def get_ovvv(self):
        return self.ovvv
    def get_capmat(self):
        self.capmat=get_gamess_capmat(self.nv)
    def transform_dipole(self):
       nbasis=self.no+self.nv
       print(nbasis)
       DIPMO = np.zeros([3,nbasis,nbasis])
       for i in range(3):
            DAO=self.dipao[i,:,:]
            mo=self.coeff
            DMO = np.einsum('ip,ij,jq->pq', mo.conj(), DAO, mo.conj())
            DIPMO[i,:,:]=DMO[:,:]
       self.dipmo=DIPMO
       return

    def int_transform_df(self,with_df):
        nocc = self.no
        nmo = self.nbasis
        nvir = nmo - nocc
        nvir_pair = nvir * (nvir + 1) // 2
        with_df = with_df
        naux = with_df.get_naoaux()
        self.naux=naux
        Loo = np.empty((naux, nocc, nocc))
        Lov = np.empty((naux, nocc, nvir))
        Lvv= np.empty((naux, nvir, nvir))
        mo = np.asarray(self.coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        p1 = 0
        Lpq = None
        for k, eri1 in enumerate(with_df.loop()):
            Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            p0, p1 = p1, p1 + Lpq.shape[0]
            Lpq = Lpq.reshape(p1 - p0, nmo, nmo)
            Loo[p0:p1] = Lpq[:, :nocc, :nocc]
            Lov[p0:p1] = Lpq[:, :nocc, nocc:]
            Lvv[p0:p1] =   Lpq[:, nocc:, nocc:]

        Lpq = None
        self.Loo = Loo
        self.Lov = Lov
        self.Lvv = Lvv

    def gen_df_integral(self,MP2=False):
        Loo=self.Loo
        Lov = self.Lov
        Lvv=self.Lvv
        nocc = self.no
        nmo = self.nbasis
        naux=self.naux
        nvir = nmo - nocc
        print(np.shape(Loo))
        Loo = Loo.reshape(naux, nocc ** 2)
        # Lvo = Lov.transpose(0,2,1).reshape(naux,nvir*nocc)
        if MP2:
            Lov = Lov.reshape(naux, nocc * nvir)
            self.ovov = lib.ddot(Lov.T, Lov).reshape(nocc, nvir, nocc, nvir)
            self.ovoo = lib.ddot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc)
        else:
         Lov = Lov.reshape(naux, nocc * nvir)
         Lvv = Lvv.reshape(naux, nvir ** 2)
         self.oooo = lib.ddot(Loo.T, Loo).reshape(nocc, nocc, nocc, nocc)
         self.ovoo = lib.ddot(Lov.T, Loo).reshape(nocc, nvir, nocc, nocc)
         self.ovov = lib.ddot(Lov.T, Lov).reshape(nocc, nvir, nocc, nvir)
         self.ovvo = self.ovov.transpose(0, 1, 3, 2)
         print('self.incore',self.incore)
         if self.incore > 3:
            self.ovvv = lib.ddot(Lov.T, Lvv).reshape(nocc, nvir, nvir, nvir)
         self.oovv = lib.ddot(Loo.T, Lvv).reshape(nocc, nocc, nvir, nvir)
         if self.incore > 4:
            self.vvvv = lib.ddot(Lvv.T, Lvv).reshape(nvir, nvir, nvir, nvir)
         start = time.time()

         self.transform_dipole()
         end = time.time()
         print("Time taken in dipole integral tranformation".format(end - start))

    def int_transform(self,twoe):
        start = time.time()
        nbasis, nvir = self.coeff.shape
        #twoe = read_2_e(nbasis,self.int_pyscf,self.inp_name)
        end = time.time()
        #print("Time taken in read_2_e is {} seconds".format(end - start))
        coeff=self.coeff
        #print(coeff)
        #print('coefficient shape %s' % str(coeff.shape))''
        opt = False
        if opt:
            start = time.time()
            self.coeff = self.coeff.copy(order ='C')
            moint= transform_ao2mo(twoe,self.coeff)
            end = time.time()
        else :
            start = time.time()
            moint= AOTOMO(twoe,self.coeff)
            end = time.time()            

        print("Time taken in AOTOMO is {} seconds".format(end - start))
        start = time.time()
        self.integral_sort(moint)
        end = time.time()
        print("Time taken in integral_sort is {} seconds".format(end - start))
        '''start = time.time()
        self.transform_dipole()
        end = time.time()
        print("Time taken in dipole integral tranformation".format(end - start))'''



    def int_transform_read(self):
        FII_data = np.fromfile('output.dat', dtype=np.float, count=n * m)

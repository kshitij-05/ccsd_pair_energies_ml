import time
from pyscf import gto, scf, cc, lib
import rintermediates as imd
from ccsd_intermediates import cc_Foo, cc_Fvv, cc_Fov, cc_Woooo, cc_Wvvvv, cc_Wvoov, cc_Wvovo
import numpy as np

einsum = lib.einsum



def init_CC_amps(nocc, nvir, eris):
    mo_e = eris.fock.diagonal().real
    start = time.time()
    eia = mo_e[:nocc, None] - mo_e[None, nocc:]
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1 = eris.fock[:nocc, nocc:].conj() / eia
    eris_ovov = np.asarray(eris.ovov)

    t2 = eris_ovov.transpose(0, 2, 1, 3).conj() / eijab
    emp2 = 2 * einsum('ijab,iajb', t2, eris_ovov)
    emp2 -= einsum('ijab,ibja', t2, eris_ovov)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in MP2 calculation is {} seconds".format(end - start))
    # lib.logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
    return emp2, t1, t2


def CC_energy(t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    start = time.time()
    e = 2 * einsum('ia,ia', fock[:nocc, nocc:], t1)
    tau = einsum('ia,jb->ijab', t1, t1)
    tau += t2
    eris_ovov = np.asarray(eris.ovov)
    e += 2 * einsum('ijab,iajb', tau, eris_ovov)
    e += -einsum('ijab,ibja', tau, eris_ovov)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in calculation of CC energy is {} seconds".format(end - start))
    return e.real





def CC_iter(t1, t2, eris, convergence):
    printops = True
    error_max_size = 8
    eris.printlevel=1
    diis_check = 1
    t1_old = t1.copy()
    t2_old = t2.copy()
    T1Set = [t1.copy()]
    T2Set = [t2.copy()]
    errors = []
    error_size = 0
    ECCSD=0
    print(f"DIIS convergence is turned on")

    for j in range(0, 60):
            OLDCC = ECCSD
            start = time.time()
            t1, t2 = update_CC_amps(t1, t2, eris, cc2=False)
            ECCSD = CC_energy(t1, t2, eris)
            DECC = abs(ECCSD - OLDCC)

            if DECC < convergence:
                print("TOTAL ITERATIONS: ", j)
                break
            if printops == True:
                print("E corr: {0:.12f}".format(ECCSD), "a.u.", '\t', "DeltaE: {0:.12f}".format(DECC))

            # Appending DIIS vectors to T1 and T2 set
            T1Set.append(t1.copy())
            T2Set.append(t2.copy())

            # calculating error vectors
            error_t1 = (T1Set[-1] - t1_old).ravel()
            error_t2 = (T2Set[-1] - t2_old).ravel()
            errors.append(np.concatenate((error_t1, error_t2)))
            t1_old = t1.copy()
            t2_old = t2.copy()

            if j >= diis_check:
                # size limit of DIIS vector
                if (len(T1Set) > error_max_size + 1):
                    del T1Set[0]
                    del T2Set[0]
                    del errors[0]

                error_size = len(T1Set) - 1

                # create error matrix B_mat
                B_mat = np.ones((error_size + 1, error_size + 1)) * -1
                B_mat[-1, -1] = 0

                for a1, b1 in enumerate(errors):
                    B_mat[a1, a1] = np.dot(b1.real, b1.real)
                    for a2, b2 in enumerate(errors):
                        if a1 >= a2: continue
                        B_mat[a1, a2] = np.dot(b1.real, b2.real)
                        B_mat[a2, a1] = B_mat[a1, a2]

                B_mat[:-1, :-1] /= np.abs(B_mat[:-1, :-1]).max()

                # create zero vector
                zero_vector = np.zeros(error_size + 1)
                zero_vector[-1] = -1

                # getting coefficients
                coeff = np.linalg.solve(B_mat, zero_vector)

                # getting extrapolated amplitudes
                t1 = np.zeros_like(t1_old)
                t2 = np.zeros_like(t2_old)
                for i in range(error_size):
                    t1 += coeff[i] * T1Set[i + 1]
                    t2 += coeff[i] * T2Set[i + 1]

                # Save extrapolated amplitudes to t_old amplitudes
                t1_old = t1.copy()
                t2_old = t2.copy()

    ECCSD = CC_energy(t1, t2, eris)
    end = time.time()
    print('Final CCSD correlation energy', ECCSD)
    if eris.printlevel:
	    print(f"Time taken for CCSD Convergence {end-start} seconds")
    return t1, t2


def update_CC_amps(t1, t2, eris, cc2=False):
    level_shift = 0.00  # A shift on virtual orbital energies to stablize the CCSD iteration
    # Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004) Eqs.(35)-(36)
    nocc, nvir = t1.shape
    fock = eris.fock
    naux = eris.naux
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + level_shift
    fov = fock[:nocc, nocc:].copy()
    foo = fock[:nocc, :nocc].copy()
    fvv = fock[nocc:, nocc:].copy()

    Foo = cc_Foo(t1, t2, eris)
    Fvv = cc_Fvv(t1, t2, eris)
    Fov = cc_Fov(t1, t2, eris)
    # Move energy terms to the other side
    Foo[np.diag_indices(nocc)] -= mo_e_o
    Fvv[np.diag_indices(nvir)] -= mo_e_v

    # T1 equation
    start = time.time()
    t1new = -2 * np.einsum('kc,ka,ic->ia', fov, t1, t1)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T1 formation_1 is {} seconds".format(end - start))
    start = time.time()
    t1new += np.einsum('ac,ic->ia', Fvv, t1)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T1 formation_2 is {} seconds".format(end - start))
    start = time.time()
    t1new += -np.einsum('ki,ka->ia', Foo, t1)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T1 formation_3 is {} seconds".format(end - start))
    start = time.time()
    t1new += 2 * np.einsum('kc,kica->ia', Fov, t2)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T1 formation_4 is {} seconds".format(end - start))
    start = time.time()
    t1new += -np.einsum('kc,ikca->ia', Fov, t2)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T1 formation_5 is {} seconds".format(end - start))
    start = time.time()
    t1new += np.einsum('kc,ic,ka->ia', Fov, t1, t1)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T1 formation_6 is {} seconds".format(end - start))
    start = time.time()
    t1new += fov.conj()
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T1 formation_7 is {} seconds".format(end - start))
    start = time.time()
    t1new += 2 * np.einsum('kcai,kc->ia', eris.ovvo, t1)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T1 formation_8 is {} seconds".format(end - start))
    start = time.time()
    t1new += -np.einsum('kiac,kc->ia', eris.oovv, t1)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T1 formation_9 is {} seconds".format(end - start))
    eris_ovvv = np.asarray(eris.get_ovvv())

    start = time.time()
    if (eris.incore < 4 and eris.df):
        tautemp = np.einsum('kd,ic->ikcd', t1, t1)
        tautemp += t2
        ttnew = 2 * lib.einsum('mkd,ikcd->mic', eris.Lov, tautemp)
        t1new += lib.einsum('mic,mac->ia', ttnew, eris.Lvv)
        ttnew = None
        ttnew = lib.einsum('mkc,ikcd->mid', eris.Lov, tautemp)
        t1new -= lib.einsum('mid,mad->ia', ttnew, eris.Lvv)
        ttnew = None
        tautemp = None
    else:
        t1new += 2 * lib.einsum('kdac,ikcd->ia', eris_ovvv, t2)
        t1new += -lib.einsum('kcad,ikcd->ia', eris_ovvv, t2)
        t1new += 2 * lib.einsum('kdac,kd,ic->ia', eris_ovvv, t1, t1)
        t1new += -lib.einsum('kcad,kd,ic->ia', eris_ovvv, t1, t1)
    eris_ovoo = np.asarray(eris.ovoo, order='C')
    t1new += -2 * lib.einsum('lcki,klac->ia', eris_ovoo, t2)
    t1new += lib.einsum('kcli,klac->ia', eris_ovoo, t2)
    t1new += -2 * lib.einsum('lcki,lc,ka->ia', eris_ovoo, t1, t1)
    t1new += lib.einsum('kcli,lc,ka->ia', eris_ovoo, t1, t1)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T1 formation_10 is {} seconds".format(end - start))

    # T2 equation
    start = time.time()
    if (eris.incore < 4 and eris.df):
        tmp4 = lib.einsum('kibc,jc->kijb', eris.oovv, -t1)
        tmp = lib.einsum('kijb,ka->ijab', tmp4, t1)
        tmp4 = None
        tmp2 = lib.einsum('mcb,jc->mjb', eris.Lvv, t1)
        tmp += lib.einsum('aim,mjb->ijab', eris.Lov.T, tmp2)
        tmp2 = None
    else:
        tmp2 = lib.einsum('kibc,ka->abic', eris.oovv, -t1)
        tmp2 += np.asarray(eris_ovvv).conj().transpose(1, 3, 0, 2)
        tmp = lib.einsum('abic,jc->ijab', tmp2, t1)
    t2new = tmp + tmp.transpose(1, 0, 3, 2)
    tmp2 = lib.einsum('kcai,jc->akij', eris.ovvo, t1)
    tmp2 += eris_ovoo.transpose(1, 3, 0, 2).conj()
    tmp = lib.einsum('akij,kb->ijab', tmp2, t1)
    t2new -= tmp + tmp.transpose(1, 0, 3, 2)
    t2new += np.asarray(eris.ovov).conj().transpose(0, 2, 1, 3)
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T2 formation_1 is {} seconds".format(end - start))

    start = time.time()
    if cc2:
        Woooo2 = np.asarray(eris.oooo).transpose(0, 2, 1, 3).copy()
        Woooo2 += lib.einsum('lcki,jc->klij', eris_ovoo, t1)
        Woooo2 += lib.einsum('kclj,ic->klij', eris_ovoo, t1)
        Woooo2 += lib.einsum('kcld,ic,jd->klij', eris.ovov, t1, t1)
        t2new += lib.einsum('klij,ka,lb->ijab', Woooo2, t1, t1)
        Wvvvv = lib.einsum('kcbd,ka->abcd', eris_ovvv, -t1)
        Wvvvv = Wvvvv + Wvvvv.transpose(1, 0, 3, 2)
        Wvvvv += np.asarray(eris.vvvv).transpose(0, 2, 1, 3)
        t2new += lib.einsum('abcd,ic,jd->ijab', Wvvvv, t1, t1)
        Lvv2 = fvv - np.einsum('kc,ka->ac', fov, t1)
        Lvv2 -= np.diag(np.diag(fvv))
        tmp = lib.einsum('ac,ijcb->ijab', Lvv2, t2)
        t2new += (tmp + tmp.transpose(1, 0, 3, 2))
        Loo2 = foo + np.einsum('kc,ic->ki', fov, t1)
        Loo2 -= np.diag(np.diag(foo))
        tmp = lib.einsum('ki,kjab->ijab', Loo2, t2)
        t2new -= (tmp + tmp.transpose(1, 0, 3, 2))
    else:
        Loo = imd.Loo(t1, t2, eris)
        Lvv = imd.Lvv(t1, t2, eris)
        Loo[np.diag_indices(nocc)] -= mo_e_o
        Lvv[np.diag_indices(nvir)] -= mo_e_v

        Woooo = imd.cc_Woooo(t1, t2, eris)
        Wvoov = imd.cc_Wvoov(t1, t2, eris)
        Wvovo = imd.cc_Wvovo(t1, t2, eris)

        if (eris.incore > 5 or not eris.df):
            Wvvvv = imd.cc_Wvvvv(t1, t2, eris)

        tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
        t2new += lib.einsum('klij,klab->ijab', Woooo, tau)
        if (eris.incore < 5 and eris.df):
            #ttemp = contract_4p_ri(tau.T, eris.Lvv.T, eris.Lov.T, t1.T, nocc, nvir, naux)
            #t2new += ttemp.transpose(2, 3, 0, 1)
            # t2new=contract_ri_4p(nocc, nvir, tau,t1, t2new, eris.Lvv,eris.Lov)

            for i in range(tau.shape[0]):
                for j in range(tau.shape[1]):
                    ttemp=lib.einsum('cd,mbd->bcm',tau[i,j,:,:],eris.Lvv)
                    t2new[i,j,:,:] +=lib.einsum('bcm,mca->ab',ttemp,eris.Lvv)
                ttemp=None
                ttemp = lib.einsum('jcd,mkd->jkcm', tau[i,:,:,:], eris.Lov)
                ttemp1 =lib.einsum('jkcm,mca->jak',ttemp,eris.Lvv)
                t2new[i,:,:,:] -= lib.einsum('jak,kb->jab', ttemp1, t1)
                ttemp1=None
                ttemp=None
                ttemp = lib.einsum('jcd,mkc->jkdm', tau[i,:,:,:], eris.Lov)
                ttemp1 = lib.einsum('jkdm,mbd->jkb', ttemp, eris.Lvv)
                t2new[i,:,:,:] -= lib.einsum('jkb,ka->jab', ttemp1, t1)
                ttemp1=None
                ttemp=None

        else:
            Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
            t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
        tmp = lib.einsum('ac,ijcb->ijab', Lvv, t2)
        t2new += (tmp + tmp.transpose(1, 0, 3, 2))
        tmp = lib.einsum('ki,kjab->ijab', Loo, t2)
        t2new -= (tmp + tmp.transpose(1, 0, 3, 2))
        tmp = 2 * lib.einsum('akic,kjcb->ijab', Wvoov, t2)
        tmp -= lib.einsum('akci,kjcb->ijab', Wvovo, t2)
        t2new += (tmp + tmp.transpose(1, 0, 3, 2))
        tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)
        t2new -= (tmp + tmp.transpose(1, 0, 3, 2))
        tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)
        t2new -= (tmp + tmp.transpose(1, 0, 3, 2))
    end = time.time()
    if eris.printlevel > 5:
        print("Time taken in T2 formation_2 is {} seconds".format(end - start))

    eia = mo_e_o[:, None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new



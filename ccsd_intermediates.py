from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
import numpy as np

# This is restricted (R)CCSD
# Ref: Hirata et al., J. Chem. Phys. 120, 2581 (2004)

### Eqs. (37)-(39) "kappa"

def cc_Foo(t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    eris_ovov = np.asarray(eris.ovov)
    Fki  = 2*lib.einsum('kcld,ilcd->ki', eris_ovov, t2)
    Fki -=   lib.einsum('kdlc,ilcd->ki', eris_ovov, t2)
    Fki += 2*lib.einsum('kcld,ic,ld->ki', eris_ovov, t1, t1)
    Fki -=   lib.einsum('kdlc,ic,ld->ki', eris_ovov, t1, t1)
    Fki += foo
    return Fki
def cc_Fvv(t1, t2, eris):
    nocc, nvir = t1.shape
    fvv = eris.fock[nocc:,nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fac  =-2*lib.einsum('kcld,klad->ac', eris_ovov, t2)
    Fac +=   lib.einsum('kdlc,klad->ac', eris_ovov, t2)
    Fac -= 2*lib.einsum('kcld,ka,ld->ac', eris_ovov, t1, t1)
    Fac +=   lib.einsum('kdlc,ka,ld->ac', eris_ovov, t1, t1)
    Fac += fvv
    return Fac

def cc_Fov(t1, t2, eris):
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    eris_ovov = np.asarray(eris.ovov)
    Fkc  = 2*np.einsum('kcld,ld->kc', eris_ovov, t1)
    Fkc -=   np.einsum('kdlc,ld->kc', eris_ovov, t1)
    Fkc += fov
    return Fkc
def cc_Woooo(t1, t2, eris):
    eris_ovoo = np.asarray(eris.ovoo)
    Wklij  = lib.einsum('lcki,jc->klij', eris_ovoo, t1)
    Wklij += lib.einsum('kclj,ic->klij', eris_ovoo, t1)
    eris_ovov = np.asarray(eris.ovov)
    Wklij += lib.einsum('kcld,ijcd->klij', eris_ovov, t2)
    Wklij += lib.einsum('kcld,ic,jd->klij', eris_ovov, t1, t1)
    Wklij += np.asarray(eris.oooo).transpose(0,2,1,3)
    return Wklij

def cc_Wvvvv(t1, t2, eris):
    # Incore
    eris_ovvv = np.asarray(eris.get_ovvv())
    Wabcd  = lib.einsum('kdac,kb->abcd', eris_ovvv,-t1)
    Wabcd -= lib.einsum('kcbd,ka->abcd', eris_ovvv, t1)
    Wabcd += np.asarray(_get_vvvv(eris)).transpose(0,2,1,3)
    return Wabcd
def cc_Wvoov(t1, t2, eris):
    eris_ovoo = np.asarray(eris.ovoo)
    if (eris.incore < 4 and eris.df):
        temp_mai=lib.einsum('mad,id->mai', eris.Lvv, t1)
        Wakic = lib.einsum('ckm,mai->akic', eris.Lov.T, temp_mai)
        temp_mai=None
    else:
     eris_ovvv = np.asarray(eris.get_ovvv())
     Wakic  = lib.einsum('kcad,id->akic', eris_ovvv, t1)
    Wakic -= lib.einsum('kcli,la->akic', eris_ovoo, t1)
    Wakic += np.asarray(eris.ovvo).transpose(2,0,3,1)
    eris_ovov = np.asarray(eris.ovov)
    Wakic -= 0.5*lib.einsum('ldkc,ilda->akic', eris_ovov, t2)
    Wakic -= 0.5*lib.einsum('lckd,ilad->akic', eris_ovov, t2)
    Wakic -= lib.einsum('ldkc,id,la->akic', eris_ovov, t1, t1)
    Wakic += lib.einsum('ldkc,ilad->akic', eris_ovov, t2)
    return Wakic

def cc_Wvovo(t1, t2, eris):

    eris_ovoo = np.asarray(eris.ovoo)
    if (eris.incore < 4 and eris.df):
        tmp_kim=lib.einsum('id,mkd->ikm', t1,eris.Lov)
        Wakci=lib.einsum('ikm,mac->akci', tmp_kim, eris.Lvv)
        tmp_kim=None
    else:
     eris_ovvv = np.asarray(eris.get_ovvv())
     Wakci  = lib.einsum('kdac,id->akci', eris_ovvv, t1)
    Wakci -= lib.einsum('lcki,la->akci', eris_ovoo, t1)
    Wakci += np.asarray(eris.oovv).transpose(2,0,3,1)
    eris_ovov = np.asarray(eris.ovov)
    Wakci -= 0.5*lib.einsum('lckd,ilda->akci', eris_ovov, t2)
    Wakci -= lib.einsum('lckd,id,la->akci', eris_ovov, t1, t1)
    return Wakci

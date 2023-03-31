import sys
sys.path.append("..")
from importall import *
import scipy
from scipy import integrate

def main():
    init_precision(128)
    print(LogMessage(), 'Testing functions at specific values')
    assert( abs(float(mp.fsub(Zfact_mp(estar_=1,sigma_=0.1), mpf(str('0.2506628274631'))))) < 1e-12)
    assert (abs(float(mp.fsub(Zfact_mp(estar_=7, sigma_=0.5), mpf(str('1.2533141373155'))))) < 1e-12)
    print(LogMessage(), 'Zfact_mp ok')
    assert (abs(float(mp.fsub(ft_mp(0.8, t=3, sigma_=0.1,alpha=mpf(-0.99), emin=mpf(0.25)), mpf(str('0.0444940561260751335503904613233'))))) < 1e-12)
    assert (abs(float(mp.fsub(ft_mp(e=1.2, sigma_=0.2, t=7), mpf(str('0.000599148'))))) < 1e-8)
    print(LogMessage(), 'ft_mp ok')

    v = mp.matrix(1,5)
    v[0]=mpf(5)
    v[1]=mpf(6)
    v[2]=mpf(4)
    v[3]=mpf(8)
    v[4]=mpf(7)
    vbar = averageVector_mp(v)
    assert((vbar[0]-6)==0)
    sqrt2 = mp.sqrt(2)
    diff = mp.fsub(vbar[1], sqrt2)
    print(LogMessage(), 'my stddv - true stddv', '{:2.2e}'.format(float(diff)))
    assert (diff < 10**(-mp.dps+1))
    print(LogMessage(), 'averageVector_mp ok')
    del v
    v = np.zeros((5,3))
    v[0][0] = 5
    v[1][0] = 6
    v[2][0] = 4
    v[3][0] = 8
    v[4][0] = 7
    v[0][1] = 5
    v[1][1] = 6
    v[2][1] = 4
    v[3][1] = 8
    v[4][1] = 7
    v[0][2] = 7
    v[1][2] = 6
    v[2][2] = 8
    v[3][2] = 4
    v[4][2] = 5
    num_boot = 100
    vboot = np.zeros((num_boot,3))
    vboot = bootstrap_fp(T_=3, nms_=5, Nb_=num_boot, in_=v, out_=vboot)
    vbootbar_a = averageVector_fp(vector=vboot[:,0], get_error=True, get_var=True)
    vbootbar_b = averageVector_fp(vector=vboot[:, 1], get_error=True, get_var=True)
    vbootbar_c = averageVector_fp(vector=vboot[:, 2], get_error=True, get_var=True)
    assert(vbootbar_a[0] - vbootbar_b[0] < 1e-8)
    assert (vbootbar_a[1] - vbootbar_b[1] < 1e-8)
    assert (vbootbar_a[0] - vbootbar_c[0] < vbootbar_a[1])
    print(LogMessage(), 'bootstrap_fp ok')
    print(LogMessage(), 'Test smearing kernel normalisations:')
    I = scipy.integrate.quad(lambda x: gauss_fp(x, 0.5, 1, norm='Half') , 0, np.inf)
    print(LogMessage(), 'Integrate in [0,inf) ', I[0], '+/-', I[1])
    I = scipy.integrate.quad(lambda x: gauss_fp(x, 0.5, 1, norm='Full'), -np.inf, np.inf)
    print(LogMessage(), 'Integrate in (-inf,inf) ', I[0], '+/-', I[1])

if __name__ == '__main__':
    main()
    end()
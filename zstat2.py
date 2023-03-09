import numpy as np
import scipy.stats as stats
from scipy.stats import norm


def ztest2(P1,P2,*args):
    '''
    ZTEST2 two tailed, two proportion z-test, pooled variance estimate.
    H = ZTEST2(P,N) performs a t-test of the hypothesis that two
    independent samples of draws from binomial processes, in the data
    vectors P1 and P2, come from distributions with equal probability of
    sucess. H is the result of this statistical test, evaluated at p <
    0.05.
    
    P1 and P2 should be vectors of [proportion sucess, total observations]

    The proportion can be expressed either as a count (i.e. 3 sucesses) or
    a proportion (i.e. 0.3). The script treats input values > 1 as counts.

    example useage: [h,p] = ztest2([0.1 100],[0.5 100])

    Evaluates the hypothesis that a sucess rate of 0.1, observed on 100
    independant draws, is significantly different from 0.5.

    See also ZTEST, TTEST2
        
    '''
    if P1 is None or P2 is None:
        print('Error: too few input arguments')
        return

    alpha = 0.05  # default to p < 0.05
    tail = 0      # code for two-sided

    # initialize
    h = float('nan')
    zStat = float('nan')
    p = float('nan')

    if len(P1) == 2 and len(P2) == 2:
        n = [P1[1], P2[1]]
        prop = [P1[0], P2[0]]

      # rescale to proportion if input seems to be a count
        if prop[0] > 1 or prop[1] > 1:
            print('\n')
            print('Treating input proportions as counts. \n')
            prop = [prop[i] / n[i] for i in range(2)]
        elif n[0] < 10 or n[1] < 10:
            print('\n')
            print('Error: n observations must be greater than 10 \n')

     
        pooledP = (prop[0] * n[0] + prop[1] * n[1]) / (n[0] + n[1])
        pooledSE = np.sqrt(pooledP * (1 - pooledP) * ((1 / n[0]) + (1 / n[1])))
        zStat = (prop[0] - prop[1]) / pooledSE
    else:
        print('\n')
        print('Error: Wrong input format \n')
        print('P1 and P2 must be 2x1 or 1x2 vectors, proportion and n \n')

    if zStat > 0:
        p = 2 * (1-stats.norm.cdf(zStat))
    else:
        p = 2 * stats.norm.cdf(zStat)
    #high and low value thresholds for the z-statistic

    thresh = norm.ppf([alpha/2, 1-alpha/2])
    h = int(zStat < thresh[0] or zStat > thresh[1])
    
    return [h, p,zStat]





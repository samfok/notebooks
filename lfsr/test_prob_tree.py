import numpy as np
from prob_tree import ProbTree
import matplotlib.pyplot as plt


def ptree_experiment(nbits, dist, nsamples=1000, title=None):
    ptree = ProbTree(nbits, lfsr_seed=0b00101010, w_dist=dist)
    samples = np.zeros(nsamples, dtype=int)
    for i in xrange(nsamples):
        samples[i] = ptree.sample()
    bincount = np.bincount(samples)
    w_emp = bincount/float(nsamples)

    apx_error = abs(ptree.w - ptree.w_apx)
    emp_error = abs(ptree.w - w_emp)
    q_level = 1./2**nbits

    fig = plt.figure(figsize=(8, 10))

    ax = fig.add_subplot(211)
    ax.plot(ptree.w, 'bo', label='desired')
    ax.plot(ptree.w_apx, 'ro', label='tree approximation')
    ax.plot(w_emp, 'go', label='measured')
    ax.legend(loc='best')
    ax.set_title('probability distributions')
    ax.set_xlim(0, len(ptree.w))
    ax.set_xticklabels([])

    ax = fig.add_subplot(212)
    ax.plot(apx_error, 'ro')
    ax.plot(emp_error, 'go')
    ax.axhline(q_level, color='k')
    ax.set_title('distribution errors')
    ax.set_xlim(0, len(ptree.w))

    if title is not None:
        fig.suptitle(title, fontsize=18)


nbits = 8
# uniform distribution
for n in [8, 16, 32, 64, 128]:
    uniform = np.ones(n)/n
    ptree_experiment(nbits, uniform, title='uniform%d' % n)

plt.show()


from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import scipy.stats as stats

from fitting.visual import fontdict
from fitting.utils import uniform_exp


path_out = Path().cwd()/'results/distribution'
path_out.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(nrows=1,  ncols=2, figsize=(9, 5))

# ------------ Log Normal ------------
x = np.linspace(0, 500, int(1e4))
y = stats.lognorm.pdf(x, s=0.66, loc=5, scale=70)
# y = stats.lognorm.cdf(x, s=0.66, loc=5, scale=70)
print(stats.lognorm.cdf(x=210, s=0.66, loc=5, scale=70))
print(stats.lognorm.cdf(x=500, s=0.66, loc=5, scale=70))
print(stats.lognorm.ppf(q=[0, 0.95], s=0.66, loc=5, scale=70))


axis = ax[0]
axis.plot(x, y)
axis.set_ylabel(r'Probability Density Function', fontdict=fontdict)
axis.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=2))
axis.set_xlabel(r'$\mathbf{T2}$ [ms]', fontdict=fontdict)


# ------------- Uniform Exponential ---------
x = np.linspace(-100, 2500, int(1e4))
pdf = np.vectorize(uniform_exp.pdf)
print("Para A", uniform_exp.ppf(q=[0.01, 0.95]))
print("Para A", uniform_exp.ppf(q=[0.00001, 0.99]))
print("Para A CDF: ", uniform_exp.cdf(500) )
axis = ax[1]
axis.plot(x, pdf(x))
axis.set_ylabel('Probability Density Function', fontdict=fontdict)
axis.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=2))
axis.set_xlabel(r'$\mathbf{S_0}$', fontdict=fontdict)


print(uniform_exp.ppf(q=[0.05, 0.95]))


fig.tight_layout()
fig.savefig(path_out/'params_dist.png', dpi=300)
fig.savefig(path_out/'params_dist.tiff', dpi=300)

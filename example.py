from processes import GaussianProcess, TProcess
import george
import numpy as np
import matplotlib.pyplot as plt

kernel = 0.01*george.kernels.ExpSquaredKernel(1/20.)
mean = lambda x: np.cos(x)
gp = GaussianProcess(kernel,)# mean=mean)
tp = TProcess(kernel, df=2.1, )#mean=mean)


t = np.linspace(0, 3, 100)
fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)
ax1.plot(t, gp.sample(t, size=100).T, alpha=0.4)
ax2.plot(t, tp.sample(t, size=100).T, alpha=0.4)


def data1():
    return np.array([1.2, 2.8]), np.array([0.5, -1])

def data2():
    x = np.sort(np.random.uniform(0, 3, size=15))
    y = gp.sample(x)
    y[5] -= 2 # outlier
    return x, y


x, y = data1()


gp.compute(x)
ax3.plot(t, gp.sample_conditional(y, t, size=10).T, alpha=0.5)
mu, cov = gp.predict(y, t)
std = np.sqrt(np.diag(cov))
ax3.fill_between(x=t, y1=mu-2*std, y2=mu+2*std, alpha=0.5, color='k')
ax3.plot(x, y, 'o')

tp.compute(x)
ax4.plot(t, tp.sample_conditional(y, t, size=10).T, alpha=0.5)
newdf, mu, cov = tp.predict(y, t)
std = np.sqrt(np.diag(cov))
ax4.fill_between(x=t, y1=mu-2*std, y2=mu+2*std, alpha=0.4, color='k')
ax4.plot(x, y, 'o')

ax1.set_ylim(-4, 4)
plt.show()
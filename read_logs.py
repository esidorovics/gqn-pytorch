import pandas as pd
import matplotlib.pyplot as plt
# df = pd.read_csv("main.log")
df = pd.read_csv("~/data/gqn/10/main.log")
# df = pd.read_csv("~/data/gqn/6/main.log")
print(df.shape)
print(df.columns)
df = df[df["step"]>150000]
step = df["step"].to_list()
ll = df["ll"].to_list()
kl = df["kl"].to_list()
elbo = df['elbo'].to_list()
sigma = df['sigma'].to_list()
mu = df['mu'].to_list()
kl_av = []
ll_av = []
elbo_av = []

average_step = 1000
for i in range(average_step, len(step)):
    kl_av.append(sum(kl[i-average_step:i])/average_step)
    ll_av.append(sum(ll[i-average_step:i])/average_step)
    elbo_av.append(sum(elbo[i-average_step:i])/average_step)
step = step[average_step:]
sigma = sigma[average_step:]
mu =mu[average_step:]

plt.subplot(231)
plt.plot(step, elbo_av)

plt.subplot(232)
plt.plot(step, ll_av)

plt.subplot(233)
plt.plot(step, kl_av)

plt.subplot(234)
plt.plot(step, mu)
plt.subplot(235)
plt.plot(step, sigma)
plt.show()

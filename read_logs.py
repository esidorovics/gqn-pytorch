import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("main.log")
print(df.shape)
print(df.columns)
df = df[df["step"]>20000]
step = df["step"].to_list()
ll = df["ll"].to_list()
kl = df["kl"].to_list()
kl_av = []
ll_av = []

average_step = 100
for i in range(average_step, len(step)):
    kl_av.append(sum(kl[i-average_step:i])/average_step)
    ll_av.append(sum(ll[i-average_step:i])/average_step)
step = step[average_step:]

l2log = pd.read_csv("l2.log")
step2 = l2log['step'].to_list()
l2 = l2log['l2'].to_list()

plt.subplot(221)
plt.plot(step, ll_av)
# plt.plot(step, ll)
plt.subplot(222)
# plt.plot(step, kl)
plt.plot(step, kl_av)
plt.subplot(223)
plt.plot(step2, l2)
plt.show()

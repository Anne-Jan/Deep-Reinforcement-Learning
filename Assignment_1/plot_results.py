import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

# Loading results
# 2 is best, 5 is also fine, rest are bad
results_text = np.load('results/2.npy')
results = []


for line in results_text:
    print(line)
    results.append(float(line[-3:]))
print(len(results))
print(max(results))

# Plotting results
# fig = plt.figure(figsize=(10, 10))
# gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1])
# ax3 = plt.subplot(gs[2])
# ax4 = plt.subplot(gs[3])
plt.plot(results)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Average Reward of 10 Testing Runs After 10 Training Episodes')
plt.show()


# Get Average of all results
avg_rewards = []

results_text1 = np.load('results/1.npy')
results_text2 = np.load('results/2.npy')
results_text3 = np.load('results/3.npy')
results_text4 = np.load('results/4.npy')
results_text5 = np.load('results/7.npy')

results1 = []
results2 = []
results3 = []
results4 = []
results5 = []
for line in results_text1:
    results1.append(float(line[-3:]))
for line in results_text2:
    results2.append(float(line[-3:])) 
for line in results_text3:
    results3.append(float(line[-3:]))
for line in results_text4:
    results4.append(float(line[-3:]))
for line in results_text5:
    results5.append(float(line[-3:]))
avg_rewards.append(np.mean(results1[-100:]))
avg_rewards.append(np.mean(results2[-100:]))
avg_rewards.append(np.mean(results3[-100:]))
avg_rewards.append(np.mean(results4[-100:]))
avg_rewards.append(np.mean(results5[-50:]))
print(avg_rewards)
import matplotlib.pyplot as plt
import csv

# Read data
x = []
y = []
csv_path = 'results/breakout_length.csv'
with open(csv_path, 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        if row[0] == "Step":
            continue
        x.append(int(row[0]))
        y.append(float(row[1]))

# Plot data
plt.plot(x, y)
plt.xlabel('Episode')
plt.ylabel('Mean Episode Length')
plt.title('Mean episode length per 30000 episodes')
plt.show()
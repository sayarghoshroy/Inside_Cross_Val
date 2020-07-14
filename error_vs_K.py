import numpy as np
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt

data_100 = []
for i in range(100):
	data_100.append(i)
data_100 = np.array(data_100)

data_10000 = []
for i in range(10000):
	data_10000.append(i)
data_10000 = np.array(data_10000)

points_100 = []
points_10000 = []
mu = 0
sigma = 1
error_100 = np.random.normal(mu, sigma, 100)
error_10000 = np.random.normal(mu, sigma, 10000)

for x in range(100):
	points_100.append((x, math.sin(x) + error_100[x]))

for x in range(10000):
	points_10000.append((x, math.sin(x) + error_10000[x]))

folds = []
for i in range(2, 100):
	if(100 % i == 0):
		folds.append(i)

mean_var_100 = []

for fold_num in folds:
	kfold = KFold(fold_num, True, 1)

	fold_means = []

	for train, test in kfold.split(data_100):
		sample_errs = [error_100[i] for i in data_100[train]]
		fold_means.append(np.mean(np.array(sample_errs)))

	mean = np.mean(fold_means)
	var = np.var(fold_means)
	mean_var_100.append((fold_num, mean, var))

print(mean_var_100)

folds = []
for i in range(2, 10000):
	if(10000 % i == 0):
		folds.append(i)

mean_var_10000 = []

for fold_num in folds:
	kfold = KFold(fold_num, True, 1)

	fold_means = []

	for train, test in kfold.split(data_10000):
		sample_errs = [error_10000[i] for i in data_10000[train]]
		fold_means.append(np.mean(np.array(sample_errs)))

	mean = np.mean(fold_means)
	var = np.var(fold_means)
	mean_var_10000.append((fold_num, mean, var))

print(mean_var_10000)

# Plotting stdevs
x = [a[0] for a in mean_var_100]
y_dev = [a[2] for a in mean_var_100]
y_mean = [a[1] for a in mean_var_100]

x_b = [a[0] for a in mean_var_10000]
y_b_dev = [a[2] for a in mean_var_10000]
y_b_mean = [a[1] for a in mean_var_10000]

fig = plt.figure()

p_1 = fig.add_subplot(221, title='100 Points: Error Variance v/s K - Using all values of K')
p_1.plot(x, y_dev, 'o', linestyle = '--', color = 'r')
p_2 = fig.add_subplot(222, title='100 Points: Error Mean v/s K - Using all values of K')
p_2.plot(x, y_mean,'o', linestyle = '--', color = 'r')


p_3 = fig.add_subplot(223, title='10000 Points: Error Variance v/s K')
p_3.plot(x_b, y_b_dev, 'o', linestyle = '--', color = 'r')
p_4 = fig.add_subplot(224, title='10000 Points: Error Mean v/s K')
p_4.plot(x_b, y_b_mean,'o', linestyle = '--', color = 'r')

plt.show()

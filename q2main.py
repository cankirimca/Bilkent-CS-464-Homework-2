import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_features = pd.read_csv('question-2-features.csv')
features = df_features.to_numpy()
df_labels = pd.read_csv('question-2-labels.csv')
labels = df_labels.to_numpy()

print("----------Question 2.2-----------")

ones = np.full((506,1),1)
x = np.append(ones,features, axis=1)


product = x.transpose() @ x
print("Rank:" , np.linalg.matrix_rank(product))

print("----------Question 2.3-----------")
lstat = features[:,-1]
lstat_with_ones = np.vstack((np.ones(506), lstat)).transpose()
coefs = np.linalg.inv(lstat_with_ones.transpose() @ lstat_with_ones) @ lstat_with_ones.transpose() @ labels
print("Coefficients:", coefs)
predictions = []
total_sq_error = 0
for index, label in enumerate(labels):
  prediction = lstat_with_ones[index] @ coefs
  predictions.append(prediction)
  total_sq_error += pow(prediction - label, 2)
mse = total_sq_error/len(labels)
print("MSE:", mse)

"""
fig, ax = plt.subplots()
ax.scatter(lstat, labels, color='blue')
ax.scatter(lstat, predictions, color='red')
ax.legend(["Ground Truth", "Prediction"])
plt.xlabel("LSTAT")
plt.ylabel("Price")
plt.title("LSTAT vs Price for Question 2.3")
plt.show()
"""

print("----------Question 2.4-----------")

lstat_with_squares = np.vstack((lstat_with_ones.transpose(), np.square(lstat))).transpose()

coefs = np.linalg.inv(lstat_with_squares.transpose() @ lstat_with_squares) @ lstat_with_squares.transpose() @ labels
print("Coefficients:", coefs)
predictions = []
total_sq_error = 0
for index, label in enumerate(labels):
  prediction = lstat_with_squares[index] @ coefs
  predictions.append(prediction)
  total_sq_error += pow(prediction - label, 2)
mse = total_sq_error/len(labels)
print("MSE:", mse)

"""
fig, ax = plt.subplots()
ax.scatter(lstat, labels, color='blue')
ax.scatter(lstat, predictions, color='red')
ax.legend(["Ground Truth", "Prediction"])
plt.xlabel("LSTAT")
plt.ylabel("Price")
plt.title("LSTAT vs Price for Question 2.4")
plt.show()
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, learning_curve
import matplotlib.pyplot as plt

# To ignore deprecated modules/functions warnings
import warnings
warnings.filterwarnings("ignore")

# Load datasets
train = pd.read_csv('train-data.csv')
train_label = pd.read_csv('train-targets.csv')
test = pd.read_csv('test-data.csv')
targets = pd.read_csv('test-targets.csv')

# Divide train and validation
X_train, X_test, y_train, y_test = train_test_split(train, train_label, test_size=0.2, random_state=1)

# Instantiate KFold
kf = KFold(n_splits=3, shuffle=False, random_state=42)

# Algorithm selected
alg = SVC(kernel='rbf')

# Parameters for the GridSearch
parameters = {
	'gamma': [1, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001, 0.0005, 0.0001],
	'C': [1e-1, 1e0, 1e1, 1e2, 1e3]
}

# Train the GridSearch
grid = GridSearchCV(alg, parameters, cv=kf)
grid.fit(X_train, y_train)

pred = grid.predict(X_test)
rep = metrics.classification_report(y_test, pred)

print(grid.best_estimator_)
print(rep)
print("Accuracy: ", metrics.accuracy_score(y_test, pred))


#################################
# PLOT THE LEARNING CURVE
#################################

plt.figure()
plt.title("Learning curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()


# Compute the scores of the learning curve
# by default the (relative) dataset sizes are: 10%, 32.5%, 55%, 77.5%, 100%
# The function automatuically executes a Kfold cross validation for each dataset size
train_sizes, train_scores, val_scores = learning_curve(grid.best_estimator_, X_train, y_train, scoring='accuracy', cv=3)

# Get the mean and std of train and validation scores over the cv folds along the varying dataset sizes
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Plot the mean  for the training scores
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

# Plot the  std for the training scores
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")

# Plot the mean  for the validation scores
plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

# Plot the std for the validation scores
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")

# Set bottom and top limits for y axis
plt.ylim(0.05,1.3)
plt.legend()
plt.show()


# Compute results on test set
chosen = grid.best_estimator_
chosen.fit(train, train_label)
test_pred = chosen.predict(test)
pd.DataFrame(test_pred).to_csv('test-pred.txt', index=False, header=None)
print("Accuracy on test: ", metrics.accuracy_score(targets, test_pred))
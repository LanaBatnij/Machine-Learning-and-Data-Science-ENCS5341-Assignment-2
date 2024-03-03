# Lana Batnij __ 1200308

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# PART 1
# Question 1

# Load file and read it
filePath = r"C:\Users\LENOVO\Downloads\data_reg.csv"
data = pd.read_csv(filePath)

# Divide the data into three sets: training, validation, and testing.
train_data, test_data = train_test_split(data, test_size=40, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=40, random_state=42)

# Scatter plot for training set
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_data['x1'], train_data['x2'], train_data['y'], c='black', marker='^', label='Training Set')

# Scatter plot for validation set
ax.scatter(val_data['x1'], val_data['x2'], val_data['y'], c='hotpink', marker='^', label='Validation Set')

# Scatter plot for testing set
ax.scatter(test_data['x1'], test_data['x2'], test_data['y'], c='purple', marker='^', label='Testing Set')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.show()
# --------------------------------------------------------------
# Question 2

# Establish a training set using features and target labels.
X_train = train_data[['x1', 'x2']].values
y_train = train_data['y'].values

# do the same for validation set
X_val = val_data[['x1', 'x2']].values
y_val = val_data['y'].values

#  For plotting the learned function's surface beside training examples
def plot_polynomial_surface(X, y, model, degree):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], y, c='black', marker='^', label='Training Set')

    # Construct a surface on the graph plot
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    X_mesh = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
    X_mesh_poly = PolynomialFeatures(degree=degree).fit_transform(X_mesh)
    y_mesh = model.predict(X_mesh_poly)
    y_mesh = y_mesh.reshape(x1_mesh.shape)

    # The learnt function surface plot
    ax.plot_surface(x1_mesh, x2_mesh, y_mesh, alpha=0.5, cmap='cividis', label=f'Degree {degree} Fit')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.legend()
    plt.title(f'Polynomial Regression - Degree {degree}')
    plt.show()

# Polynomial regression calculation and plotting function
def polynomial_regression(X_train, y_train, X_val, y_val, degrees):
    val_errors = []

    for degree in degrees:
        # Change features as polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        # Establish a linear regression model.
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        # Validation set predictions
        y_val_pred = model.predict(X_val_poly)

        # On the validation set, compute the mean squared error.
        val_error = mean_squared_error(y_val, y_val_pred)
        val_errors.append(val_error)

        plot_polynomial_surface(X_train, y_train, model, degree)

    # Plot validation error vs polynomial degree curve
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, val_errors, marker='^', color='purple')
    plt.title('Validation Error vs Polynomial Degree')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error (Validation)')
    plt.grid(True)
    plt.show()

# Degrees for polynomial regression
degrees = range(1, 11)

# Polynomial_regression function
polynomial_regression(X_train, y_train, X_val, y_val, degrees)

# --------------------------------------------------------------
# Question 3

# Polynomial features of degree 8
poly = PolynomialFeatures(degree=8)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

# To test regularization parameters
alphas = [0.001, 0.005, 0.01, 0.1, 10]

mse_values = []
for alpha in alphas:
    # Ridge regression model
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, y_train)

    y_val_pred = model.predict(X_val_poly)

    mse = mean_squared_error(y_val, y_val_pred)
    mse_values.append(mse)

# Plot MSE on validation vs regularization parameter
plt.figure(figsize=(8, 5))
plt.plot(alphas, mse_values, marker='^', color='purple')
plt.xscale('log')  # Use a logarithmic scale for better visualization
plt.title('MSE on Validation vs Regularization Parameter')
plt.xlabel('Regularization Parameter (alpha)')
plt.ylabel('Mean Squared Error (Validation)')
plt.grid(True)
plt.show()

# The best alpha
best_alpha = alphas[np.argmin(mse_values)]
print(f'Best Regularization Parameter (alpha): {best_alpha}')
# --------------------------------------------------------------
# PART 2
# Question 1

# Get the classification data and read it
train_cls_path = r"C:\Users\LENOVO\Downloads\train_cls.csv"
test_cls_path = r"C:\Users\LENOVO\Downloads\test_cls.csv"

train_cls_data = pd.read_csv(train_cls_path)
test_cls_data = pd.read_csv(test_cls_path)

print("Columns of train_cls_data:", train_cls_data.columns)

# Features should be collected for the training set
X_train_cls = train_cls_data[['x1', 'x2']].values

label_encoder = LabelEncoder()
y_train_cls = label_encoder.fit_transform(train_cls_data['class'].values)

# Do the same for testing set
X_test_cls = test_cls_data[['x1', 'x2']].values

y_test_cls = label_encoder.transform(test_cls_data['class'].values)

# Develop and apply the logistic regression model
model = LogisticRegression()
model.fit(X_train_cls, y_train_cls)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(X_train_cls[y_train_cls == 0][:, 0], X_train_cls[y_train_cls == 0][:, 1], color='hotpink', label='Class 0 (Training)', marker='o')
plt.scatter(X_train_cls[y_train_cls == 1][:, 0], X_train_cls[y_train_cls == 1][:, 1], color='purple', label='Class 1 (Training)', marker='^')
plt.title('Training Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test_cls[y_test_cls == 0][:, 0], X_test_cls[y_test_cls == 0][:, 1], color='hotpink', label='Class 0 (Testing)', marker='o')
plt.scatter(X_test_cls[y_test_cls == 1][:, 0], X_test_cls[y_test_cls == 1][:, 1], color='purple', label='Class 1 (Testing)', marker='^')
plt.title('Testing Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.tight_layout()
plt.show()

# The decision border on the training set's scatter plot.
plt.figure(figsize=(10, 6))
plt.scatter(X_train_cls[y_train_cls == 0][:, 0], X_train_cls[y_train_cls == 0][:, 1], color='hotpink', label='Class 0', marker='o')
plt.scatter(X_train_cls[y_train_cls == 1][:, 0], X_train_cls[y_train_cls == 1][:, 1], color='purple', label='Class 1', marker='^')

# Decision boundary
x_min, x_max = X_train_cls[:, 0].min() - 1, X_train_cls[:, 0].max() + 1
y_min, y_max = X_train_cls[:, 1].min() - 1, X_train_cls[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Make sure that Z is of the correct data type
Z = Z.astype(float)

# Determine whether all predictions are the same; if so, the contour plot will not work.
if np.all(Z == Z[0]):
    print("Contour plot not possible; all predictions are the same.")
else:
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='black', levels=[0], linewidths=2, label='Decision Boundary (Linear)')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Training accuracy calculations
y_train_cls_pred = model.predict(X_train_cls)
train_cls_accuracy = accuracy_score(y_train_cls, y_train_cls_pred)
print(f'Training Accuracy: {train_cls_accuracy}')

# Testing accuracy calculations
y_test_cls_pred = model.predict(X_test_cls)
test_cls_accuracy = accuracy_score(y_test_cls, y_test_cls_pred)
print(f'Testing Accuracy: {test_cls_accuracy}')
# --------------------------------------------------------------
# Question 2

# Generate polynomial features degree=2.
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_cls)
X_test_poly = poly.transform(X_test_cls)

model = LogisticRegression()
model.fit(X_train_poly, y_train_cls)

plt.figure(figsize=(10, 6))
plt.scatter(X_train_cls[y_train_cls == 0][:, 0], X_train_cls[y_train_cls == 0][:, 1], color='hotpink', label='Class 0', marker='o')
plt.scatter(X_train_cls[y_train_cls == 1][:, 0], X_train_cls[y_train_cls == 1][:, 1], color='purple', label='Class 1', marker='^')

# Decision boundary
x_min, x_max = X_train_cls[:, 0].min() - 1, X_train_cls[:, 0].max() + 1
y_min, y_max = X_train_cls[:, 1].min() - 1, X_train_cls[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
X_contour = np.c_[xx.ravel(), yy.ravel()]
X_contour_poly = poly.transform(X_contour)
Z = model.predict(X_contour_poly)
Z = Z.astype(float)
if np.all(Z == Z[0]):
    print("Contour plot not possible; all predictions are the same.")
else:
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='black', levels=[0], linewidths=2, label='Decision Boundary (Quadratic)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

y_train_cls_pred = model.predict(X_train_poly)
train_cls_accuracy = accuracy_score(y_train_cls, y_train_cls_pred)
print(f'Training Accuracy: {train_cls_accuracy}')

y_test_cls_pred = model.predict(X_test_poly)
test_cls_accuracy = accuracy_score(y_test_cls, y_test_cls_pred)
print(f'Testing Accuracy: {test_cls_accuracy}')

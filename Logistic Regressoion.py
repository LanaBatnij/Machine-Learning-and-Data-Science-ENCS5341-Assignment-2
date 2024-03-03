import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the classification data
train_cls_path = r"C:\Users\LENOVO\Downloads\train_cls.csv"
test_cls_path = r"C:\Users\LENOVO\Downloads\test_cls.csv"

train_cls_data = pd.read_csv(train_cls_path)
test_cls_data = pd.read_csv(test_cls_path)

# Print columns of train_cls_data
print("Columns of train_cls_data:", train_cls_data.columns)

# Extract features for training set (classification)
X_train_cls = train_cls_data[['x1', 'x2']].values

# Encode the target labels
label_encoder = LabelEncoder()
y_train_cls = label_encoder.fit_transform(train_cls_data['class'].values)

# Extract features for testing set (classification)
X_test_cls = test_cls_data[['x1', 'x2']].values

# Encode the target labels
y_test_cls = label_encoder.transform(test_cls_data['class'].values)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train_cls, y_train_cls)

# Plot training set
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_train_cls[y_train_cls == 0][:, 0], X_train_cls[y_train_cls == 0][:, 1], color='hotpink', label='Class 0 (Training)', marker='o')
plt.scatter(X_train_cls[y_train_cls == 1][:, 0], X_train_cls[y_train_cls == 1][:, 1], color='purple', label='Class 1 (Training)', marker='^')
plt.title('Training Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot testing set
plt.subplot(1, 2, 2)
plt.scatter(X_test_cls[y_test_cls == 0][:, 0], X_test_cls[y_test_cls == 0][:, 1], color='hotpink', label='Class 0 (Testing)', marker='o')
plt.scatter(X_test_cls[y_test_cls == 1][:, 0], X_test_cls[y_test_cls == 1][:, 1], color='purple', label='Class 1 (Testing)', marker='^')
plt.title('Testing Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.show()

# Plot decision boundary on scatter plot of the training set
plt.figure(figsize=(10, 6))
plt.scatter(X_train_cls[y_train_cls == 0][:, 0], X_train_cls[y_train_cls == 0][:, 1], color='hotpink', label='Class 0', marker='o')
plt.scatter(X_train_cls[y_train_cls == 1][:, 0], X_train_cls[y_train_cls == 1][:, 1], color='purple', label='Class 1', marker='^')

# Decision boundary
x_min, x_max = X_train_cls[:, 0].min() - 1, X_train_cls[:, 0].max() + 1
y_min, y_max = X_train_cls[:, 1].min() - 1, X_train_cls[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Ensure Z is of the correct data type
Z = Z.astype(float)

# Check if all predictions are the same; if so, contour plot won't work
if np.all(Z == Z[0]):
    print("Contour plot not possible; all predictions are the same.")
else:
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='black', levels=[0], linewidths=2, label='Decision Boundary (Linear)')

# Set labels for axes
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Compute training accuracy
y_train_cls_pred = model.predict(X_train_cls)
train_cls_accuracy = accuracy_score(y_train_cls, y_train_cls_pred)
print(f'Training Accuracy: {train_cls_accuracy}')

# Compute testing accuracy
y_test_cls_pred = model.predict(X_test_cls)
test_cls_accuracy = accuracy_score(y_test_cls, y_test_cls_pred)
print(f'Testing Accuracy: {test_cls_accuracy}')


#------------------------------------------------------------------------
# Create polynomial features (degree=2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_cls)
X_test_poly = poly.transform(X_test_cls)

# Create and fit the logistic regression model with quadratic decision boundary
model = LogisticRegression()
model.fit(X_train_poly, y_train_cls)

# Plot decision boundary on scatter plot of the training set
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

# Ensure Z is of the correct data type
Z = Z.astype(float)

# Check if all predictions are the same; if so, contour plot won't work
if np.all(Z == Z[0]):
    print("Contour plot not possible; all predictions are the same.")
else:
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='black', levels=[0], linewidths=2, label='Decision Boundary (Quadratic)')

# Set labels for axes
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Compute training accuracy
y_train_cls_pred = model.predict(X_train_poly)
train_cls_accuracy = accuracy_score(y_train_cls, y_train_cls_pred)
print(f'Training Accuracy: {train_cls_accuracy}')

# Compute testing accuracy
y_test_cls_pred = model.predict(X_test_poly)
test_cls_accuracy = accuracy_score(y_test_cls, y_test_cls_pred)
print(f'Testing Accuracy: {test_cls_accuracy}')

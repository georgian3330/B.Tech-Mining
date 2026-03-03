import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time

# 1. Import Dataset 
housing = fetch_california_housing()
X = housing.data[:10000, [0]]  # Only MedianIncome feature
y = housing.target[:10000].reshape(-1, 1)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias term
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

# 2. Cost Function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# 3. Batch Gradient Descent 
def batch_gradient_descent(X, y, theta, learning_rate=0.1, iterations=100):
    m = len(y)
    cost_history = []
    theta_path = []
    for i in range(iterations):
        gradients = (1/m) * X.T @ (X @ theta - y)
        theta = theta - learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
        theta_path.append(theta.copy())
    return theta, cost_history, theta_path

# 4. Stochastic Gradient Descent 
def stochastic_gradient_descent(X, y, theta, learning_rate=0.1, epochs=100):
    m = len(y)
    cost_history = []
    theta_path = []
    for epoch in range(epochs):
        for i in range(m):
            rand_index = np.random.randint(m)
            xi = X[rand_index:rand_index+1]
            yi = y[rand_index:rand_index+1]
            gradients = xi.T @ (xi @ theta - yi)
            theta = theta - learning_rate * gradients
        cost_history.append(compute_cost(X, y, theta))
        theta_path.append(theta.copy())
    return theta, cost_history, theta_path

# 5. Normal Equation (with timing)
start_time = time.time()
theta_normal = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train
ne_time = time.time() - start_time

# 6. Train Models (with timing)
theta_init = np.random.randn(2, 1)

#Compute initial cost using initial theta
initial_cost = compute_cost(X_train_b, y_train, theta_init.copy())

# Batch GD (timed)
start_time = time.time()
theta_bgd, cost_bgd, path_bgd = batch_gradient_descent(
    X_train_b, y_train, theta_init.copy(), learning_rate=0.1, iterations=100
)
bgd_time = time.time() - start_time

# SGD (timed)
start_time = time.time()
theta_sgd, cost_sgd, path_sgd = stochastic_gradient_descent(
    X_train_b, y_train, theta_init.copy(), learning_rate=0.1, epochs=100
)
sgd_time = time.time() - start_time

# Prepend initial cost to both histories
cost_bgd = [initial_cost] + cost_bgd
cost_sgd = [initial_cost] + cost_sgd

# 7. Test Models
y_pred_bgd = X_test_b @ theta_bgd
y_pred_sgd = X_test_b @ theta_sgd
y_pred_ne = X_test_b @ theta_normal

print("\n=== Performance Metrics (10,000 samples) ===")
print(f"MSE (Batch GD): {mean_squared_error(y_test, y_pred_bgd):.4f}")
print(f"MSE (SGD): {mean_squared_error(y_test, y_pred_sgd):.4f}")
print(f"MSE (Normal Eq.): {mean_squared_error(y_test, y_pred_ne):.4f}")

print("\n=== Execution Time ===")
print(f"Batch GD: {bgd_time:.4f} seconds")
print(f"SGD: {sgd_time:.4f} seconds")
print(f"Normal Equation: {ne_time:.4f} seconds")

# 8. Plot 3D Cost Function
theta0_vals = np.linspace(-1, 5, 100)
theta1_vals = np.linspace(-1, 5, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        t = np.array([[t0], [t1]])
        J_vals[i, j] = compute_cost(X_train_b, y_train, t)

T0, T1 = np.meshgrid(theta0_vals, theta1_vals)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T0, T1, J_vals.T, cmap='viridis', alpha=0.8)
ax.set_xlabel('Theta 0 (Bias)')
ax.set_ylabel('Theta 1 (Weight)')
ax.set_zlabel('Cost')
ax.set_title('3D Cost Function Surface (10,000 samples)')
plt.show()

# 9. Plot GD Path on Cost Surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T0, T1, J_vals.T, cmap='viridis', alpha=0.6)
bgd_t0 = [t[0,0] for t in path_bgd]
bgd_t1 = [t[1,0] for t in path_bgd]
bgd_cost = [compute_cost(X_train_b, y_train, t) for t in path_bgd]
ax.plot(bgd_t0, bgd_t1, bgd_cost, marker='o', color='red', label='BGD Path')
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Cost')
ax.set_title('BGD Optimization Path')
ax.legend()
plt.show()

# 10. Plot SGD Path on Cost Surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T0, T1, J_vals.T, cmap='viridis', alpha=0.6)
sgd_t0 = [t[0,0] for t in path_sgd]
sgd_t1 = [t[1,0] for t in path_sgd]
sgd_cost = [compute_cost(X_train_b, y_train, t) for t in path_sgd]
ax.plot(sgd_t0, sgd_t1, sgd_cost, marker='o', color='orange', label='SGD Path')
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Cost')
ax.set_title('SGD Optimization Path')
ax.legend()
plt.show()

# 11. Plot Error Convergence
plt.figure(figsize=(8,5))
plt.plot(range(len(cost_bgd)), cost_bgd, label=f'Batch GD')
plt.plot(range(len(cost_sgd)), cost_sgd, label=f'SGD')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Training Steps')
plt.legend()
plt.grid(True)
plt.show()
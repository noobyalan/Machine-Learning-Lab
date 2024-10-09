# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def read_data(addr):
    data = np.loadtxt(addr, delimiter=',')
    n = data.shape[0]
    
    ###### You may modify this section to change the model
    X = np.concatenate([np.ones([n, 1]), data[:,0:6]], axis=1)
    ###### You may modify this section to change the model
    
    Y = None
    if "train" in addr:
        Y = np.expand_dims(data[:, 6], axis=1)
    
    return (X, Y, n)

def cost_gradient(W, X, Y, n):
    # Logistic regression cost and gradient
    Y_hat = 1 / (1 + np.exp(-X @ W))
    j = -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / n
    G = X.T @ (Y_hat - Y) / n
    return (j, G)

def error(W, X, Y):
    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat >= 0.5] = 1
    return (1 - np.mean(np.equal(Y_hat, Y)))

def train(W, X, Y, lr, n, iterations):
    fold_size = n // 10
    J_all = []
    E_trn_all = []
    E_val_all = []

    for fold in range(10):
        X_val = X[fold * fold_size:(fold + 1) * fold_size]
        Y_val = Y[fold * fold_size:(fold + 1) * fold_size]
        X_trn = np.concatenate([X[:fold * fold_size], X[(fold + 1) * fold_size:]])
        Y_trn = np.concatenate([Y[:fold * fold_size], Y[(fold + 1) * fold_size:]])
        
        J = np.zeros([iterations, 1])
        E_trn = np.zeros([iterations, 1])
        E_val = np.zeros([iterations, 1])
        W_fold = W.copy()

        for i in range(iterations):
            (J[i], G) = cost_gradient(W_fold, X_trn, Y_trn, X_trn.shape[0])
            W_fold = W_fold - lr * G
            E_trn[i] = error(W_fold, X_trn, Y_trn)
            E_val[i] = error(W_fold, X_val, Y_val)
        
        J_all.append(J)
        E_trn_all.append(E_trn)
        E_val_all.append(E_val)
    
    # Average the validation error across folds
    avg_E_val = np.mean(np.array(E_val_all), axis=0)
    print(f"Final Validation Error: {avg_E_val[-1]}")

    return (W, J_all, E_trn_all, E_val_all)

def predict(W):
    (X, _, _) = read_data("test_data.csv")
    Y_hat = 1 / (1 + np.exp(-X @ W))
    Y_hat[Y_hat < 0.5] = 0
    Y_hat[Y_hat >= 0.5] = 1
    idx = np.expand_dims(np.arange(1, 201), axis=1)
    np.savetxt("predict.csv", np.concatenate([idx, Y_hat], axis=1), header="Index,ID", comments='', delimiter=',')

# Training parameters
iterations = 1000  # Set appropriate training loops
lr = 0.001  # Set appropriate learning rate

# Reading data
(X, Y, n) = read_data("train.csv")
W = np.random.random([X.shape[1], 1])

# Train with 10-fold validation
(W, J_all, E_trn_all, E_val_all) = train(W, X, Y, lr, n, iterations)

# Plotting loss and error
plt.figure()
plt.plot(range(iterations), np.mean(np.array(J_all), axis=0))
plt.figure()
plt.ylim(0, 1)
plt.plot(range(iterations), np.mean(np.array(E_trn_all), axis=0), "b", label='Training Error')
plt.plot(range(iterations), np.mean(np.array(E_val_all), axis=0), "r", label='Validation Error')
plt.legend()
plt.show()

# Prediction on test data
predict(W)

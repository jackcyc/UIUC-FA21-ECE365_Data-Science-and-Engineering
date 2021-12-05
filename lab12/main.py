import numpy as np


class Question1(object):
    def pca_reduce_dimen(self, X):
        k = 2
        n_data = X.shape[1]
        out = np.zeros((k, n_data))

        Xmean = X.mean(axis=1).reshape((X.shape[0], 1))
        C = (X - Xmean) @ (X - Xmean).T / n_data
        # eigen decomposition
        eigens, U = np.linalg.eigh(C)
        W = np.fliplr(U).T

        W = W[:k, :]
        out = W @ X
        return out

    def pca_project(self, X, k):
        n_data = X.shape[1]
        pca_reduced = np.zeros((k, n_data))

        Xmean = X.mean(axis=1).reshape((X.shape[0], 1))
        C = (X - Xmean) @ (X - Xmean).T / n_data
        # eigen decomposition
        eigens, U = np.linalg.eigh(C)
        W = np.fliplr(U).T
        W = W[:k, :]
        pca_reduced = W @ X  # shape = (k, n)

        # reconstruct: pca -> origin
        filtered = np.zeros((X.shape[0], n_data))
        filtered = W.T @ pca_reduced  # shape = (dim, n)

        return filtered


class Question2(object):
    def wiener_filter(self, data_noisy, C, mu, sigma):
        dim = data_noisy.shape[0]
        var = sigma**2

        # # eigen decomposition
        # eig_val, eig_vec = np.linalg.eigh(C)
        # U_ = eig_vec.copy()
        # U_ /= (1 + (var / eig_val))
        # filtered = ((data_noisy.T - mu.reshape(1, dim)) @ eig_vec @ U_.T).T + mu.reshape(dim, 1)

        #
        filtered = mu + C @ np.linalg.inv(C + var * np.identity(dim)) @ (data_noisy - mu)

        return filtered


class Question3(object):
    def embedding(self, A):
        eig_val, eig_vec = np.linalg.eigh(A)
        eig_val = eig_val[::-1]
        eig_vec = np.fliplr(eig_vec)

        return eig_vec, eig_val

import numpy as np

class Question1(object):
    def pca(self,data):
        """ Implement PCA via the eigendecomposition or the SVD.

        Parameters:
        1. data     (N,d) numpy ndarray. Each row as a feature vector.

        Outputs:
        1. W        (d,d) numpy array. PCA transformation matrix (Note that each **row** of the matrix should be a principal component)
        2. s        (d,) numpy array. Vector consisting of the amount  of variance explained in the data by each PCA feature.
        Note that the PCA features are ordered in **decreasing** amount of variance explained, by convention.
        """
        W = np.zeros((data.shape[1],data.shape[1]))
        s = np.zeros(data.shape[1])
        # Put your code below


        return (W,s)

    def pcadimreduce(self,data,W,k):
        """ Implements dimension reduction via PCA.

        Parameters:
        1. data     (N,d) numpy ndarray. Each row as a feature vector.
        2. W        (d,d) numpy array. PCA transformation matrix
        3. k        number. Number of PCA features to retain

        Outputs:
        1. reduced_data  (N,k) numpy ndarray, where each row contains PCA features corresponding to its input feature.
        """
        reduced_data = np.zeros((data.shape[0],k))
        # Put your code below


        return reduced_data

    def pcareconstruct(self,pcadata,W):
        """ Implements dimension reduction via PCA.

        Parameters:
        1. pcadata  (N,k) numpy ndarray. Each row as a PCA vector. (e.g. generated from pcadimreduce)
        2. W        (d,d) numpy array. PCA transformation matrix

        Outputs:
        1. reconstructed_data  (N,d) numpy ndarray, where the i-th row contains the reconstruction of the original i-th input feature vector (in `data`) based on the PCA features contained in `pcadata`.
        """
        reconstructed_data = np.zeros((pcadata.shape[0],W.shape[0]))
        # Put your code below


        return reconstructed_data

from sklearn.decomposition import PCA

class Question2(object):
    def unexp_var(self,X,k):
        """Returns an numpy array with the fraction of unexplained variance on X by retaining the first k principal components for k =1,...200.
        Parameters:
        1. X        The input image

        Returns:
        1. pca      The PCA object fit on X
        2. unexpv   A (k,) numpy vector, where the i-th element contains the percentage of unexplained variance on X by retaining i+1 principal components
        """
        pca = None
        unexpv = np.zeros(k)
        # Put your code below


        return (pca,unexpv)

    def pca_approx(self,X_t,pca,i):
        """Returns an approimation of `X_t` using the the first `i`  principal components (learned from `X`).

        Parameters:
            1. X_t      The input image to be approximated
            2. pca      The PCA object to use for the transform
            3. i        Number of principal components to retain

        Returns:
            1. recon_img    The reconstructed approximation of X_t using the first i principal components learned from X (As a sanity check it should be of size (1,4096))
        """
        recon_img = np.zeros((1,4096))
        # Put your code below


        assert(recon_img.shape==(1,4096))
        return recon_img

from sklearn import neighbors

class Question3(object):
    def pca_classify(self,traindata,trainlabels,valdata,vallabels,k):
        """Returns validation errors using 1-NN on the PCA features using 1,2,...,k PCA features, the minimum validation error, and number of PCA features used.

        Parameters:
            1. traindata       (Nt, d) numpy ndarray. The features in the training set.
            2. trainlabels     (Nt,) numpy array. The responses in the training set.
            3. valdata         (Nv, d) numpy ndarray. The features in the validation set.
            4. valabels        (Nv,) numpy array. The responses in the validation set.
            5. k               Integer. Maximum number of PCA features to retain

        Returns:
            1. ve              A length k numpy array, where ve[i] is the validation error using the first i+1 features (i=0,...,255).
            2. min_ve          Minimum validation error
            3. min_pca_feat    Number of PCA features to retain. Integer.
        """

        ve = np.zeros(k)
        # Put your code below


        return (ve, min_ve, min_pca_feat)

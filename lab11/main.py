import numpy as np
from sklearn import svm


class Question1(object):
    def svm_classifiers(self, X, y):
        svm_linear = svm.SVC(kernel="linear").fit(X, y)
        svm_non_linear = svm.SVC(kernel="rbf").fit(X, y)
        return svm_linear, svm_non_linear

    def acc_prec_recall(self, y_pred, y_test):
        TP = np.logical_and(y_test == 1, y_pred == 1).sum()
        FP = np.logical_and(y_test == 0, y_pred == 1).sum()
        FN = np.logical_and(y_test == 1, y_pred == 0).sum()
        TN = np.logical_and(y_test == 0, y_pred == 0).sum()
        acc = (y_pred == y_test).sum() / len(y_pred)
        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
        return acc, prec, recall


class Question2(object):
    def CCF_1d(self, x1, x2):
        assert(len(x1) == len(x2))
        d = len(x1)
        ccf = np.zeros(d)
        tmp = np.zeros(d) 
        for m in range(d):
            ccf[m] = (x1 * np.roll(x2, -m)).sum()
        return ccf

    def align_1d(self, x1, x2):
        assert(len(x1) == len(x2))
        d = len(x1)
        shift = np.argmax(self.CCF_1d(x1, x2))
        aligned_sig = np.roll(x2, -shift)

        return aligned_sig


class Question3(object):
    def CCF_2d(self, x1, x2):
        assert(x1.shape == x2.shape)
        d = x1.shape[0]
        ccf = np.zeros((d, d))
        for m in range(d):
            for n in range(d):
                ccf[m, n] = (x1 * np.roll(np.roll(x2,-m,0), -n, 1)).sum()
        return ccf

    def align_2d(self, x1, x2):
        assert(x1.shape == x2.shape)
        d = x1.shape[0]
        shift_yx = np.unravel_index(np.argmax(self.CCF_2d(x1, x2)), x1.shape)
        aligned_img = np.roll(np.roll(x2,-shift_yx[0], 0), -shift_yx[1], 1)
        return aligned_img

    def response_signal(self, ref_images, query_image):
        d = query_image.shape[0]
        M = ref_images.shape[2]

        resp = np.zeros(M)
        for idx in range(M):
            ccf = self.CCF_2d(ref_images[:, :, idx], query_image)
            normalized_ccf = ccf - ccf.sum()/(d**2 ) 
            resp[idx] = np.max(normalized_ccf)
        return resp

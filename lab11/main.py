import numpy as np
from sklearn import svm


class Question1(object):
    def svm_classifiers(self, X, y):
        svm_linear = None
        svm_non_linear = None
        return svm_linear, svm_non_linear

    def acc_prec_recall(self, y_pred, y_test):
        acc = None
        prec = None
        recall = None
        return acc, prec, recall


class Question2(object):
    def CCF_1d(self, x1, x2):
        ccf = None
        return ccf

    def align_1d(self, x1, x2):
        aligned_sig = None
        return aligned_sig


class Question3(object):
    def CCF_2d(self, x1, x2):
        ccf = None
        return ccf

    def align_2d(self, x1, x2):
        aligned_img = None
        return aligned_img

    def response_signal(self, ref_images, query_image):
        resp = None
        return resp

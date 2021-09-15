import numpy as np
import scipy.spatial.distance as dist
from scipy import stats
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time

class Question1(object):
    def bayesClassifier(self,data,pi,means,cov):
        # labels = None
        C_inv = np.linalg.inv(cov)
        # a = means @ C_inv @ data.T
        # b = np.sum ((means @ C_inv) * means, axis=1)
        a = np.matmul(np.matmul(means, C_inv), data.T)
        b = np.sum(np.matmul(means, C_inv) * means, axis=1)
        labels = np.argmax(np.log(pi) + a.T - 0.5*b, axis=1)
        return labels

    def classifierError(self,truelabels,estimatedlabels):
        error = np.array(truelabels!=estimatedlabels).sum()
        error /= len(truelabels)
        return error


class Question2(object):
    def trainLDA(self,trainfeat,trainlabel):
        nlabels = int(trainlabel.max())+1 # Assuming all labels up to nlabels exist.
        pi = np.zeros(nlabels)            # Store your prior in here
        means = np.zeros((nlabels,trainfeat.shape[1]))            # Store the class means in here
        cov = np.zeros((trainfeat.shape[1],trainfeat.shape[1]))   # Store the covariance matrix in here
        # Put your code below
        Num = len(trainlabel)
        for i in range(nlabels):
            N_l = (trainlabel==i).sum()
            pi[i] = N_l / Num
            means[i] = mean_l = trainfeat[trainlabel==i].sum(axis=0) / N_l
            mean_err = trainfeat[trainlabel==i] - mean_l
            cov += np.matmul(mean_err.T, mean_err)
        cov /= Num - nlabels
        # Don't change the output!
        return (pi,means,cov)

    def estTrainingLabelsAndError(self,trainingdata,traininglabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        lpi, lmeans, lcov = self.trainLDA(trainingdata,traininglabels)
        esttrlabels = q1.bayesClassifier(trainingdata, lpi, lmeans, lcov)
        trerror = q1.classifierError(traininglabels, esttrlabels)
        # Don't change the output!
        return (esttrlabels, trerror)

    def estValidationLabelsAndError(self,trainingdata,traininglabels,valdata,vallabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        lpi, lmeans, lcov = self.trainLDA(trainingdata,traininglabels)

        estvallabels = q1.bayesClassifier(valdata, lpi, lmeans, lcov)
        valerror = q1.classifierError(vallabels, estvallabels)
        # Don't change the output!
        return (estvallabels, valerror)


class Question3(object):
    def kNN(self,trainfeat,trainlabel,testfeat, k):
        distances = dist.cdist(testfeat, trainfeat, 'euclidean')
        args = np.argpartition(distances, k, axis=1)[:, :k]
        points_labels = np.array([trainlabel[i] for i in args]).reshape(args.shape)
        labels, cnt = stats.mode(points_labels, axis=1)
        return labels.reshape(labels.shape[0])

    def kNN_errors(self,trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()
        trainingError = np.zeros(4)
        validationError = np.zeros(4)
        k_array = [1,3,4,5]

        for i in range(len(k_array)):
            # Please store the two error arrays in increasing order with k
            # This function should call your previous self.kNN() function.
            # Put your code below
            esttrlabels = self.kNN(trainingdata, traininglabels, trainingdata, k_array[i])
            trainingError[i] = q1.classifierError(traininglabels, esttrlabels)
            estvallabels = self.kNN(trainingdata, traininglabels, valdata, k_array[i])
            validationError[i] = q1.classifierError(vallabels, estvallabels)
        # Don't change the output!
        return (trainingError, validationError)

class Question4(object):
    def sklearn_kNN(self,traindata,trainlabels,valdata,vallabels):
        # classifier, valerror, fitTime, predTime = (None, None, None, None)
        classifier = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='brute')
        start = time.time()
        classifier.fit(traindata, trainlabels)
        fitTime = time.time() - start
        start = time.time()
        estvallabels = classifier.predict(valdata)
        predTime = time.time() - start
        q1 = Question1()
        valerror = q1.classifierError(vallabels, estvallabels)

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

    def sklearn_LDA(self,traindata,trainlabels,valdata,vallabels):
        classifier, valerror, fitTime, predTime = (None, None, None, None)
        classifier = LinearDiscriminantAnalysis()
        start = time.time()
        classifier.fit(traindata, trainlabels)
        fitTime = time.time() - start
        start = time.time()
        estvallabels = classifier.predict(valdata)
        predTime = time.time() - start
        q1 = Question1()
        valerror = q1.classifierError(vallabels, estvallabels)

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

###

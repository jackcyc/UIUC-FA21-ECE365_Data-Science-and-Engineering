import numpy as np
from skimage.transform import radon, iradon, resize
import matplotlib.pyplot as plt


class Question1(object):
    def complete_ortho_matrix(self, M):
        orth_mat = np.concatenate((M, np.cross(M[:, 0], M[:, 1]).reshape((3, 1))), axis=1)
        return orth_mat

    def recover_ortho_matrix(self, M):
        U, S, W = np.linalg.svd(M)
        orth_mat = U @ W
        return orth_mat

    def comp_rec_ortho_matrix(self, M):
        # M_ = np.concatenate((M, np.zeros((3, 1))), axis=1)
        M_ = self.complete_ortho_matrix(M)
        orth_mat = self.recover_ortho_matrix(M_)
        # print(np.trace((M-orth_mat[:, :2]) @ (M-orth_mat[:, :2]).T))
        # print(np.trace((M_-orth_mat) @ (M_-orth_mat).T))
        return orth_mat


class Question2(object):
    def template_matching(self, noisy_proj, I0, M, Tmax):
        dim, N = noisy_proj.shape
        t = 0
        theta = np.linspace(0., 360., M, endpoint=False)
        while t < Tmax:
            # update the projection angle
            sinogram = radon(I0, theta=theta)

            corr = noisy_proj.T @ sinogram / np.linalg.norm(sinogram, axis=0)

            l = np.argmax(corr, axis=1)
            theta_i = theta[l]

            # ss = np.zeros((dim, M))
            # for idx, i in enumerate(l):
            #     ss[:, i] = noisy_proj[:, idx]
            # plt.imshow(ss)
            # plt.hist(l)
            # plt.show()

            # update the image
            I0 = iradon(noisy_proj, theta_i)
            # plt.imshow(I0)
            # plt.show()

            t += 1

        theta = theta_i
        I_rec = I0
        return I_rec, theta

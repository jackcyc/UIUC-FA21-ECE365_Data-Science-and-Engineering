import numpy as np


class Question1(object):
    def rotate_matrix(self, theta):
        R_theta = None
        return R_theta

    def rotate_2d(self, points, theta):
        rot_points = None
        return rot_points

    def combine_rotation(self, theta1, theta2):
        err = None
        return err


class Question2(object):
    def rotate_matrix_x(self, theta):
        R_x = None
        return R_x

    def rotate_matrix_y(self, theta):
        R_y = None
        return R_y

    def rotate_matrix_z(self, theta):
        R_z = None
        return R_z

    def rot_matrix(self, alpha, beta, gamma):
        R = None
        return R

    def rotate_point(self, points, R):
        rot_points = None
        return rot_points


class Question3(object):
    def rotate_x_axis(self, image_size, theta):
        rot_points = None
        return rot_points

    def nudft2(self, img, grid_f):
        img_f = None
        return img_f

    def gen_projection(self, img, theta):
        # grid on which to compute the fourier transform
        points_rot = None

        # Put your code here
        # ...


        # Don't change the rest of the code and the output!
        ft_img = self.nudft2(img, points_rot)
        proj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ft_img)))
        proj = np.real(proj)
        return proj


class Question4(object):
    def nudft3(self, vol, grid_f):
        vol_f = None
        return vol_f

    def gen_projection(self, vol, R_theta):
        vol_sz = vol.shape[0]
        # grid on which to compute the fourier transform
        xy_plane_rot = None

        # Put your code here
        # ...

        # Don't change the rest of the code and the output!
        ft_vol = self.nudft3(vol, xy_plane_rot)
        ft_vol = np.reshape(ft_vol, [vol_sz, vol_sz])
        proj_img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ft_vol)))
        proj_img = np.real(proj_img)
        return proj_img

    def apply_ctf(self, img, ctf):
        # Nothing to add here!
        fm = np.fft.fftshift(np.fft.fftn(img))
        cm = np.real(np.fft.ifftn(np.fft.ifftshift(fm * ctf)))
        return cm

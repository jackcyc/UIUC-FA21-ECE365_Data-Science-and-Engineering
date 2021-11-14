import numpy as np


class Question1(object):
    def rotate_matrix(self, theta):
        theta = np.deg2rad(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_theta = np.array([
            [cos_theta, -1*sin_theta],
            [sin_theta, cos_theta]
        ])
        return R_theta

    def rotate_2d(self, points, theta):
        rot_points = self.rotate_matrix(theta) @ points
        return rot_points

    def combine_rotation(self, theta1, theta2):
        rot_mat_opt1 = self.rotate_matrix(theta1 + theta2)
        rot_mat_opt2 = self.rotate_matrix(theta1) @ self.rotate_matrix(theta2)
        err = np.linalg.norm(rot_mat_opt1 - rot_mat_opt2)
        return err


class Question2(object):
    def rotate_matrix_x(self, theta):
        theta = np.deg2rad(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_x = np.array([
            [1, 0, 0],
            [0, cos_theta, -1*sin_theta],
            [0, sin_theta, cos_theta]
        ])
        return R_x

    def rotate_matrix_y(self, theta):
        theta = np.deg2rad(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_y = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-1*sin_theta, 0, cos_theta]
        ])
        return R_y

    def rotate_matrix_z(self, theta):
        theta = np.deg2rad(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R_z = np.array([
            [cos_theta, -1*sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
        return R_z

    def rot_matrix(self, alpha, beta, gamma):
        R = self.rotate_matrix_z(gamma) @ self.rotate_matrix_y(beta) @ self.rotate_matrix_z(alpha)
        return R

    def rotate_point(self, points, R):
        rot_points = R @ points
        return rot_points


class Question3(object):
    def rotate_x_axis(self, image_size, theta):
        points = np.zeros((2, image_size))
        points[0] = np.linspace(-1*(image_size-1)/2, (image_size-1)/2., image_size)
        rot_points = Question1().rotate_2d(points, theta)
        return rot_points

    def nudft2(self, img, grid_f):
        image_size = img.shape[0]
        N = (image_size-1) / 2

        y = x = np.linspace(-1*N, N, image_size)
        xx, yy = np.meshgrid(x, y)

        myconst = -1j * (2 * np.pi) / (2 * N + 1)

        img_f = np.zeros((image_size), dtype="complex_")
        for idx in range(image_size):
            k_x, k_y = grid_f[:, idx]
            img_f[idx] = (img * np.exp(myconst * (xx * k_x + yy * k_y))).sum()

        return img_f

    def gen_projection(self, img, theta):
        # grid on which to compute the fourier transform
        points_rot = None

        # Put your code here
        # ...
        image_size = img.shape[0]
        points_rot = self.rotate_x_axis(image_size, theta)

        # Don't change the rest of the code and the output!
        ft_img = self.nudft2(img, points_rot)
        proj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ft_img)))
        proj = np.real(proj)
        return proj


class Question4(object):
    def nudft3(self, vol, grid_f):
        # assume vol's shape[0] == shape[1] == shape[2]
        vol_size = vol.shape[0]
        N = (vol_size-1) / 2

        z = y = x = np.linspace(-1*N, N, vol_size)
        xx, yy, zz = np.meshgrid(x, y, z)

        myconst = -1j * (2 * np.pi) / (2 * N + 1)

        vol_f = np.zeros((vol_size*vol_size), dtype="complex_")
        for idx in range(vol_size*vol_size):
                k_x, k_y, k_z = grid_f[:, idx]
                vol_f[idx] = (vol * np.exp(myconst * (xx * k_x + yy * k_y + zz * k_z))).sum()

        return vol_f

    def gen_projection(self, vol, R_theta):
        vol_sz = vol.shape[0]
        # grid on which to compute the fourier transform
        xy_plane_rot = None

        # Put your code here
        # ...
        vol_size = vol.shape[0]
        N = (vol_size-1) / 2

        y = x = np.linspace(-1*N, N, vol_size)
        xx, yy = np.meshgrid(x, y)
        
        points = np.zeros((3, vol_size, vol_size))
        points[0] = xx
        points[1] = yy
        points = points.reshape((3, -1))
        xy_plane_rot = Question2().rotate_point(points, R_theta)

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

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import cbrt


def lin_array(vec):
    return np.isclose(np.max(np.diff(vec)), np.min(np.diff(vec)))


def poly2(a, b, c, x):
    return a * np.power(x, 2) + b * np.power(x, 1) + c


def polyn(param, x):
    a = 0
    for i in range(len(param)):
        a += param[i] * np.power(x, len(param) - i - 1)
    return a


def poly1(a, b, x):
    return a * np.power(x, 1) + b


def powerl2(x, x0, c):
    return x0 + c * np.power(x, -2)


def powerl(x, x0, c, gamma):
    return x0 + c * np.power(x, gamma)


def is_list(val):
    return isinstance(val, (list, np.ndarray))


def are_not_list(list):
    for i in list:
        if is_list(i):
            return False
    return True


def are_all_list(list):
    for i in list:
        if not (is_list(i)):
            return False
    return True


def are_same_size(list):
    val = np.shape(list[0])
    for i in list:
        if np.shape(i) != val:
            return False
    return True


def find_cubic_roots(
    a,
    c,
    d,
):
    a, c, d = a + 0j, c + 0j, d + 0j

    Q = (3 * a * c) / (9 * a**2)
    R = (-27 * a**2 * d) / (54 * a**3)
    D = Q**3 + R**2
    S = 0  # NEW CALCULATION FOR S STARTS HERE
    if np.isreal(R + np.sqrt(D)):
        S = cbrt(np.real(R + np.sqrt(D)))
    else:
        S = (R + np.sqrt(D)) ** (1 / 3)
    T = 0  # NEW CALCULATION FOR T STARTS HERE
    if np.isreal(R - np.sqrt(D)):
        T = cbrt(np.real(R - np.sqrt(D)))
    else:
        T = (R - np.sqrt(D)) ** (1 / 3)
    result = []
    result1 = +(S + T)
    if np.isreal(result1):
        result.append(np.real(result1))
    result2 = -(S + T) / 2 + 0.5j * np.sqrt(3) * (S - T)
    if np.isreal(result2):
        result.append(np.real(result2))
    result3 = -(S + T) / 2 - 0.5j * np.sqrt(3) * (S - T)
    if np.isreal(result3):
        result.append(np.real(result3))

    return np.array(result)


class Disk_model:
    """
    Class of a disk model, defined by:
    - rin, rout : Internal and external radii
    - epsilon : H/R ratio
    - alpha : emissivity law
    """

    def __init__(
        self,
        rin=1,
        rout=100,
        epsilon=0.2,
        alpha=1,
        Mstar=1,
        side="both",
        vphi_corr=False,
        gamma=0,
    ):
        """
        class constructor
        """
        if (rin < 0) or (rout < 0):
            sys.exit("Error : Radius value must be positive !")
        if rin >= rout:
            sys.exit("Error : Internal radius is bigger than external radius !")
        if epsilon < 0:
            sys.exit("Error : H/R ratio must be > 0  !")
        if side not in ["both", "top", "bottom"]:
            sys.exit(
                'Error : Unknown side parameter, must be either "both", "bottom" or "top"'
            )
        self.rin = rin
        self.rout = rout
        self.epsilon = epsilon
        self.alpha = alpha
        self.side = side
        self.vphi_corr = vphi_corr
        self.Mstar = Mstar
        self.gamma = gamma

    def create_setup(self, step, dphi):
        self.step = step
        self.dphi = dphi

        lenr = (self.rout - self.rin) / step
        lenphi = 360 / dphi
        if self.side == "both":
            lenr = lenr * 2
        print("Setting up the resolution")
        print("Final array size : (%d , %d , %d)" % (lenr, lenphi, 1))

    def print_parameters(self):
        print("rin = ", self.rin)
        print("rout = ", self.rout)
        print("epsilon = ", self.epsilon)
        print("alpha = ", self.alpha)
        print("side = ", self.side)
        print("vphi_corr = ", self.vphi_corr)

    def create_profile(self):
        """
        This function creates:
        - Z(L) profile
        - R(L) profile
        - VR(L) profile
        - VZ(L) profile
        - Vphi(L) profile
        - I(L) profile
        with L the longitudinal parameter (with an indentation of dl)
        vertically stack each line and return the resulting array
        """
        phi_1D = np.arange(0, 360, self.dphi) * np.pi / (180.0)

        dl = self.step
        opening = np.arctan(1 / self.epsilon)
        lmax = self.rout / (np.sin(opening))
        lmin = self.rin / (np.sin(opening))
        if self.side == "top":
            ll = np.arange(lmin, lmax + dl, dl)
        elif self.side == "bottom":
            ll = np.arange(-lmin, -lmax - dl, -dl)
        elif self.side == "both":
            ll1 = np.arange(lmin, lmax + dl, dl)
            ll2 = np.arange(-lmin, -lmax - dl, -dl)
            ll = np.append(ll1, ll2)
        r = ll * np.sin(opening) * np.sign(ll)
        z = ll * np.cos(opening)
        if self.vphi_corr:
            vphi = (30 * np.sqrt(self.Mstar)) * r * r / (np.power(ll, 1.5))
        elif not (self.vphi_corr):
            vphi = (30 * np.sqrt(self.Mstar)) / np.sqrt(r)
        vp = self.gamma * vphi
        vr = -vp * np.sin(opening)
        vz = -vp * np.cos(opening)

        dv = dl * np.abs(ll) * dl
        I = dv * np.power(r, -self.alpha)

        r = r[np.newaxis, :]
        z = z[np.newaxis, :]
        vr = vr[np.newaxis, :]
        vz = vz[np.newaxis, :]
        vphi = vphi[np.newaxis, :]
        I = I[np.newaxis, :]
        profile = np.dstack((r, z, vr, vz, vphi, I))
        return phi_1D, profile

    def plot_dynamics(self, plot_J=False, savename=""):
        _, param = self.create_profile()
        cmap = ["inferno", "cividis", "viridis"]
        cbar_label = [
            r"V$_R$ (kms$^{-1}$)",
            r"V$_Z$ (kms$^{-1}$)",
            r"V$_{\Phi}$ (kms$^{-1}$)",
        ]
        if plot_J:
            param[:, :, 4] = param[:, :, 4] * param[:, :, 0]
            cbar_label[2] = r"RV$_{\Phi}$ (aukms$^{-1}$)"
        fig, ax = plt.subplots(
            nrows=1, ncols=3, sharey=True, sharex=True, figsize=(15, 4)
        )
        for i in range(3):
            im = ax[i].scatter(
                param[:, :, 0], param[:, :, 1], c=param[:, :, i + 2], s=1, cmap=cmap[i]
            )
            cbar = plt.colorbar(im, ax=ax[i], pad=0)
            cbar.set_label(cbar_label[i], rotation=0, labelpad=-20, y=1.07, ha="center")
        ax[0].set_xlabel("R (au)")
        ax[0].set_ylabel("Z (au)")
        if savename:
            plt.savefig(
                savename + ".png",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
                dpi=400,
            )
        plt.show()


class WDS_model:
    """
    Class of a wind driven shell model (lee 200), defined by:
    - C : Parabolic parameter (au-2)
    - tau : 1/V0 : determine the apparent acceleration (yr-1)
    - eta : Parameter of collimation (see de Valon et. al 2020)
    - J : Angular momentum of the shell, assumed to be constant along the outflow
    """

    def __init__(self, C=0, tau=0, eta=1, J=0, I0=1, alpha=0):
        """
        class constructor
        """
        if (not (is_list(C)) and is_list(tau)) or (not (is_list(tau)) and is_list(C)):
            sys.exit("C and tau must both be either array or float !")

        elif is_list(C) and is_list(tau):
            if len(C) != len(tau):
                sys.exit("Error : C and tau must have the same lengh !")
            if not (is_list(eta)):
                eta = np.ones_like(tau) * eta
            if not (is_list(J)):
                J = np.ones_like(tau) * J
            if not (is_list(I0)):
                I0 = np.ones_like(tau) * I0
            if not (is_list(alpha)):
                alpha = np.ones_like(tau) * alpha
        self.C = C
        self.tau = tau
        self.eta = eta
        self.J = J
        self.I0 = I0
        self.alpha = alpha

    def create_setup(self, zmax, step, dphi):
        self.step = step
        self.dphi = dphi
        self.zmax = zmax
        lenr = (zmax) / step
        lenphi = 360 / dphi
        print("Setting up the resolution")
        print("Final array size : (%d , %d , %d)" % (lenr, lenphi, len(self.C)))

    def print_parameters(self):
        print("C = ", self.C)
        print("tau = ", self.tau)
        print("eta = ", self.eta)
        print("J = ", self.J)

    def create_profile(self):
        """
        This function creates:
        - Z(Z) profile
        - R(Z) profile
        - VR(Z) profile
        - VZ(Z) profile
        - Vphi(Z) profile
        - I(Z) profile
        with Z the longitudinal parameter (with an indentation of dl)
        vertically stack each line and return the resulting array
        """
        dl = self.step
        phi_1D = np.arange(0, 360, self.dphi) * np.pi / (180.0)

        v0 = (1 / self.tau) * (1.5e8) / (3600 * 24 * 365)
        dl = self.step
        z1D = np.arange(dl, dl + self.zmax, dl)
        z, c = np.meshgrid(z1D, self.C)
        _, eta = np.meshgrid(z1D, self.eta)
        _, j = np.meshgrid(z1D, self.J)
        _, v0 = np.meshgrid(z1D, v0)
        _, I0 = np.meshgrid(z1D, self.I0)
        _, alpha = np.meshgrid(z1D, self.alpha)

        r = np.sqrt(z / c)

        dv = dl * dl * r / np.sqrt(z)
        I = dv * I0 * np.power(z, -alpha * 1.0)

        vz = v0 * z
        vr = v0 * eta * r
        vphi = j / r

        profile = np.dstack((r, z, vr, vz, vphi, I))
        return phi_1D, profile

    def plot_dynamics(self, plot_J=False, savename=""):
        _, param = np.copy(self.create_profile())
        cmap = ["inferno", "cividis", "viridis"]
        cbar_label = [
            r"V$_R$ (kms$^{-1}$)",
            r"V$_Z$ (kms$^{-1}$)",
            r"V$_{\Phi}$ (kms$^{-1}$)",
        ]
        if plot_J:
            param[:, :, 4] = param[:, :, 4] * param[:, :, 0]
            cbar_label[2] = r"RV$_{\Phi}$ (aukms$^{-1}$)"
        fig, ax = plt.subplots(
            nrows=1, ncols=3, sharey=True, sharex=True, figsize=(15, 4)
        )
        for i in range(3):
            im = ax[i].scatter(
                param[:, :, 0], param[:, :, 1], c=param[:, :, i + 2], s=1, cmap=cmap[i]
            )
            cbar = plt.colorbar(im, ax=ax[i], pad=0)
            cbar.set_label(cbar_label[i], rotation=0, labelpad=-20, y=1.07, ha="center")
        ax[0].set_xlabel("R (au)")
        ax[0].set_ylabel("Z (au)")
        if savename:
            plt.savefig(
                savename + ".png",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
                dpi=400,
            )
        plt.show()


class DW_model:
    """
    Class of a conical disk wind model defined by:
    - r0 : radius of ejection (au)
    - theta : angle of the ejection with respect to the Z axis (degree)
    - VP : Poloidal velocity of the layer (in km/s)
    - J : Angular momentum of the layer

    VP and J are assumed to be constant along the outflow, the trajectory is
    assumed to be constant
    All values must be either a single value (if only one layer) or an array/VZ_list
    (in the case of a shear)
    """

    def __init__(self, r0, theta, vp, J, alpha=1, I0=1):
        """
        class constructor
        """
        if is_list(r0) and is_list(theta) and is_list(vp) and is_list(J):
            if not (is_list(alpha)):
                alpha = np.ones_like(r0) * alpha
            if not (is_list(I0)):
                I0 = np.ones_like(r0) * I0
            if not (are_same_size([r0, theta, vp, J, alpha, I0])):
                sys.exit("Error : All arrays must have the same size !")
        elif are_not_list([r0, theta, vp, J, alpha, I0]):
            r0 = np.array([r0])
            theta = np.array([theta])
            vp = np.array([vp])
            J = np.array([J])
            alpha = np.array([alpha])
            I0 = np.array([I0])
        else:
            sys.exit("Error : input value must correspond in type !")

        self.r0 = r0
        self.theta = theta
        self.vp = vp
        self.J = J
        self.alpha = alpha
        self.I0 = I0

    def print_parameters(self):
        print("r0 = ", self.r0)
        print("theta = ", self.theta)
        print("vp = ", self.vp)
        print("J = ", self.J)
        print("alpha = ", self.alpha)
        print("I0 = ", self.I0)

    def create_setup(self, zmax, step, dphi):
        self.step = step
        self.dphi = dphi
        self.zmax = zmax
        lenr = (zmax) / step
        lenphi = 360 / dphi
        print("Setting up the resolution")
        print("Final array size : (%d , %d , %d)" % (lenr, lenphi, len(self.vp)))

    def create_profile(self):
        """
        This function creates:
        - Z(Z) profile
        - R(Z) profile
        - VR(Z) profile
        - VZ(Z) profile
        - Vphi(Z) profile
        - I(Z) profile
        with Z the longitudinal parameter (with an indentation of dl)
        vertically stack each line and return the resulting array
        """
        dl = self.step
        phi_1D = np.arange(0, 360, self.dphi) * np.pi / (180.0)

        z1D = np.arange(dl, dl + self.zmax, dl)
        z, r0 = np.meshgrid(z1D, self.r0)
        _, theta = np.meshgrid(z1D, self.theta * np.pi / 180)
        _, j = np.meshgrid(z1D, self.J)
        _, vp = np.meshgrid(z1D, self.vp)
        _, I0 = np.meshgrid(z1D, self.I0)
        _, alpha = np.meshgrid(z1D, self.alpha)

        r = r0 + z * np.tan(theta)

        dv = dl * dl * r

        I = dv * I0 * np.power(z * 1.0, -alpha)

        vz = vp * np.cos(theta)
        vr = vp * np.sin(theta)
        vphi = j / r

        profile = np.dstack((r, z, vr, vz, vphi, I))
        return phi_1D, profile

    def plot_dynamics(self, plot_J=False, savename=""):
        _, param = np.copy(self.create_profile())
        cmap = ["inferno", "cividis", "viridis"]
        cbar_label = [
            r"V$_R$ (kms$^{-1}$)",
            r"V$_Z$ (kms$^{-1}$)",
            r"V$_{\Phi}$ (kms$^{-1}$)",
        ]
        if plot_J:
            param[:, :, 4] = param[:, :, 4] * param[:, :, 0]
            cbar_label[2] = r"RV$_{\Phi}$ (aukms$^{-1}$)"
        fig, ax = plt.subplots(
            nrows=1, ncols=3, sharey=True, sharex=True, figsize=(15, 4)
        )
        for i in range(3):
            im = ax[i].scatter(
                param[:, :, 0], param[:, :, 1], c=param[:, :, i + 2], s=1, cmap=cmap[i]
            )
            cbar = plt.colorbar(im, ax=ax[i], pad=0)
            cbar.set_label(cbar_label[i], rotation=0, labelpad=-20, y=1.07, ha="center")
        ax[0].set_xlabel("R (au)")
        ax[0].set_ylabel("Z (au)")
        if savename:
            plt.savefig(
                savename + ".png",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
                dpi=400,
            )
        plt.show()


class Infall_model:
    """
    Class of a conical disk wind model defined by:
    - Rd : Size of the disk (au)
    - M : Mass of the central star
    - theta0 : starting angle of infall
    All value must be single values
    """

    def __init__(self, rd, M, theta0, alpha=1, J_sign=1):
        """
        class constructor
        """
        if is_list(rd):
            sys.exit("Error : Rd must be a single value")
        elif is_list(M):
            sys.exit("Error : M must be a single value")
        elif is_list(alpha):
            sys.exit("Error : alpha must be a single value")
        elif is_list(theta0):
            sys.exit("Error : theta0 must be a single value")
        elif is_list(J_sign):
            sys.exit("Error : J_sign must be a single value")
        if np.abs(J_sign) != 1:
            sys.exit("Error : J_sign either be 1 or -1")

        else:
            rd = [rd]
            M = [M]
            alpha = [alpha]
            theta0 = [theta0 * np.pi / 180]

        self.rd = rd
        self.M = M
        self.theta0 = theta0
        self.alpha = alpha
        self.J_sign = J_sign

    def create_setup(self, zmax, step, dphi):
        self.step = step
        self.dphi = dphi
        self.zmax = zmax
        lenr = (zmax) / step
        lenphi = 360 / dphi
        print("Setting up the resolution")
        print("Final array size : (%d , %d , %d)" % (lenr, lenphi, 1))

    def print_parameters(self):
        print("rd = ", self.rd)
        print("M = ", self.M)
        print("theta0 = ", self.theta0)
        print("alpha = ", self.alpha)

    def create_profile(self):
        """
        Zmax is Rmax here.
        """
        dl = self.step
        theta0 = self.theta0
        M = self.M
        rd = self.rd
        alpha = self.alpha
        zmax = self.zmax
        print(zmax)
        Rmax = zmax / np.cos(theta0)
        R = np.arange(rd * np.power(np.sin(theta0), 2), Rmax, dl)
        phi_1D = np.arange(0, 360, self.dphi) * np.pi / (180.0)

        R, theta0 = np.meshgrid(R, theta0)
        _, M = np.meshgrid(R, M)
        _, rd = np.meshgrid(R, rd)
        _, alpha = np.meshgrid(R, alpha)

        costheta = -rd * np.cos(theta0) * np.power(np.sin(theta0), 2) / R + np.cos(
            theta0
        )
        z = R * costheta
        theta = np.arccos(costheta)
        r = R * np.sin(theta)
        V_sph_r = (-30 * np.sqrt(M) / np.sqrt(R)) * np.sqrt(
            1 + np.cos(theta) / np.cos(theta0)
        )
        V_sph_theta = (
            (30 * np.sqrt(M) / np.sqrt(R))
            * (np.cos(theta0) - np.cos(theta))
            * np.sqrt(
                (np.cos(theta0) + np.cos(theta))
                / (np.cos(theta0) * np.power(np.sin(theta), 2))
            )
        )
        vphi = (
            self.J_sign
            * (30 * np.sqrt(M) / np.sqrt(R))
            * (np.sin(theta0) / (np.sin(theta)))
            * np.sqrt(1 - np.cos(theta) / np.cos(theta0))
        )

        vr = V_sph_r * np.sin(theta) + V_sph_theta * np.cos(theta)
        vz = V_sph_r * np.cos(theta) - V_sph_theta * np.sin(theta)

        dv = dl * R * dl
        I = dv * np.power(R, -alpha)

        profile = np.dstack((r, z, vr, vz, vphi, I))
        return phi_1D, profile

    def plot_dynamics(self, plot_J=True, savename=""):
        _, param = np.copy(self.create_profile())
        cmap = ["inferno", "cividis", "viridis"]
        cbar_label = [
            r"V$_R$ (kms$^{-1}$)",
            r"V$_Z$ (kms$^{-1}$)",
            r"V$_{\Phi}$ (kms$^{-1}$)",
        ]
        if plot_J:
            param[:, :, 4] = np.copy(param[:, :, 4] * param[:, :, 0])
            cbar_label[2] = r"RV$_{\Phi}$ (aukms$^{-1}$)"
        fig, ax = plt.subplots(
            nrows=1, ncols=3, sharey=True, sharex=True, figsize=(15, 4)
        )
        for i in range(3):
            im = ax[i].scatter(
                param[:, :, 0], param[:, :, 1], c=param[:, :, i + 2], s=1, cmap=cmap[i]
            )
            cbar = plt.colorbar(im, ax=ax[i], pad=0)
            cbar.set_label(cbar_label[i], rotation=0, labelpad=-20, y=1.07, ha="center")
        ax[0].set_xlabel("R (au)")
        ax[0].set_ylabel("Z (au)")
        if savename:
            plt.savefig(
                savename + ".png",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
                dpi=400,
            )
        plt.show()


class BS_model:
    """
    Class of a conical disk wind model defined by:
    - Rd : Size of the disk (au)
    - M : Mass of the central star
    - theta0 : starting angle of infall
    All value must be single values
    """

    def __init__(self, mdot, v0, rho, vj, zj, alpha, model="mixed"):
        """
        class constructor
        """

        if model not in ["mixed", "post-shock"]:
            sys.exit('model must be either "mixed" or "post-shock"')
        self.mdot = mdot
        self.v0 = v0
        self.rho = rho
        self.vj = vj
        self.zj = zj
        self.model = model
        self.alpha = alpha

    def print_parameters(self):
        print("mdot = ", self.mdot)
        print("v0 = ", self.v0)
        print("rho = ", self.rho)
        print("vj = ", self.vj)
        print("zj = ", self.zj)

    def create_setup(self, step, dphi):
        self.step = step
        self.dphi = dphi
        print("Setting up the resolution")

    def create_profile(self):
        mdot = self.mdot
        v0 = self.v0
        rho = self.rho
        vj = self.vj
        zj = self.zj
        alpha = self.alpha
        dl = self.step
        mH = 1.66054e-27
        mdot = mdot * 2e30 / (24 * 3600 * 365)
        v0 = v0 * 1e3
        vj = vj * 1e3
        rho = rho * 1.4 * mH * (np.power(1e2, 3))
        zj = zj * 1.5e11

        phi_1D = np.arange(0, 360, self.dphi) * np.pi / (180.0)

        L0 = np.sqrt(3 * mdot * v0 / (np.pi * rho * np.power(vj, 2)))
        gamma = vj / v0

        rbf = find_cubic_roots(
            1 / (gamma * np.power(L0, 3)), 1 / L0, -v0 * zj / (L0 * vj)
        )
        nl = int(rbf / (dl * 1.5e11))
        Rb = np.linspace(dl * 1.5e11, rbf, nl)

        r, v0 = np.meshgrid(Rb, v0)
        _, vj = np.meshgrid(Rb, vj)
        _, zj = np.meshgrid(Rb, zj)

        z = zj - np.power(r / L0, 2) * r
        if self.model == "mixed":
            vr = v0 / (1 + 3 / gamma * np.power(r / L0, 2))
            vz = vj / (1 + 3 / gamma * np.power(r / L0, 2))
        elif self.model == "post-shock":
            vr = vj * 3 * np.power(r / L0, 2) / (1 + 9 * np.power(r / L0, 4))
            vz = vj / (1 + 9 * np.power(r / L0, 4))
        vphi = np.zeros_like(z)
        dv = dl * r * dl
        I = dv * np.power(r, -alpha)
        r = r / 1.5e11
        z = z / 1.5e11
        vr = vr / 1e3
        vz = vz / 1e3

        profile = np.dstack((r, z, vr, vz, vphi, I))
        return phi_1D, profile

    def plot_dynamics(self, plot_J=True, savename=""):
        _, param = np.copy(self.create_profile())
        cmap = ["inferno", "cividis", "viridis"]
        cbar_label = [
            r"V$_R$ (kms$^{-1}$)",
            r"V$_Z$ (kms$^{-1}$)",
            r"V$_{\Phi}$ (kms$^{-1}$)",
        ]
        if plot_J:
            param[:, :, 4] = np.copy(param[:, :, 4] * param[:, :, 0])
            cbar_label[2] = r"RV$_{\Phi}$ (aukms$^{-1}$)"
        fig, ax = plt.subplots(
            nrows=1, ncols=3, sharey=True, sharex=True, figsize=(15, 4)
        )
        for i in range(3):
            im = ax[i].scatter(
                param[:, :, 0], param[:, :, 1], c=param[:, :, i + 2], s=1, cmap=cmap[i]
            )
            cbar = plt.colorbar(im, ax=ax[i], pad=0)
            cbar.set_label(cbar_label[i], rotation=0, labelpad=-20, y=1.07, ha="center")
        ax[0].set_xlabel("R (au)")
        ax[0].set_ylabel("Z (au)")
        if savename:
            plt.savefig(
                savename + ".png",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
                dpi=400,
            )
        plt.show()


class Your_model:
    """
    Particular model defined by the user, it must be defined by :
    -  Z
    -  R
    -  VR
    -  VZ
    -  VPHI
    -  I

    Each parameter can be either an 1D array or a 2D array. but must have the same size.
    """

    def __init__(self, r, z, vr, vz, vphi, i):
        """
        class constructor
        """
        if are_not_list([r, z, vr, vz, vphi, i]):
            sys.exit("Error : all input must be arrays.")
        else:
            if not (are_same_size([r, z, vr, vz, vphi, i])):
                sys.exit("Error : all input must have the same size.")
            elif len(np.shape(r)) == 1:
                r = np.expand_dims(np.asarray(r), axis=0)
                z = np.expand_dims(np.asarray(z), axis=0)
                vr = np.expand_dims(np.asarray(vr), axis=0)
                vz = np.expand_dims(np.asarray(vz), axis=0)
                vphi = np.expand_dims(np.asarray(vphi), axis=0)
                i = np.expand_dims(np.asarray(i), axis=0)
            elif len(np.shape(r)) == 2:
                r = np.asarray(r)
                z = np.asarray(z)
                vr = np.asarray(vr)
                vz = np.asarray(vz)
                vphi = np.asarray(vphi)
                i = np.asarray(i)
            else:
                sys.exit("Error : array must be either 1D or 2D.")

        self.r = r
        self.z = z
        self.vr = vr
        self.vz = vz
        self.vphi = vphi
        self.i = i

    def create_setup(self, dphi):
        self.dphi = dphi
        print("Setting up the resolution")

    def create_profile(self):
        phi_1D = np.arange(0, 360, self.dphi) * np.pi / (180.0)
        profile = np.dstack(
            (self.r, self.z, self.vr, self.vz, self.vphi, self.i * self.r)
        )
        return phi_1D, profile

    def plot_dynamics(self, plot_J=True, savename=""):
        _, param = np.copy(self.create_profile(0, 0))
        cmap = ["inferno", "cividis", "viridis"]
        cbar_label = [
            r"V$_R$ (kms$^{-1}$)",
            r"V$_Z$ (kms$^{-1}$)",
            r"V$_{\Phi}$ (kms$^{-1}$)",
        ]
        if plot_J:
            param[:, :, 4] = param[:, :, 4] * param[:, :, 0]
            cbar_label[2] = r"RV$_{\Phi}$ (aukms$^{-1}$)"
        fig, ax = plt.subplots(
            nrows=1, ncols=3, sharey=True, sharex=True, figsize=(15, 4)
        )
        for i in range(3):
            im = ax[i].scatter(
                param[:, :, 0], param[:, :, 1], c=param[:, :, i + 2], s=1, cmap=cmap[i]
            )
            cbar = plt.colorbar(im, ax=ax[i], pad=0)
            cbar.set_label(cbar_label[i], rotation=0, labelpad=-20, y=1.07, ha="center")
        ax[0].set_xlabel("R (au)")
        ax[0].set_ylabel("Z (au)")
        if savename:
            plt.savefig(
                savename + ".png",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0,
                dpi=400,
            )
        plt.show()

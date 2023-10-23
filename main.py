import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator

class ColorCorrections:
    """
    A class for calculating color corrections for astronomical observations.

    This class provides methods for computing color corrections based on redshift
    and cosmological parameters using data from Poggianti tables.

    Args:
        redshift_range (numpy.ndarray): An array of redshift values.
        h_0 (float): The Hubble constant in units of km/s/Mpc.
        omega_m (float): The matter density parameter.

    Attributes:
        redshift_range (numpy.ndarray): An array of redshift values.
        h_0 (float): The Hubble constant in units of km/s/Mpc.
        omega_m (float): The matter density parameter.
        cosmology (astropy.cosmology.FlatLambdaCDM): A cosmology object
            based on H0 and Om0.
        z_k (numpy.ndarray): Redshift values from the k-correction data.
        kcorr (numpy.ndarray): k-correction values from the k-correction data.
        z_e (numpy.ndarray): Redshift values from the e-correction data.
        ecorr (numpy.ndarray): e-correction values from the e-correction data.

    Note:
        The class loads redshift, k-correction, and e-correction data from external
        files ('kcorr.dat' and 'ecorr.dat') for further calculations.
        The loaded data is specific to r-band E (red) galaxies and is relevant for
        intrinsic alignments (IA) in astronomy. Data is taken form Poggianti 1997
        (check https://arxiv.org/abs/astro-ph/9608029 for details).
    """

    def __init__(self, redshift_range, h_0, omega_m):
        self.redshift_range = redshift_range
        self.h_0 = h_0
        self.omega_m = omega_m
        self.cosmology = FlatLambdaCDM(H0=self.h_0 * u.km / u.s / u.Mpc, Om0=self.omega_m)

        # Load the redshift and k and e corrections from Poggianti tables
        # this particular dataset is for r-band E (red) galaxies
        # which is relevant for IA
        self.z_k, self.kcorr, _, _, _ = np.loadtxt('data_input/kcorr.dat', unpack=True)
        self.z_e, self.ecorr, _, _, _ = np.loadtxt('data_input/ecorr.dat', unpack=True)
        

    def poggianti1997_time(self, redshift):
        """Poggianti, 1997, eq. 4.
        Consult https://arxiv.org/abs/astro-ph/9608029 for more info.

        Get time since Big Bang at redshift z, in the decelerating universe
        of the paper.

        Parameters
        ----------
        redshift : float or array
            Redshift

        Return
        ------
        t : astropy.units.Quantity
            Time since Big Bang, in Gyr
        """
        q0 = 0.225  # used in Poggianti 1997
        H0 = 50 * u.km / u.s / u.Mpc  # used in Poggianti 1997
        term1 = -4. * q0 / (H0 * np.power(1. - 2. * q0, 1.5))

        root_val = np.sqrt((1. + 2. * q0 * redshift) / (1. - 2. * q0))
        term2 = root_val
        term3 = 2.0 * (1. - (1. + 2. * q0 * redshift) / (1. - 2. * q0))
        term4 = 0.25 * np.log(np.abs((1 + root_val) / (1. - root_val)))

        # Compute the final result for time
        time = term1 * (term2 / term3 + term4)

        return time
        

    def poggianti1997_lookback_time(self, redshift):
        """Lookback time in the decelerating universe of Poggianti, 1997
        (see https://arxiv.org/abs/astro-ph/9608029 for more info).

        Parameters
        ----------
        redshift : float or array
            Redshift

        Return
        ------
        t : astropy.units.Quantity
            Lookback time, in Gyr
        """
        return self.poggianti1997_time(0) - self.poggianti1997_time(redshift)
        

    def lookback_time_to_redshift(self, lb_time):
        """Get redshift (in the accelerating universe) from lookback time

        Parameters
        ----------
        lb_time : astropy.units.Quantity
            Lookback time

        Return
        ------
        z : float or array
            Redshift
        """
        # This is practically the inverse of lookback time function,
        # to resolve the redshift from the lookback time.
        # It can be reimplemented manually with scipy.optimize.root_scalar
        return z_at_value(self.cosmology.lookback_time, lb_time)
        

    def poggianti1997_to_accelerating_redshift(self, redshift):
        """Get redshift in the accelerating universe from redshift
        in the decelerating universe of Poggianti, 1997"""
        return self.lookback_time_to_redshift(self.poggianti1997_lookback_time(redshift))
        

    def make_kcorr_spline(self):
        """Make a spline for k-correction
        Return
        ------
        spline : scipy.interpolate.CubicSpline
            Cubic spline for k-correction, input is redshift
        """

        z = self.z_k
        kcorr = self.kcorr
        z = np.r_[0, z]
        kcorr = np.r_[0, kcorr]
        # spline = CubicSpline(z, kcorr, bc_type='natural', extrapolate=False)
        spline = Akima1DInterpolator(z, kcorr)

        slope = spline.derivative()(z[-1])
        intercept = kcorr[-1] - slope * z[-1]

        z_end_point = 1000
        kcorr_end_point = slope * z_end_point + intercept

        z = np.r_[z, z_end_point]
        kcorr = np.r_[kcorr, kcorr_end_point]

        # return CubicSpline(z, kcorr, bc_type='natural', extrapolate=False)
        return Akima1DInterpolator(z, kcorr)
        

    def make_ecorr_spline(self, original_z=False):
        """Make a spline for e-corrections.

        Parameters
        ----------
        original_z : bool
            If True, use original redshifts from the paper (decelerating universe).
            If False, use redshifts in the accelerating universe.

        Return
        ------
        spline : scipy.interpolate.CubicSpline
            Cubic spline for e-correction, input is redshift
        """

        z = self.z_e
        ecorr = self.ecorr
        if not original_z:
            z = self.lookback_time_to_redshift(self.poggianti1997_lookback_time(z))

        z = np.r_[0, z]
        ecorr = np.r_[0, ecorr]
        # return CubicSpline(z, ecorr, bc_type='natural', extrapolate=False)
        return Akima1DInterpolator(z, ecorr)
        

    def get_color_corrections(self):
        """
        Calculate color corrections for a given redshift range.
    
        This function calculates the k-corrections and e-corrections
        using precomputed splines for a given redshift range.
    
        Returns:
        k_corrections (numpy.ndarray): Array of k-corrections.
        e_corrections (numpy.ndarray): Array of e-corrections.
        """
        # Calculate the k-corrections using a precomputed spline
        kcorr_spline = self.make_kcorr_spline()
        k_corrections = kcorr_spline(self.redshift_range)
    
        # Calculate the e-corrections using a precomputed spline
        ecorr_spline = self.make_ecorr_spline(original_z=False)
        e_corrections = ecorr_spline(self.redshift_range)
    
        return k_corrections, e_corrections


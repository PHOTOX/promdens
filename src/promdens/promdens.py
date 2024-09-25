#!/usr/bin/env python3
"""Promoted Density Approach code for including laser pulse effects into initial conditions for nonadiabatic dynamics.

© Jiri Janos 2024
"""

# /// script
# requires-python = ">=3.7"
# dependencies = [
#     "numpy>=1.15",
#     "matplotlib~=3.0",
# ]
# ///

import argparse
import dataclasses
from pathlib import Path
from timeit import default_timer as timer

import numpy as np

ENERGY_UNITS = ['a.u.', 'eV', 'nm', 'cm-1']
TDM_UNITS = ['a.u.', 'debye']
METHODS = ['pda', 'pdaw']
ENVELOPE_TYPES = ['gauss', 'lorentz', 'sech', 'sin', 'sin2']
NEG_PROB_HANDLING = ['error', 'ignore', 'abs']
FILE_TYPES = ['file']

DESC = "Promoted density approach for initial conditions"


### functions and classes ###
def print_header():
    print("\n##########################################################\n"
          f"###  {DESC}  ###\n"
          "###                   * * * * *                        ###\n"
          "###       version 1.0         Jiri Janos 2024          ###\n"
          "##########################################################\n")


def print_footer():
    print('\nPromoted density approached calculation finished.'
          '\n - "May the laser pulses be with you."\n')
    print("       %.                                \n"
          "        %.                    #%%%%%%%%  \n"
          "         %.        %%        %%%%%%%%%%% \n"
          "          %.    %%%%%       %%%%%%%%%%%% \n"
          "           %  %%%%%%%%%    %%%%%%%%%%%%* \n"
          "            %  %%%%%%%%%   %%%%%%%%%%%   \n"
          "             %    %%%%%%%  %%%%%%%%      \n"
          "              %%%%%%%%%%%  %%%%%%%       \n"
          "              %%+%%%%%%%%% %%%%%%%       \n"
          "                   %%%%%%% %%%%%%        \n"
          "                %%%%%%%%%% %%%%%%        \n"
          "                 %%%%%%%%%%%%%%%         \n"
          "                %%%%%%%%%%%%%%           \n"
          "                                           ")


def positive_int(str_value: str) -> int:
    """Convert a string into a positive integer.

    This is a helper type conversion function
    for user input type checking by argparse, see:
    https://docs.python.org/3/library/argparse.html#type

    raises: ValueError if string is not a positive integer
    """
    val = int(str_value)
    if val <= 0:
        raise ValueError(f"'{val}' is not a positive integer")
    return val


def positive_float(str_value: str) -> float:
    """Convert a string into a positive float.

    raises: ValueError if string is not a positive real number
    """
    val = float(str_value)
    if val <= 0:
        raise ValueError(f"'{val}' is not a positive real number")
    return val


@dataclasses.dataclass
class LaserPulse:
    omega: float
    fwhm: float
    envelope_type: str
    t0: float
    lchirp: float

    equation: str = dataclasses.field(init=False)
    tmin: float = dataclasses.field(init=False)
    tmax: float = dataclasses.field(init=False)

    def __str__(self):
        return (f"Laser pulse parameters:\n"
                f"  E(t) = {self.equation}\n"
                f"  - envelope type = '{self.envelope_type}'\n"
                f"  - omega = {self.omega:.6f} a.u.\n"
                f"  - FWHM = {self.fwhm:.6f} a.u.\n"
                f"  - t0 = {self.t0:.6f} a.u.\n"
                f"  - chirp = {self.lchirp:.3e} a.u.")

    def __post_init__(self):
        if self.envelope_type not in ENVELOPE_TYPES:
            msg = f'Invalid envelope type "{self.envelope_type}"'
            raise ValueError(msg)

        if self.envelope_type == 'gauss':
            self.equation = "exp(-2*ln(2)*(t-t0)^2/fwhm^2)*cos((omega+chirp*t)*t)"
            temporal_extent = 2.4*self.fwhm
        elif self.envelope_type == 'lorentz':
            self.equation = "(1+4/(1+sqrt(2))*(t/fwhm)^2)^-1*cos((omega+chirp*t)*t)"
            temporal_extent = 8*self.fwhm
        elif self.envelope_type == 'sech':
            self.equation = "sech(2*ln(1+sqrt(2))*t/fwhm)*cos((omega+chirp*t)*t)"
            temporal_extent = 4.4*self.fwhm
        elif self.envelope_type == 'sin':
            self.equation = "sin(pi/2*(t-t0+fwhm)/fwhm)*cos((omega+chirp*t)*t) in range [t0-fwhm,t0+fwhm]"
            temporal_extent = self.fwhm
        elif self.envelope_type == 'sin2':
            self.equation = "sin(pi/2*(t-t0+T)/T)^2*cos((omega+chirp*t)*t) in range [t0-T,t0+T] where T=1.373412575*fwhm"
            temporal_extent = 1/(2 - 4/np.pi*np.arcsin(2**(-1/4)))*self.fwhm

        self.tmin = self.t0 - temporal_extent
        self.tmax = self.t0 + temporal_extent

    def field_cos(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate oscillations of the electric field with the cos function.

        :param t: array of time values in atomic units
        :return: array of cos((w + lchirp*t)*t)
        """
        return np.cos((self.omega + self.lchirp*t)*t)

    def calc_field_envelope(self, t: np.ndarray) -> np.ndarray:
        """
        Calculate field envelope on a grid of time values.

        :param t: array of time values (a.u.)
        :return: array of envelope of the electric field
        """
        if self.envelope_type == 'gauss':
            return np.exp(-2*np.log(2)*(t - self.t0)**2/self.fwhm**2)
        elif self.envelope_type == 'lorentz':
            return (1 + 4/(1 + np.sqrt(2))*((t - self.t0)/self.fwhm)**2)**-1
        elif self.envelope_type == 'sech':
            return 1/np.cosh(2*np.log(1 + np.sqrt(2))*(t - self.t0)/self.fwhm)
        elif self.envelope_type == 'sin':
            field = np.zeros(shape=np.shape(t))
            trange = np.logical_and(t >= self.tmin, t <= self.tmax)
            field[trange] = np.sin(np.pi/2*(t[trange] - self.t0 + self.fwhm)/self.fwhm)
            return field
        elif self.envelope_type == 'sin2':
            T = 1/(2 - 4/np.pi*np.arcsin(2**(-1/4)))*self.fwhm
            field = np.zeros(shape=np.shape(t))
            trange = np.logical_and(t >= self.tmin, t <= self.tmax)
            field[trange] = np.sin(np.pi/2*(t[trange] - self.t0 + T)/T)**2
            return field

    def wigner_transform(self, tprime, de):
        """
        Wigner transform of the pulse. The current implementation uses the pulse envelope formulation to simplify calculations.
        The integral is calculated numerically. Analytic formulas could be implemented here.

        :param tprime: the excitation time t' (a.u.)
        :param de: excitation energy (a.u.)
        :return: Wigner pulse transform
        """

        loc_omega = self.omega + 2*self.lchirp*tprime
        # analytic integrals if available
        if self.envelope_type == 'gauss':
            integral = 16**(-(tprime - self.t0)**2/self.fwhm**2)*np.exp(
                -self.fwhm**2*(de - loc_omega)**2/np.log(16))*self.fwhm*np.sqrt(np.pi/np.log(2))
        elif self.envelope_type == 'sin':
            if de == loc_omega:
                loc_omega += 1e-10  # because we can divide by 0 but the integral is constant for small loc_omega
            w = de - loc_omega
            if tprime < self.t0 and tprime > self.t0 - self.fwhm:
                integral = (np.pi*(-2*self.fwhm*w*np.cos(2*(tprime - self.t0 + self.fwhm)*w)*np.sin(
                    np.pi*(tprime - self.t0)/self.fwhm) + np.pi*np.cos(np.pi*(tprime - self.t0)/self.fwhm)*np.sin(
                    2*(tprime - self.t0 + self.fwhm)*w))/(w*(np.pi**2 - 4*self.fwhm**2*w**2)))
            elif tprime >= self.t0 and tprime < self.t0 + self.fwhm:
                integral = (np.pi*(2*self.fwhm*w*np.cos(2*(-tprime + self.t0 + self.fwhm)*w)*np.sin(
                    np.pi*(tprime - self.t0)/self.fwhm) + np.pi*np.cos(np.pi*(tprime - self.t0)/self.fwhm)*np.sin(
                    2*(-tprime + self.t0 + self.fwhm)*w))/(w*(np.pi**2 - 4*self.fwhm**2*w**2)))
            else:
                integral = 0
        else:  # numerical integrals for the remaining envelopes
            # setting an adaptive integration step according to the frequency of the integrand oscillations (de - omega)
            # If de == loc_omega, there are no oscillations and we integrate only the envelope intensity
            if de == loc_omega:
                dt = self.fwhm/500
            else:  # out of resonance, we set the integration time step based on the frequency of oscillations
                # dt = 2*np.pi/np.abs(de - loc_omega)/50 # this would be the best formula but we use the one below
                # because if we are too far from resonance, the numeric integral is extremely expensive just to give us zero
                dt = 2*np.pi/np.min([np.abs(de - loc_omega), loc_omega])/50

            ds = np.min([dt, self.fwhm/500])  # time step for integration

            # integration ranges for different pulse envelopes
            # ideally, we would integrate from -infinity to infinity, yet this is not very computationally efficient
            # empirically, it was found out that efficient integration varies for different pulses
            # analytic formulas should be implemented in the future to avoid that
            factor = {
                'lorentz': 50,
                'sech'   : 20,
                'sin2'   : 4, }

            # instead of calculating the complex integral int_{-inf}^{inf}[E(t+s/2)E(t-s/2)exp(i(w-de)s)]ds we use the
            # properties of even and odd fucntions and calculate 2*int_{0}^{inf}[E(t+s/2)E(t-s/2)cos((w-de)s)]ds
            s = np.arange(0, factor[self.envelope_type]*self.fwhm, step=ds)
            # Note: We assume here that de is in atomic units, otherwise it needs to be divided by hbar
            cos = np.cos((de - loc_omega)*s)
            integral = 2*np.trapz(x=s,
                y=cos*self.calc_field_envelope(tprime + s/2)*self.calc_field_envelope(tprime - s/2))

        return integral


class InitialConditions:
    """The main class around which the code is build containing all the data and functions.
    The class:
       - reads the input file and stores excitation energies and magnitudes of transition dipole moment |mu_ij|
       - converts units of the inputed energies and TDMs
       - calculates absorption spectrum with Nuclear Ensemble Method (NEA)
       - calculates the laser pulse electric field and its spectrum
       - creates initial conditions in the excited states based on PDA using wigner pulse envelope transformation
       - calculates weights and convolution function for windowing based on PDAW
    More details are provided in the functions below."""

    # conversion factor between units
    evtoau = 0.036749405469679
    nmtoau = 45.56335
    cm1toau = 0.0000045563352812122295
    fstoau = 41.341374575751
    debtoau = 0.393456
    autocm = 8.478354e-30  # dipole moment conversion

    def __init__(self, nsamples=0, nstates=1, input_type='file'):
        """
        Initialization of the class.
        :param nsamples: number of inputed samples (position-momentum pairs), if set 0 then maximum number provided will be taken
        :param nstates: number of excited states considered in the calculation
        :param input_type: input file type, currently only 'file'; possible extensions to other sampling codes
        """
        self.nsamples = nsamples
        self.nstates = nstates
        self.input_type = input_type

    def read_input_data(self, fname: str, energy_unit: str, tdm_unit: str) -> None:
        """
        Reading the input data: index of traj, excitation energies and magnitudes of transition dipole moments.
        :param fname: name of the input file
        :return: store all the data within the class
        """
        print(f"* Reading data from file '{fname}' of input file type '{self.input_type}'.")
        try:
            input = np.loadtxt(fname, dtype=float).T  # reading input file with numpy
        except FileNotFoundError as err:
            print(f"\nERROR: Input file '{fname}' not found!\n (Error: {err})")
            exit(1)
        except ValueError as err:
            print(err)
            print(f"\nERROR: Incorrect value type encountered in the input file '{fname}'!\n (Error: {err})")
            exit(1)
        except Exception as err:
            print(f"\nERROR: Unexpected error: {err}, type: {type(err)}")
            exit(1)

        if np.shape(input)[0] < self.nstates*2 + 1:  # check enough columns provided in the file for required nstates
            print(f"\nERROR: Not enough columns provided in the input file '{fname}'! "
                  f"\nExpected {self.nstates*2 + 1} columns for {self.nstates} excited states.")
            exit(1)

        if self.nsamples == 0:  # use all samples loaded if user input nsamples is 0
            self.nsamples = np.shape(input)[1]
            print(f"  - Number of samples loaded from the input file: {self.nsamples}")
        elif self.nsamples > np.shape(input)[1]:  # check if requested nsamples is not larger than provided in the input
            print(
                f"  - Number of samples loaded from the input file {np.shape(input)[1]} instead of requested {self.nsamples}")
            self.nsamples = np.shape(input)[1]

        self.traj_index = np.array(input[0, :self.nsamples], dtype=int)  # saving indexes of trajectories
        self.de = input[1:self.nstates*2:2, :self.nsamples]  # saving excitation energies
        self.tdm = input[2:self.nstates*2 + 1:2, :self.nsamples]  # saving transition dipole moments

        # converting energy and tdm units to a.u.
        self.convert_units(energy_unit=energy_unit, tdm_unit=tdm_unit)

    def convert_units(self, energy_unit: str, tdm_unit: str) -> None:
        """
        Convert all the data into atomic units which are used throughout the code.
        :param energy_unit: energy unit on the input
        :param tdm_unit: unit of transition dipole moments on the input
        """

        if not (energy_unit == 'a.u.' and tdm_unit == 'a.u.'):
            print("* Converting units.")

        if energy_unit == 'eV':
            self.de *= self.evtoau
            print("  - 'eV' -> 'a.u.'")
        elif energy_unit == 'nm':
            self.de *= self.nmtoau
            print("  - 'nm' -> 'a.u.'")
        elif energy_unit == 'cm-1':
            self.de *= self.cm1toau
            print("  - 'cm-1' -> 'a.u.'")

        if tdm_unit == 'debye':
            self.tdm *= self.debtoau
            print("  - 'debye' -> 'a.u.'")

    def calc_spectrum(self):
        """
        Calculating spectrum with the Nuclear Ensemble Approach. The calculated spectrum is in absorption cross-section
        units (cm^2*molecule^-1). Conversion factor to molar absorption coefficient (dm^3*mol^-1*cm^-1) is 6.022140e20 / ln(10).
        """

        def gauss(e, de, tdm, h):
            """
            Gaussian function used in the spectrum calculation
            :param e: energy axis (a.u.)
            :param de: excitation energy (centre of the Gaussian, a.u.)
            :param tdm: transition dipole moment (a.u.)
            :param h: width of the Gaussian (a.u.)
            :return: Gaussian as a function of energy with intensity proportional to TDM
            """
            return de*tdm**2*np.exp(-(e - de)**2/2/h**2)

        print("* Calculating spectrum with the Nuclear Ensemble Approach.")

        # calculating coefficient for intensity of the spectrum
        eps0 = 8.854188e-12
        hbar = 6.626070e-34/(2*np.pi)
        c = 299792458
        int_coeff = np.pi*self.autocm**2*1e4/(3*hbar*eps0*c)/self.nsamples/np.sqrt(2*np.pi)

        # width for the Gaussians set by Silverman’s rule of thumb: https://doi.org/10.1039/C8CP00199E
        emin, emax = np.min(self.de), np.max(self.de)
        self.spectrum = np.zeros(shape=(self.nstates + 2, 10000), dtype=float)
        h = (4/3/self.nsamples)**0.2*np.std(self.de)  # width h for all data to get energy range of the spectrum
        self.spectrum[0] = np.linspace(emin - 2*h, emax + 2*h, np.shape(self.spectrum)[1])

        for state in range(self.nstates):
            h = (4/3/self.nsamples)**0.2*np.std(self.de[state])  # width h for individual states

            for ic in range(self.nsamples):
                self.spectrum[state + 1] += gauss(self.spectrum[0], self.de[state, ic], self.tdm[state, ic], h)

            self.spectrum[state + 1] *= int_coeff/h

        # calculating total spectrum (summing over all states)
        self.spectrum[-1] = np.sum(self.spectrum[1:-1], axis=0)

    def calc_field(self, pulse: LaserPulse) -> None:
        """
        Calculate electric field E(t) and its spectrum E(w) with Fourier transform.

        :param pulse: LaserPulse dataclass containing laser pulse parameters (frequency, fwhm...)
        """
        self.pulse = pulse

        # calculating the field
        dt = 2*np.pi/self.pulse.omega/50
        self.field_t = np.arange(self.pulse.tmin, self.pulse.tmax, dt)  # time array for the field in a.u.
        self.field_envelope = self.pulse.calc_field_envelope(self.field_t)
        self.field = self.field_envelope*self.pulse.field_cos(self.field_t)

        # calculating the FT of the field (pulse spectrum)
        extra_t = 20*self.pulse.fwhm
        t_ft = np.arange(self.pulse.tmin - extra_t, self.pulse.tmax + extra_t,
            dt)  # Set up a new longer time array for FT
        field = self.pulse.calc_field_envelope(t_ft)*self.pulse.field_cos(t_ft)  # calculate field for FT
        self.field_ft = np.abs(np.fft.rfft(field))  # FT
        self.field_ft /= np.max(self.field_ft)  # normalizing to have maximum at 0
        self.field_ft_omega = 2*np.pi*np.fft.rfftfreq(len(t_ft), dt)

        if self.is_maxwell_fulfilled():
            print("  - WARNING: Pulse is too short and integral of E(t) is not equal to 0 - Maxwell's equations are "
                  "not fulfilled. This means\n    that the representation of the pulse as envelope times cos(wt) is not physical."
                  " See the original reference for more details.")
        else:
            print("  - Integral of E(t) from -infinity to infinity is equal to 0 - pulse is physically realizable.")

    def is_maxwell_fulfilled(self):
        """Check if the pulse fulfills Maxwell's equations
        (integral from -infinity to infinity of E(t) = E(w=0) 0)
        """
        # the integral is equal to spectrum at zero frequency
        if self.field_ft_omega[0] == 0:
            integral = self.field_ft[0]
        else:  # in case the first element is not zero frequency (which should not be at the current version of python)
            integral = self.field_ft[self.field_ft_omega == 0]
        # empirical threshold which considers the spectrum has maximum equal to 1
        if integral > 0.01:
            return False
        else:
            return True

    def sample_initial_conditions(self, nsamples_ic: int, neg_handling: str, preselect: bool, seed=None,
                                  output_fname='pda.dat'):
        """
        Sample time-dependent initial conditions using the Promoted Density Approach.
        :param nsamples_ic: number of initial conditions to be sampled
        :return: store the initial conditions within the class and save output
        """

        print(f"* Sampling {nsamples_ic} initial conditions considering the laser pulse.")

        def progress(percent, width, n, str=''):
            """Function to print progress of calculation."""
            left = width*percent//n
            right = width - left
            print(f'\r{str}[', '#'*left, ' '*right, '] %d'%(percent*100/n) + '%', sep='', end='', flush=True)

        # variable storing initial conditions
        samples = np.zeros((5, nsamples_ic))  # index, excitation time, initial excited state, de, tdm

        # setting maximum random number generated during sampling by taking the maximum TDM and the maximum of
        # wigner transform (which is at tprime = t0 and de = w + lch*t0)
        resonant_de = self.pulse.omega + self.pulse.lchirp*self.pulse.t0
        rnd_max = np.max(self.tdm**2)*self.pulse.wigner_transform(self.pulse.t0, de=resonant_de)*1.01

        # preselection of initial conditions based on pulse spectrum in order to avoid long calculation of the Wigner
        # distribution for samples far from resonance (the more out of resonance with the field, the more the integrand
        # oscillates and the finer grid is needed, although the result is zero; will be resolved with analytic integrals)
        # True - sample will be skipped; False - sample will be considered for sampling of initial conditions
        if preselect:
            preselected = np.interp(x=self.de, xp=self.field_ft_omega,
                fp=self.field_ft) < 1e-6  # considering field_ft is normalized and positive
            print(
                f"  - Discarding {np.sum(preselected):d} samples that are not within the pulse spectrum [sigma(dE) < 10^-6].")
        else:
            preselected = np.zeros(shape=(self.nstates, self.nsamples), dtype=bool)

        # setting up random generator with a seed (None means random seed form OS taken)
        rng = np.random.default_rng(seed=seed)

        i, nattempts, start = 0, 0, timer()  # i: loop index; nattempts: to calculate efficiency of the sampling; start: time t0
        while i < nsamples_ic:  # while is used in case we need to restart the loop when probability exceeds rnd_max
            # randomly selecting index of traj and excited state
            rnd_index = rng.integers(low=0, high=self.nsamples, dtype=int)
            rnd_state = rng.integers(low=0, high=self.nstates, dtype=int)

            # checking if the sample was preselected for discarding
            if preselected[rnd_state, rnd_index]:
                continue

            nattempts += 1

            # selecting randomly excitation time and random uniform number
            rnd_time = rng.uniform(low=self.pulse.tmin, high=self.pulse.tmax)
            rnd = rng.uniform(low=0, high=rnd_max)  # random number to be compared with Wig. dist.

            # excitation probability
            de = self.de[rnd_state, rnd_index]
            tdm = self.tdm[rnd_state, rnd_index]
            prob = tdm**2*self.pulse.wigner_transform(rnd_time, de)

            # check and handle negative probabilities
            if prob < -1e-12*rnd_max:  # check negative value bigger than integration precision
                if neg_handling == 'error':
                    print(
                        f"\nERROR: Negative probability ({prob/rnd_max*100:.1e}%) encountered! Check flag 'neg_handling' "
                        f"for more option how to handle negative probabilities. See also manual and ref XXX for more "
                        f"information.\n")
                    exit(1)
                elif neg_handling == 'ignore':
                    continue
                elif neg_handling == 'abs':
                    prob = np.abs(prob)

            if prob > rnd_max:  # check if the probability is higher than rnd_max, restart while loop
                print(f"\n - rnd_max ({rnd_max}) is smaller than probability ({prob} for sample "
                      f"{self.traj_index[rnd_index]} on state {rnd_state}). Increasing rnd_max and reruning.")
                rnd_max *= 1.2  # increase rnd_max
                samples = np.zeros((5, nsamples_ic))
                i = 0  # reset index
            elif rnd <= prob:  # check if the initial condition was accepted and save it
                samples[0, i] = self.traj_index[rnd_index]
                samples[1, i] = rnd_time
                samples[2, i] = rnd_state + 1
                samples[3, i] = self.de[rnd_state, rnd_index]
                samples[4, i] = self.tdm[rnd_state, rnd_index]
                i += 1
                progress(i, 50, nsamples_ic, str='  - Sampling progress: ')

        # saving samples within the object
        samples = samples[:, samples[0].argsort()]  # sorting according to sample index
        self.ics = samples

        print(
            f"\n  - Time: {timer() - start:.3f} s\n  - Success rate of random sampling: {nsamples_ic/nattempts*100:.5f}%")

        # getting unique initial conditions for each excited state
        unique_states, unique = np.zeros(shape=(self.nstates), dtype=int), []
        for state in range(self.nstates):
            unique.append(np.array(np.unique(samples[0, samples[2] == state + 1]), dtype=int))  # unique sample indexes
            unique_states[state] = len(unique[state])  # number of unique sample indexes for given state

        # print indexes of trajectories that must be propagated
        if self.nstates == 1:
            print(f"  - Selected {unique_states[0]} unique ICs (unique sample indexes) from "
                  f"{self.nsamples} provided. Unique ICs to be propagated:\n   ", *np.array(unique[0], dtype=str))
        else:
            print(f"  - Selected {np.sum(unique_states)} unique ICs over {self.nstates} states from {self.nsamples} "
                  f"positions and velocities provided. \n    Unique IC corresponds to a unique sample index and initial "
                  f"electronic state.")
            for state in range(self.nstates):
                if unique_states[state] == 0:
                    continue
                print(f"  - State {state + 1} - {unique_states[state]} unique ICs to be propagated: \n   ",
                    *np.array(unique[state], dtype=str))

        # save the selected samples
        header = (f"Sampling: number of ICs = {nsamples_ic}, number of unique ICs = {np.sum(unique_states):d}\n"
                  f"Field parameters: omega = {self.pulse.omega:.5e} a.u., "
                  f"linear_chirp = {self.pulse.lchirp:.5e} a.u., fwhm = {self.pulse.fwhm/self.fstoau:.3f} fs, "
                  f"t0 = {self.pulse.t0/self.fstoau:.3f} fs, envelope type = '{self.pulse.envelope_type}'\n"
                  f"index        exc. time (a.u.)   el. state     dE (a.u.)       |tdm| (a.u.)")
        np.savetxt(output_fname, samples.T, fmt=['%8d', '%18.8f', '%12d', '%16.8f', '%16.8f'], header=header)
        print(f"  - Output saved to file '{output_fname}'")

    def windowing(self, output_fname='pdaw.dat'):
        """
        Performs Promoted Density Approach for Windowing (PDAW). The function calculates normalized weights and outputs
        the convolution functions I(t).
        :return: Prints analysis of windowing weights together with convolutions functions and saves all the weights
                 to an output file.
        """

        print("* Generating weights and convolution for windowing.")

        # determine and print convolution function
        # determine and print convolution function
        conv_eq = {
            'gauss'  : "I(t) = exp(-4*ln(2)*(t-t0)^2/fwhm^2)",
            'lorentz': "I(t) = (1+4/(1+sqrt(2))*(t/fwhm)^2)^-2",
            'sech'   : "I(t) = sech(2*ln(1+sqrt(2))*t/fwhm)^2",
            'sin'    : "I(t) = sin(pi/2*(t-t0+fwhm)/fwhm)^2 in range [t0-fwhm,t0+fwhm]",
            'sin2'   : "I(t) = sin(pi/2*(t-t0+T)/T)^4 in range [t0-T,t0+T] where T=1.373412575*fwhm", }
        print(f"  - Convolution: {conv_eq[self.pulse.envelope_type]}\t\t\t [The function must be normalized!]\n"
              f"  - Parameters:  fwhm = {self.pulse.fwhm/self.fstoau:.3f} fs, "
              f"t0 = {self.pulse.t0/self.fstoau:.3f} fs")

        print("  - Calculating normalized weights:")
        # creating a field for weights
        self.weights = np.zeros((self.nstates, self.nsamples))  # sample index, weights in different states

        # generating weights for all states and samples
        for state in range(0, self.nstates):
            self.weights[state] = self.tdm[state]**2*np.interp(self.de[state], self.field_ft_omega, self.field_ft)**2

            # analysis
            sorted = np.sort(self.weights[state, :]/np.sum(self.weights[state, :]))[
                     ::-1]  # sorting from the largest weight to smallest
            print(
                f"    > State {state + 1} -  analysis of normalized weights (weights/sum of weights on state {state + 1}):\n"
                f"      - Largest weight: {np.max(self.weights[state, :]):.3e}\n"
                f"      - Number of ICs making up 90% of S{state + 1} weights: {np.sum(np.cumsum(sorted) < 0.9) + 1:d}\n"
                f"      - Number of ICs with weights bigger than 0.001: {np.sum(self.weights[state, :] > 0.001):d}")

        # normalization of weights at given state
        self.weights /= np.sum(self.weights)

        # creating a variable for printing with first column being sample indexes
        arr_print = np.zeros((self.nstates + 1, self.nsamples))  # index, weights in different states
        arr_print[0, :] = self.traj_index
        arr_print[1:, :] = self.weights

        spacer = ' '*8
        weights_str = spacer.join([f'weight S{s + 1}' for s in range(self.nstates)])
        header = (f"Convolution: {conv_eq[self.pulse.envelope_type]}\n"
                  f"Parameters:  fwhm = {self.pulse.fwhm/self.fstoau:.3f} fs, "
                  f"t0 = {self.pulse.t0/self.fstoau:.3f} fs\n"
                  f"index        {weights_str}")
        np.savetxt(output_fname, arr_print.T, fmt=['%8d'] + ['%16.5e']*self.nstates, header=header)

        print(f"  - Weights saved to file '{output_fname}'")


def plot_spectrum(ics: InitialConditions) -> None:
    import matplotlib.pyplot as plt

    print("  - Plotting UV/vis absorption spectrum")
    colors = list(plt.cm.viridis(np.linspace(0.35, 0.9, ics.nstates)))
    if ics.nstates > 1:
        colors.append(plt.cm.viridis(0.2))  # color for the total spectrum
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle("Characteristics of initial conditions (ICs) loaded")
    plt.get_current_fig_manager().set_window_title('UV/vis absorption spectrum')  # modify the window name from Figure x

    for state in range(ics.nstates):
        axs[0].plot(ics.traj_index, ics.de[state]/ics.evtoau, color=colors[state], alpha=0.6)
        axs[0].scatter(ics.traj_index, ics.de[state]/ics.evtoau, color=colors[state], s=5)
    axs[0].set_xlim(np.min(ics.traj_index) - 1, np.max(ics.traj_index) + 1)
    axs[0].set_xlabel("IC index")
    axs[0].set_ylabel(r"$\Delta E$ (eV)")
    axs[0].set_title(r"Excitation energies")
    axs[0].minorticks_on()
    axs[0].tick_params('both', direction='in', which='both', top=True, right=True)

    for state in range(ics.nstates):
        axs[1].plot(ics.traj_index, ics.tdm[state], color=colors[state], alpha=0.6)
        axs[1].scatter(ics.traj_index, ics.tdm[state], color=colors[state], s=5)
    axs[1].set_xlim(np.min(ics.traj_index) - 1, np.max(ics.traj_index) + 1)
    axs[1].set_xlabel("IC index")
    axs[1].set_ylabel(r"$|\mu|$ (a.u.)")
    axs[1].set_title(r"Transition dipole moments")
    axs[1].minorticks_on()
    axs[1].tick_params('both', direction='in', which='both', top=True, right=True)

    axs[2].plot(ics.spectrum[0]/ics.evtoau, ics.spectrum[-1], color=colors[-1], label='Total spectrum')
    axs[2].fill_between(ics.spectrum[0]/ics.evtoau, ics.spectrum[-1]*0, ics.spectrum[-1], color=colors[-1], alpha=0.2)
    if ics.nstates > 1:
        for state in range(ics.nstates):
            axs[2].plot(ics.spectrum[0]/ics.evtoau, ics.spectrum[state + 1], color=colors[state], linestyle='--',
            label=r"S$_\mathregular{%d}$"%(state+1))
            axs[2].fill_between(ics.spectrum[0]/ics.evtoau, 0, ics.spectrum[state + 1], color=colors[state], alpha=0.2)
    axs[2].set_xlim(np.min(ics.spectrum[0]/ics.evtoau), np.max(ics.spectrum[0]/ics.evtoau))
    axs[2].set_ylim(0, np.max(ics.spectrum[-1])*1.2)
    axs[2].set_xlabel(r"$E$ (eV)")
    axs[2].set_ylabel(r"$\sigma$ (cm$^2\cdot$molecule$^{-1}$)")
    axs[2].set_title(r"Absorption spectrum")
    axs[2].legend(frameon=True, labelspacing=0.1, edgecolor='white')
    axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)
    axs[2].minorticks_on()
    axs[2].tick_params('both', direction='in', which='both', top=True, right=True)

    plt.tight_layout()
    plt.savefig('spectrum', dpi=300)
    plt.show(block=False)


def plot_field(ics: InitialConditions) -> None:
    import matplotlib.pyplot as plt

    print("  - Plotting pulse characteristics")
    colors = plt.cm.viridis([0.35, 0.6, 0.0])
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle("Pulse characteristics")
    plt.get_current_fig_manager().set_window_title('Pulse characteristics')  # modify the window name from Figure x

    axs[0].plot(ics.field_t/ics.fstoau, ics.field, color=colors[0], linewidth=0.5, label='Field')
    axs[0].plot(ics.field_t/ics.fstoau, ics.field_envelope, color=colors[0], alpha=0.4)
    axs[0].plot(ics.field_t/ics.fstoau, -ics.field_envelope, color=colors[0], alpha=0.4)
    axs[0].fill_between(ics.field_t/ics.fstoau, ics.field_envelope, -ics.field_envelope, color=colors[0],
        label='Envelope', alpha=0.2)
    axs[0].set_xlim(np.min(ics.field_t/ics.fstoau), np.max(ics.field_t/ics.fstoau))
    axs[0].set_ylim(np.min(-ics.field_envelope)*1.2, np.max(ics.field_envelope)*1.2)
    axs[0].set_xlabel(r"Time (fs)")
    axs[0].set_ylabel(r"$\vec{E}$")
    axs[0].set_title(r"Laser pulse field")
    axs[0].legend(frameon=True, labelspacing=0.1, edgecolor='white', loc='upper left')
    axs[0].minorticks_on()
    axs[0].tick_params('both', direction='in', which='both', top=True, right=True)

    for state in range(ics.nstates):
        axs[1].plot(ics.spectrum[0]/ics.evtoau, ics.spectrum[state + 1]/np.max(ics.spectrum[-1]), color=colors[-1],
            linestyle='--', linewidth=1, alpha=0.5)
    axs[1].plot(ics.spectrum[0]/ics.evtoau, ics.spectrum[-1]/np.max(ics.spectrum[-1]), color=colors[1],
        label='Absorption spectrum')
    axs[1].fill_between(ics.spectrum[0]/ics.evtoau, ics.spectrum[-1]*0, ics.spectrum[-1]/np.max(ics.spectrum[-1]),
        color=colors[1], alpha=0.2)
    axs[1].plot(ics.field_ft_omega/ics.evtoau, ics.field_ft, color=colors[0], label='Pulse spectrum')
    axs[1].fill_between(ics.field_ft_omega/ics.evtoau, ics.field_ft*0, ics.field_ft, color=colors[0], alpha=0.2)
    axs[1].set_xlim(np.min(ics.spectrum[0]/ics.evtoau), np.max(ics.spectrum[0]/ics.evtoau))
    axs[1].set_ylim(0, 1.2)
    axs[1].set_xlabel(r"$E$ (eV)")
    axs[1].set_ylabel(r"$\epsilon$")
    axs[1].set_title(r"Pulse spectral intensity")
    axs[1].legend(frameon=True, labelspacing=0.1, edgecolor='white')
    axs[1].minorticks_on()
    axs[1].tick_params('both', direction='in', which='both', top=True, right=True)

    plt.tight_layout()
    plt.savefig('field', dpi=300)
    plt.show(block=False)

    # In case the pulse does not fulfil Maxwell's equations, plot the whole pulse spectrum and explain.
    if not ics.is_maxwell_fulfilled():
        print("  - Plotting Maxwell eq. violation (pulse spectrum at 0 frequency)")
        fig, axs = plt.subplots(1, 1, figsize=(4, 3.5))
        fig.suptitle("Pulse spectrum nonzero at zero frequency!")
        plt.get_current_fig_manager().set_window_title('Maxwell eq. violation')  # modify the window name from Figure x

        axs.plot(ics.field_ft_omega/ics.evtoau, ics.field_ft, color=colors[0], label='Pulse spectrum')
        axs.fill_between(ics.field_ft_omega/ics.evtoau, ics.field_ft*0, ics.field_ft, color=colors[0], alpha=0.2)
        axs.axvline(0, color='black', lw=0.5)
        axs.scatter(ics.field_ft_omega[0]/ics.evtoau, ics.field_ft[0], color='black')
        axs.set_xlim(-0.1, ics.field_ft_omega[np.argmax(ics.field_ft)*2]/ics.evtoau)
        axs.set_ylim(0, 1.2)
        axs.set_xlabel(r"$E$ (eV)")
        axs.set_ylabel(r"Pulse spectrum")
        axs.set_title(
            r"$\int_{-\infty}^\infty \vec{E}(t) \mathregular{d}t = \mathcal{F}[\vec{E}(t)]|_{\omega=0} \neq 0$"
            "\nViolation of Maxwell's equations!")
        axs.minorticks_on()
        axs.tick_params('both', direction='in', which='both', top=True, right=True)

        plt.tight_layout()
        plt.savefig('field_maxwell_violation', dpi=300)
        plt.show(block=False)


def plot_pda(ics: InitialConditions) -> None:
    import matplotlib.pyplot as plt

    print("  - Plotting PDA initial conditions")
    colors = plt.cm.viridis([0.35, 0.6])
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle("Excitations in time")
    plt.get_current_fig_manager().set_window_title('PDA initial conditions')  # modify the window name from Figure x

    # setting the other plots around the main plot
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.00
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    axs = fig.add_axes(rect_scatter)

    emin, emax = np.min(ics.spectrum[0]/ics.evtoau), np.max(ics.spectrum[0]/ics.evtoau)
    tmin, tmax = np.min(ics.field_t/ics.fstoau), np.max(ics.field_t/ics.fstoau)

    axs.hist2d(ics.ics[3]/ics.evtoau, ics.ics[1]/ics.fstoau, range=[[emin, emax], [tmin, tmax]], bins=(100, 100),
        cmap=plt.cm.viridis, density=True, )
    if ics.pulse.lchirp != 0:
        axs.plot((ics.pulse.omega + 2*ics.pulse.lchirp*ics.field_t)/ics.evtoau, ics.field_t/ics.fstoau, color="white",
            linestyle="--", label=r"$\omega(t)$", )
        axs.legend(frameon=True, framealpha=0.4, labelspacing=0.1)

    ax_histy = fig.add_axes(rect_histy, sharey=axs)
    ax_histy.plot(ics.field_envelope**2, ics.field_t/ics.fstoau, color=colors[0], label="Pulse \nintensity")
    ax_histy.fill_betweenx(ics.field_t/ics.fstoau, ics.field_envelope**2, 0, color=colors[0], alpha=0.2)
    ax_histy.set_xlim(0, 1.2)
    ax_histy.legend(frameon=True, framealpha=0.9, labelspacing=0.1, edgecolor='white')

    ax_histx = fig.add_axes(rect_histx, sharex=axs)
    ax_histx.plot(ics.spectrum[0]/ics.evtoau, ics.spectrum[-1]/np.max(ics.spectrum[-1]), color=colors[1],
        label='Absorption spectrum')
    ax_histx.fill_between(ics.spectrum[0]/ics.evtoau, ics.spectrum[-1]*0, ics.spectrum[-1]/np.max(ics.spectrum[-1]),
        color=colors[1], alpha=0.2)
    ax_histx.plot(ics.field_ft_omega/ics.evtoau, ics.field_ft**2, color=colors[0], label='Pulse spec. intensity')
    ax_histx.fill_between(ics.field_ft_omega/ics.evtoau, ics.field_ft*0, ics.field_ft**2, color=colors[0], alpha=0.2)
    ax_histx.set_ylim(0, 1.2)
    ax_histx.legend(frameon=True, labelspacing=0.1, edgecolor='white')

    ax_histx.tick_params("both", which='both', direction='in', labelbottom=False)
    ax_histy.tick_params("both", which='both', direction='in', labelleft=False)
    ax_histy.set_xticks([])
    ax_histx.set_yticks([])
    axs.set_xlabel(r"$\Delta E$ (eV)")
    axs.set_ylabel(r"Time (fs)")
    axs.minorticks_on()
    axs.tick_params("both", which='both', direction='in', top=True, right=True, color='white')

    plt.savefig('pda', dpi=300)
    plt.show()


def plot_pdaw(ics: InitialConditions) -> None:
    import matplotlib.pyplot as plt

    print("  - Plotting PDAW weights")
    colors = list(plt.cm.viridis(np.linspace(0.35, 0.9, ics.nstates)))
    if ics.nstates > 1:
        colors.append(plt.cm.viridis(0.2))  # color for the total spectrum
    fig, axs = plt.subplots(1, 1, figsize=(4, 3.5))
    fig.suptitle("Selected initial conditions and their weights")
    plt.get_current_fig_manager().set_window_title('PDAW weights')  # modify the window name from Figure x

    axs.plot(ics.spectrum[0]/ics.evtoau, ics.spectrum[-1]/np.max(ics.spectrum[-1]), color=colors[-1],
        label='Absorption spectrum')
    axs.fill_between(ics.spectrum[0]/ics.evtoau, ics.spectrum[-1]*0, ics.spectrum[-1]/np.max(ics.spectrum[-1]),
        color=colors[-1], alpha=0.2)

    maxw = np.max(ics.weights)
    if ics.nstates > 1:
        for state in range(ics.nstates):
            # plotting spectrum
            axs.plot(ics.spectrum[0]/ics.evtoau, ics.spectrum[state + 1]/np.max(ics.spectrum[-1]), color=colors[state],
                linestyle='--')
            axs.fill_between(ics.spectrum[0]/ics.evtoau, 0, ics.spectrum[state + 1]/np.max(ics.spectrum[-1]),
                color=colors[state], alpha=0.2)
            # weights of initial conditions plotted as sticks with points
            axs.scatter(ics.de[state, :]/ics.evtoau, ics.weights[state, :]/maxw, color=colors[state], s=5)
            for index in range(ics.nsamples):
                axs.plot([ics.de[state, index]/ics.evtoau]*2, [0, ics.weights[state, index]/maxw], color=colors[state])

    axs.plot(ics.field_ft_omega/ics.evtoau, ics.field_ft**2, color='black', alpha=0.5, label='Pulse intensity spectrum')
    axs.set_xlim(np.min(ics.spectrum[0]/ics.evtoau), np.max(ics.spectrum[0]/ics.evtoau))
    axs.set_ylim(0, 1.3)
    axs.set_xlabel(r"$E$ (eV)")
    axs.set_ylabel(r"$\epsilon$")
    axs.set_title(r"Pulse spectrum")
    axs.legend(frameon=True, labelspacing=0.1, edgecolor='white', loc='upper left')
    axs.minorticks_on()
    axs.tick_params('both', direction='in', which='both', top=True, right=True)

    plt.tight_layout()
    plt.savefig('pdaw', dpi=300)
    plt.show()


def parse_cmd_args():
    ### setting up parser ###
    parser = argparse.ArgumentParser(description=DESC, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--method", default='pda', choices=METHODS,
        help="Select either Promoted density approach (PDA) to generate initial conditions with excitation times or "
             "PDA for windowing (PDAW) to generate weights and convolution parameters.")
    parser.add_argument("-np", "--npsamples", default=1000, type=positive_int,
        help="Number of initial conditions generated with PDA.")

    inp_file = parser.add_argument_group("Input file handling")
    inp_file.add_argument("-n", "--nsamples", default=0, type=positive_int,
        help="Number of data points from the input file considered. By default all initial conditions provided in the input file are taken.")
    inp_file.add_argument("-ns", "--nstates", default=1, type=positive_int, help="Number of excited states considered.")
    inp_file.add_argument("-eu", "--energy_unit", choices=ENERGY_UNITS, default='a.u.',
        help="Units in which excitation energies are provided.")
    inp_file.add_argument("-tu", "--tdm_unit", choices=TDM_UNITS, default='a.u.',
        help="Units in which magnitudes of transition dipole moments (|mu_ij|) are provided.")
    inp_file.add_argument("-ft", "--file_type", choices=FILE_TYPES, default='file', help="Input file type.")

    field = parser.add_argument_group('Field parameters')
    field.add_argument("-w", "--omega", required=True, type=positive_float, help="Frequency of the field in a.u.")
    field.add_argument("-f", "--fwhm", required=True, type=positive_float,
        help="Full Width at Half Maximum (FWHM) parameter for the pulse intensity envelope in fs.")
    field.add_argument("-env", "--envelope_type", choices=ENVELOPE_TYPES, default='gauss', help="Field envelope type.")
    field.add_argument("-lch", "--linear_chirp", default=0.0, type=float,
        help="Linear chirp [w(t) = w+lch*t] of the field frequency in a.u.")
    field.add_argument("-t0", "--t0", default=0.0, type=float, help="Time of the field maximum in fs.")

    parser.add_argument("-neg", "--neg_handling", choices=NEG_PROB_HANDLING, default='error',
        help="Procedures how to handle negative probabilities.")
    parser.add_argument("-s", "--random_seed", default=None, type=positive_int,
        help="Seed for the random number generator. Default: generate random seed from OS.")
    parser.add_argument("-ps", "--preselect", action="store_true",
        help="Preselect samples within pulse spectrum for PDA. This option provides significant speed "
             "up if the pulse spectrum covers only small part of the absorption spectrum as it avoids expensive "
             "calculation of W for non-resonant cases. The lost of accuracy should be minimal, yet we still "
             "recommend to use this option only if the calculation is too expensive, e.g. for very long pulses.")
    parser.add_argument("-p", "--plot", action="store_true",
        help="Plot the input data and calculated results and save them as png images.")
    parser.add_argument("input_file", help="Input file name.")

    return parser.parse_args()


def main():
    # Parse the command line parameters and print them
    config = parse_cmd_args()
    print_header()
    print("* Input parameters:")
    for key, value in vars(config).items():
        add = ''
        if key == 'nsamples' and value == 0:
            add = '(All input data will be used)'
        if config.method == 'pdaw' and key in ('npsamples', 'random_seed', 'preselect', 'neg_handling'):
            continue
        print(f"  - {key:20s}: {value}   {add}")

    if not Path(config.input_file).is_file():
        print(f"ERROR: file '{config.input_file}' not found!")
        exit(1)

    # convert pulse input to atomic units
    fstoau = 41.341374575751
    pulse = LaserPulse(omega=config.omega, fwhm=config.fwhm*fstoau, envelope_type=config.envelope_type,
        lchirp=config.linear_chirp, t0=config.t0*fstoau, )
    print(f"* {pulse}")

    ics = InitialConditions(nsamples=config.nsamples, nstates=config.nstates, input_type=config.file_type)

    ics.read_input_data(fname=config.input_file, energy_unit=config.energy_unit, tdm_unit=config.tdm_unit)

    # calculating spectrum with nuclear ensemble approach
    ics.calc_spectrum()

    if config.plot:
        plot_spectrum(ics)

    # calculating the field and its spectrum
    ics.calc_field(pulse=pulse)
    if config.plot:
        plot_field(ics)

    # perform either PDA or PDAW and plot
    if config.method == 'pda':
        ics.sample_initial_conditions(nsamples_ic=config.npsamples, neg_handling=config.neg_handling,
            preselect=config.preselect, seed=config.random_seed)
        if config.plot:
            plot_pda(ics)

    elif config.method == 'pdaw':
        ics.windowing()
        if config.plot:
            plot_pdaw(ics)

    print_footer()


if __name__ == '__main__':
    main()

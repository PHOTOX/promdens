"""Code for filtering of initial conditions for nonadiabatic dynamics with a laser pulse.

© Jiri Janos 2024"""

# todo: energy per pulse!

# importing python libraries
import argparse
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np


### functions and classes ###
class initial_conditions:
    """Class containing initial conditions. The whole code is based on this class, which
       - reads the input file
       - saves excitation energies and trasition dipole moment |mu_ij|
       - calculates spectrum with NEM
       - calculated the laser pulse and its spectrum
       - performes a selection of initial conditions in the excited states using wigner transformation of the pulse
       All data loaded and calculated are saved in this object. More details are provided in the functions below."""

    # constants in atomic units
    hbar = 1.0

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
        :param nsamples: number of samples considered, if set 0 then maximum number provided will be taken
        :param nstates: number of excited states considered
        :param input_type: data read form file, other options are 'Newton-X' etc (to be done)
        """
        self.nsamples = nsamples
        self.nstates = nstates
        self.input_type = input_type
        # flags that everything was calculated
        self.input_read = False
        self.units_converted = False
        self.spectrum_calculated = False
        self.field_calculated = False

    def read_input_data(self, fname='ics.dat'):
        """
        Reading the input data: index of traj, excitation energies and transition dipole moments.
        :param fname: file with the input data
        :return: store all the data in the class
        """
        print(f"* Reading data from file '{fname}' of type '{self.input_type}'.")
        if self.input_type == 'file':
            try:
                input = np.loadtxt(fname, dtype=float).T  # reading input file
            except FileNotFoundError as err:
                print(f"\nERROR: File with input data '{fname}' not found!\n (Error: {err})")
                exit(1)
            except ValueError as err:
                print(err)
                print(f"\nERROR: Incorrect value type encountered in '{fname}!\n (Error: {err})")
                exit(1)
            except Exception as err:
                print(f"\nERROR: Unexpected error: {err}, type: {type(err)}")
                exit(1)

            if np.shape(input)[0] < self.nstates * 2 + 1:  # check enough columns provided in the file
                print(f"\nERROR: Not enough columns provided in file '{fname}'! "
                      f"Expected {self.nstates * 2 + 1} columns for {self.nstates} excited states.")
                exit(1)

            if self.nsamples == 0:  # use all samples loaded if user input nsamples is 0
                self.nsamples = np.shape(input)[1]
                print(f"  - Number of ICs loaded from the input file: {self.nsamples}")
            elif self.nsamples > np.shape(input)[1]:
                print(f"  - Number of ICs loaded from the input file {np.shape(input)[1]} "
                      f"instead of requested {self.nsamples}")
                self.nsamples = np.shape(input)[1]

            self.traj_index = np.array(input[0, :self.nsamples], dtype=int)  # saving indexes of trajectories
            self.de = input[1:self.nstates * 2:2, :self.nsamples]  # saving excitation energies
            self.tdm = input[2:self.nstates * 2 + 1:2, :self.nsamples]  # saving transition dipole moments
        else:
            print(f"\nERROR: File type '{self.input_type}' not supported!")
            exit(1)

        self.input_read = True

    def convert_units(self, energy_units, tdm_units):
        """
        Converting all the data into atomic units which are used throughout the code.
        :param energy_units: energy units of the input data
        :return: store data converted to atomic units
        """
        print(f"* Converting units.")
        if energy_units == 'eV':
            self.de *= self.evtoau
        elif energy_units == 'nm':
            self.de *= self.nmtoau
        elif energy_units == 'cm-1':
            self.de *= self.cm1toau

        if tdm_units == 'Debye':
            self.tdm *= self.debtoau
        self.units_converted = True

    def calc_spectrum(self):
        """
        Calculating spectrum with the Nuclear Ensemble Approach. Currently, the spectrum is calculated in arbitrary
        units. The calculated spectrum is in absorption cross-section units (cm^2*molecule^-1). Conversion to molar
        absorption coefficient (dm^3*mol^-1*cm^-1) can be done with factor 6.022140e20 / ln(10).
        """

        def gauss(e, de, tdm, h):
            """
            Gaussian function used in spectra calculation
            :param e: energy axis (a.u.)
            :param de: excitation energy (centre of the Gaussian, a.u.)
            :param tdm: transition dipole moment (a.u.)
            :param h: width of the Gaussian (a.u.)
            :return: Gaussian as a function of energy with intensity proportional to TDM
            """
            return de * tdm ** 2 * np.exp(-(e - de) ** 2 / 2 / h ** 2)

        print(f"* Calculating spectrum with the Nuclear Ensemble Approach.")

        # checking if all the necessary preceding calculations were executed
        if not self.input_read:
            print(f"ERROR: Field yet not calculated. Please first use 'calc_field()'!")
            exit(1)
        elif not self.units_converted:
            print(f"ERROR: Units not converted yet. Please first use 'convert_units()'!")
            exit(1)

        # coefficient for intensity of the spectrum
        eps0 = 8.854188e-12
        hbar = 6.626070e-34 / (2 * np.pi)
        c = 299792458
        int_coeff = np.pi * self.autocm ** 2 * 1e4 / (3 * hbar * eps0 * c) / self.nsamples / np.sqrt(2 * np.pi)

        # width for the Gaussians set by Silverman’s rule of thumb: https://doi.org/10.1039/C8CP00199E
        emin, emax = np.min(self.de), np.max(self.de)
        self.spectrum = np.zeros(shape=(self.nstates + 2, 10000), dtype=float)
        h = (4 / 3 / self.nsamples) ** 0.2 * np.std(self.de)  # width h for all data to get energy range of the spectrum
        self.spectrum[0] = np.linspace(emin - 2 * h, emax + 2 * h, np.shape(self.spectrum)[1])

        for state in range(self.nstates):
            h = (4 / 3 / self.nsamples) ** 0.2 * np.std(self.de[state])  # width h for individual states

            for ic in range(self.nsamples):
                self.spectrum[state + 1] += gauss(self.spectrum[0], self.de[state, ic], self.tdm[state, ic], h)

            self.spectrum[state + 1] *= int_coeff / h

        self.spectrum[-1] = np.sum(self.spectrum[1:-1], axis=0)

        self.spectrum_calculated = True

    def calc_field_envelope(self, t):
        """
        Calculating field envelope. The parameters of the field are taken form the class.
        :param t: time axis (a.u.)
        :return: envelope of the field
        """
        if self.field_envelope_type == 'gauss':
            return np.exp(-2 * np.log(2) * (t - self.field_t0) ** 2 / self.field_fwhm ** 2)
        elif self.field_envelope_type == 'lorentz':
            return (1 + 4 / (1 + np.sqrt(2)) * ((t - self.field_t0) / self.field_fwhm) ** 2) ** -1
        elif self.field_envelope_type == 'sech':
            return 1 / np.cosh(2 * np.log(1 + np.sqrt(2)) * (t - self.field_t0) / self.field_fwhm)
        elif self.field_envelope_type == 'sin':
            field = np.zeros(shape=np.shape(t))
            for k in range(np.shape(t)[0]):
                if t[k] >= self.tmin and t[k] <= self.tmax:
                    field[k] = np.sin(np.pi / 2 * (t[k] - self.field_t0 + self.field_fwhm) / self.field_fwhm)
            return field
        elif self.field_envelope_type == 'sin2':
            T = 1.373412575 * self.field_fwhm
            field = np.zeros(shape=np.shape(t))
            for k in range(np.shape(t)[0]):
                if t[k] >= self.tmin and t[k] <= self.tmax:
                    field[k] = np.sin(np.pi / 2 * (t[k] - self.field_t0 + T) / T) ** 2
            return field

    def field_cos(self, t):
        """
        Calculating oscilatoin of the field with the cos function.
        :param t: time
        :return: cos((w + lchirp*t)*t)
        """
        return np.cos((self.field_omega + self.field_lchirp * t) * t)

    def calc_field(self, omega, fwhm, t0=0.0, lchirp=0.0, envelope_type='gauss'):
        """
        Calculating electric field as a function of time and it spectrum with Fourier transform.
        :param omega: frequency of the field (a.u.)
        :param fwhm: full width half maximum of the intensity envelope (a.u.)
        :param t0: centre of the envelope in time (a.u.)
        :param lchirp: linear chirp parameter w = lchirp*t + omega (a.u.)
        :param envelope_type: envelope shape types: Gaussian, Lorentzian, sech
        :return: store the electric field as a function of time and the pulse spectrum
        """

        print(f"* Calculating laser pulse field using envelope type '{envelope_type}', omega={omega:.6f} a.u.,"
              f" fwhm={fwhm:.6f} a.u., t0={t0:.6f} a.u., lchirp={lchirp:.6f} a.u.")
        # saving field parameters in the class
        self.field_omega = omega
        self.field_lchirp = lchirp
        self.field_envelope_type = envelope_type
        self.field_t0 = t0
        self.field_fwhm = fwhm

        # print field function and determine maximum and minimum times for the field
        if self.field_envelope_type == 'gauss':
            print("  - E(t) = exp(-2*ln(2)*(t-t0)^2/fwhm^2)*cos((omega+lchirp*t)*t)")
            self.tmin, self.tmax = self.field_t0 - 2.4 * self.field_fwhm, self.field_t0 + 2.4 * self.field_fwhm
        elif self.field_envelope_type == 'lorentz':
            print("  - E(t) = (1+4/(1+sqrt(2))*(t/fwhm)^2)^-1*cos((omega+lchirp*t)*t)")
            self.tmin, self.tmax = self.field_t0 - 8 * self.field_fwhm, self.field_t0 + 8 * self.field_fwhm
        elif self.field_envelope_type == 'sech':
            print("  - E(t) = 1/cosh(2*ln(1+sqrt(2))*t/fwhm)*cos((omega+lchirp*t)*t)")
            self.tmin, self.tmax = self.field_t0 - 4.4 * self.field_fwhm, self.field_t0 + 4.4 * self.field_fwhm
        elif self.field_envelope_type == 'sin':
            print("  - E(t) = sin(pi/2*(t-t0+fwhm)/fwhm)*cos((omega+lchirp*t)*t) in range [t0-fwhm,t0+fwhm]")
            self.tmin, self.tmax = self.field_t0 - self.field_fwhm, self.field_t0 + self.field_fwhm
        elif self.field_envelope_type == 'sin2':
            print(
                "  - E(t) = sin(pi/2*(t-t0+T)/T)^2*cos((omega+lchirp*t)*t) in range [t0-T,t0+T] where T=1.373412575*fwhm")
            T = 1.373412575 * self.field_fwhm
            self.tmin, self.tmax = self.field_t0 - T, self.field_t0 + T

        # calculating the field
        self.field_t = np.arange(self.tmin, self.tmax, 2 * np.pi / omega / 50)  # time array for the field in a.u.
        self.field_envelope = self.calc_field_envelope(self.field_t)
        self.field = self.field_envelope * self.field_cos(self.field_t)

        # calculating the FT of the field
        dt = 2 * np.pi / omega / 50
        t_ft = np.arange(self.tmin - 20 * self.field_fwhm, self.tmax + 20 * self.field_fwhm, dt)
        field = self.calc_field_envelope(t_ft) * self.field_cos(t_ft)
        self.field_ft = np.abs(np.fft.rfft(field))  # FT
        self.field_ft /= np.max(self.field_ft)  # normalizing
        self.field_ft_omega = 2 * np.pi * np.fft.rfftfreq(len(t_ft), dt)

        self.field_calculated = True

    def pulse_wigner(self, tprime, de):
        """
        Wigner transform of the pulse envelope as originally propposed by Martínez-Mesa and Saalfrank.
        :param tprime: time at which the molecule is excited (a.u.)
        :param de: excitation energy (a.u.)
        :return:
        """

        if not self.field_calculated:
            print(f"ERROR: Input data not read yet. Please first use 'read_input_data()'!")
            exit(1)

        # attempt for an adaptive integration step according to the oscillation
        # however, for plotting on a grid it doesn't work as the edges have dE-omega = 12 or more so I needed to add a
        # thresh with omega. Implementing this for the sampling shouldn't be a problem because I will have only relevant
        # excitation energies.
        loc_omega = self.field_omega + self.field_lchirp * tprime
        if de != loc_omega:
            T = 2 * np.pi / (np.min([np.abs(de - loc_omega), loc_omega]))
        else:  # in case they are equal, I cannot divide by 0
            T = 2 * np.pi / loc_omega
        ds = np.min([T / 50, self.field_fwhm / 500])

        # integration ranges for different pulse envelopes
        # ideally, we would integrate from -infinity to infinity, yet this is not very computationally efficient
        # empirically, it was found out that efficient integration varies for different pulses
        if self.field_envelope_type == 'gauss':
            factor = 7.5
        elif self.field_envelope_type == 'lorentz':
            factor = 50
        elif self.field_envelope_type == 'sech':
            factor = 20
        elif self.field_envelope_type == 'sin':
            factor = 3
        elif self.field_envelope_type == 'sin2':
            factor = 4

        # instead of calculating the complex integral int_{-inf}^{inf}[E(t+s/2)E(t-s/2)exp(i(w-de)s)]ds we use the
        # properties of even and odd fucntions and calculate 2*int_{0}^{inf}[E(t+s/2)E(t-s/2)cos((w-de)s)]ds
        s = np.arange(0, factor * self.field_fwhm, step=ds)
        cos = np.cos((de / self.hbar - loc_omega) * s)
        integral = 2 * np.trapz(x=s, y=cos * self.calc_field_envelope(tprime + s / 2) * self.calc_field_envelope(
            tprime - s / 2))

        W = 1 / 2 / np.pi / self.hbar * integral
        if W < -1e-5:  # threshold for numerical accuracy
            print(f"\nERROR: Negative probability encountred ({W})! This is usually the case for all pulses but 'gauss'"
                  f" and is conected to ill-defined wigner distribution.\n")
            exit(1)
        return W

    def sample_initial_conditions(self, new_ic_nsamples):
        """
        Sample time-dependent initial condition from the excited state distribution.
        :param new_ic_nsamples: number of samples to be sampled
        :return: store the new initial conditions in the class and also save output
        """

        print(f"* Sampling {new_ic_nsamples:d} initial conditions considering the laser pulse.")

        if not self.input_read:
            print(f"\nERROR: Field yet not calculated. Please first use 'calc_field()'!")
            exit(1)
        elif not self.units_converted:
            print(f"\nERROR: Units not converted yet. Please first use 'convert_units()'!")
            exit(1)

        def progress(percent, width, n, str=''):
            """Function to print progress of calculation."""
            left = width * percent // n
            right = width - left
            print(f'\r{str}[', '#' * left, ' ' * right, '] %d' % (percent * 100 / n) + '%', sep='', end='', flush=True)

        # variable for selected samples
        samples = np.zeros((5, new_ic_nsamples))  # index, initial time, initial excited state, de, tdm

        # setting maximum random number generated during sampling
        rnd_max = np.max(self.tdm ** 2) * self.pulse_wigner(t0, de=omega + lchirp * t0) * 1.01

        i, nattempts = 0, 0  # i: loop index; nattempts: to calculate efficiency of the sampling
        while i < new_ic_nsamples:  # while loop is used because in case we need to restart it with higher rnd_max
            nattempts += 1

            # index of sample
            rnd_index = np.random.randint(low=0, high=self.nsamples, dtype=int)
            rnd_time = np.random.uniform(low=self.tmin, high=self.tmax)
            rnd_state = np.random.randint(low=0, high=self.nstates, dtype=int)

            rnd = np.random.uniform(low=0, high=rnd_max)  # random number to be compared with Wig. dist.

            prob = self.tdm[rnd_state, rnd_index] ** 2 * self.pulse_wigner(rnd_time, self.de[rnd_state, rnd_index])

            if prob > rnd_max:  # check if the probability is not higher than rnd_max
                print(f"\n - rnd_max ({rnd_max}) is smaller than probability ({prob} for sample "
                      f"{self.traj_index[rnd_index]} on state {rnd_state}). Increasing rnd_max and reruning.")
                rnd_max *= 1.2
                samples = np.zeros((5, new_ic_nsamples))
                i = 0
            elif rnd <= prob:  # check if the point is sampled
                samples[0, i] = self.traj_index[rnd_index]
                samples[1, i] = rnd_time
                samples[2, i] = rnd_state + 1
                samples[3, i] = self.de[rnd_state, rnd_index]
                samples[4, i] = self.tdm[rnd_state, rnd_index]
                i += 1
                progress(i, 50, new_ic_nsamples, str='  Sampling progress: ')

        # saving samples within the object
        self.filtered_ics = samples

        print(f"\n  - Success rate of random sampling: {new_ic_nsamples / nattempts * 100:.5f} %")

        # getting unique initial conditions for each excited state
        unique_states, unique = np.zeros(shape=(self.nstates), dtype=int), []
        for state in range(self.nstates):
            unique.append(np.array(np.unique(samples[0, samples[2] == state + 1]), dtype=int))
            unique_states[state] = len(unique[state])

        # print indexes of trajectories that must be propagated
        if self.nstates == 1:
            print(f"  - Selected {unique_states[0]} unique ICs from {self.nsamples} provided. Unique ICs to be "
                  f"propagated:\n   ", *np.array(unique[0], dtype=str))
        else:
            print(f"  - Selected {np.sum(unique_states)} unique ICs over {self.nstates} state from {self.nsamples} "
                  f"positions and velocities provided.")
            for state in range(self.nstates):
                if unique_states[state] == 0:
                    continue
                print(f"  - State {state + 1} - {unique_states[state]} unique ICs to be propagated: \n   ",
                      *np.array(unique[state], dtype=str))

        # save the selected samples
        np.savetxt(f'IC_sampling.dat', samples.T, fmt=['%8d', '%18.8f', '%8d', '%16.8f', '%16.8f'],
                   header=f"nsamples = {new_ic_nsamples:d}, unique nsamples = {len(unique):d}, "
                          f"omega = {self.field_omega:.4f} a.u., "
                          f"linearchirp = {self.field_lchirp:.10f} a.u., fwhm = {self.field_fwhm / self.fstoau:.3f} fs, "
                          f"t0 = {self.field_t0 / self.fstoau:.3f} fs, envelope type = '{self.field_envelope_type}'\n"
                          f"index        time (a.u.)    el. state     de (a.u.)        tdm (arb.unit)")
        print(f"  - Output saved to file 'IC_sampling.dat'.")


### setting up parser ###
parser = argparse.ArgumentParser(description="Parser for this code",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--nsamples", default=0, type=int,
                    help="Number of initial conditions considered for sampling. 0 takes all initial conditions provided in the input file.")
parser.add_argument("-nf", "--nfsamples", default=1000, type=int,
                    help="Number of filtered initial conditions that will be calculated.")
parser.add_argument("-ns", "--nstates", default=1, type=int, help="Number of excited states considered.")
parser.add_argument("-ft", "--file_type", default='file', help="Input file type. Options are 'file'.")
parser.add_argument("-p", "--plot", action="store_true",
                    help="Plot along the code to see loaded data and results. Plots are saved as png images.")
parser.add_argument("-eu", "--energy_units", default='a.u.',
                    help="Units in which energies are provided. Options are 'a.u.', 'eV', 'nm', 'cm-1'. "
                         "If input file type is Newton-X, defaults for Newton-X will be taken.")
parser.add_argument("-tu", "--tdm_units", default='a.u.',
                    help="Units in which transition dipole moments (|mu_ij|) are provided. Options are 'a.u.', 'debye'. "
                         "If input file type is Newton-X, defaults for Newton-X will be taken.")
parser.add_argument("-w", "--omega", default=0.1, type=float, help="Frequency of the field omega in a.u.")
parser.add_argument("-lch", "--linear_chirp", default=0.0, type=float,
                    help="Linear chirp [w(t) = w+lch*t] of the field frequency in a.u.")
parser.add_argument("-f", "--fwhm", default=10.0, type=float,
                    help="Full Width Half Maximum (FWHM) parameter in fs for the pulse intentsity envelope.")
parser.add_argument("-t0", "--t0", default=0.0, type=float, help="Time of the maximum of the field in fs.")
parser.add_argument("-env", "--envelope_type", default='gauss',
                    help="Type of field envelope. Options are 'gauss', 'lorentz', 'sech', 'sin2'.")
parser.add_argument("input_file", help="Input file name.")

### entering code ###
print(f"\n#######################################################\n"
      f"###  Filtering initial conditions with laser pulse  ###\n"
      f"###                 * * * * *                       ###\n"
      f"###      version BETA       Jiri Janos 2024         ###\n"
      f"#######################################################\n")

# parsing the input and creating variables from it
print(f"* Parsing the input.")
config = vars(parser.parse_args())
for item in config:
    add = ''
    if item == 'nsamples' and config[item] == 0:
        add = '(max number provided in the input will be used)'
    print(f"  - {item:20s}: {config[item]}   {add}")

# storing input into variables used in the code
nsamples = config['nsamples']
new_nsamples = config['nfsamples']
nstates = config['nstates']
plotting = config['plot']
energy_units = config['energy_units']
tdm_units = config['tdm_units']
fwhm = config['fwhm']
omega = config['omega']
lchirp = config['linear_chirp']
t0 = config['t0']
envelope_type = config['envelope_type']
ftype = config['file_type']
fname = config['input_file']

# converting pulse input to a.t.u.
fstoau = 41.341374575751
t0 *= fstoau
fwhm *= fstoau

# checking input
if not energy_units in ['a.u.', 'eV', 'nm', 'cm-1']:
    print(f"\nERROR: {energy_units} is not available unit for energy!")
    exit(1)

if not tdm_units in ['a.u.', 'debye']:
    print(f"\nERROR: {tdm_units} is not available unit for transition dipole moment!")
    exit(1)

if not ftype in ['file']:
    print(f"\nERROR: {ftype} is not available file type!")
    exit(1)

if not envelope_type in ['gauss', 'lorentz', 'sech', 'sin', 'sin2']:
    print(f"\nERROR: {envelope_type} is not available envelope type!")
    exit(1)

if not exists(fname):
    print(f"\nERROR: file '{fname}' not found!")
    exit(1)

if nsamples < 0:
    print(f"\nERROR: nsamples is smaller than 0 ({nsamples})!")
    exit(1)

if new_nsamples <= 0:
    print(f"\nERROR: nfsamples is smaller than 0 ({new_nsamples})!")
    exit(1)

if nstates <= 0:
    print(f"\nERROR: invalid number of excited states (nstates={nstates})!")
    exit(1)

### code ###
# creating object initial conditions
ics = initial_conditions(nsamples=nsamples, nstates=nstates, input_type=ftype)

# reading input data
ics.read_input_data(fname=fname)

# converting units to make everything in atomic units
ics.convert_units(energy_units=energy_units, tdm_units=tdm_units)

# calculating spectrum
ics.calc_spectrum()

# plotting loaded data
if plotting:
    print("  - Plotting Figure 1")
    colors = list(plt.cm.viridis(np.linspace(0.35, 0.9, ics.nstates)))
    if ics.nstates > 1: colors.append(plt.cm.viridis(0.2))  # color for the total spectrum
    plt.rcParams["font.family"] = 'Helvetica'
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle("Characteristics of initial conditions (ICs) loaded")

    for state in range(ics.nstates):
        axs[0].plot(ics.traj_index, ics.de[state] / ics.evtoau, color=colors[state], alpha=0.6,
                    label=r"S$_\mathregular{%d}$" % state)
        axs[0].scatter(ics.traj_index, ics.de[state] / ics.evtoau, color=colors[state], s=5)
    axs[0].set_xlim(np.min(ics.traj_index) - 1, np.max(ics.traj_index) + 1)
    axs[0].set_xlabel("IC index")
    axs[0].set_ylabel(r"$\Delta E$ (eV)")
    axs[0].set_title(r"Excitation energies")
    axs[0].legend(frameon=False, labelspacing=0.1)
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

    axs[2].plot(ics.spectrum[0] / ics.evtoau, ics.spectrum[-1], color=colors[-1], label='Total spectrum')
    axs[2].fill_between(ics.spectrum[0] / ics.evtoau, ics.spectrum[-1] * 0, ics.spectrum[-1], color=colors[-1],
                        alpha=0.2)
    if ics.nstates > 1:
        for state in range(ics.nstates):
            axs[2].plot(ics.spectrum[0] / ics.evtoau, ics.spectrum[state + 1], color=colors[state], linestyle='--')
            axs[2].fill_between(ics.spectrum[0] / ics.evtoau, ics.spectrum[state + 1] * 0, ics.spectrum[state + 1],
                                color=colors[state], alpha=0.2)
    axs[2].set_xlim(np.min(ics.spectrum[0] / ics.evtoau), np.max(ics.spectrum[0] / ics.evtoau))
    axs[2].set_ylim(0, np.max(ics.spectrum[-1]) * 1.2)
    axs[2].set_xlabel(r"$E$ (eV)")
    axs[2].set_ylabel(r"$\sigma$ (cm$^2\cdot$molecule$^{-1}$)")
    axs[2].set_title(r"Absorption spectrum")
    axs[2].legend(frameon=False, labelspacing=0.1)
    axs[2].minorticks_on()
    axs[2].tick_params('both', direction='in', which='both', top=True, right=True)

    plt.tight_layout()
    plt.savefig('spectrum', dpi=300)
    plt.show(block=False)

# calculating the field
ics.calc_field(omega=omega, fwhm=fwhm, t0=t0, lchirp=lchirp, envelope_type=envelope_type)

# plotting field
if plotting:
    print("  - Plotting Figure 2")
    colors = plt.cm.viridis([0.35, 0.6, 0.0])
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle("Field characteristics")

    axs[0].plot(ics.field_t / ics.fstoau, ics.field, color=colors[0], linewidth=0.5, label='Field')
    axs[0].plot(ics.field_t / ics.fstoau, ics.field_envelope, color=colors[0], alpha=0.4)
    axs[0].plot(ics.field_t / ics.fstoau, -ics.field_envelope, color=colors[0], alpha=0.4)
    axs[0].fill_between(ics.field_t / ics.fstoau, ics.field_envelope, -ics.field_envelope, color=colors[0],
                        label='Envelope', alpha=0.2)
    axs[0].set_xlim(np.min(ics.field_t / ics.fstoau), np.max(ics.field_t / ics.fstoau))
    axs[0].set_ylim(np.min(-ics.field_envelope) * 1.2, np.max(ics.field_envelope) * 1.2)
    axs[0].set_xlabel(r"$t$ (fs)")
    axs[0].set_ylabel(r"$\vec{E}$")
    axs[0].set_title(r"Laser pulse field")
    axs[0].legend(frameon=False, labelspacing=0.1, loc='upper left')
    axs[0].minorticks_on()
    axs[0].tick_params('both', direction='in', which='both', top=True, right=True)

    for state in range(ics.nstates):
        axs[1].plot(ics.spectrum[0] / ics.evtoau, ics.spectrum[state + 1] / np.max(ics.spectrum[-1]), color=colors[-1],
                    linestyle='--', linewidth=1, alpha=0.5)
    axs[1].plot(ics.spectrum[0] / ics.evtoau, ics.spectrum[-1] / np.max(ics.spectrum[-1]), color=colors[1],
                label='Absorption spectrum')
    axs[1].fill_between(ics.spectrum[0] / ics.evtoau, ics.spectrum[-1] * 0, ics.spectrum[-1] / np.max(ics.spectrum[-1]),
                        color=colors[1], alpha=0.2)
    axs[1].plot(ics.field_ft_omega / ics.evtoau, ics.field_ft, color=colors[0], label='Pulse spectrum')
    axs[1].fill_between(ics.field_ft_omega / ics.evtoau, ics.field_ft * 0, ics.field_ft, color=colors[0], alpha=0.2)
    axs[1].set_xlim(np.min(ics.spectrum[0] / ics.evtoau), np.max(ics.spectrum[0] / ics.evtoau))
    axs[1].set_ylim(0, 1.2)
    axs[1].set_xlabel(r"$E$ (eV)")
    axs[1].set_ylabel(r"$\epsilon$")
    axs[1].set_title(r"Pulse spectrum")
    axs[1].legend(frameon=False, labelspacing=0.1)
    axs[1].minorticks_on()
    axs[1].tick_params('both', direction='in', which='both', top=True, right=True)

    plt.tight_layout()
    plt.savefig('field', dpi=300)
    plt.show(block=False)

# sampling
ics.sample_initial_conditions(new_ic_nsamples=new_nsamples)

if plotting:
    print("  - Plotting Figure 3")
    colors = plt.cm.viridis([0.35, 0.6])
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle("Excitations in time")

    # setting the other plots around the main plot
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.00
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    axs = fig.add_axes(rect_scatter)

    emin, emax = np.min(ics.spectrum[0] / ics.evtoau), np.max(ics.spectrum[0] / ics.evtoau)
    tmin, tmax = np.min(ics.field_t / ics.fstoau), np.max(ics.field_t / ics.fstoau)

    h = axs.hist2d(ics.filtered_ics[3] / ics.evtoau, ics.filtered_ics[1] / ics.fstoau,
                   range=[[emin, emax], [tmin, tmax]], bins=(100, 100), cmap=plt.cm.viridis, density=True)
    if lchirp != 0:
        axs.plot((omega + lchirp * ics.field_t) / ics.evtoau, ics.field_t / ics.fstoau, color='white', linestyle='--',
                 label=r"$\omega(t)$")
        axs.legend(frameon=True, framealpha=0.4, labelspacing=0.1)

    ax_histy = fig.add_axes(rect_histy, sharey=axs)
    ax_histy.plot(ics.field, ics.field_t / ics.fstoau, color=colors[0], linewidth=0.5, label='Field')
    ax_histy.plot(ics.field_envelope, ics.field_t / ics.fstoau, color=colors[0], alpha=0.4)
    ax_histy.plot(-ics.field_envelope, ics.field_t / ics.fstoau, color=colors[0], alpha=0.4)
    ax_histy.fill_betweenx(ics.field_t / ics.fstoau, ics.field_envelope, -ics.field_envelope, color=colors[0],
                           label='Envelope', alpha=0.2)
    ax_histy.set_xlim(np.min(-ics.field_envelope) * 1.2, np.max(ics.field_envelope) * 1.2)
    ax_histy.set_xlabel(r"$\vec{E}$")
    ax_histy.legend(frameon=True, framealpha=0.9, labelspacing=0.1)

    ax_histx = fig.add_axes(rect_histx, sharex=axs)
    ax_histx.plot(ics.spectrum[0] / ics.evtoau, ics.spectrum[-1] / np.max(ics.spectrum[-1]), color=colors[1],
                  label='Absorption spectrum')
    ax_histx.fill_between(ics.spectrum[0] / ics.evtoau, ics.spectrum[-1] * 0,
                          ics.spectrum[-1] / np.max(ics.spectrum[-1]), color=colors[1], alpha=0.2)
    ax_histx.plot(ics.field_ft_omega / ics.evtoau, ics.field_ft, color=colors[0], label='Pulse spectrum')
    ax_histx.fill_between(ics.field_ft_omega / ics.evtoau, ics.field_ft * 0, ics.field_ft, color=colors[0], alpha=0.2)
    ax_histx.set_ylim(0, 1.2)
    ax_histx.set_ylabel(r"$\epsilon$")
    ax_histx.legend(frameon=False, labelspacing=0.1)

    ax_histx.tick_params("both", which='both', direction='in', labelbottom=False)
    ax_histy.tick_params("both", which='both', direction='in', labelleft=False)
    ax_histy.set_xticks([])
    ax_histx.set_yticks([])
    axs.set_xlabel(r"$\Delta E$ (eV)")
    axs.set_ylabel(r"$t$ (fs)")
    axs.minorticks_on()
    axs.tick_params("both", which='both', direction='in', top=True, right=True, color='white')

    plt.savefig('ic_filtering', dpi=300)
    plt.show()

print('\nFiltering of initial conditions finished.'
      '\n - "May the laser pulses be with you."\n')

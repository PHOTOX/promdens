"""Code for filtering of initial conditions for nonadiabatic dynamics with the laser pulse.

© Jiri Janos 2024"""

# todo: add multiple excited states
# todo: change from sigma to FWHM and implement Lorentzian and sech2

import argparse
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np


### functions adn classes ###
class initial_conditions:
    """Class containing initial conditions.
    nsamples = 0            # number of samples considered, if set 0 then maximum number provided will be taken
    input_type = 'file'     # data read form file, other options are 'Newton-X' etc (to be done)
    """

    # constants in a.u.
    hbar = 1.0
    c = 137.03599

    # conversions
    evtoau = 0.036749405469679
    fstoau = 41.341374575751

    def __init__(self, nsamples=0, input_type='file'):
        self.nsamples = nsamples
        self.input_type = input_type
        # flags that everything was calculated
        self.input_read = False
        self.units_converted = False
        self.spectrum_calculated = False
        self.field_calculated = False

    def read_input_data(self, fname='ics.dat'):
        """Reading the input data: index of traj, excitation energies and transition dipole moments."""
        print(f"* Reading data from file '{fname}' of type '{self.input_type}'.")
        if self.input_type == 'file':
            try:
                # reading input file
                input = np.loadtxt(fname, dtype=float).T
                if self.nsamples == 0:
                    self.nsamples = np.shape(input)[1]
                    print(f"  - Number of ICs loaded from the input file: {self.nsamples}")
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
            self.traj_index = np.array(input[0], dtype=int)  # indexes of trajectories
            self.de = input[1]  # excitation energies
            self.tdm = input[2]  # transition dipole moments
        else:
            print(f"\nERROR: File type '{self.input_type}' not supported!")
            exit(1)

        self.input_read = True

    def convert_units(self, energy_units):
        print(f"* Converting units.")
        if energy_units == 'eV':
            self.de *= self.evtoau
        self.units_converted = True

    def calc_spectrum(self):

        print(f"* Calculating spectrum with Nuclear ensemble approach.")

        if not self.input_read:
            print(f"ERROR: Field yet not calculated. Please first use 'calc_field()'!")
            exit(1)
        elif not self.units_converted:
            print(f"ERROR: Units not converted yet. Please first use 'convert_units()'!")
            exit(1)

        def gauss(e, de, tdm, h):
            return de*tdm**2*np.exp(-(e - de)**2/2/h**2)

        h = (4/3/self.nsamples)**0.2*np.std(self.de)
        emin, emax = np.min(self.de), np.max(self.de)
        npoints = 10000
        self.spectrum = np.zeros(shape=(2, npoints), dtype=float)
        self.spectrum[0] = np.linspace(emin - 2*h, emax + 2*h, npoints)

        for ic in range(self.nsamples):
            self.spectrum[1] += gauss(self.spectrum[0], self.de[ic], self.tdm[ic], h)

        # todo: multiplication of factors should be checked with stepans code
        self.spectrum[1] *= np.pi/(3*self.hbar*self.c*self.nsamples*h*np.sqrt(2*np.pi))

        self.spectrum_calculated = True

    def calc_field_envelope(self, t):
        if self.field_envelope_type == 'gauss':
            return np.exp(-(t - self.field_t0)**2/2/self.field_sigma**2)  # envelope of the field
        else:
            print(f"ERROR: unavailable envelope '{envelope_type}'!")
            exit(1)

    def calc_field(self, omega, sigma, t0=0.0, lchirp=0.0, envelope_type='gauss'):

        print(f"* Calculating laser pulse field using envelope type '{envelope_type}', omega={omega:.6f} a.u.,"
              f" sigma={sigma:.6f} a.u., t0={t0:.6f} a.u.")
        # saving field parameters
        self.field_sigma = sigma
        self.field_omega = omega
        self.field_lchirp = lchirp
        self.field_envelope_type = envelope_type
        self.field_t0 = t0
        self.field_fwhm = 2.35*sigma

        # calculating the field
        self.field_t = np.arange(t0 - 1.5*self.field_fwhm, t0 + 1.5*self.field_fwhm, 2*np.pi/omega/50)  # time array for the field in a.u.
        self.field_envelope = self.calc_field_envelope(self.field_t)
        self.field = self.field_envelope*np.cos((omega + lchirp*self.field_t)*self.field_t)  # field

        # calculating the FT of the field
        dt = 2*np.pi/omega/50
        t_ft = np.arange(t0 - 20*self.field_fwhm, t0 + 20*self.field_fwhm, dt)
        field = np.exp(-(t_ft - t0)**2/2/sigma**2)*np.cos((omega + lchirp*t_ft)*t_ft)
        self.field_ft = np.abs(np.fft.rfft(field))  # FT
        self.field_ft /= np.max(self.field_ft)  # normalizing
        self.field_ft_omega = 2*np.pi*np.fft.rfftfreq(len(t_ft), dt)

        self.field_calculated = True

    def pulse_wigner(self, tprime, de):

        if not self.field_calculated:
            print(f"ERROR: Input data not read yet. Please first use 'read_input_data()'!")
            exit(1)

        # attempt for an adaptive integration step according to the oscillation
        # however, for plotting on a grid it doesn't work as the edges have dE-omega = 12 or more so I needed to add a
        # thresh with omega. Implementing this for the sampling shouldn't be a problem because I will have only relevant
        # excitation energies.
        loc_omega = self.field_omega + self.field_lchirp*tprime
        if de != loc_omega:
            T = 2*np.pi/(np.min([np.abs(de - loc_omega), loc_omega]))
        else:  # in case they are equal, I cannot divide by 0
            T = 2*np.pi/loc_omega
        ds = np.min([T/200, self.field_fwhm/200])
        s = np.arange(self.field_t0 - 10*self.field_fwhm, self.field_t0 + 10*self.field_fwhm, step=ds)

        iexp = np.exp(1j*(de/self.hbar - loc_omega)*s)
        integral = np.trapz(x=s, y=iexp*self.calc_field_envelope(tprime + s/2)*self.calc_field_envelope(tprime - s/2))
        W = 1/2/np.pi/self.hbar*integral
        if np.abs(np.imag(W)) > 1e-12:
            print(f"WARNING: imaginary part of the wigner pulse transform is nonzero {np.abs(np.imag(W))}!")
        return np.real(W)

    def sample_initial_conditions(self, new_ic_nsamples):

        print(f"* Sampling {new_ic_nsamples:d} initial conditions considering the laser pulse.")

        if not self.input_read:
            print(f"\nERROR: Field yet not calculated. Please first use 'calc_field()'!")
            exit(1)
        elif not self.units_converted:
            print(f"\nERROR: Units not converted yet. Please first use 'convert_units()'!")
            exit(1)

        def progress(percent, width, n, str=''):
            """Function to print progress of calculation."""
            left = width*percent//n
            right = width - left
            print(f'\r{str}[', '#'*left, ' '*right, '] %d'%(percent*100/n) + '%', sep='', end='', flush=True)

        # variable for selected samples
        samples = np.zeros((4, new_ic_nsamples))  # index, initial time, de, tdm #todo: possibly index of state once

        # setting maximum random number generated during sampling
        rnd_max = np.max(self.tdm**2)*self.pulse_wigner(t0, de=omega)*1.05

        nattempts = 0  # to calculate efficiency of the sampling
        for i in range(new_ic_nsamples):
            while True:
                nattempts += 1

                # index of sample
                rnd_index = np.random.randint(low=0, high=self.nsamples, dtype=int)
                rnd_time = np.random.uniform(low=-1.5*self.field_fwhm, high=1.5*self.field_fwhm)
                rnd = np.random.uniform(low=0, high=rnd_max)  # random number to be compared with Wig. dist.

                prob = self.tdm[rnd_index]**2*self.pulse_wigner(rnd_time, self.de[rnd_index])

                # check if the distribution is not negative or higher than rnd_max
                if prob < -1e-4*rnd_max:  # this threshold is for numerical accuracy
                    print(f"\nERROR: Sampling from negative distribution!\n"
                          f"For sample {self.traj_index[rnd_index]}) the probability is {prob}.\n"
                          f"Integration step for the pulse Wigner transformation probably necessary.")
                    exit(1)
                elif prob > rnd_max:
                    print(f"\nERROR: rnd_max ({rnd_max}) is smaller than probability ({prob} for sample "
                          f"{self.traj_index[rnd_index]}). Increase rnd_max and rerun again.")
                    exit(1)

                # check if the point is sampled
                if rnd <= prob:
                    samples[0, i] = self.traj_index[rnd_index]
                    samples[1, i] = rnd_time
                    samples[2, i] = self.de[rnd_index]
                    samples[3, i] = self.tdm[rnd_index]
                    break
            progress(i + 1, 50, new_ic_nsamples, str='  Sampling progress: ')
        # save samples withing the object
        self.filtered_ics = samples
        # getting unique indexes
        unique = np.array(np.unique(samples[0]), dtype=int)

        print(f"\n  - Success rate of random sampling: {new_ic_nsamples/nattempts*100:.5f} %")

        # save the selected samples
        np.savetxt(f'IC_sampling.dat', samples.T, fmt=['%8d', '%18.8f', '%16.8f', '%16.8f'],
                   header=f"nsamples = {new_ic_nsamples:d}, unique nsamples = {len(unique):d}, "
                          f"omega = {self.field_omega:.4f} a.u., "
                          f"linearchirp = {self.field_lchirp:.10f} a.u., sigma = {self.field_sigma/self.fstoau:.3f} fs, "
                          f"t0 = {self.field_t0/self.fstoau:.3f} fs\n"
                          f"index        time (a.u.)        de (a.u.)        tdm (arb.unit)")
        print(f"  - Output saved to file 'IC_sampling.dat'.")

        # print indexes of trajectories that must be run
        print(f"  - Selected {len(unique)} unique ICs from {self.nsamples} provided. Unique ICs that are necessary "
              f"to propagate:\n   ", *np.array(unique, dtype=str))


### setting up parser ###
parser = argparse.ArgumentParser(description="Parser for this code", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n", "--nsamples", default=0, type=int,
                    help="Number of initial conditions considered for sampling. 0 takes all initial conditions provided in the input file.")
parser.add_argument("-nf", "--nfsamples", default=1000, type=int, help="Number of filtered initial conditions that will be calculated.")
parser.add_argument("-ft", "--file_type", default='file', help="Input file type. Options are 'file'.")
parser.add_argument("-p", "--plot", action="store_true", help="Plot along the code to see loaded data and results")
parser.add_argument("-eu", "--energy_units", default='a.u.',
                    help="Units in which energies are provided. Options are 'a.u.', 'eV'. If input file type is Newton-X, defaults for Newton-X will be taken.")
parser.add_argument("-w", "--omega", default=0.1, type=float, help="Frequency of the field omega in a.u.")
parser.add_argument("-lch", "--linear_chirp", default=0.0, type=float, help="Linear chirp [w(t) = w+lch*t] of the field frequency in a.u.")
parser.add_argument("-s", "--sigma", default=10.0, type=float, help="Sigma width parameter for the pulse in fs.")
parser.add_argument("-t0", "--t0", default=0.0, type=float, help="Time of the maximum of the field in fs.")
parser.add_argument("-env", "--envelope_type", default='gauss', help="Type of field envelope. Options are 'gauss'.")
parser.add_argument("file", help="Input file name.")

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
plotting = config['plot']
energy_units = config['energy_units']
sigma = config['sigma']
omega = config['omega']
lchirp = config['linear_chirp']
t0 = config['t0']
envelope_type = config['envelope_type']
ftype = config['file_type']
fname = config['file']

# converting pulse input to a.t.u.
fstoau = 41.341374575751
t0 *= fstoau
sigma *= fstoau

# checking input
if not energy_units in ['a.u.', 'eV']:
    print(f"\nERROR: {energy_units} is not available unit for energy!")
    exit(1)

if not ftype in ['file']:
    print(f"\nERROR: {ftype} is not available file type!")
    exit(1)

if not envelope_type in ['gauss']:
    print(f"\nERROR: {envelope_type} is not available envelope type!")
    exit(1)

if not exists(fname):
    print(f"\nERROR: file '{fname}' not found!")
    exit(1)

if nsamples < 0:
    print(f"\nERROR: nsamples is smaller than 0 ({nsamples})!")
    exit(1)

if new_nsamples < 0:
    print(f"\nERROR: nsamples is smaller than 0 ({new_nsamples})!")
    exit(1)

### code ###
# creating object initial conditions
ics = initial_conditions(nsamples=nsamples, input_type=ftype)

# reading input data
ics.read_input_data(fname=fname)

# converting units to make everything in atomic units
ics.convert_units(energy_units=energy_units)

# calculating spectrum
ics.calc_spectrum()

# plotting loaded data
if plotting:
    print("  - Plotting Figure 1")
    colors = plt.cm.viridis(0.35)
    plt.rcParams["font.family"] = 'Helvetica'
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle("Characteristics of initial conditions (ICs) loaded")

    axs[0].plot(ics.traj_index, ics.de/ics.evtoau, color=colors, alpha=0.6)
    axs[0].scatter(ics.traj_index, ics.de/ics.evtoau, color=colors, s=5)
    axs[0].set_xlim(np.min(ics.traj_index) - 1, np.max(ics.traj_index) + 1)
    axs[0].set_xlabel("IC index")
    axs[0].set_ylabel(r"$\Delta E$ (eV)")
    axs[0].set_title(r"Excitation energies")

    axs[1].plot(ics.traj_index, ics.tdm, color=colors, alpha=0.6)
    axs[1].scatter(ics.traj_index, ics.tdm, color=colors, s=5)
    axs[1].set_xlim(np.min(ics.traj_index) - 1, np.max(ics.traj_index) + 1)
    axs[1].set_xlabel("IC index")
    axs[1].set_ylabel(r"$|\mu|$ (arb. unit)")
    axs[1].set_title(r"Transition dipole moments")

    axs[2].plot(ics.spectrum[0]/ics.evtoau, ics.spectrum[1], color=colors)
    axs[2].fill_between(ics.spectrum[0]/ics.evtoau, ics.spectrum[1]*0, ics.spectrum[1], color=colors, alpha=0.2)
    axs[2].set_xlim(np.min(ics.spectrum[0]/ics.evtoau), np.max(ics.spectrum[0]/ics.evtoau))
    axs[2].set_ylim(0, np.max(ics.spectrum[1])*1.2)
    axs[2].set_xlabel(r"$E$ (eV)")
    axs[2].set_ylabel(r"$\epsilon$ (arb. unit)")
    axs[2].set_title(r"Absorption spectrum")

    plt.tight_layout()
    plt.savefig('spectrum', dpi=300)
    plt.show(block=False)

# calculating the field
ics.calc_field(omega=omega, sigma=sigma, t0=t0, lchirp=lchirp, envelope_type=envelope_type)

# plotting field
if plotting:
    print("  - Plotting Figure 2")
    colors = plt.cm.viridis([0.35, 0.6])
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle("Field characteristics")

    axs[0].plot(ics.field_t/ics.fstoau, ics.field, color=colors[0], linewidth=0.5, label='Field')
    axs[0].plot(ics.field_t/ics.fstoau, ics.field_envelope, color=colors[0], alpha=0.4)
    axs[0].plot(ics.field_t/ics.fstoau, -ics.field_envelope, color=colors[0], alpha=0.4)
    axs[0].fill_between(ics.field_t/ics.fstoau, ics.field_envelope, -ics.field_envelope, color=colors[0], label='Envelope', alpha=0.2)
    axs[0].set_xlim(np.min(ics.field_t/ics.fstoau), np.max(ics.field_t/ics.fstoau))
    axs[0].set_ylim(np.min(-ics.field_envelope)*1.2, np.max(ics.field_envelope)*1.2)
    axs[0].set_xlabel(r"$t$ (fs)")
    axs[0].set_ylabel(r"$\vec{E}$ (arb. unit)")
    axs[0].set_title(r"Laser pulse field")
    axs[0].legend(frameon=False, labelspacing=0.1)

    axs[1].plot(ics.spectrum[0]/ics.evtoau, ics.spectrum[1]/np.max(ics.spectrum[1]), color=colors[1], label='Absorption spectrum')
    axs[1].fill_between(ics.spectrum[0]/ics.evtoau, ics.spectrum[1]*0, ics.spectrum[1]/np.max(ics.spectrum[1]), color=colors[1], alpha=0.2)
    axs[1].plot(ics.field_ft_omega/ics.evtoau, ics.field_ft, color=colors[0], label='Pulse spectrum')
    axs[1].fill_between(ics.field_ft_omega/ics.evtoau, ics.field_ft*0, ics.field_ft, color=colors[0], alpha=0.2)
    axs[1].set_xlim(np.min(ics.spectrum[0]/ics.evtoau), np.max(ics.spectrum[0]/ics.evtoau))
    axs[1].set_ylim(0, 1.2)
    axs[1].set_xlabel(r"$E$ (eV)")
    axs[1].set_ylabel(r"$\epsilon$ (arb. unit)")
    axs[1].set_title(r"Pulse spectrum")
    axs[1].legend(frameon=False, labelspacing=0.1)

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

    emin, emax = np.min(ics.spectrum[0]/ics.evtoau), np.max(ics.spectrum[0]/ics.evtoau)
    tmin, tmax = np.min(ics.field_t/ics.fstoau), np.max(ics.field_t/ics.fstoau)

    h = axs.hist2d(ics.filtered_ics[2]/ics.evtoau, ics.filtered_ics[1]/ics.fstoau, range=[[emin, emax], [tmin, tmax]], bins=(100, 100),
                   cmap=plt.cm.viridis, density=True)
    if lchirp != 0:
        axs.plot((omega + lchirp*ics.field_t)/ics.evtoau, ics.field_t/ics.fstoau, color='white', linestyle='--', label=r"$\omega(t)$")
        axs.legend(frameon=True, framealpha=0.4, labelspacing=0.1)

    ax_histy = fig.add_axes(rect_histy, sharey=axs)
    ax_histy.plot(ics.field, ics.field_t/ics.fstoau, color=colors[0], linewidth=0.5, label='Field')
    ax_histy.plot(ics.field_envelope, ics.field_t/ics.fstoau, color=colors[0], alpha=0.4)
    ax_histy.plot(-ics.field_envelope, ics.field_t/ics.fstoau, color=colors[0], alpha=0.4)
    ax_histy.fill_betweenx(ics.field_t/ics.fstoau, ics.field_envelope, -ics.field_envelope, color=colors[0], label='Envelope', alpha=0.2)
    ax_histy.set_xlim(np.min(-ics.field_envelope)*1.2, np.max(ics.field_envelope)*1.2)
    ax_histy.set_xlabel(r"$\vec{E}$ (arb. unit)")
    ax_histy.legend(frameon=True, framealpha=0.9, labelspacing=0.1)

    ax_histx = fig.add_axes(rect_histx, sharex=axs)
    ax_histx.plot(ics.spectrum[0]/ics.evtoau, ics.spectrum[1]/np.max(ics.spectrum[1]), color=colors[1], label='Absorption spectrum')
    ax_histx.fill_between(ics.spectrum[0]/ics.evtoau, ics.spectrum[1]*0, ics.spectrum[1]/np.max(ics.spectrum[1]), color=colors[1], alpha=0.2)
    ax_histx.plot(ics.field_ft_omega/ics.evtoau, ics.field_ft, color=colors[0], label='Pulse spectrum')
    ax_histx.fill_between(ics.field_ft_omega/ics.evtoau, ics.field_ft*0, ics.field_ft, color=colors[0], alpha=0.2)
    ax_histx.set_ylim(0, 1.2)
    ax_histx.set_ylabel(r"$\epsilon$ (arb. unit)")
    ax_histx.legend(frameon=False, labelspacing=0.1)

    ax_histx.tick_params(axis="x", which='both', direction='in', labelbottom=False)
    ax_histy.tick_params(axis="y", direction='in', labelleft=False)
    axs.set_xlabel(r"$\Delta E$ (eV)")
    axs.set_ylabel(r"$t$ (fs)")
    plt.savefig('ic_filtering', dpi=300)
    plt.show()

print("\nFiltering of initial conditions finished.\n")

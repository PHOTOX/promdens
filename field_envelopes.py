"""Plotting field envelopes available in IC_filtering.py"""

import matplotlib.pyplot as plt
import numpy as np


class envelopes:
    # constants in atomic units
    hbar = 1.0

    # conversion factor between units
    evtoau = 0.036749405469679
    fstoau = 41.341374575751

    def field_cos(self, t):
        """
        Calculating oscilatoin of the field with the cos function.
        :param t: time
        :return: cos((w + lchirp*t)*t)
        """
        return np.cos((self.field_omega + self.field_lchirp*t)*t)

    def calc_field_envelope(self, t):
        """
        Calculating field envelope. The parameters of the field are taken form the class.
        :param t: time axis (a.u.)
        :return: envelope of the field
        """
        if self.field_envelope_type == 'gauss':
            return np.exp(-2*np.log(2)*(t - self.field_t0)**2/self.field_fwhm**2)
        elif self.field_envelope_type == 'lorentz':
            return (1 + 4/(1 + np.sqrt(2))*((t - self.field_t0)/self.field_fwhm)**2)**-1
        elif self.field_envelope_type == 'sech':
            return 1/np.cosh(2*np.log(1 + np.sqrt(2))*(t - self.field_t0)/self.field_fwhm)
        elif self.field_envelope_type == 'sin':
            field = np.zeros(shape=np.shape(t))
            for k in range(np.shape(t)[0]):
                if t[k] >= self.tmin and t[k] <= self.tmax:
                    field[k] = np.sin(np.pi/2*(t[k] - self.field_t0 + self.field_fwhm)/self.field_fwhm)
            return field
        elif self.field_envelope_type == 'sin2':
            T = 1.373412575*self.field_fwhm
            field = np.zeros(shape=np.shape(t))
            for k in range(np.shape(t)[0]):
                if t[k] >= self.tmin and t[k] <= self.tmax:
                    field[k] = np.sin(np.pi/2*(t[k] - self.field_t0 + T)/T)**2
            return field

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
            self.tmin, self.tmax = self.field_t0 - 2.4*self.field_fwhm, self.field_t0 + 2.4*self.field_fwhm
        elif self.field_envelope_type == 'lorentz':
            print("  - E(t) = (1+4/(1+sqrt(2))*(t/fwhm)^2)^-1*cos((omega+lchirp*t)*t)")
            self.tmin, self.tmax = self.field_t0 - 8*self.field_fwhm, self.field_t0 + 8*self.field_fwhm
        elif self.field_envelope_type == 'sech':
            print("  - E(t) = 1/cosh(2*ln(1+sqrt(2))*t/fwhm)*cos((omega+lchirp*t)*t)")
            self.tmin, self.tmax = self.field_t0 - 4.4*self.field_fwhm, self.field_t0 + 4.4*self.field_fwhm
        elif self.field_envelope_type == 'sin':
            print("  - E(t) = sin(pi/2*(t-t0+fwhm)/fwhm)*cos((omega+lchirp*t)*t) in range [t0-fwhm,t0+fwhm]")
            self.tmin, self.tmax = self.field_t0 - self.field_fwhm, self.field_t0 + self.field_fwhm
        elif self.field_envelope_type == 'sin2':
            print("  - E(t) = sin(pi/2*(t-t0+T)/T)^2*cos((omega+lchirp*t)*t) in range [t0-T,t0+T] where T=1.373412575*fwhm")
            T = 1.373412575*self.field_fwhm
            self.tmin, self.tmax = self.field_t0 - T, self.field_t0 + T

        # calculating the field
        self.field_t = np.arange(self.tmin, self.tmax, 2*np.pi/omega/50)  # time array for the field in a.u.
        self.field_envelope = self.calc_field_envelope(self.field_t)
        self.field = self.field_envelope*self.field_cos(self.field_t)

        # calculating the FT of the field
        dt = 2*np.pi/omega/50
        t_ft = np.arange(self.tmin - 20*self.field_fwhm, self.tmax + 20*self.field_fwhm, dt)
        field = self.calc_field_envelope(t_ft)*self.field_cos(t_ft)
        self.field_ft = np.abs(np.fft.rfft(field))  # FT
        self.field_ft /= np.max(self.field_ft)  # normalizing
        self.field_ft_omega = 2*np.pi*np.fft.rfftfreq(len(t_ft), dt)

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
        loc_omega = self.field_omega + self.field_lchirp*tprime
        if de != loc_omega:
            T = 2*np.pi/(np.min([np.abs(de - loc_omega), loc_omega]))
        else:  # in case they are equal, I cannot divide by 0
            T = 2*np.pi/loc_omega
        ds = np.min([T/50, self.field_fwhm/500])

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
        s = np.arange(0, factor*self.field_fwhm, step=ds)
        cos = np.cos((de/self.hbar - loc_omega)*s)
        integral = 2*np.trapz(x=s, y=cos*self.calc_field_envelope(tprime + s/2)*self.calc_field_envelope(tprime - s/2))

        W = 1/2/np.pi/self.hbar*integral

        return W


# input data
fwhm = 10  # in fs
omega = 1  # in a.u.
lchirp = 0  # in a.u.
t0 = 0  # in fs
envelope_types = ['sin', 'sin2', 'gauss', 'sech', 'lorentz']

# conversion of units
evtoau = 0.036749405469679
fstoau = 41.341374575751
t0 *= fstoau
fwhm *= fstoau

# plotting different envelopes of electric field and intensity, and their spectra
print("\nEnvelope profiles and spectra calculated")
colors = plt.cm.viridis(np.linspace(0, 0.9, len(envelope_types)))
fig, axs = plt.subplots(2, 2, figsize=(7, 7))

for i in range(len(envelope_types)):
    envelope_type = envelope_types[i]

    field = envelopes()
    field.calc_field(omega=omega, fwhm=fwhm, t0=t0, lchirp=lchirp, envelope_type=envelope_type)
    field.field_ft_omega -= omega

    axs[0, 0].plot(field.field_t/field.fstoau, field.field_envelope, color=colors[i], label=f"'{envelope_type}'")
    axs[0, 1].plot(field.field_t/field.fstoau, field.field_envelope**2, color=colors[i], label=f"'{envelope_type}'")
    axs[1, 0].plot(field.field_ft_omega/field.evtoau, field.field_ft, color=colors[i], label=f"'{envelope_type}'")
    axs[1, 1].plot(field.field_ft_omega/field.evtoau, field.field_ft**2, color=colors[i], label=f"'{envelope_type}'")

axs[0, 0].axhline(0.5, color='black', linestyle='--', lw=0.5)
axs[0, 0].set_xlim(-3*fwhm/fstoau, 3*fwhm/fstoau)
axs[0, 0].set_ylim(0, 1.2)
axs[0, 0].set_xlabel(r"$t$ (fs)")
axs[0, 0].set_ylabel(r"$\varepsilon(t)$")
axs[0, 0].set_title(r"Pulse envelope")
axs[0, 0].legend(frameon=False, labelspacing=0.1, loc='upper right')

fwhm2 = fwhm/2/fstoau
axs[0, 1].plot([-fwhm2, -fwhm2, fwhm2, fwhm2], [0, 0.5, 0.5, 0], color='black', linestyle='--')
axs[0, 1].text(0, 0.2, r"$\tau_\mathregular{FWHM}$", horizontalalignment='center', color='black')
axs[0, 1].set_xlim(-3*fwhm/fstoau, 3*fwhm/fstoau)
axs[0, 1].set_ylim(0, 1.2)
axs[0, 1].set_xlabel(r"$t$ (fs)")
axs[0, 1].set_ylabel(r"$I(t)$")
axs[0, 1].set_title(r"Pulse intensity")
axs[0, 1].legend(frameon=False, labelspacing=0.1, loc='upper right')

axs[1, 0].set_xlim(-2*np.pi/fwhm*fstoau, 2*np.pi/fwhm*fstoau)
axs[1, 0].set_ylim(0, 1.2)
axs[1, 0].set_xlabel(r"$E$ (eV)")
axs[1, 0].set_ylabel(r"$|\varepsilon(\omega)|$")
axs[1, 0].set_title(r"Envelope spectrum")
axs[1, 0].legend(frameon=False, labelspacing=0.1, loc='upper right')

axs[1, 1].set_xlim(-2*np.pi/fwhm*fstoau, 2*np.pi/fwhm*fstoau)
axs[1, 1].set_ylim(0, 1.2)
axs[1, 1].set_xlabel(r"$E$ (eV)")
axs[1, 1].set_ylabel(r"$S(\omega)$")
axs[1, 1].set_title(r"Spectral intensity")
axs[1, 1].legend(frameon=False, labelspacing=0.1, loc='upper right')

plt.tight_layout()
plt.savefig('implemented_envelopes', dpi=300)
plt.show(block=False)

# plotting Wigner transformations of the field
print("\nWigner transfroms of the pulses")
fig, axs = plt.subplots(2, len(envelope_types), figsize=(2.5*len(envelope_types), 5), sharex=True, sharey=True)

# create a 2D map
grid = 250
e = np.linspace(-1.5*np.pi/fwhm*fstoau, 1.5*np.pi/fwhm*fstoau, grid)
t = np.linspace(-2.1*fwhm/fstoau, 2.1*fwhm/fstoau, grid)
e2d, t2d = np.meshgrid(e, t)

for i in range(len(envelope_types)):
    envelope_type = envelope_types[i]
    pulse_wigner = np.zeros(np.shape(e2d))
    field = envelopes()
    field.calc_field(omega=omega, fwhm=fwhm, t0=t0, lchirp=lchirp, envelope_type=envelope_type)

    for j in range(len(t)):
        for k in range(len(e)):
            pulse_wigner[j, k] = field.pulse_wigner(tprime=t2d[j, k]*fstoau, de=e2d[j, k]*evtoau + omega)

    pulse_wigner /= np.max(np.abs(pulse_wigner))
    pc = axs[0, i].pcolormesh(e2d, t2d, pulse_wigner, cmap='RdBu', vmin=-1, vmax=1)
    if i == len(envelope_types) - 1: fig.colorbar(pc, ax=axs[0, i], shrink=0.92, fraction=0.05)

    pc = axs[1, i].pcolormesh(e2d, t2d, pulse_wigner, cmap='Blues', norm='log', vmin=1e-5, vmax=1)
    if i == len(envelope_types) - 1: fig.colorbar(pc, ax=axs[1, i], shrink=0.92, fraction=0.05)
    pc = axs[1, i].pcolormesh(e2d, t2d, -pulse_wigner, cmap='Reds', norm='log', vmin=1e-5, vmax=1)

    axs[1, i].set_xlabel(r"$\Delta E$ (eV)")
    axs[0, i].set_title(f"'{envelope_type}'")
    axs[0, i].tick_params('both', direction='in', which='both', top=True, right=True)
    axs[1, i].tick_params('both', direction='in', which='both', top=True, right=True)

axs[0, 0].set_ylabel(r"$t$ (fs)")
axs[1, 0].set_ylabel(r"$t$ (fs)")

axs[0, 0].text(x=np.min(e) + 0.03, y=np.max(t) - 0.15*fwhm/fstoau, s=r"$\mathcal{W}(t,\Delta E)$", va="top", ha="left")
axs[1, 0].text(x=np.min(e) + 0.03, y=np.max(t) - 0.15*fwhm/fstoau, s=r"$\ln[\mathcal{W}(t,\Delta E)]$", va="top", ha="left")

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig('envelope_wigner_transform', dpi=300)
plt.show()

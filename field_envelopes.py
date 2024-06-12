"""Plotting field envelopes available in IC_filtering.py"""

import matplotlib.pyplot as plt
import numpy as np


class envelopes:
    # conversion factor between units
    evtoau = 0.036749405469679
    fstoau = 41.341374575751

    def field_cos(self, t):
        """
        Calculating oscilatoin of the field with the cos function.
        :param t: time
        :return: cos((w + lchirp*t)*t)
        """
        return np.cos((self.field_omega + self.field_lchirp * t) * t)

    def calc_field_envelope(self, t):
        """
        Calculating field envelope. The parameters of the field are taken form the class.
        :param t: time axis (a.u.)
        :return: envelope of the field
        """
        if self.field_envelope_type == 'gauss':
            return np.exp(-2 * np.log(2) * (t - self.field_t0) ** 2 / self.field_fwhm ** 2)
        elif self.field_envelope_type == 'lorentz':
            return (1 + 4 / (1 + np.sqrt(2)) * (t / self.field_fwhm) ** 2) ** -1
        elif self.field_envelope_type == 'sech':
            return 1 / np.cosh(2 * np.log(1 + np.sqrt(2)) * t / self.field_fwhm)
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
            print("  E(t) = exp(-2*ln(2)*(t-t0)^2/fwhm^2)*cos((omega+lchirp*t)*t)")
            self.tmin, self.tmax = self.field_t0 - 2.4 * self.field_fwhm, self.field_t0 + 2.4 * self.field_fwhm
        elif self.field_envelope_type == 'lorentz':
            print("  E(t) = (1+4/(1+sqrt(2))*(t/fwhm)^2)^-1*cos((omega+lchirp*t)*t)")
            self.tmin, self.tmax = self.field_t0 - 8 * self.field_fwhm, self.field_t0 + 8 * self.field_fwhm
        elif self.field_envelope_type == 'sech':
            print("  E(t) = 1/cosh(2*ln(1+sqrt(2))*t/fwhm)*cos((omega+lchirp*t)*t)")
            self.tmin, self.tmax = self.field_t0 - 4.4 * self.field_fwhm, self.field_t0 + 4.4 * self.field_fwhm
        elif self.field_envelope_type == 'sin':
            print("  E(t) = sin(pi/2*(t-t0+fwhm)/fwhm)*cos((omega+lchirp*t)*t) in range [t0-fwhm,t0+fwhm]")
            self.tmin, self.tmax = self.field_t0 - self.field_fwhm, self.field_t0 + self.field_fwhm
        elif self.field_envelope_type == 'sin2':
            print(
                "  E(t) = sin(pi/2*(t-t0+T)/T)^2*cos((omega+lchirp*t)*t) in range [t0-T,t0+T] where T=1.373412575*fwhm")
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


fwhm = 10
omega = 1
lchirp = 0
t0 = 0
envelope_types = ['sin', 'sin2', 'gauss', 'sech', 'lorentz']

fstoau = 41.341374575751
t0 *= fstoau
fwhm *= fstoau

colors = plt.cm.viridis(np.linspace(0, 0.9, len(envelope_types)))
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))

for i in range(len(envelope_types)):
    envelope_type = envelope_types[i]

    field = envelopes()
    field.calc_field(omega=omega, fwhm=fwhm, t0=t0, lchirp=lchirp, envelope_type=envelope_type)
    field.field_ft_omega -= omega

    axs[0].plot(field.field_t / field.fstoau, field.field_envelope, color=colors[i], label=f"'{envelope_type}'")
    axs[1].plot(field.field_t / field.fstoau, field.field_envelope ** 2, color=colors[i], label=f"'{envelope_type}'")
    axs[2].plot(field.field_ft_omega / field.evtoau, field.field_ft, color=colors[i], label=f"'{envelope_type}'")

axs[0].set_xlim(-3 * fwhm / fstoau, 3 * fwhm / fstoau)
axs[0].set_ylim(0, 1.2)
axs[0].set_xlabel(r"$t$ (fs)")
axs[0].set_ylabel(r"$\vec{E}$ (arb. unit)")
axs[0].set_title(r"Pulse amplitude envelope")
axs[0].legend(frameon=False, labelspacing=0.1, loc='upper right')

fwhm2 = fwhm / 2 / fstoau
axs[1].plot([-fwhm2, -fwhm2, fwhm2, fwhm2], [0, 0.5, 0.5, 0], color='black', linestyle='--')
axs[1].text(0, 0.2, r"$\tau_\mathregular{FWHM}$", horizontalalignment='center', color='black')
axs[1].set_xlim(-3 * fwhm / fstoau, 3 * fwhm / fstoau)
axs[1].set_ylim(0, 1.2)
axs[1].set_xlabel(r"$t$ (fs)")
axs[1].set_ylabel(r"$I$ (arb. unit)")
axs[1].set_title(r"Pulse intensity envelope")
axs[1].legend(frameon=False, labelspacing=0.1, loc='upper right')

axs[2].set_xlim(-2 * np.pi / fwhm * fstoau, 2 * np.pi / fwhm * fstoau)
axs[2].set_ylim(0, 1.2)
axs[2].set_xlabel(r"$E$ (eV)")
axs[2].set_ylabel(r"$\epsilon$ (arb. unit)")
axs[2].set_title(r"Pulse spectrum")
axs[2].legend(frameon=False, labelspacing=0.1, loc='upper right')

plt.tight_layout()
plt.savefig('implemented_envelopes', dpi=300)
plt.show()

"""
Analysis of laser pulse envelopes available in promdens.py
"""

import matplotlib.pyplot as plt
import matplotlib.colors

import numpy as np

from promdens.promdens import LaserPulse, InitialConditions

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
colormap = plt.cm.viridis(np.linspace(0, 0.9, len(envelope_types)))
colors = dict(zip(envelope_types, colormap))
fig, axs = plt.subplots(2, 2, figsize=(7, 7))

for envelope_type in envelope_types:
    print(f"* {envelope_type}")

    pulse = LaserPulse(omega=omega, fwhm=fwhm, t0=t0, lchirp=lchirp, envelope_type=envelope_type)
    field = InitialConditions()
    field.calc_field(pulse=pulse)
    field.field_ft_omega -= omega

    color = colors[envelope_type]
    axs[0, 0].plot(field.field_t/field.fstoau, field.field_envelope, color=color, label=envelope_type)
    axs[0, 1].plot(field.field_t/field.fstoau, field.field_envelope**2, color=color, label=envelope_type)
    axs[1, 0].plot(field.field_ft_omega/field.evtoau, field.field_ft, color=color, label=envelope_type)
    axs[1, 1].plot(field.field_ft_omega/field.evtoau, field.field_ft**2, color=color, label=envelope_type)

axs[0, 0].axhline(0.5, color='black', linestyle='--', lw=0.5)
axs[0, 0].set_xlim(-3*fwhm/fstoau, 3*fwhm/fstoau)
axs[0, 0].set_ylim(0, 1.2)
axs[0, 0].set_xlabel(r"$t$ (fs)")
axs[0, 0].set_ylabel(r"$\varepsilon(t)$")
axs[0, 0].set_title("Pulse envelope")
axs[0, 0].legend(frameon=False, labelspacing=0.1, loc='upper right')

fwhm2 = fwhm/2/fstoau
axs[0, 1].plot([-fwhm2, -fwhm2, fwhm2, fwhm2], [0, 0.5, 0.5, 0], color='black', linestyle='--')
axs[0, 1].text(0, 0.2, r"$\tau_\mathregular{FWHM}$", horizontalalignment='center', color='black')
axs[0, 1].set_xlim(-3*fwhm/fstoau, 3*fwhm/fstoau)
axs[0, 1].set_ylim(0, 1.2)
axs[0, 1].set_xlabel(r"$t$ (fs)")
axs[0, 1].set_ylabel(r"$I(t)$")
axs[0, 1].set_title("Pulse intensity")
axs[0, 1].legend(frameon=False, labelspacing=0.1, loc='upper right')

axs[1, 0].set_xlim(-2*np.pi/fwhm*fstoau, 2*np.pi/fwhm*fstoau)
axs[1, 0].set_ylim(0, 1.2)
axs[1, 0].set_xlabel(r"$E$ (eV)")
axs[1, 0].set_ylabel(r"$|\varepsilon(\omega)|$")
axs[1, 0].set_title("Envelope spectrum")
axs[1, 0].legend(frameon=False, labelspacing=0.1, loc='upper right')

axs[1, 1].set_xlim(-2*np.pi/fwhm*fstoau, 2*np.pi/fwhm*fstoau)
axs[1, 1].set_ylim(0, 1.2)
axs[1, 1].set_xlabel(r"$E$ (eV)")
axs[1, 1].set_ylabel(r"$S(\omega)$")
axs[1, 1].set_title("Spectral intensity")
axs[1, 1].legend(frameon=False, labelspacing=0.1, loc='upper right')

plt.tight_layout()
plt.savefig('implemented_envelopes', dpi=300)
plt.show(block=False)

# plotting Wigner transformations of the field
print("\nWigner transforms of the pulses")
fig, axs = plt.subplots(2, len(envelope_types), figsize=(2.5*len(envelope_types), 5), sharex=True, sharey=True)

# create a 2D map
grid = 250
e = np.linspace(-1.5*np.pi/fwhm*fstoau, 1.5*np.pi/fwhm*fstoau, grid)
t = np.linspace(-2.1*fwhm/fstoau, 2.1*fwhm/fstoau, grid)
e2d, t2d = np.meshgrid(e, t)

for i, envelope_type in enumerate(envelope_types):
    print(f"* {envelope_type}")
    pulse_wigner = np.zeros(np.shape(e2d))
    pulse = LaserPulse(omega=omega, fwhm=fwhm, t0=t0, lchirp=lchirp, envelope_type=envelope_type)

    for j in range(len(t)):
        for k in range(len(e)):
            pulse_wigner[j, k] = pulse.wigner_transform(tprime=t2d[j, k]*fstoau, de=e2d[j, k]*evtoau + omega)

    pulse_wigner /= np.max(np.abs(pulse_wigner))
    pc = axs[0, i].pcolormesh(e2d, t2d, pulse_wigner, cmap='RdBu', vmin=-1, vmax=1)
    if i == len(envelope_types) - 1:
        fig.colorbar(pc, ax=axs[0, i], shrink=0.92, fraction=0.05)

    # Use log scale for the colormap
    # https://matplotlib.org/stable/users/explain/colors/colormapnorms.html#logarithmic
    lognorm = matplotlib.colors.LogNorm(vmin=1e-5, vmax=1)
    # Plot negative and positive parts separately in blue and red, respectively
    pc = axs[1, i].pcolormesh(e2d, t2d, pulse_wigner, cmap='Blues', norm=lognorm)
    if i == len(envelope_types) - 1:
        fig.colorbar(pc, ax=axs[1, i], shrink=0.92, fraction=0.05)
    pc = axs[1, i].pcolormesh(e2d, t2d, -pulse_wigner, cmap='Reds', norm=lognorm)

    axs[1, i].set_xlabel(r"$\Delta E$ (eV)")
    axs[0, i].set_title(envelope_type)
    axs[0, i].tick_params('both', direction='in', which='both', top=True, right=True)
    axs[1, i].tick_params('both', direction='in', which='both', top=True, right=True)

axs[0, 0].set_ylabel(r"$t$ (fs)")
axs[1, 0].set_ylabel(r"$t$ (fs)")

axs[0, 0].text(x=np.min(e) + 0.03, y=np.max(t) - 0.15*fwhm/fstoau, s=r"$\mathcal{W}_E(t,\Delta E)$", va="top", ha="left")
axs[1, 0].text(x=np.min(e) + 0.03, y=np.max(t) - 0.15*fwhm/fstoau, s=r"$\ln[\mathcal{W}_E(t,\Delta E)]$", va="top", ha="left")

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
plt.savefig('envelope_wigner_transform', dpi=300)
plt.show()

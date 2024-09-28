# PROMDENS: Promoted Density Approach code

`promdens` is a Python code implementing the Promoted Density Approach (PDA) and its version for windowing (PDAW) freely available to the scientific community under MIT license.
Derivation of PDA and PDAW and its benchmark against quantum dynamics can be found at [ArXiV](https://arxiv.org/abs/2408.17359) or soon at JPCL.

### Installation
The code is published on PyPI and can be installed with pip

```console
pip install promdens
```

After installation, the code is available as a script via the `promdens` command. To print help, run:

```console
promdens --help
```

The minimum supported Python version is 3.7.
The code depends on `numpy` and `matplotlib` libraries that are automatically installed by pip.
However, since pip by default installs packages into a global Python environment,
it can break previously installed packages e.g. by installing an incompatible version of numpy.

Therefore, we recommend using tools like [pipx](https://pipx.pypa.io/stable/) or [uv](https://docs.astral.sh/uv) which install the dependencies into an isolated Python environment but still make the `promdens` command globally available.
See the example for pipx

```console
pip install pipx
pipx install promdens
```

and uv 

```console
pip install uv
uv tool install promdens
```


### Usage
The code requires information about the method (PDA or PDAW), the number of excited states to consider, 
the number of initials conditions to be generated, and the characteristics of the laser pulse, such as the envelope type 
(Gaussian, Lorentzian, sech, etc.), the pulse frequency, the linear chirp parameter, and the full width at half maximum parameter. 
The code can be launched from a terminal with a series of flags as follows

```console
promdens --method pda --energy_unit a.u. --tdm_unit debye --nstates 2 --fwhm 3 --omega 0.355 --npsamples 10 --envelope_type gauss input_file.dat
```

The input file should contain information about the excitation energies and magnitudes of the transition dipole moments 
for each pair of sampled nuclear positions and momenta (labelled by an index number).
In the following, we provide an example of the input file for the first two excited states of protonated formaldimine:
```
#index    dE12 (a.u.)   mu_12 (Debye)   dE13 (a.u.)   mu_13 (Debye)
1         0.32479719       0.1251       0.40293672       1.351
2         0.32070472       0.2434       0.40915241       1.289
3         0.34574925       0.7532       0.38595754       1.209
4         0.33093699       0.1574       0.36679075       1.403
5         0.31860215       0.1414       0.36973886       1.377
6         0.31057768       0.0963       0.40031651       1.390
7         0.33431888       0.1511       0.40055704       1.358
8         0.31621589       0.0741       0.36644659       1.425
9         0.32905912       0.5865       0.36662982       1.277
10        0.31505412       0.2268       0.35529522       1.411
```

Using this input file and running the command line above, the user receives the following output file called `pda.dat` containing information about excitation times and initial excited states:
```
# Sampling: number of ICs = 10, number of unique ICs = 5
# Field parameters: omega = 3.55000e-01 a.u., linear_chirp = 0.00000e+00 a.u., fwhm = 3.000 fs, t0 = 0.000 fs, envelope type = 'gauss'
# index        exc. time (a.u.)   el. state     dE (a.u.)       |tdm| (a.u.)
       3        15.09731061            1       0.34574925       0.29635106
       3        25.94554064            1       0.34574925       0.29635106
       3        61.98106992            1       0.34574925       0.29635106
       4         7.38522206            2       0.36679075       0.55201877
       8       -14.27561557            2       0.36644659       0.56067480
       9       155.72500917            2       0.36662982       0.50244331
       9       -44.31379959            2       0.36662982       0.50244331
      10        94.19109952            2       0.35529522       0.55516642
      10        -9.13220842            2       0.35529522       0.55516642
      10        31.75086044            2       0.35529522       0.55516642
```
Inspecting this output file shows that the code generated 10 initial conditions accounting for the effect of the laser pulse, yet only 5 unique ground-state samples (pairs of nuclear positions and momenta) were used: indexes 3, 9, and 10 were selected more than once. The initial conditions are also spread over both excited states. The user should then run only 5 nonadiabatic simulations: initiating the nuclear position-momentum pair with index 3 in the first excited state and the nuclear position-momentum pairs with indexes 4, 8, 9, and 10 in the second excited state.

If the same command would be used with PDAW instead of PDA (`--method pdaw`), the output file would look like
```
# Convolution: I(t) = exp(-4*ln(2)*(t-t0)^2/fwhm^2)
# Parameters:  fwhm = 3.000 fs, t0 = 0.000 fs
# index        weight S1        weight S2
       1      1.78475e-05      9.66345e-07
       2      1.56842e-05      2.59858e-08
       3      6.31027e-02      1.29205e-03
       4      1.79107e-04      1.62817e-01
       5      2.31817e-06      1.01665e-01
       6      2.96548e-08      3.90152e-06
       7      3.81650e-04      3.33694e-06
       8      2.36147e-07      1.75628e-01
       9      1.47188e-03      1.37747e-01
      10      1.33347e-06      3.55670e-01
```
The code provides the pulse intensity and weights necessary for the convolution described in Eq. (14) in the [article](https://arxiv.org/abs/2408.17359). Note that the intensity should be normalized before used in convolution. If only a restricted amount of trajectories can be calculated, the user should choose the indexes and initial excited states corresponding to the largest weights in the file. For example, if we could run only 10 trajectories of protonated formaldimin, we would run ground-state position-momentum pairs with indexes 3, 4, 7, and 9 starting in $S_1$ and indexes 3, 4, 5, 8, 9, and 10 starting in $S_2$.

If the user selects option `--plot`, the code will produce a series of plots analyzing the provided data and calculated results, e.g. the absorption spectrum calculated with the nuclear ensemble method, the pulse spectrum or the Wigner pulse transform.

The work on a more detailed manual is currently in progress. If you have any questions, do not hesitate to contact the developers.

### Analytic formulas for pulse envelope Wigner transform

The code is based on the Wigner pulse transform which requires evaluating the Wigner integral

$$\mathcal{W}_E(t^\prime,\omega)=\int _{-\infty}^{\infty} E\left(t^\prime+\frac{s}{2}\right) E^*\left(t^\prime-\frac{s}{2}\right) \mathrm{e}^{-i\omega s} \mathrm{d} s$$

To simplify the integral evaluation, we implemented simpler Wigner pulse envelope transform (see the [article](https://arxiv.org/abs/2408.17359) for more details)

$$\mathcal{W}_\varepsilon(t,\omega) = \int _{-\infty}^{\infty}  \varepsilon\left(t+\frac{s}{2}\right) \varepsilon\left(t-\frac{s}{2}\right) \mathrm{e}^{-i\omega s}  \mathrm{d} s$$

where $\varepsilon$ is the pulse envelope. While `lorentz`, `sin2` and `sech` the $\mathcal{W}_\varepsilon$ is still calculated numerically by employing the trapezoid rule for the integral, the `gauss` and `sin` envelope Wigner transforms are calculated analytically according to analytic formulas. In the following analytic formulas, we apply a substitution 
$\Omega = \Delta E/\hbar - \omega$.

#### Gaussian envelope
$$\mathcal{W}_\varepsilon(t^\prime,\omega)=\tau\sqrt{\frac{\pi}{\ln2}}16^{-\frac{(t^\prime - t_0)^2}{\tau^2}}\exp\left(-\frac{\tau^2\omega^2}{\ln16}\right)$$

#### Sinusoidal envelope
* $\pi\frac{-2\tau\Omega\cos(2(t^\prime - t_0 + \tau)\Omega)\sin(\pi(t^\prime - t_0)/\tau) +\pi\cos(\pi(t^\prime - t_0)/\tau)\sin(2(t^\prime - t_0 + \tau)\Omega))}{\Omega(\pi^2 - 4\tau^2\Omega^2)}$            if $t^\prime < t_0$ and $t^\prime > t_0 - \tau $
* $\pi\frac{2\tau\Omega\cos(2(-t^\prime + t_0 + \tau)\Omega)\sin(\pi(t^\prime - t_0)/\tau) +\pi\cos(\pi(t^\prime - t_0)/\tau)\sin(2(-t^\prime + t_0 + \tau)\Omega))}{\Omega(\pi^2 - 4\tau^2\Omega^2)}$            if $t^\prime \ge t_0$ and $t^\prime < t_0 - \tau $
* $0$            elsewhere
"""This test aims to compare to previous PDA and PDAW results that were published and compared to exact QD.
We use NaI as an example with two different laser pulses. If this test fails, the results produced by the code are unreliable!"""

from pathlib import Path

import numpy as np

from promdens.promdens import InitialConditions, LaserPulse

# path to the tests folder
path = Path(__file__).parent.absolute()

def test_pdaw_nai(tmp_path):
    ics = InitialConditions(nsamples=1000, nstates=1, input_type='file')

    ics.read_input_data(fname=path/'NaI_reference/test_data_nai.dat', energy_unit='a.u.', tdm_unit='a.u.')

    fstoau = 41.341374575751
    pulse = LaserPulse(omega=0.13520905, fwhm=20*fstoau, envelope_type='gauss', lchirp=0, t0=0)

    ics.calc_field(pulse=pulse)

    ics.windowing()

    # moving output to a temporary path dedicated to this test
    tmp_folder = tmp_path / "pdaw"
    tmp_folder.mkdir()
    Path('pdaw.dat').rename(tmp_folder / 'pdaw.dat')

    # the following command was used to generate the reference
    # promdens -m pdaw -n 1000 -w 0.13520905 -f 20 test_data_nai.dat
    # the reference was compared to exact QD
    reference = np.genfromtxt(path/'NaI_reference/test_pdaw_reference.dat').T[1]
    weights = ics.weights[0]

    assert len(weights) == len(weights)
    # comparing all weights, the threshold 1e-17 was based on numerical differences created by switching between numpy 1.26 and 2.1
    for i in range(len(weights)):
        assert (weights[i] - reference[i]) < 1e-17, f"pdaw weight[{i}] does not match: delta={weights[i] - reference[i]}"


def test_pda_nai(tmp_path):
    ics = InitialConditions(nsamples=1000, nstates=1, input_type='file')

    ics.read_input_data(fname=path/'NaI_reference/test_data_nai.dat', energy_unit='a.u.', tdm_unit='a.u.')

    fstoau = 41.341374575751
    pulse = LaserPulse(omega=0.14294844, fwhm=100*fstoau, envelope_type='gauss', lchirp=2e-6, t0=0)

    ics.calc_field(pulse=pulse)

    ics.sample_initial_conditions(nsamples_ic=10, neg_handling='error', preselect=True, seed=123456789)

    # moving output to a temporary path dedicated to this test
    tmp_folder = tmp_path / "pda"
    tmp_folder.mkdir()
    Path('pda.dat').rename(tmp_folder / 'pda.dat')

    # the following command was used to generate the reference
    # promdens -np 10 -w 0.14294844 -lch 2e-6 -f 100 -ps --random_seed 123456789 -n 1000 test_data_nai.dat
    # the reference was compared to exact QD
    reference = np.genfromtxt(path/'NaI_reference/test_pda_reference.dat').T
    pda = ics.ics

    assert len(pda) == len(reference)
    # testing all generated ICs, same indexes and excitation times are required
    for i in range(len(pda)):
        # comparing indexes
        assert pda[0, i] == reference[0, i]
        # comparing excitation times, the threshold 1e-12 was based on numerical differences created by switching between numpy 1.26 and 2.1
        assert (pda[1, i] - reference[1, i]) < 1e-12, f"pda excitation time for index {i} does not match: delta = {pda[1, i] - reference[1, i]}"

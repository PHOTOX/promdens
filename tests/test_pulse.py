import numpy as np
import pytest
from inline_snapshot import snapshot

from promdens import ENVELOPE_TYPES, InitialConditions, LaserPulse

def test_invalid_envelope_type(make_pulse):
    with pytest.raises(ValueError):
        make_pulse(envelope_type='invalid')


def test_field_cos(make_pulse):
    pulse = make_pulse(omega=0.1, lchirp=0.0)
    t = np.array([-10., -1.0, 0.0, 1.0, 10.0])
    cos = pulse.field_cos(t)

    s = snapshot(
        {
            0: 0.5403023058681398,
            1: 0.9950041652780258,
            2: 1.0,
            3: 0.9950041652780258,
            4: 0.5403023058681398,
        }
    )

    assert len(cos) == len(t)
    for i, val in enumerate(cos):
        assert val == s[i]

    # For t=0, cos = 1.0
    assert cos[2] == 1.0
    # cos is an even function so should be symmetrical
    assert cos[0] == cos[-1]
    assert cos[1] == cos[-2]


def test_field_cos_with_chirp(make_pulse):
    pulse = make_pulse(omega=0.1, lchirp=0.01)
    t = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    cos = pulse.field_cos(t)

    s = snapshot(
        {
            0: 1.0,
            1: 0.9959527330119943,
            2: 1.0,
            3: 0.9939560979566968,
            4: -0.4161468365471424,
        }
    )

    assert len(cos) == len(t)
    for i, val in enumerate(cos):
        assert val == s[i]

    # For t=0, cos = 1.0
    assert cos[2] == 1.0
    # chirp breaks the even symmetry of the cos function
    assert cos[0] != cos[-1]
    assert cos[1] != cos[-2]


@pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
def test_field_envelope(make_pulse, envelope_type):
    fwhm = 15.0
    t0 = 0
    pulse = make_pulse(fwhm=fwhm, t0=t0, envelope_type=envelope_type)

    t = np.array([-64, -16, -4.0, 0.0, 4.0, 16, 64])

    envelope = pulse.calc_field_envelope(t)

    s = snapshot(
        {
            "gauss": {
                0: 1.0960549698939707e-11,
                1: 0.20653303211845794,
                2: 0.9061223550383886,
                3: 1.0,
                4: 0.9061223550383886,
                5: 0.20653303211845794,
                6: 1.0960549698939707e-11,
            },
            "lorentz": {
                0: 0.03209025202400356,
                1: 0.346604598571491,
                2: 0.8945978172388421,
                3: 1.0,
                4: 0.8945978172388421,
                5: 0.346604598571491,
                6: 0.03209025202400356,
            },
            "sech": {
                0: 0.001083121940752962,
                1: 0.2981611144195047,
                2: 0.8988518704633442,
                3: 1.0,
                4: 0.8988518704633442,
                5: 0.2981611144195047,
                6: 0.001083121940752962,
            },
            "sin": {
                0: 0.0,
                1: 0.0,
                2: 0.9135454576426009,
                3: 1.0,
                4: 0.913545457642601,
                5: 0.0,
                6: 0.0,
            },
            "sin2": {
                0: 0.0,
                1: 0.11811460943703252,
                2: 0.90982893698841,
                3: 1.0,
                4: 0.90982893698841,
                5: 0.11811460943703297,
                6: 0.0,
            },
        }
    )
    s_tmin = snapshot(
        {
            "gauss": -36.0,
            "lorentz": -120.0,
            "sech": -66.0,
            "sin": -15.0,
            "sin2": -20.60118862336883,
        }
    )
    s_tmax = snapshot(
        {
            "gauss": 36.0,
            "lorentz": 120.0,
            "sech": 66.0,
            "sin": 15.0,
            "sin2": 20.60118862336883,
        }
    )

    assert len(envelope) == len(t)

    assert pulse.tmin == s_tmin[envelope_type]
    assert pulse.tmax == s_tmax[envelope_type]
    assert pulse.tmin < t0
    assert pulse.tmax > t0
    assert pulse.tmin == -pulse.tmax

    # Maximum at t0 is always 1.0
    assert envelope[3] == 1.0
    for i, value in enumerate(envelope):
        # NOTE: If you need to regenerate the snapshot with --inline-snapshot=fix,
        # use the strict comparison.
        # assert s[envelope_type][i] == value
        assert s[envelope_type][i] == pytest.approx(value, abs=1e-15)

    # Envelope is an even function around t0
    assert envelope[0] == pytest.approx(envelope[-1], abs=1e-15)
    assert envelope[1] == pytest.approx(envelope[-2], abs=1e-15)
    assert envelope[2] == pytest.approx(envelope[-3], abs=1e-15)


@pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
def test_field_envelope_shifted(make_pulse, envelope_type):
    fwhm = 20.0
    t0 = 2.0
    pulse = make_pulse(fwhm=fwhm, t0=t0, envelope_type=envelope_type)

    t = np.array([-60, -12, 0.0, 2.0, 4.0, 16, 64])

    envelope = pulse.calc_field_envelope(t)

    s = snapshot(
        {
            "gauss": {
                0: 1.6375836113696163e-06,
                1: 0.5069797398950145,
                2: 0.9862327044933592,
                3: 1.0,
                4: 0.9862327044933592,
                5: 0.5069797398950145,
                6: 1.6375836113696163e-06,
            },
            "lorentz": {
                0: 0.05909337989549738,
                1: 0.5519194543083392,
                2: 0.9837014998966422,
                3: 1.0,
                4: 0.9837014998966422,
                5: 0.5519194543083392,
                6: 0.05909337989549738,
            },
            "sech": {
                0: 0.008468626120468338,
                1: 0.5367937766740093,
                2: 0.9846622513085862,
                3: 1.0,
                4: 0.9846622513085862,
                5: 0.5367937766740093,
                6: 0.008468626120468338,
            },
            "sin": {
                0: 0.0,
                1: 0.45399049973954675,
                2: 0.9876883405951378,
                3: 1.0,
                4: 0.9876883405951378,
                5: 0.45399049973954686,
                6: 0.0,
            },
            "sin2": {
                0: 0.0,
                1: 0.4847980881794031,
                2: 0.9869760345887517,
                3: 1.0,
                4: 0.9869760345887517,
                5: 0.4847980881794033,
                6: 0.0,
            },
        }
    )

    s_tmin = snapshot(
        {
            "gauss": -46.0,
            "lorentz": -158.0,
            "sech": -86.0,
            "sin": -18.0,
            "sin2": -25.468251497825108,
        }
    )
    s_tmax = snapshot(
        {
            "gauss": 50.0,
            "lorentz": 162.0,
            "sech": 90.0,
            "sin": 22.0,
            "sin2": 29.468251497825108,
        }
    )

    assert len(envelope) == len(t)

    assert pulse.tmin == s_tmin[envelope_type]
    assert pulse.tmax == s_tmax[envelope_type]
    assert pulse.tmin < t0
    assert pulse.tmax > t0
    assert t0 - pulse.tmin == pulse.tmax - t0

    # Maximum at t0 is always 1.0
    assert envelope[3] == 1.0
    for i, value in enumerate(envelope):
        # NOTE: If you need to regenerate the snapshot with --inline-snapshot=fix,
        # use the strict comparison.
        # assert s[envelope_type][i] == value
        # assert value == s[envelope_type][i]
        assert s[envelope_type][i] == pytest.approx(value, abs=1e-15)

    # Envelope is an even function around t0
    assert envelope[0] == pytest.approx(envelope[-1], abs=1e-15)
    assert envelope[1] == pytest.approx(envelope[-2], abs=1e-15)
    assert envelope[2] == pytest.approx(envelope[-3], abs=1e-15)


@pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
def test_pulse_wigner(make_pulse, envelope_type):
    pulse = make_pulse(envelope_type=envelope_type)
    tprimes = [0.0, 0.01, 1.0, 10]
    de = 0.2

    s = snapshot(
        {
            "gauss": {
                0: 2.1212693629580945,
                1: 2.1206813037329235,
                2: 0.1325793351848809,
                3: 8.214810584209081e-121,
            },
            "lorentz": {
                0: 2.4140684913342527,
                1: 2.4136641848000515,
                2: 0.8921213029327091,
                3: -0.004348713041567712,
            },
            "sech": {
                0: 2.257217042037135,
                1: 2.2567480186684,
                2: 0.4657983359088488,
                3: 3.608906131731995e-14,
            },
            "sin": {
                0: 1.9947778380166663,
                1: 1.993803973718274,
                2: 0.0,
                3: 0.0,
            },
            "sin2": {
                0: 2.0539072127434963,
                1: 2.0531922533816345,
                2: 0.023026289443473916,
                3: 0.0,
            },
        }
    )

    for i, tprime in enumerate(tprimes):
        assert s[envelope_type][i] == pytest.approx(pulse.wigner_transform(tprime, de), abs=1e-15, rel=1e-15)


@pytest.mark.parametrize("envelope_type", ENVELOPE_TYPES)
def test_pulse_wigner_de_equals_omega(make_pulse, envelope_type):
    chirp = 0.01
    omega = 0.3
    pulse = make_pulse(omega=omega, lchirp=chirp, envelope_type=envelope_type)
    tprimes = [0.0, 0.01, 1.0, 10]
    de = 0.2

    s = snapshot(
        {
            "gauss": {
                0: 2.1289340388624525,
                1: 2.1283438548324582,
                2: 0.13305837742890328,
                3: 8.244492746146238e-121,
            },
            "lorentz": {
                0: 2.4406313983925805,
                1: 2.440227083180404,
                2: 0.9185975447328396,
                3: 0.0146041084386692,
            },
            "sech": {
                0: 2.26918531421302,
                1: 2.268715316453524,
                2: 0.47140452079102957,
                3: 7.663844407334862e-14,
            },
            "sin": {
                0: 2.0,
                1: 1.9990196998183594,
                2: 0.0,
                3: 0.0
            },
            "sin2": {
                0: 2.0601188623368833,
                1: 2.059400368248726,
                2: 0.0230352473532616,
                3: 0.0,
            },
        }
    )

    for i, tprime in enumerate(tprimes):
        effective_omega = omega + 2 * chirp * tprime
        de = effective_omega
        assert s[envelope_type][i] == pytest.approx(pulse.wigner_transform(tprime, de), abs=1e-15, rel=1e-15)

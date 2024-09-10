import numpy as np
import pytest
from inline_snapshot import snapshot

from promdens.promdens import ENVELOPE_TYPES, LaserPulse

def test_invalid_envelope_type(make_pulse):
    with pytest.raises(ValueError):
        make_pulse(envelope_type='invalid')


@pytest.mark.parametrize("envelope", ENVELOPE_TYPES)
def test_envelope_types(make_pulse, envelope):
    # For now just test that we can create a pulse
    # for all available envelope types, more to come.
    pulse = make_pulse(envelope_type=envelope)

    assert envelope in str(pulse)


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
    fwhm = 10.
    t0 = -1
    pulse = make_pulse(fwhm=fwhm, t0=t0, envelope_type=envelope_type)

    # TODO: We need more points here (e.g. beyond extent)
    # Sanity check: for t=t0 the value should be always 1.0
    t = np.array([-5.0, -1.0, 4.0])

    envelope = pulse.calc_field_envelope(t)

    s = snapshot(
        {
            "gauss": {0: 0.8010698775896221, 1: 1.0, 2: 0.7071067811865476},
            "lorentz": {0: 0.7904534221560047, 1: 1.0, 2: 0.7071067811865475},
            "sech": {0: 0.7942475773796802, 1: 1.0, 2: 0.7071067811865476},
            "sin": {0: 0.8090169943749475, 1: 1.0, 2: 0.7071067811865476},
            "sin2": {0: 0.8049055147147319, 1: 1.0, 2: 0.7071067811865475},
        }
    )

    assert len(envelope) == len(t)
    assert envelope[0] == s[envelope_type][0]
    assert envelope[1] == s[envelope_type][1]
    assert envelope[2] == s[envelope_type][2]

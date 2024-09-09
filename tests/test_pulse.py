import numpy as np
import pytest
from inline_snapshot import snapshot

from promdens.promdens import ENVELOPE_TYPES, LaserPulse

def test_invalid_envelope_type(make_pulse):
    with pytest.raises(ValueError):
        make_pulse(envelope_type='invalid')

#@pytest.mark.parametrize("envelope", ENVELOPE_TYPES)
def test_field_cos(make_pulse):
    pulse = make_pulse(lchirp=0.01, omega=0.1)
    t = np.array([0.0, 1.0, 10.0])
    cos = pulse.field_cos(t)

    s = snapshot({0: 1.0, 1: 0.9939560979566968, 2: -0.4161468365471424})

    assert len(cos) == len(t)
    assert cos[0] == s[0]
    assert cos[1] == s[1]
    assert cos[2] == s[2]

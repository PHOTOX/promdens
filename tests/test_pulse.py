import numpy as np
import pytest
from inline_snapshot import snapshot

from promdens.promdens import LaserPulse

def test_invalid_envelope_type(make_pulse):
    with pytest.raises(ValueError):
        make_pulse(envelope_type='invalid')

#ENVELOPE_TYPES = ['gauss', 'lorentz', 'sech', 'sin', 'sin2']
def test_field_cos(make_pulse):
    pulse = make_pulse(lchirp=0.01)
    t = np.array([0.0, 1.0, 10.0])
    cos = pulse.field_cos(t)

    assert len(cos) == len(t)
    assert cos[0] == snapshot(1.0)
    assert cos[1] == snapshot(0.9939560979566968)
    assert cos[2] == snapshot(-0.4161468365471424)

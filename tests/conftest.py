import pytest

from promdens import LaserPulse


@pytest.fixture
def make_pulse():
    def _make_pulse(omega=0.1, fwhm=1.0, lchirp=0, t0=0, envelope_type="gauss"):
        return LaserPulse(omega=omega, fwhm=fwhm, lchirp=lchirp, t0=t0, envelope_type=envelope_type)

    return _make_pulse

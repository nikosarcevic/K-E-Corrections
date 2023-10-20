import numpy as np

from k_e_corrections import KECorrections


def test_corrections():
    corrections = KECorrections(
        redshift_range=np.linspace(0.1, 1, 10),
        h_0=70,
        omega_m=0.3,
    )

    kcorr, ecorr = corrections.get_color_corrections()

    assert np.all(np.isfinite(kcorr))
    assert np.all(np.isfinite(ecorr))
    assert np.all(kcorr > 0)
    assert np.all(ecorr < 0)

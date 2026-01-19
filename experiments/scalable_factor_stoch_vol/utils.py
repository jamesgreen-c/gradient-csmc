def _check_param_shapes(m0, P0, Q, phi):
    assert isinstance(m0, tuple) and len(m0) == 2, "m0 must be a tuple of (h0, d0)"
    assert isinstance(Q, tuple) and len(Q) == 2, "sigma0 must be a tuple of (Sigma_h, Sigma_d)"
    assert isinstance(phi, tuple) and len(phi) == 2, "phi must be a tuple of (phi_h, phi_d)"
    assert isinstance(P0, tuple) and len(P0) == 2, "P0 must be a tuple of (P_h, P_d)"
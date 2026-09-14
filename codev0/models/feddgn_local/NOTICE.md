# Design provenance

- **A-DGN backbone:** follows Eq. (5) of *Anti-Symmetric DGN: a stable
  architecture for Deep Graph Networks* (ICLR 2023): shared recurrent weights,
  anti-symmetric self interaction `W-W^T-gamma I`, generic graph aggregation,
  residual Euler step. The default local path additionally matches the released
  `gcn_norm=True` model's self-loop handling, Kaiming initialization, linear
  input embedding, and linear readout.
- **Force interface:** intentionally kept outside the intrinsic field.  The
  tensor/callable API follows the existing local PH-DGN adaptation, while the
  state-dependent callable semantics are compatible with the forcing roles in
  SONAR and port-Hamiltonian graph simulators.
- **Not copied from SONAR / PH-DGN:** no p/q split, no wave velocity, no learned
  resistance, no Hamiltonian energy, no separate damping network, no block MLP.
  Those mechanisms mainly serve deep/long-range propagation and are not needed
  in this local-backbone stage.

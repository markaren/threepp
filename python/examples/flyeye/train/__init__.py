"""flyeye.train: refit the flyvis free parameters on rendered flight.

The connectome is fixed (wiring, synapse counts, signs). The free parameters are the same
~700 floats flyvis fits by gradient descent: `time_const` and `bias` per cell type (65 each)
and `syn_strength` per edge type (604). `model.TrainableLobe` is `optic_lobe.OpticLobe` with
those three groups as `nn.Parameter`s; `decoder.py` is the readout the loss goes through;
`data.py` reads the Phase 2 recordings; `train.py` fits; `evaluate.py` scores a checkpoint
through `scoring.py` on held-out flights.

Checkpoints are npz files in the same format as `data/flyeye_model.npz`, so `OpticLobe`,
`members.py` and `score_members.py` load them unchanged.
"""

from pathlib import Path

RETRAIN_ROOT = Path(r"C:\dev\_flyeye\retrain")
TRAIN_SCEN = ("straight_3", "straight_10", "yaw", "roll")
TEST_SCEN = ("straight_30", "pitch", "approach")

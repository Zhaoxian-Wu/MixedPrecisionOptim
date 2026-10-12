"""Moonlight: native Muon with RMS scaling and native AdamW auxiliaries."""

from MixedPrecisionOptim.MpOptimizerMixin import MpOptimizerMixin
from MixedPrecisionOptim.Muon import NativeMatrixWithAdamW


class DigitalMoonlight(NativeMatrixWithAdamW):
    """Use the native 0.2 * sqrt(max(rows, columns)) matrix scaling."""

    matrix_lr_adjustment = "match_rms_adamw"


class MpMoonlight(MpOptimizerMixin, DigitalMoonlight):
    digital_optimizer_cls = DigitalMoonlight


__all__ = ["DigitalMoonlight", "MpMoonlight"]

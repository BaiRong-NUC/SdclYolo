"""Opt-in bounded scaling for future screens, never for historical replays."""

import numpy as np


class FitEnvelope:
    def __init__(self, lower, upper):
        self.lower = np.array(lower, dtype=np.float64, copy=True)
        self.upper = np.array(upper, dtype=np.float64, copy=True)
        if (self.lower.ndim != 1 or not self.lower.size or
                self.upper.shape != self.lower.shape or
                not np.isfinite(self.lower).all() or
                not np.isfinite(self.upper).all() or
                np.any(self.lower > self.upper)):
            raise ValueError("Invalid finite fit envelope.")
        self.center = .5*self.lower+.5*self.upper
        self.radius = .5*self.upper-.5*self.lower
        magnitude = np.maximum(1., np.maximum(abs(self.lower), abs(self.upper)))
        self.active = self.radius > 64*np.finfo(np.float64).eps*magnitude
        for value in (self.lower, self.upper, self.center, self.radius, self.active):
            value.flags.writeable = False

    @staticmethod
    def _matrix(values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError("Expected a finite nonzero-width feature matrix.")
        return values

    @classmethod
    def fit(cls, values):
        values = cls._matrix(values)
        if not len(values):
            raise ValueError("Cannot fit an empty population.")
        return cls(values.min(0), values.max(0))

    def transform(self, values):
        values = self._matrix(values)
        if values.shape[1] != len(self.lower):
            raise ValueError("Feature width differs from fitted envelope.")
        # Clip in raw units first, preventing tiny fit spreads from amplifying new scenes.
        bounded = np.clip(values[:, self.active], self.lower[self.active], self.upper[self.active])
        output = np.zeros(values.shape, dtype=np.float64)
        scaled = (bounded-self.center[self.active])/self.radius[self.active]
        scaled = np.where(bounded == self.lower[self.active], -1., scaled)
        output[:, self.active] = np.where(bounded == self.upper[self.active], 1., scaled)
        return np.clip(output, -1., 1.)

    def to_dict(self):
        return {"kind": "fit_envelope_v1", "lower": self.lower.tolist(), "upper": self.upper.tolist()}

    @classmethod
    def from_dict(cls, state):
        if set(state) != {"kind", "lower", "upper"} or state["kind"] != "fit_envelope_v1":
            raise ValueError("Unknown fit envelope state.")
        return cls(state["lower"], state["upper"])

"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tar_scenario import Instance  # noqa: E402


@pytest.fixture
def tiny():
    """Zwei Haltepunkte, ein OD-Paar mit 100 Personen, 5 km, ÖV 10 min, Auto 12 min, eine Kante, alle Personen fahren 2-mal im Monat.
    Handrechnung siehe tests/test_model.py."""
    return Instance(
        seed=0, n=2, hub_pct=0, grid=np.array([[0, 0], [25, 0]]), demand=np.array([[0, 10], [10, 0]]), hubs=[0],
        I=np.array([0]), J=np.array([1]), persons=np.array([100.0]), l_km=np.array([5.0]), t_pt=np.array([10.0]), t_car=np.array([12.0]),
        eu=np.array([0]), ev=np.array([1]), elen=np.array([5.0]), inc=np.array([[1.0]]), inner=np.array([True, True]), z=np.array([1.0]),
        pk=np.array([1.0, 0, 0, 0, 0, 0, 0]), share_unit=np.array([0.0]),
    )

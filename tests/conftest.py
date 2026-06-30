"""Shared pytest fixtures for test suite."""

import pytest
from rdkit import Chem


@pytest.fixture
def ethanol_mol():
    """RDKit molecule fixture for ethanol (CCO)."""
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def benzene_mol():
    """RDKit molecule fixture for benzene (c1ccccc1)."""
    return Chem.MolFromSmiles("c1ccccc1")


@pytest.fixture
def caffeine_mol():
    """RDKit molecule fixture for caffeine."""
    return Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")


@pytest.fixture
def mol_params_default():
    """Default mol_params configuration for fingerprints."""
    return {
        "fingerprint": True,
        "fingerprint_morgan": True,
        "fingerprint_rdkit": True,
        "fingerprint_maccs": True,
        "morgan_radius": 3,
        "morgan_nbits": 2048,
        "rdkit_nbits": 2048,
    }


@pytest.fixture
def mol_params_custom_radius():
    """mol_params with custom Morgan radius."""
    return {
        "fingerprint": True,
        "fingerprint_morgan": True,
        "fingerprint_rdkit": False,
        "fingerprint_maccs": False,
        "morgan_radius": 2,
        "morgan_nbits": 2048,
        "rdkit_nbits": 2048,
    }


@pytest.fixture
def mol_params_custom_nbits():
    """mol_params with custom nbits values."""
    return {
        "fingerprint": True,
        "fingerprint_morgan": True,
        "fingerprint_rdkit": True,
        "fingerprint_maccs": True,
        "morgan_radius": 3,
        "morgan_nbits": 1024,
        "rdkit_nbits": 1024,
    }

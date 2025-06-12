"""
@author Nels Frazier

@date June 12, 2025

Tests for functionality of the adjacency module.
"""

import pytest
import numpy as np
import geopandas as gpd
from pathlib import Path
import tempfile
from ..adjacency import create_matrix, matrix_to_zarr, index_matrix
import zarr
import os

# TODO consider generating random flowpaths and networks for more robust testing
# then figure out a way to reporduce the same random flowpaths and networks
# when a failure occurs so it can be debugged easily

@pytest.fixture
def simple_flowpaths():
    """Create a simple flowpaths GeoDataFrame for testing."""
    data = {
        'toid': ['nex-1', 'nex-1'],  # Terminal flowpath has NaN
        'geometry': [None]  # Simplified for testing
    }
    index = ['wb-1', 'wb-2']
    fp = gpd.GeoDataFrame(data, index=index)
    fp.index.name = 'id'
    return fp

@pytest.fixture
def simple_network():
    """Create a simple network GeoDataFrame for testing."""
    data = {
        'toid': [np.nan, 'nex-1', 'nex-1'],  # Terminal nexus has NaN
        'geometry': [None]  # Simplified for testing
    }
    index = ['nex-1', 'wb-1', 'wb-2']
    network = gpd.GeoDataFrame(data, index=index)
    network.index.name = 'id'
    return network

@pytest.fixture
def complex_flowpaths():
    """Create a more complex flowpaths GeoDataFrame for testing."""
    data = {
        'toid': ['nex-10', 'nex-10', 'nex-10', 'nex-11', 'nex-12', 'nex-12'],
        'geometry': [None]
    }
    index = ['wb-10', 'wb-11', 'wb-12', 'wb-13', 'wb-14', 'wb-15']
    fp = gpd.GeoDataFrame(data, index=index)
    fp.index.name = 'id'
    return fp


@pytest.fixture
def complex_network(complex_flowpaths):
    """Create a more complex network GeoDataFrame for testing."""
    data = {
        'toid': ['wb-13', 'wb-14', np.nan] + complex_flowpaths['toid'].tolist(),
        'geometry': [None]
    }
    index = ['nex-10', 'nex-11', 'nex-12'] + complex_flowpaths.index.tolist()
    network = gpd.GeoDataFrame(data, index=index)
    network.index.name = 'id'
    
    return network

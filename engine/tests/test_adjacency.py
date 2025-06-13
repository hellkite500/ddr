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
class TestAdjanceyMatrix:
    """Test cases for the create_matrix function."""

    @pytest.mark.parametrize("fp, network, ghost", [
        ('simple_flowpaths', 'simple_network', True),
        ('complex_flowpaths', 'complex_network', True),
        ('simple_flowpaths', 'simple_network', False),
        ('complex_flowpaths', 'complex_network', False)
    ])
    def test_create_matrix(self, fp, network, ghost, request):
        """Test basic functionality of create_matrix."""
        fp = request.getfixturevalue(fp)
        network = request.getfixturevalue(network)
        matrix, ts_order = create_matrix(fp, network, ghost)
        
        # Validate matrix properties
        # Check that matrix is square and has correct dimensions
        assert matrix.shape[0] == matrix.shape[1]
        # Check that matrix contains only 0s and 1s
        assert np.all((matrix == 0) | (matrix == 1))
        # Check that diagonal is all zeros (no self-loops)
        assert np.all(np.diag(matrix) == 0)
        # Check that matrix is lower triangular
        assert np.allclose(matrix, np.tril(matrix))

        paths = set(fp.index)
        # NOTE this works becuase crete_matrix will modify the
        # flowpath and network index when ghost is True
        num_paths = len(paths)
        assert matrix.shape[0] == num_paths
        # Check that ts_order contains all flowpath IDs
        assert len(ts_order) == num_paths
        if not ghost:
            ts_order = filter(lambda s: not s.startswith("ghost-"), ts_order)
        assert set(ts_order) == paths

    def test_empty_dataframes(self):
        """Test behavior with empty dataframes."""
        empty_fp = gpd.GeoDataFrame({'toid': []}, index=[])
        empty_network = gpd.GeoDataFrame({'toid': []}, index=[])
        empty_fp.index.name = 'id'
        empty_network.index.name = 'id'
        
        matrix, ts_order = create_matrix(empty_fp, empty_network)
        
        assert matrix.shape == (0, 0)
        assert len(ts_order) == 0

    @pytest.mark.parametrize("ghost", [True, False])
    def test_single_flowpath(self, ghost):
        """Test with a single flowpath (terminal)."""
        single_fp = gpd.GeoDataFrame({'toid': ['nex-1']}, index=['wb-1'])
        single_network = gpd.GeoDataFrame({'toid': [np.nan]}, index=['nex-1'], dtype='object')
        single_fp.index.name = 'id'
        single_network.index.name = 'id'
        
        matrix, ts_order = create_matrix(single_fp, single_network, ghost)
        
        if ghost:
            assert matrix.shape == (2, 2)
            assert ts_order == ['wb-1', 'ghost-0']
        else:
            assert matrix.shape == (1, 1)
            assert ts_order == ['wb-1']

    @pytest.mark.parametrize("fp, network, ghost", [
        ('simple_flowpaths', 'simple_network', True),
        ('complex_flowpaths', 'complex_network', True),
        ('simple_flowpaths', 'simple_network', False),
        ('complex_flowpaths', 'complex_network', False)
    ])
    def test_topology(self, fp, network, ghost, request):
        """Test matrix topology."""

        fp = request.getfixturevalue(fp)
        network = request.getfixturevalue(network)
        # NJF an interesting result of the adjancency matrix is that
        # columns that are all zeros are tailwater/terminal segments
        # and the rows that are all zeros are headwater segments
        matrix, ts_order = create_matrix(fp, network, ghost)
        
        fp = fp.reindex(ts_order)
        # Check matrix properties
        # NOTE this works becuase crete_matrix will modify the
        # flowpath and network index when ghost is True
        paths = set(fp.index)
        num_paths = len( paths )
        # These are checked in test_create_matrix
        # I don't think we need to check them again
        # unless a new network/fixture is used for this function
        # which isn't in create_matrix, so for now I'll leave these
        # commented out here...
        # assert matrix.shape[0] == num_paths
        # assert np.allclose(matrix, np.tril(matrix))
        # assert len(ts_order) == num_paths
        # assert set(ts_order) == paths

        # Test the values in the matrix
        idx = fp.index
        for i in range(len(fp)):
            nex = fp.loc[idx[i]]['toid']
            if isinstance(nex, float) and np.isnan(nex):
                continue
            ds = network.loc[nex]['toid']
            if isinstance(ds, float) and np.isnan(ds):
                continue
            j = idx.get_loc(ds)
            assert matrix[j, i] == 1, f"Expected 1 at ({i}, {j}), got {matrix[i, j]}"

    def test_missing_nexus_handled(self, capsys):
        """Test that missing nexus IDs are handled gracefully."""
        fp = gpd.GeoDataFrame({'toid': ['nex-missing']}, index=['wb-1'])
        network = gpd.GeoDataFrame({'toid': []}, index=[])
        fp.index.name = 'id'
        network.index.name = 'id'
        
        matrix, ts_order = create_matrix(fp, network)
        
        # Should capture the print statement about terminal nex
        captured = capsys.readouterr()
        assert "Terminal nex???" in captured.out



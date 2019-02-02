import copy

from fastparquet import ParquetFile, parquet_thrift
from fastparquet.writer import write_common_metadata
from six import string_types

import hilbert_frame.hilbert_curve as hc
import numpy as np
import pandas as pd
import dask.dataframe as dd
import os
import shutil
import json


def data2coord(vals, val_range, side_length):
    if isinstance(vals, (list, tuple)):
        vals = np.array(vals)

    x_width = val_range[1] - val_range[0]
    return ((vals - val_range[0]) * (side_length / x_width)
            ).astype(np.int64).clip(0, side_length - 1)


def compute_distance(df, x, y, p, x_range, y_range):
    side_length = 2 ** p
    x_coords = data2coord(df[x], x_range, side_length)
    y_coords = data2coord(df[y], y_range, side_length)
    return hc.distance_from_coordinates(p, x_coords, y_coords)


def compute_extents(df, x, y):
    x_min = df[x].min()
    x_max = df[x].max()
    y_min = df[y].min()
    y_max = df[y].max()
    return pd.DataFrame({'x_min': x_min,
                         'x_max': x_max,
                         'y_min': y_min,
                         'y_max': y_max},
                        index=[0])


class HilbertFrame2D(object):

    @staticmethod
    def from_dataframe(df,
                       filename,
                       x='x',
                       y='y',
                       p=10,
                       npartitions=None,
                       shuffle=None,
                       persist=False,
                       engine='auto',
                       compression='default'):

        # Validate dirname
        if (not isinstance(filename, string_types) or
                not filename.endswith('.parquet')):
            raise ValueError(
                'filename must be a string ending with a .parquet extension')

        # Remove any existing directory
        if os.path.exists(filename):
            shutil.rmtree(filename)

        # Normalize to dask dataframe
        if isinstance(df, pd.DataFrame):
            ddf = dd.from_pandas(df, npartitions=4)
        elif isinstance(df, dd.DataFrame):
            ddf = df
        else:
            raise ValueError("""
df must be a pandas or dask DataFrame instance.
Received value of type {typ}""".format(typ=type(df)))

        # Compute npartitions if needed
        if npartitions is None:
            # Make partitions of ~8 million rows with a minimum of 8
            # partitions
            max(int(np.ceil(len(df) / 2**23)), 8)

        # Compute data extents
        extents = ddf.map_partitions(
            compute_extents, x, y).compute()

        x_range = (float(extents['x_min'].min()),
                   float(extents['x_max'].max()))

        y_range = (float(extents['y_min'].min()),
                   float(extents['y_max'].max()))

        # Compute distance of points in integer hilbert space
        ddf = ddf.assign(distance=ddf.map_partitions(
            compute_distance, x=x, y=y, p=p,
            x_range=x_range, y_range=y_range))

        # Set index to distance
        ddf = ddf.set_index('distance',
                            npartitions=npartitions,
                            shuffle=shuffle)

        # Build partitions grid
        # Uses distance divisions computed above, but does not revisit data
        distance_divisions = [int(d) for d in ddf.divisions]

        # Save other properties as custom metadata in the parquet file
        props = dict(
            version='1.0',
            x=x,
            y=y,
            p=p,
            distance_divisions=distance_divisions,
            x_range=x_range,
            y_range=y_range,
        )

        # Drop distance index
        ddf = ddf.reset_index(drop=True)

        # Save ddf to parquet
        dd.to_parquet(ddf, filename, engine=engine, compression=compression)

        # Open file
        pf = ParquetFile(filename)

        # Add a new property to the file metadata
        new_fmd = copy.copy(pf.fmd)
        new_kv = parquet_thrift.KeyValue()
        new_kv.key = 'hilbert_frame'
        new_kv.value = json.dumps(props)
        new_fmd.key_value_metadata.append(new_kv)

        # Overwrite file metadata
        fn = os.path.join(filename, '_metadata')
        write_common_metadata(fn, new_fmd, no_row_groups=False)

        fn = os.path.join(filename, '_common_metadata')
        write_common_metadata(fn, new_fmd)

        # Construct HilbertFrame2D from file
        return HilbertFrame2D(filename, persist=persist)

    @staticmethod
    def build_partition_grid(distance_grid, dask_divisions, p):

        search_divisions = np.array(
            list(dask_divisions[1:-1]))

        side_length = 2 ** p
        partition_grid = np.zeros([side_length] * 2, dtype='int')
        for i in range(side_length):
            for j in range(side_length):
                partition_grid[i, j] = np.searchsorted(
                    search_divisions,
                    distance_grid[i, j],
                    sorter=None,
                    side='right')
        return partition_grid

    @staticmethod
    def build_distance_grid(p):
        side_length = 2 ** p
        distance_grid = np.zeros([side_length] * 2, dtype='int')
        for i in range(side_length):
            for j in range(side_length):
                distance_grid[i, j] = (
                    hc.distance_from_coordinates(p, i, j))
        return distance_grid

    def __init__(self, filename, persist=False):

        # Open hilbert properties
        # Reopen file
        pf = ParquetFile(filename)

        # Access custom metadata
        props = json.loads(pf.key_value_metadata['hilbert_frame'])

        # Set all props as attributes
        self.x = props['x']
        self.y = props['y']
        self.p = props['p']
        self.x_range = props['x_range']
        self.y_range = props['y_range']
        self.distance_divisions = props['distance_divisions']

        # Compute grids
        self.distance_grid = HilbertFrame2D.build_distance_grid(self.p)
        self.partition_grid = HilbertFrame2D.build_partition_grid(
            self.distance_grid, self.distance_divisions, self.p)

        # Compute simple derived properties
        n = 2
        self.side_length = 2 ** self.p
        self.max_distance = 2 ** (n * self.p) - 1
        self.x_width = self.x_range[1] - self.x_range[0]
        self.y_width = self.y_range[1] - self.y_range[0]
        self.x_bin_width = self.x_width / self.side_length
        self.y_bin_width = self.y_width / self.side_length

        # Read parquet file
        self.ddf = dd.read_parquet(filename)

        # Persist if requested
        if persist:
            self.ddf = self.ddf.persist()

    def range_query(self, query_x_range, query_y_range):
        # Compute bounds in hilbert coords
        expanded_x_range = [query_x_range[0],
                            query_x_range[1] + self.x_bin_width]

        expanded_y_range = [query_y_range[0],
                            query_y_range[1] + self.y_bin_width]

        query_x_range_coord = data2coord(expanded_x_range,
                                         self.x_range,
                                         self.side_length)

        query_y_range_coord = data2coord(expanded_y_range,
                                         self.y_range,
                                         self.side_length)

        partition_query = self.partition_grid[
            slice(*query_x_range_coord), slice(*query_y_range_coord)]

        query_partitions = sorted(np.unique(partition_query))

        if query_partitions:
            partition_dfs = [self.ddf.get_partition(p) for p in
                             query_partitions]
            query_frame = dd.concat(partition_dfs)
            return query_frame
        else:
            return self.ddf.loc[1:0]

    @property
    def hilbert_distance(self):
        x = self.x
        y = self.y
        p = self.p
        x_range = self.x_range
        y_range = self.y_range
        return self.ddf.map_partitions(
            compute_distance, x=x, y=y, p=p, x_range=x_range, y_range=y_range)


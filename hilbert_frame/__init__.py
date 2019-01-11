from six import string_types

import hilbert_frame.hilbert_curve as hc
import numpy as np
import pandas as pd
import dask.dataframe as dd
import os
import shutil
import pickle


def data2coord(vals, val_range, side_length):
    if isinstance(vals, (list, tuple)):
        vals = np.array(vals)

    x_width = val_range[1] - val_range[0]
    return ((vals - val_range[0]) * (side_length / x_width)
            ).astype(np.int64).clip(0, side_length - 1)


def compute_distance(df, x, y, p, x_range, y_range, side_length):
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
                       dirname,
                       x='x',
                       y='y',
                       p=6,
                       npartitions=8,
                       shuffle=None,
                       persist=False,
                       engine='auto',
                       compression='default'):

        # Validate dirname
        if (not isinstance(dirname, string_types) or
                not dirname.endswith('.hframe')):
            raise ValueError(
                'dirname must be a string ending with a .hframe extension')

        # Remove any existing directory
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

        # Create output directory
        os.mkdir(dirname)

        # Normalize to dask dataframe
        if isinstance(df, pd.DataFrame):
            ddf = dd.from_pandas(df, npartitions=4)
        elif isinstance(df, dd.DataFrame):
            ddf = df
        else:
            raise ValueError("""
df must be a pandas or dask DataFrame instance.
Received value of type {typ}""".format(typ=type(df)))

        # Only support 2D spaces for now
        n = 2
        # TODO: validate x/y/p/npartitions/persist

        side_length = 2 ** p

        # Compute data extents
        extents = ddf.map_partitions(
            compute_extents, x, y).compute()
        x_range = (extents['x_min'].min(), extents['x_max'].max())
        y_range = (extents['y_min'].min(), extents['y_max'].max())

        # Compute distance of points in integer hilbert space
        ddf = ddf.assign(distance=ddf.map_partitions(
            compute_distance, x=x, y=y, p=p,
            x_range=x_range, y_range=y_range, side_length=side_length))

        # Set index to distance
        ddf = ddf.set_index('distance',
                            npartitions=npartitions,
                            shuffle=shuffle)

        # Build distance grid
        distance_grid = np.zeros([side_length] * 2, dtype='int')
        for i in range(side_length):
            for j in range(side_length):
                distance_grid[i, j] = (
                    hc.distance_from_coordinates(p, i, j))

        # Build partitions grid
        # Uses distance divisions computed above, but does not revisit data
        search_divisions = np.array(
            list(ddf.divisions[1:-1]))

        partition_grid = np.zeros([side_length] * 2, dtype='int')
        for i in range(side_length):
            for j in range(side_length):
                partition_grid[i, j] = np.searchsorted(
                    search_divisions,
                    distance_grid[i, j],
                    sorter=None,
                    side='right')

        # Save ddf to parquet
        dd.to_parquet(ddf, os.path.join(dirname, 'frame.parquet'),
                      engine=engine, compression=compression)

        # Save other properties as pickle file
        props = dict(
            x=x,
            y=y,
            p=p,
            npartitions=npartitions,
            x_range=x_range,
            y_range=y_range,
            distance_grid=distance_grid,
            partition_grid=partition_grid
        )

        with open(os.path.join(dirname, 'props.pkl'), 'wb') as f:
            pickle.dump(props, f)

        # Construct HilbertFrame2D from directory
        return HilbertFrame2D(dirname, persist=persist)

    def __init__(self, dirname, persist=False):

        with open(os.path.join(dirname, 'props.pkl'), 'rb') as f:
            props = pickle.load(f)

        # Set all props as attributes
        self.x = props['x']
        self.y = props['y']
        self.p = props['p']
        self.x_range = props['x_range']
        self.y_range = props['y_range']
        self.distance_grid = props['distance_grid']
        self.partition_grid = props['partition_grid']

        # Compute simple derived properties
        n = 2
        self.side_length = 2 ** self.p
        self.max_distance = 2 ** (n * self.p) - 1
        self.x_width = self.x_range[1] - self.x_range[0]
        self.y_width = self.y_range[1] - self.y_range[0]
        self.x_bin_width = self.x_width / self.side_length
        self.y_bin_width = self.y_width / self.side_length

        # Read parquet file
        self.ddf = dd.read_parquet(os.path.join(dirname, 'frame.parquet'))

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

        distance_query = self.distance_grid[
            slice(*query_x_range_coord), slice(*query_y_range_coord)]

        distance_ranges = []
        prev_partition = None
        for p in query_partitions:
            ds = distance_query[partition_query == p]
            ds_min, ds_max = ds.min(), ds.max()

            if (p - 1 == prev_partition or
                    distance_ranges and distance_ranges[-1][1] == ds_min-1):
                # Merge consecutive partitions to reduce the number of loc
                # operations needed
                distance_ranges[-1] = (distance_ranges[-1][0], ds_max)
            else:
                distance_ranges.append((ds_min, ds_max))

            prev_partition = p

        partition_subframes = []
        for d_range in distance_ranges:
            dmin, dmax = d_range
            partition_subframes.append(self.ddf.loc[dmin:dmax])

        if partition_subframes:
            query_frame = dd.concat(partition_subframes)
            return query_frame
        else:
            return self.ddf.loc[1:0]

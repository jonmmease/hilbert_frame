from six import string_types

from hilbert_curve import HilbertCurve
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
    return ((vals - val_range[0]) *
            (side_length / x_width)
            ).astype('int').clip(0, side_length - 1)


class HilbertFrame2D(object):

    @staticmethod
    def from_dataframe(df,
                       dirname,
                       x='x',
                       y='y',
                       p=6,
                       npartitions=8,
                       persist=False):

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
        N = 2
        # TODO: validate x/y/p/npartitions/persist

        # Build Hilber Curve calculator
        # TODO: replace with numba implementation
        hilbert_curve = HilbertCurve(p, N)

        side_length = 2 ** p
        max_distance = 2 ** (N * p) - 1

        # Compute data extents
        # TODO: replace with single map_partitions call
        x_range = (df[x].min().compute(), df[x].max().compute())
        y_range = (df[y].min().compute(), df[y].max().compute())
        x_width = x_range[1] - x_range[0]
        y_width = y_range[1] - y_range[0]

        # Compute bin widths
        x_bin_width = x_width / side_length
        y_bin_width = y_width / side_length

        # Compute x/y coords in integer hilbert space
        ddf = ddf.assign(
            x_coord=data2coord(ddf[x], x_range, side_length),
            y_coord=data2coord(ddf[y], y_range, side_length))

        # Compute distances
        ddf['distance'] = ddf.map_partitions(lambda pd_df: pd_df.apply(
            lambda s: hilbert_curve.distance_from_coordinates(
                [s['x_coord'], s['y_coord']]), axis=1))

        # Compute divisions
        ddf = ddf.set_index('distance',
                            npartitions=npartitions,
                            compute=True)

        # Build distance grid
        distance_grid = np.zeros([side_length] * 2, dtype='int')
        for i in range(side_length):
            for j in range(side_length):
                distance_grid[i, j] = (
                    hilbert_curve.distance_from_coordinates([i, j]))

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
        dd.to_parquet(ddf, os.path.join(dirname, 'frame.parquet'))

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
        N = 2
        self.side_length = 2 ** self.p
        self.max_distance = 2 ** (N * self.p) - 1
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
        for p in query_partitions:
            ds = distance_query[partition_query == p]
            distance_ranges.append((ds.min(), ds.max()))

        partition_subframes = []
        for p, d_range in zip(query_partitions, distance_ranges):
            dmin, dmax = d_range
            partition_subframes.append(self.ddf.loc[dmin:dmax])

        if partition_subframes:
            query_frame = dd.concat(partition_subframes)
            return query_frame
        else:
            return self.ddf.loc[1:0]

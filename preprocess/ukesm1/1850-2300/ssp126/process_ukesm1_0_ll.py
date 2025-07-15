#!/usr/bin/env python

import argparse
import xarray as xr
import numpy as np
import os
import logging
import warnings
from pathlib import Path
from xarray.coders import CFDatetimeCoder


def compute_yearly_mean(in_file: Path, out_file: Path, lat_threshold: float = -48.0):
    """Compute annual mean for data south of a latitude threshold and save to NetCDF."""
    if out_file.exists():
        return
    logging.info(f"Processing {in_file} to {out_file}")

    coder = CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(in_file, decode_times=coder)
    ds = ds.rename({'longitude': 'lon',
                    'latitude': 'lat',
                    'vertices_longitude': 'lon_vertices',
                    'vertices_latitude': 'lat_vertices'})
    ds = ds.drop_vars('time_bnds')

    # crop to Southern Ocean
    minLat = ds.lat.min(dim='i')
    mask = minLat <= lat_threshold
    yIndices = np.nonzero(mask.values)[0]
    ds = ds.isel(j=yIndices)

    for coord in ['lev_bnds', 'lon_vertices', 'lat_vertices']:
        ds.coords[coord] = ds[coord]

    # annual mean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ds = ds.groupby('time.year').mean('time', keep_attrs=True)

    # convert back to CF-compliant time
    ds = ds.rename({'year': 'time'})
    ds['time'] = 365.0 * ds.time
    ds.time.attrs['bounds'] = "time_bnds"
    ds.time.attrs['units'] = "days since 0000-01-01 00:00:00"
    ds.time.attrs['calendar'] = "noleap"
    ds.time.attrs['axis'] = "T"
    ds.time.attrs['long_name'] = "time"
    ds.time.attrs['standard_name'] = "time"

    timeBounds = np.zeros((ds.sizes['time'], ds.sizes['bnds']))
    timeBounds[:, 0] = ds.time.values
    timeBounds[:, 1] = ds.time.values + 365
    ds['time_bnds'] = (('time', 'bnds'), timeBounds)
    ds['time_bnds'].attrs['units'] = "days since 0000-01-01 00:00:00"
    ds['time_bnds'].attrs['calendar'] = "noleap"

    time_var = ds.variables['time']
    bnds_var = ds.variables['time_bnds']
    decoded_time = coder.decode(time_var, time_var.attrs)
    decoded_bnds = coder.decode(bnds_var, bnds_var.attrs)
    ds = ds.assign_coords(time=decoded_time, time_bnds=decoded_bnds)

    encoding = {'time': {'units': 'days since 0000-01-01'}}
    ds.to_netcdf(out_file, encoding=encoding)


def process_periods(dates, model, run, scenario, out_dir, file_type):
    """Process a set of periods for a given variable and scenario."""
    files = []
    for date in dates:
        in_file = out_dir / f"{file_type}_Omon_{model}_{scenario}_{run}_gn_{date}.nc"
        out_file = out_dir / f"{file_type}_annual_{model}_{scenario}_{run}_{date}.nc"
        compute_yearly_mean(in_file, out_file)
        files.append(out_file)
    return files


def combine_files(hist_files: list[Path], scenario_files: list[Path], out_file: Path, start_year: int):
    """Combine historical and scenario files into a single NetCDF file for a given period."""
    if out_file.exists():
        return
    logging.info(f"Combining files into {out_file}")
    coder = CFDatetimeCoder(use_cftime=True)
    ds = xr.open_mfdataset(hist_files + scenario_files, combine='nested', concat_dim='time', decode_times=coder)
    mask = ds['time.year'] >= start_year
    tIndices = np.nonzero(mask.values)[0]
    ds = ds.isel(time=tIndices)
    encoding = {'time': {'units': 'days since 0000-01-01'}}
    ds.to_netcdf(out_file, encoding=encoding)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', dest='out_dir', metavar='DIR', required=True, type=str, help='output directory')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    out_dir = Path(args.out_dir)

    model = 'UKESM1-0-LL'
    run = 'r4i1p1f2'
    scenario = 'ssp126'

    hist_dates = {
        'thetao': [
            '195001-199912',
            '200001-201412'
            ], 
        'so': [
            '195001-199912',
            '200001-201412'
            ]
        }
    scenario_dates = {
        'thetao': [
            '201501-204912', 
            '205001-209912', 
            '210001-210012', 
            '210101-214912', 
            '215001-219912', 
            '220001-224912', 
            '225001-229912', 
            '230001-230012'
            ],
        'so': [
            '201501-204912', 
            '205001-209912', 
            '210001-210012', 
            '210101-214912', 
            '215001-219912', 
            '220001-224912', 
            '225001-229912', 
            '230001-230012'
            ]
    }

    hist_files = {}
    scenario_files = {}
    for var in hist_dates:
        hist_files[var] = process_periods(hist_dates[var], model, run, 'historical', out_dir, var)
    for var in scenario_dates:
        scenario_files[var] = process_periods(scenario_dates[var], model, run, scenario, out_dir, var)
    
    start_year = 1995
    for var in ['so', 'thetao']:
        out_file = out_dir / f"{var}_annual_{model}_{scenario}_{run}_{start_year}01-230012.nc"
        combine_files(hist_files[var], scenario_files[var], out_file, start_year)


if __name__ == "__main__":
    main()

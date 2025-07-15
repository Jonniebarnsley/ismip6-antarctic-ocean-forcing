#!/usr/bin/env python

import argparse
import xarray
import numpy
import os
import warnings
from xarray.coding.times import CFDatetimeCoder

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-o', dest='out_dir', metavar='DIR', required=True,
                    type=str, help='output directory')
args = parser.parse_args()


def compute_yearly_mean(inFileName, outFileName):
    # crop to below 48 S and take annual mean over the data
    if os.path.exists(outFileName):
        return
    print('{} to {}'.format(inFileName, outFileName))

    ds = xarray.open_dataset(inFileName)
    ds = ds.rename({'longitude': 'lon',
                    'latitude': 'lat',
                    'vertices_longitude': 'lon_vertices',
                    'vertices_latitude': 'lat_vertices'})

    ds = ds.drop_vars('time_bnds')

    # crop to Southern Ocean
    minLat = ds.lat.min(dim='i')
    mask = minLat <= -48.
    yIndices = numpy.nonzero(mask.values)[0]
    ds = ds.isel(j=yIndices)

    for coord in ['lev_bnds', 'lon_vertices', 'lat_vertices']:
        ds.coords[coord] = ds[coord]

    # annual mean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ds = ds.groupby('time.year').mean('time', keep_attrs=True)

    # convert back to CF-compliant time
    ds = ds.rename({'year': 'time'})
    ds['time'] = 365.0*ds.time
    ds.time.attrs['bounds'] = "time_bnds"
    ds.time.attrs['units'] = "days since 0000-01-01 00:00:00"
    ds.time.attrs['calendar'] = "noleap"
    ds.time.attrs['axis'] = "T"
    ds.time.attrs['long_name'] = "time"
    ds.time.attrs['standard_name'] = "time"

    timeBounds = numpy.zeros((ds.sizes['time'], ds.sizes['bnds']))
    timeBounds[:, 0] = ds.time.values
    timeBounds[:, 1] = ds.time.values + 365
    ds['time_bnds'] = (('time', 'bnds'), timeBounds)
    ds['time_bnds'].attrs['units'] = "days since 0000-01-01 00:00:00"
    ds['time_bnds'].attrs['calendar'] = "noleap"
    
    coder = CFDatetimeCoder(use_cftime=True)
    time_var = ds.variables['time']
    bnds_var = ds.variables['time_bnds']
    decoded_time = coder.decode(time_var, time_var.attrs)
    decoded_bnds = coder.decode(bnds_var, bnds_var.attrs)
    ds = ds.assign_coords(time=decoded_time, time_bnds=decoded_bnds)

    encoding = {'time': {'units': 'days since 0000-01-01'}}
    ds.to_netcdf(outFileName, encoding=encoding)


model = 'UKESM1-0-LL'
run = 'r4i1p1f2'

dates = {'thetao': [#'185001-189912',
                    #'190001-194912',
                    #'195001-199912',
                    '200001-201412'],
         'so': [#'185001-189912',
                #'190001-194912',
                #'195001-199912',
                '200001-201412']}

histFiles = {}
for field in dates:
    histFiles[field] = []
    for date in dates[field]:
        inFileName = '{}/{}_Omon_{}_historical_{}_gn_{}.nc'.format(
            args.out_dir, field, model, run, date)

        outFileName = '{}/{}_annual_{}_historical_{}_{}.nc'.format(
            args.out_dir, field, model, run, date)

        compute_yearly_mean(inFileName, outFileName)
        histFiles[field].append(outFileName)

# dates = {'thetao': ['201501-204912'],
#          'so': ['201501-204912']}
# for scenario in ['ssp585']:
#     scenarioFiles = {}
#     for field in dates:
#         scenarioFiles[field] = []
#         for date in dates[field]:
#             inFileName = '{}/{}_Omon_{}_{}_{}_gn_{}.nc'.format(
#                 args.out_dir, field, model, scenario, run, date)

#             outFileName = '{}/{}_annual_{}_{}_{}_{}.nc'.format(
#                 args.out_dir, field, model, scenario, run, date)

#             compute_yearly_mean(inFileName, outFileName)
#             scenarioFiles[field].append(outFileName)

#     for field in ['so', 'thetao']:
#         outFileName = \
#             '{}/{}_annual_{}_{}_{}_185001-201412.nc'.format(
#                 args.out_dir, field, model, scenario, run)
#         if not os.path.exists(outFileName):
#             print(outFileName)

#             # combine it all into a single data set
#             ds = xarray.open_mfdataset(histFiles[field] + scenarioFiles[field],
#                                        combine='nested', concat_dim='time',
#                                        use_cftime=True)

#             mask = ds['time.year'] <= 2014
#             tIndices = numpy.nonzero(mask.values)[0]
#             ds = ds.isel(time=tIndices)
#             encoding = {'time': {'units': 'days since 0000-01-01'}}
#             ds.to_netcdf(outFileName, encoding=encoding)

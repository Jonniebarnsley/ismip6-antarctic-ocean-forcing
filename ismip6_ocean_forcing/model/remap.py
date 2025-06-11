import xarray
import os
import numpy
import progressbar
from pyremap import get_polar_descriptor_from_file, LatLonGridDescriptor, \
    LatLon2DGridDescriptor, Remapper

from ismip6_ocean_forcing.remap.interp1d import remap_vertical
from ismip6_ocean_forcing.remap.res import get_res, get_horiz_res
from ismip6_ocean_forcing.thermal_forcing.main import \
    potential_to_in_situ_temperature


def remap_model(config, modelFolder):
    _fix_units_and_periodicity(config, modelFolder)
    _interp_z(config, modelFolder)
    _remap(config, modelFolder)


def _fix_units_and_periodicity(config, modelFolder):

    modelName = config.get('model', 'name')
    tIndexMin = config.getint('output', 'tIndexMin')
    tIndexMax = config.getint('output', 'tIndexMax')
    if tIndexMax == -1:
        tIndexMax = None
    else:
        tIndexMax += 1

    renameDict = {}
    for varName in ['lat', 'lon', 'z', 'time', 'temperature', 'salinity',
                    'z_bnds']:
        oldVarName = config.get('model', varName)
        if varName == oldVarName:
            continue

        renameDict[oldVarName] = varName

    inFileNames = {}
    outFileNames = {}
    bothExist = True
    for fieldName in ['temperature', 'salinity']:
        inFileNames[fieldName] = \
            config.get('model', '{}FileName'.format(fieldName))

        outFileNames[fieldName] = \
            '{}/{}_{}_periodic.nc'.format(modelFolder, modelName, fieldName)
        if not os.path.exists(outFileNames[fieldName]):
            bothExist = False

    if bothExist:
        return

    print('  Add a periodic image in longitude and fix units...')

    datasets = {}
    for fieldName in inFileNames:
        inFileName = inFileNames[fieldName]
        outFileName = outFileNames[fieldName]
        print('    {}'.format(outFileName))
        keepList = ['lat', 'lon', 'z', 'time', 'z_bnds', fieldName]

        ds = xarray.open_dataset(inFileName)
        for name in renameDict:
            if name in ds:
                newName = renameDict[name]
                if newName in ds:
                    ds = ds.drop_vars(newName)
                ds = ds.rename({name: renameDict[name]})

        ds = ds.isel(time=slice(tIndexMin, tIndexMax))

        for coord in ds.coords:
            if coord not in keepList:
                ds = ds.drop_vars(coord)

        dropList = []
        for var in ds.data_vars:
            if var not in keepList:
                dropList.append(var)
        ds = ds.drop_vars(dropList)

        ds.z.attrs['bounds'] = 'z_bnds'

        if numpy.amax(ds.z.values) > 0.:
            attrs = ds.z.attrs
            attrs['positive'] = 'up'
            ds['z'] = -ds.z
            ds.z.attrs = attrs
            attrs = ds.z_bnds.attrs
            ds['z_bnds'] = -ds.z_bnds
            ds.z_bnds.attrs = attrs

        if fieldName == 'temperature':
            if ds.temperature.attrs['units'] == 'K':
                attrs = ds.temperature.attrs
                attrs['units'] = 'degrees C'
                ds['temperature'] = ds.temperature - 273.15
                ds.temperature.attrs = attrs

        if fieldName == 'salinity':
            if 'units' not in ds.salinity.attrs:
                # Let's hope it's PSU...
                ds.salinity.attrs['units'] = 'PSU'

        if len(ds.lon.dims) == 1:
            lonDim = ds.lon.dims[0]
            lonRange = ds.lon[-1].values - ds.lon[0].values
            if numpy.abs(lonRange - 360.) > 1e-10:
                # Needs a periodic image
                ds = _add_periodic_lon(ds, lonDim)
        else:
            assert(len(ds.lon.dims) == 2)
            lonDim = ds.lon.dims[1]
            lonRange = ds.lon[0, -1].values - ds.lon[0, 0].values
            if numpy.abs(lonRange - 360.) > 1e-10:
                # Needs a periodic image
                ds = _add_periodic_lon(ds, lonDim)

        datasets[fieldName] = ds

    dsTemp = potential_to_in_situ_temperature(datasets['temperature'],
                                              datasets['salinity'])
    dsTemp.to_netcdf(outFileNames['temperature'])
    datasets['salinity'].to_netcdf(outFileNames['salinity'])


def _add_periodic_lon(ds, lonDim):

    nLon = ds.sizes[lonDim]
    lonIndices = xarray.DataArray(numpy.append(numpy.arange(nLon), [0]),
                                  dims=('newLon',))
    ds.load()
    ds = ds.isel({lonDim: lonIndices})
    ds = ds.rename({'newLon': lonDim})
    return ds


def _interp_z(config, modelFolder):

    modelName = config.get('model', 'name')

    inFileNames = {}
    outFileNames = {}
    for fieldName in ['temperature', 'salinity']:
        inFileNames[fieldName] = \
            '{}/{}_{}_periodic.nc'.format(modelFolder, modelName, fieldName)

        outFileNames[fieldName] = \
            '{}/{}_{}_interp_z.nc'.format(modelFolder, modelName, fieldName)

    remap_vertical(config, inFileNames, outFileNames, extrap=True)


def _remap(config, modelFolder):

    res = get_res(config)
    hres = get_horiz_res(config)
    modelName = config.get('model', 'name')

    inFileNames = {}
    outFileNames = {}
    bothExist = True
    for fieldName in ['temperature', 'salinity']:
        inFileNames[fieldName] = \
            '{}/{}_{}_interp_z.nc'.format(modelFolder, modelName, fieldName)

        outFileNames[fieldName] = \
            '{}/{}_{}_{}.nc'.format(modelFolder, modelName, fieldName, res)
        if not os.path.exists(outFileNames[fieldName]):
            bothExist = False

    if bothExist:
        return

    print('  Remapping to {} grid...'.format(res))
    for fieldName in inFileNames:
        inFileName = inFileNames[fieldName]
        outFileName = outFileNames[fieldName]
        if os.path.exists(outFileName):
            continue
        outGridFileName = 'ismip6/{}_grid.nc'.format(hres)
        print('    {}'.format(outFileName))
        progressDir = '{}/progress_remap_{}'.format(modelFolder, fieldName)

        try:
            os.makedirs(progressDir)
        except OSError:
            pass

        ds = xarray.open_dataset(inFileName)

        if len(ds.lon.dims) == 1:
            inDescriptor = LatLonGridDescriptor.read(
                    inFileName, latVarName='lat', lonVarName='lon')
        else:
            assert(len(ds.lon.dims) == 2)
            inDescriptor = LatLon2DGridDescriptor.read(
                    inFileName, latVarName='lat', lonVarName='lon')
        inDescriptor.regional = True
        outDescriptor = get_polar_descriptor_from_file(outGridFileName,
                                                       projection='antarctic')

        mappingFileName = '{}/map_{}_to_{}.nc'.format(
                modelName.lower(), inDescriptor.meshName,
                outDescriptor.meshName)

        remapper = Remapper(inDescriptor, outDescriptor, mappingFileName)

        remapper.build_mapping_file(method='bilinear')

        ds = ds.drop_vars(['lat', 'lon'])

        nt = ds.sizes['time']

        widgets = ['  ', progressbar.Percentage(), ' ',
                   progressbar.Bar(), ' ', progressbar.ETA()]
        print(f' remapping: {fieldName}')
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=nt).start()

        for tIndex in range(nt):
            progressFileName = '{}/{}_t_{}.nc'.format(
                    progressDir, modelName, tIndex)
            if os.path.exists(progressFileName):
                bar.update(tIndex+1)
                continue

            dsIn = ds.isel(time=tIndex)
            dsOut = remapper.remap(dsIn, renormalizationThreshold=0.1)

            dsOut = dsOut.transpose('z', 'y', 'x')

            for attrName in ['units', 'standard_name', 'long_name']:
                if attrName in ds[fieldName].attrs:
                    dsOut[fieldName].attrs[attrName] = \
                        ds[fieldName].attrs[attrName]
            dsOut.z.attrs = ds.z.attrs

            dsOut.to_netcdf(progressFileName)

            bar.update(tIndex+1)
        bar.finish()

        dsOut = xarray.open_mfdataset(
            '{}/{}_t_*.nc'.format(progressDir, modelName), combine='nested',
            concat_dim='time')
        #Fully load netcdf files.
        dsOut = dsOut.load()

        dsOut['z_bnds'] = ds.z_bnds

        dsOut.to_netcdf(outFileName)
        #Close xarray object.
        dsOut.close()

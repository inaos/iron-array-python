import netCDF4
import gdal, gdalconst
import numpy as np
import re, os, math, datetime


# src_dir = 'C:/Users/ecanoscheurich/Documents/MasterData/datahub/source/uni-bonn/COSMO-REA6/TOT_PREC/'
src_dir = "./rea6/tot_prec/"
tgt_dir = "./ia-data/rea6/tot_prec/"
parameter = "precipitation"
unit = "mm"

file_post = ".grb"


def producer(time):
    # read and process grb radar files
    itBand = 0
    dataOld = 0
    itFile = 0

    for file in files:
        print(" processing file (" + str(itFile) + ") = " + file)
        # time and data processing
        try:
            dataset = gdal.Open(src_dir + year + "/" + file, gdalconst.GA_ReadOnly)
            nt = dataset.RasterCount

            for step in range(nt):
                print(
                    "  dataset "
                    + file
                    + " processing step "
                    + str(step)
                    + " from source, adding step "
                    + str(itBand)
                    + " of target"
                )

                # time
                timeStr = dataset.GetRasterBand(step + 1).GetMetadata_Dict("")["GRIB_REF_TIME"]
                timeInt = int(re.findall(r"\d+", timeStr)[0])
                fctInt = 0
                try:
                    # fix for forecast time issue
                    fctStr = dataset.GetRasterBand(step + 1).GetMetadata_Dict("")[
                        "GRIB_FORECAST_SECONDS"
                    ]
                    print("   fix: forecast time = " + fctStr)
                    fctInt = int(re.findall(r"\d+", fctStr)[0])
                    timeInt += fctInt
                    print("   fix: adding forecast time to reference time")
                except:
                    print("   fix: no forecast time found")
                timeInstance = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=timeInt)
                time[itBand] = netCDF4.date2num(timeInstance, time.units, calendar="standard")

                # data
                data = dataset.GetRasterBand(step + 1).ReadAsArray(0, 0, nx, ny).astype(np.float32)
                # fix for forecast time issue and deaccumulate every six hours
                if (timeInstance.hour - 1) % 6 != 0:
                    print("   fix: deaccumulating data at forecast time " + str(fctInt) + " s")
                    yield data - dataOld
                else:
                    yield data
                dataOld = data

        except:
            print("Error while processing data of file " + file)
        itFile += 1


def transformPoint(lat_, lon_):
    fac = math.pi / 180.0
    lat_ *= fac
    lon_ *= fac

    lat0 = -39.25 * fac
    lon0 = +18.0 * fac

    nu = -(90.0 * fac + lat0)
    phi = -lon0

    x_ = math.cos(lon_) * math.cos(lat_)
    y_ = math.sin(lon_) * math.cos(lat_)
    z_ = math.sin(lat_)

    x = math.cos(nu) * math.cos(phi) * x_ + math.sin(phi) * y_ + math.sin(nu) * math.cos(phi) * z_
    y = -math.cos(nu) * math.sin(phi) * x_ + math.cos(phi) * y_ - math.sin(nu) * math.sin(phi) * z_
    z = -math.sin(nu) * x_ + math.cos(nu) * z_

    lat = math.asin(z) / fac
    lon = math.atan2(y, x) / fac

    return lat, lon


years = ("2018",)
for year in years:

    files = sorted(os.listdir(src_dir + year + "/"))
    nt_ = 0
    nx = 0
    ny = 0
    for file in files:
        dataset = gdal.Open(src_dir + year + "/" + file, gdalconst.GA_ReadOnly)
        nx = dataset.RasterXSize
        ny = dataset.RasterYSize
        nt_ += dataset.RasterCount

    # create evaporation variable
    rootgrp = netCDF4.Dataset(tgt_dir + year + ".nc", "w", format="NETCDF4")

    dataset = gdal.Open(src_dir + year + "/" + file, gdalconst.GA_ReadOnly)
    nx = dataset.RasterXSize
    ny = dataset.RasterYSize

    # create dimensions
    time = rootgrp.createDimension("time", None)
    xc = rootgrp.createDimension("xc", nx)
    yc = rootgrp.createDimension("yc", ny)

    # assume affine transformation of the projected coordinates and get the related data
    gt = dataset.GetGeoTransform()

    # projected coordinates (only works for orthogonal dimensions)
    xc = rootgrp.createVariable("xc", np.float32, ("xc"))
    xc.axis = "X"
    xc.long_name = "longitude in rotated system"
    xc.units = "degrees_east"
    for ix in range(nx):
        xc[ix] = gt[0] + ix * gt[1]

    yc = rootgrp.createVariable("yc", np.float32, ("yc"))
    yc.axis = "Y"
    yc.long_name = "latidute in rotated system"
    yc.units = "degrees_north"
    for iy in range(ny):
        yc[iy] = gt[3] + iy * gt[5]

    # # geographical coordinates
    # lat = rootgrp.createVariable('latitude', np.float32, ('yc', 'xc'))
    # lat.standard_name = 'latitude'
    # lat.long_name = 'latitude'
    # lat.units = 'degrees_north'
    #
    # lon = rootgrp.createVariable('longitude', np.float32, ('yc', 'xc'))
    # lon.standard_name = 'longitude'
    # lon.long_name = 'longitude'
    # lon.units = 'degrees_east'
    #
    # for ix in range(nx):
    #     for iy in range(ny):
    #         lon_ = gt[0] + ix * gt[1] + iy * gt[2]
    #         lat_ = gt[3] + ix * gt[4] + iy * gt[5]
    #         lat[iy, ix], lon[iy, ix] = transformPoint(lat_, lon_)

    # create time variable
    time = rootgrp.createVariable("time", np.float64, ("time"))
    time.units = "seconds since 1970-01-01 00:00:00"
    time.long_name = "time"
    time.calendar = "gregorian"

    precipitation = rootgrp.createVariable(
        "precipitation",
        np.float32,
        ("time", "yc", "xc"),
        zlib=True,
        complevel=4,
        chunksizes=[1, ny, nx],
    )  #  least_significant_digit=1,

    for i, block in enumerate(producer(time)):
        precipitation[i, :, :] = block

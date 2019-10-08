import iarray as ia
import gdal, gdalconst
import numpy as np
import re, os, datetime


# src_dir = 'C:/Users/ecanoscheurich/Documents/MasterData/datahub/source/uni-bonn/COSMO-REA6/TOT_PREC/'
src_dir = './rea6/tot_prec/'
tgt_dir = './ia-data/rea6/tot_prec/'
parameter = 'precipitation'
unit = 'mm'

file_post = '.grb'


def producer():
    # read and process grb radar files
    itBand = 0
    dataOld = 0
    itFile = 0

    for file in files:
        print(' processing file (' + str(itFile) + ') = ' + file)
        # time and data processing
        try:
            dataset = gdal.Open(src_dir + year + '/' + file, gdalconst.GA_ReadOnly)
            nt = dataset.RasterCount

            for step in range(nt):
                print('  dataset ' + file + ' processing step ' + str(step) + ' from source, adding step ' + str(
                    itBand) + ' of target')

                # time
                timeStr = dataset.GetRasterBand(step + 1).GetMetadata_Dict('')['GRIB_REF_TIME']
                timeInt = int(re.findall(r'\d+', timeStr)[0])
                fctInt = 0
                try:
                    # fix for forecast time issue
                    fctStr = dataset.GetRasterBand(step + 1).GetMetadata_Dict('')['GRIB_FORECAST_SECONDS']
                    print('   fix: forecast time = ' + fctStr)
                    fctInt = int(re.findall(r'\d+', fctStr)[0])
                    timeInt += fctInt
                    print('   fix: adding forecast time to reference time')
                except:
                    print('   fix: no forecast time found')
                timeInstance = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=timeInt)

                # data
                data = dataset.GetRasterBand(step + 1).ReadAsArray(0, 0, nx, ny).astype(np.float32)
                # fix for forecast time issue and deaccumulate every six hours
                if (timeInstance.hour - 1) % 6 != 0:
                    print('   fix: deaccumulating data at forecast time ' + str(fctInt) + ' s')
                    yield data - dataOld
                else:
                    yield data
                dataOld = data

        except:
            print('Error while processing data of file ' + file)
        itFile += 1


years = ("2018",)
for year in years:

    files = sorted(os.listdir(src_dir + year + '/'))
    nt_ = 0
    nx = 0
    ny = 0
    for file in files:
        dataset = gdal.Open(src_dir + year + '/' + file, gdalconst.GA_ReadOnly)
        nx = dataset.RasterXSize
        ny = dataset.RasterYSize
        nt_ += dataset.RasterCount

    # create evaporation variable
    # dtshape = ia.dtshape(shape=[nt_, nx, ny], pshape=[10, 40, 40], dtype=np.float32)
    dtshape = ia.dtshape(shape=[nt_, ny, nx], pshape=[1, ny, nx], dtype=np.float32)
    precipitation = ia.empty(dtshape, filename=tgt_dir + year + ".iarray", clevel=9)

    producer_iter = producer()
    for (_, precip_block) in precipitation.iter_write_block():
        precip_block[:, :, :] = next(producer_iter)


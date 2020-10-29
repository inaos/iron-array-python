from multiprocessing import Pool
import numpy as np
import iarray as ia
from time import time

NPROCS = 2

# Dimensions and type properties for the arrays
# 'Small' arrays config follows...
# shape = (200, 2000, 250)
shape = (20, 2000, 250)
# pshape = (100, 200, 50)
pshape = (10, 500, 50)
# This config generates containers of more than 2 GB in size
# shape = (250, 4000, 400)
# pshape = (200, 400, 50)
dtype = np.float32

# Compression properties
clib = ia.LZ4
clevel = 9
filters = [ia.TRUNC_PREC, ia.SHUFFLE]
filters_meta = [10, 0]
nthreads = 2


def f(sl, x):
    new_pos = tuple(int(inner_sl.start / pshape[i]) for i, inner_sl in enumerate(sl))
    return (new_pos, np.average(x))


def f1(sl, x):
    new_pos = tuple(int(inner_sl.start / pshape[i]) for i, inner_sl in enumerate(sl))
    return (new_pos, np.average(x))


def f2(sl, x):
    new_pos = tuple(int(inner_sl / pshape_) for inner_sl, pshape_ in zip(sl, pshape))
    return (new_pos, np.average(x))


if __name__ == "__main__":
    # Create content for populating arrays
    dtshape = ia.DTShape(shape=shape, pshape=pshape, dtype=np.float32)
    carr = ia.linspace(dtshape, 0, 10, clevel=clevel, clib=clib, nthreads=nthreads)
    content = ia.iarray2numpy(carr)

    t0 = time()
    with Pool(processes=NPROCS) as pool:
        results = []
        for i in range(0, shape[0], pshape[0]):
            for j in range(0, shape[1], pshape[1]):
                for k in range(0, shape[2], pshape[2]):
                    blockslice = (
                        slice(i, i + pshape[0]),
                        slice(j, j + pshape[0]),
                        slice(k, k + pshape[0]),
                    )
                    block = content[blockslice]
                    results.append(pool.apply_async(f, (blockslice, block)))
        rnp = [res.get(timeout=1) for res in results]
        print("rnp->", rnp)
    t1 = time()
    print(
        "Time for average (numpy): %.3f" % (t1 - t0),
    )

    t0 = time()
    with Pool(processes=NPROCS) as pool:
        results = []
        for i in range(0, shape[0], pshape[0]):
            for j in range(0, shape[1], pshape[1]):
                for k in range(0, shape[2], pshape[2]):
                    blockslice = (
                        slice(i, i + pshape[0]),
                        slice(j, j + pshape[0]),
                        slice(k, k + pshape[0]),
                    )
                    block = carr[blockslice]
                    results.append(pool.apply_async(f1, (blockslice, block)))
        rnp = [res.get(timeout=1) for res in results]
        print("rnp->", rnp)
    t1 = time()
    print(
        "Time for average (iarray): %.3f" % (t1 - t0),
    )

    # t0 = time()
    # with Pool(processes=NPROCS) as pool:
    #     results = []
    #     for info, block in carr.iter_read_block():
    #         # print("block->", type(block))
    #         results.append(pool.apply_async(f2, (info.elemindex, block)))
    #     riter = [res.get(timeout=1) for res in results]
    #     print("riter->", riter)
    # t1 = time()
    # print("Time for average (iter_read_block): %.3f" % (t1 - t0),)

    # t0 = time()
    # with Pool(processes=NPROCS) as pool:
    #     results = []
    #     for block, info in carr.iter_read(blockshape=pshape):
    #         sframe = cat.from_numpy(block, pshape=pshape, clevel=clevel, cname=cname,
    #                                 filters=filters, filters_meta=filters_meta,
    #                                 cnthreads=nthreads, dnthreads=nthreads).to_sframe()
    #         results.append(pool.apply_async(f2, (info.slice, sframe)))
    #     rsframe = [res.get(timeout=1) for res in results]
    # t1 = time()
    # print("Time for average (iter_read + sframe): %.3f" % (t1 - t0),)

from pydantic import BaseModel
import iarray as ia
from fastapi import FastAPI, Response, HTTPException
import numpy as np
from iarray import iarray_ext as ext


app = FastAPI()


class Get_block(BaseModel):
    array_id: str
    nchunk: int
    start: int
    nitems: int
    size: int


@app.get("/v1/catalog/")
async def catalog():
    return list(ia.global_catalog.keys())


@app.get("/v1/meta/")
async def meta(array_id: str):
    cat = ia.global_catalog
    if array_id not in cat.keys():
        raise HTTPException(status_code=404, detail=array_id + " not in catalog")
    arr = cat[array_id]
    return {"shape": arr.shape, "chunks": arr.chunks, "blocks": arr.blocks, "dtype": np.dtype(arr.dtype).str}


@app.post("/v1/blocks/")
async def blocks(params: Get_block):
    cat = ia.global_catalog
    if params.array_id not in cat.keys():
        raise HTTPException(status_code=404, detail=cat[params.array_id] + " not in catalog")
    iarr = cat[params.array_id]
    try:
        res = ext._server_job(iarr, params.nchunk, params.start, params.nitems, params.size)
    except:
        raise HTTPException(status_code=500, detail="could not get the block")

    return Response(content=res)

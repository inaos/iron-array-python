import sys
import os
import iarray as ia
import uvicorn


# python iarr_server.py host port "path to folder where all the arrays to view are"
host = str(sys.argv[1])
port = int(sys.argv[2])
path = str(sys.argv[3])


# Add arrays to catalog
for file in os.listdir(path):
    array_id = os.path.join(path, file)
    if file.endswith('.iarr'):
        ia.global_catalog[file] = ia.open(array_id)

uvicorn.run(ia.http_server.app, host=host, port=port)

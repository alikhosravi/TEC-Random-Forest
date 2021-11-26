# BBOX

Problem definition:

The spatial data users often are required to obtain the coordinates of the minimum bounding box of vector and raster data in a way that be readable for machines. 


Advantages:

The output support programmers with a spreadsheet that contains the bounding box coordinates so, there is no need for GIS software or spatial databases.

Usage:

First,
    I) Set the GIS data path on line 6 (eg ``` path = './census.shp' ```)
Or
    II) Place the BBox.py script in the same directory as the geospatial data and set path variable to null (``` path = './' ```)

Then, run the script within the virtual env:
```
python BBox.py
```

Dependencies:

GDAL (a translator library for raster and vector geospatial data formats)
More information about GDAL is avilable at (https://gdal.org/)

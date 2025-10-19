import geopandas
import pandas as pd
import pystac_client
from odc.stac import stac_load
from typing import List

def get_imagery_from_stac(
                          collection: List[str],
                          catalog:str,
                          kwargs_search: dict,
                          kwargs_load: dict,
                          zarr_store: str=None,
                          ) -> pd.DataFrame:
    catalog = pystac_client.Client.open(catalog)
    search = catalog.search(collections=collection, **kwargs_search)
    items = search.item_collection()
    if len(items)==0:
        print(f'number of detected scenes: {len(items)}')
        return None

    df = geopandas.GeoDataFrame.from_features(items.to_dict(), crs=kwargs_load['crs'])
    item_list = list(items)
    ds = stac_load(item_list, groupby='id', **kwargs_load)
    print(f'num of scenes loaded by stac_load() is {len(ds["time"])}')

    if zarr_store:
        ds.to_zarr(zarr_store,
                   mode="w",
                   consolidated=True,
                   zarr_format=2)
    return ds, df

if __name__ == "__main__":
    zarr_store = '../../.dataset/s2_saar.zarr'
    collection = ['sentinel-2-l2a',]
    catalog = "https://stac.dataspace.copernicus.eu/v1/"
    bbox = [6.711273,48.998240,7.216644,49.351072]

    kwargs_search = dict(
        bbox=bbox,
        datetime="2024-04-01/2024-09-30",
        query={"eo:cloud_cover": {"gte": 0,"lte": 5}},
    )
    kwargs_load = dict(
        bbox=bbox,
        bands=["B04_10m", "B03_10m", "B02_10m"],
        chunks={"x": 256, "y": 256, "time": 1},
        resolution=10,
        dtype="uint16",
        nodata=0)
    get_imagery_from_stac(collection, catalog,  kwargs_search, kwargs_load, zarr_store)
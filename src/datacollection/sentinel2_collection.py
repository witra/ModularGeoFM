import geopandas
import pystac_client
from odc.stac import stac_load

def get_S2(zarr_store,
           kwargs_search,
           kwargs_load):
        catalog = pystac_client.Client.open("https://stac.dataspace.copernicus.eu/v1/")
        search = catalog.search(collections=["sentinel-2-l2a"], **kwargs_search)
        items = search.item_collection()
        df = geopandas.GeoDataFrame.from_features(items.to_dict(), crs="epsg:4326")
        print(f'number of detected scenes: {df.shape[0]}')

        items = [item for item in items]
        xx = stac_load(items, groupby='id', **kwargs_load)
        print(f'num of time after join by stac_load() is {len(xx["time"])}')

        xx.to_zarr(zarr_store,
                   mode="w",
                   consolidated=True,
                   zarr_format=2)
        print('Done get_s2() !')
        return df

if __name__ == "__main__":
    zarr_store = '../../.dataset/s2_saar.zarr'
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
    get_S2(zarr_store, kwargs_search, kwargs_load)
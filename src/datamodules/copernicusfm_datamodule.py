import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from xbatcher import BatchGenerator
import xarray as xr


class CopernicusFMDataset(IterableDataset):
    def __init__(self,
                 xr_dataset,
                 input_dims,
                 input_overlap,
                 verify_fn = None,
                 batch_size_gen=1,
                 ) -> None:
        self.xr_dataset = xr_dataset
        self.input_dims = input_dims
        self.input_overlap = input_overlap or {}
        self.batch_size_gen = batch_size_gen
        self.verify_fn = verify_fn or self.basic_filter

    def create_batch_generator(self, subset):
        return BatchGenerator(
            subset,
            input_dims=self.input_dims,
            input_overlap=self.input_overlap
        )

    def basic_filter(self, patch, threshold=0.05):
        """
        Check if the data contain no_data not more than threshold
        :param patch:
        :type patch:
        :param threshold:
        :type threshold:
        :return:
        :rtype:
        """
        total_elements = patch.size
        num_zeros = np.sum(patch == 0)
        num_nans = np.isnan(patch).sum()
        invalid_fraction = (num_zeros + num_nans) / total_elements
        return invalid_fraction < threshold

    def __iter__(self):
        worker_info = get_worker_info()
        print(f'worker info: {worker_info}')
        if worker_info is None:
            subset = self.xr_dataset
        else:
            # 🔀 Divide dataset across workers (if possible)
            total_workers = worker_info.num_workers
            worker_id = worker_info.id

             # split across `time` dimension
            time = self.xr_dataset.coords['time']
            split_indices = torch.linspace(0, len(time), total_workers + 1, dtype=torch.int)
            start = split_indices[worker_id].item()
            end = split_indices[worker_id + 1].item()
            subset = self.xr_dataset.isel(time=slice(start, end))

        batch_gen = iter(self.create_batch_generator(subset))
        iteration = 1
        while True:
            batch_data = []
            while len(batch_data) < self.batch_size_gen:
                try:
                    patch = next(batch_gen).values.squeeze()
                    status = False
                    if self.verify_fn:
                        status = self.verify_fn(patch) # T/F
                    if status:
                        batch_data.append(patch)
                        iteration = iteration + 1
                except StopIteration:
                    print("Generator exhausted — stopping batch collection.")
                    if batch_data:  # yield remaining data if any
                        batch_data = np.stack(batch_data, axis=0)
                        yield torch.from_numpy(batch_data).squeeze()
                    return None
            batch_data = np.stack(batch_data, axis=0) # stack the selected patches into batch
            print('batch_Data shape', batch_data.shape)
            print('current worker is', worker_id)
            yield torch.from_numpy(batch_data).squeeze()

class CopernicusFMDataModule(L.LightningDataModule):
    def __init__(self,
                 train_xr_dataset,
                 input_dims,
                 input_overlap,
                 verify_fn=None,
                 batch_size_gen=1,
                 num_workers=4):
        super().__init__()
        self.train_xr_dataset = train_xr_dataset
        self.input_dims = input_dims
        self.input_overlap = input_overlap
        self.batch_size_gen = batch_size_gen
        self.verify_fn = verify_fn
        self.train_ds = CopernicusFMDataset(train_xr_dataset,
                                            input_dims=self.input_dims,
                                            input_overlap=self.input_overlap,
                                            verify_fn=self.verify_fn,
                                            batch_size_gen=self.batch_size_gen)
        if num_workers > len(self.train_xr_dataset.coords['time']):
            self.num_workers = len(self.train_xr_dataset.coords['time'])
        else:
            self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            dataset= self.train_ds,
            batch_size=None,
            num_workers=self.num_workers
        )
    def val_dataloader(self):
        pass
    def predict_dataloader(self):
        pass

if __name__ == "__main__":
    store = "../../.dataset/my_s2_zarr_v2.zarr"
    ds = xr.open_zarr(store)

    ds['image'] = ds[["B02", "B03", "B04"]].to_array(dim='band')
    ds['image'].coords['band'] = ["B02", "B03", "B04"]
    ds['image'] = ds['image'].transpose('band', 'x', 'y', 'time')
    datamodule = CopernicusFMDataModule(
        train_xr_dataset=ds['image'],
        input_dims={'band': 3, "x": 256, "y": 256, 'time': 1},
        input_overlap={"x": 0, "y": 0},
        batch_size_gen=10,
        num_workers=4
    )

    for i, batch in enumerate(datamodule.train_dataloader()):
        print('last print', i, batch.shape)
        if i==3:
            break


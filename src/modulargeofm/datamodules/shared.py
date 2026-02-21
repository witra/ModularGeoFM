import torch
from xbatcher import BatchGenerator
 
def create_batch_generator(subset, input_dims, input_overlap):
    """
    Create a batch generator from an xarray subset. 
    
    Parameters 
    ---------- 
    subset : xarray.Dataset 
        The dataset subset representing a specific temporal slice. 
    
    Returns 
    ------- 
    xbatcher.BatchGenerator 
        Iterator yielding patches of the dataset subset.
    """
    return BatchGenerator(
        subset,
        input_dims=input_dims,
        input_overlap=input_overlap,
        )

def filter_y(batch_patches:torch.tensor, threshold=None):
    """ 
    Filtering function for rejecting invalid label patches. At least there is 2 classes. 
    And filter if background is greater than threshold

    Returns 
    ------- 
    bool 
        ``True`` if patch is valid, ``False`` otherwise. 
    """
    batch_size = batch_patches.shape[0]
    patches_flat = batch_patches.view(batch_size, -1)
    has_multiple_labels = torch.any(patches_flat != patches_flat[:, :1], dim=1)

    if threshold is not None:
        # Calculate the fraction of the most common label
        invalid_mask = (patches_flat == 0) | torch.isnan(patches_flat)
        invalid_fraction = invalid_mask.sum(dim=1).float() / patches_flat.shape[1]
        valid_mask = (invalid_fraction < threshold).to(torch.int)
        return has_multiple_labels.bool() & valid_mask.bool()
    return has_multiple_labels.bool()

def filter_x(batch_patches:torch.tensor, threshold=0.05):
    """ 
    Basic filtering function for rejecting invalid patches. 
    
    Checks if the fraction of zeros or NaN values in the input tensor exceeds a defined threshold. 
    
    Parameters 
    ---------- 
        patch : torch.Tensor Input patch tensor. 
        threshold : float, default=0.05 Maximum allowed fraction of invalid elements. 
    
    Returns 
    ------- 
    bool 
        ``True`` if patch is valid, ``False`` otherwise. 
    """
    batch_size = batch_patches.shape[0]
    patches_flat = batch_patches.view(batch_size, -1)
    invalid_mask = (patches_flat == 0) | torch.isnan(patches_flat)
    invalid_fraction = invalid_mask.sum(dim=1).float() / patches_flat.shape[1]
    valid_mask = (invalid_fraction < threshold).to(torch.int)
    return valid_mask.bool()


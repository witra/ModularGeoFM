# Evaluation of Satellite Data Pipeline

This guide provides an instruction to reproduce the experiment explained on [my medium story](https://medium.com/@tahjudil.witra/6373f7351257)  

1. Install the modulargeofm package
2. Provide necessary dataset or Download the necessary dataset from the provided HuggingFace page
 and place it in the appropriate directory.
3. Update the `zarr_dir` variable in the following scripts to point to the location of your dataset. 
    - `running_test_on_OnTheFly.py`
    - `running_test_on_Precomputed.py`
4. Run the provided scripts:
    ```bash
    python running_test_on_OnTheFly.py
    python running_test_on_Precomputed.py
5. Generate the summary information and visualizations using the notebook `vis_gpu_util.ipynb`  
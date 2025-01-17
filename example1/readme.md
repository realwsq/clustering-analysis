### step 0:

- Trained RRR model is saved in a json file and can be loaded with `pd.read_json` function. 
- The model json file is saved in the folder `local_folder_lf` and names as `RRRglobal_full.json`.
- `local_folder_lf` is specified in `clus_perarea.py`. 

- The loaded model is a pd.DataFrame with
    - rows contain information of single neurons.
    - columns contain 
        * `eid`: session id
        * `RRRglobal_r2` and `meanact_r2`: r2 of RRR model and null model (for neuron inclusion)
        * `RRRglobal_beta`: time-varying coefficients of the shape (ncoef+1, T), +1 is the time-varying bias

### step 1:

- run `python -m example1.clus_perarea` to perform the clustering analysis for each individual brain area.
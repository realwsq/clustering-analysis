This subfolder includes a **complete** example of performing the clustering analysis in the **variable selectivity space**, using the single neuron coefficients estimated from an encoding model that predicts the neural responses using the relevant task variables.

This subfolder reproduces main results in the paper [1], including Fig. 3bcde, Suppl. Fig. 7abcd, and Suppl. Fig. 4 and 5.

The main pipeline of the clustering analysis is visualized in Fig. 3a of the paper [1] (see Methods for more details).

### before running the code:

#### preparing single neuron selectivity profiles (inputs for the clustering analysis):

In this example, we use the coefficients of the trained reduced-rank regression encoding model to characterize the selectivity profile of each neuron. 
- Trained RRR model is saved in `example1/neural_selectivity_data/RRRglobal_full.json`. 
- The loaded model is a pd.DataFrame with
    - rows contain information of single neurons.
    - columns contain 
        * `RRRglobal_beta`: time-varying coefficients of the shape (n_vars+1, n_times). The last dimension (+1) corresponds to the time-varying bias, which we will not use in the clustering analysis.
        * `eid`: session id, which will be used to detect the suspicious clusters whose neurons are (almost) exclusively from one single session
        * `RRRglobal_r2` and `meanact_r2`: r2 of RRR model and baeline model, which will be used for neuron inclusion. We only want to include neurons whose neural responses are well-captured, that is, whose `RRRglobal_r2` is significantly higher than `meanact_r2`.

#### updating the path to your own:
- make sure you update the path `local_folder_lf` in `example1/clus_perarea.py` and `example1/clus_permodule` to your own path, where the intermediate clustering results will be saved.

### performing the clustering analysis for each brain area:

- cd to the main folder and run `python -m example1.clus_perarea`. 
    - The results will be saved in the `clusfig_folder` folder (under the path f`{local_folder_lf}/clus`). The name of the folder is specified according to the setups used in the clustering analysis.
    - The final results (i.e., correlating functional clustering quality of each area with its anatomical hierarchy) will be saved in f`{clusfig_folder}/results.pdf`.
    - The clustering results of each area will also be saved in the `clusfig_folder` folder.
- try to vary setups in `example1/clus_perarea.py` to see how the clustering results change (e.g., try to include time profiles for clustering, change `min_Deltar2` for neuron inclusion, etc).

### performing the clustering analysis for each brain module:

- cd to the main folder and run `python -m example1.clus_permodule`. 
    - The results will be saved in the `clusfig_folder` folder (under the path f`{local_folder_lf}/clus_module`). The name of the folder is specified according to the setups used in the clustering analysis.
    - The final results (i.e., functional clustering quality of each brain module) will be saved in f`{clusfig_folder}/results.pdf`.
    - The clustering results of each module will also be saved in the `clusfig_folder` folder.

### References:
[1] Posani, L., Wang, S., Muscinelli, S. P., Paninski, L., & Fusi, S. (2025). Rarely categorical, always high-dimensional: how the neural code changes along the cortical hierarchy. bioRxiv, 2024-11.
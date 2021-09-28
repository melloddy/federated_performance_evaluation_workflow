# Performance Evaluation Script: release-2-(classification-and-regression)

Performance evaluation scripts from the Single- and Multi-pharma outputs.
These evaluation scripts allow us to verify whether predictions performed using a model generated (referred to as single-pharma model hereafter) during a federated run can be reproduced when executed locally (prediction step) within the pharma partner's IT infrastructure. 
Second the evaluation assesses whether the predictions from the federated model improve over predictions from the single-pharma model.

## Requirements
On your local IT infrastructure you'd need 

1. Python 3.6 or higher
2. Local Conda installation (e.g. miniconda)
3. Git installation
4. melloddy_tuner environment from WP1 code: https://github.com/melloddy/MELLODDY-TUNER
5. sparsechem https://github.com/melloddy/SparseChem (with sparse-predict functionality).

## Year 2 run analysis (performance_evaluation.py): single- vs. multi-partner evaluation


### Evaluate CLS/CLSAUX/REG models

```
$  python performance_evaluation.py -h
usage: performance_evaluation.py [-h] [--y_cls Y_CLS] [--y_clsaux Y_CLSAUX] [--y_regr Y_REGR] [--y_censored_regr Y_CENSORED_REGR]
                                    [--y_cls_single_partner Y_CLS_SINGLE_PARTNER] [--y_clsaux_single_partner Y_CLSAUX_SINGLE_PARTNER]
                                    [--y_regr_single_partner Y_REGR_SINGLE_PARTNER] [--y_cls_multi_partner Y_CLS_MULTI_PARTNER]
                                    [--y_clsaux_multi_partner Y_CLSAUX_MULTI_PARTNER] [--y_regr_multi_partner Y_REGR_MULTI_PARTNER] [--folding_cls FOLDING_CLS]
                                    [--folding_clsaux FOLDING_CLSAUX] [--folding_regr FOLDING_REGR] [--t8c_cls T8C_CLS] [--t8c_clsaux T8C_CLSAUX] [--t8r_regr T8R_REGR]
                                    [--weights_cls WEIGHTS_CLS] [--weights_clsaux WEIGHTS_CLSAUX] [--weights_regr WEIGHTS_REGR] [--run_name RUN_NAME]
                                    [--significance_analysis {0,1}] [--verbose {0,1}] [--validation_fold {0,1,2,3,4} [{0,1,2,3,4} ...]]
                                    [--aggr_binning_scheme_perf AGGR_BINNING_SCHEME_PERF [AGGR_BINNING_SCHEME_PERF ...]]
                                    [--aggr_binning_scheme_perf_delta AGGR_BINNING_SCHEME_PERF_DELTA [AGGR_BINNING_SCHEME_PERF_DELTA ...]]

MELLODDY Performance Evaluation

optional arguments:
  -h, --help            show this help message and exit
  --y_cls Y_CLS         Classification activity file (npz) (e.g. cls_T10_y.npz)
  --y_clsaux Y_CLSAUX   Aux classification activity file (npz) (e.g. cls_T10_y.npz)
  --y_regr Y_REGR       Activity file (npz) (e.g. reg_T10_y.npz)
  --y_censored_regr Y_CENSORED_REGR
                        Censored activity file (npz) (e.g. reg_T10_censor_y.npz)
  --y_cls_single_partner Y_CLS_SINGLE_PARTNER
                        Yhat cls prediction output from an single-partner run (e.g. <single pharma dir>/<cls_prefix>-class.npy)
  --y_clsaux_single_partner Y_CLSAUX_SINGLE_PARTNER
                        Yhat clsaux prediction from an single-partner run (e.g. <single pharma dir>/<clsaux_prefix>-class.npy)
  --y_regr_single_partner Y_REGR_SINGLE_PARTNER
                        Yhat regr prediction from an single-partner run (e.g. <single pharma dir>/<regr_prefix>-regr.npy)
  --y_cls_multi_partner Y_CLS_MULTI_PARTNER
                        Classification prediction output for comparison (e.g. pred from the multi-partner run)
  --y_clsaux_multi_partner Y_CLSAUX_MULTI_PARTNER
                        Classification w/ aux prediction output for comparison (e.g. pred from the multi-partner run)
  --y_regr_multi_partner Y_REGR_MULTI_PARTNER
                        Regression prediction output for comparison (e.g. pred from the multi-partner run)
  --folding_cls FOLDING_CLS
                        Folding file (npy) (e.g. cls_T11_fold_vector.npy)
  --folding_clsaux FOLDING_CLSAUX
                        Folding file (npy) (e.g. cls_T11_fold_vector.npy)
  --folding_regr FOLDING_REGR
                        Folding file (npy) (e.g. reg_T11_fold_vector.npy)
  --t8c_cls T8C_CLS     T8c file for classification in the results_tmp/classification folder
  --t8c_clsaux T8C_CLSAUX
                        T8c file for classification w/ auxiliary in the results_tmp/classification folder
  --t8r_regr T8R_REGR   T8r file for regression in the results_tmp/regression folder
  --weights_cls WEIGHTS_CLS
                        CSV file with columns task_id and weight (e.g. cls_weights.csv)
  --weights_clsaux WEIGHTS_CLSAUX
                        CSV file with columns task_id and weight (e.g cls_weights.csv)
  --weights_regr WEIGHTS_REGR
                        CSV file with columns task_id and weight (e.g. reg_weights.csv)
  --run_name RUN_NAME   Run name directory for results from this output (timestemp used if not specified)
  --significance_analysis {0,1}
                        Run significant analysis (1 = Yes, 0 = No sig. analysis
  --verbose {0,1}       Verbosity level: 1 = Full; 0 = no output
  --validation_fold {0,1,2,3,4} [{0,1,2,3,4} ...]
                        Validation fold to used to calculate performance
  --aggr_binning_scheme_perf AGGR_BINNING_SCHEME_PERF [AGGR_BINNING_SCHEME_PERF ...]
                        Shared aggregated binning scheme for performances
  --aggr_binning_scheme_perf_delta AGGR_BINNING_SCHEME_PERF_DELTA [AGGR_BINNING_SCHEME_PERF_DELTA ...]
                        Shared aggregated binning scheme for delta performances

```


#### Step 1.1. Prepare MP and SP cls, clsaux & reg outputs.
E.g. for each model type (cls,clsaux and reg) and each variant (different cp's) do:
```
gunzip -c <cls-hash>.tar.gz | tar xvf -
gunzip -c <clsaux-hash>.tar.gz | tar xvf -
gunzip -c <reg-hash>.tar.gz | tar xvf -
```

#### Step 1.2. Select the optimal cls & clsaux mp models.
Identify the optimal performing model based on the AUCPR reported in perf/perf.json.

E.g: a very crude example could be:
```
$ head  <clsaux_models_to_compre*>/perf/perf.json
==> <model1>/perf/perf.json <==
{"all": 0.6}
==> <model2>/perf/perf.json <==
{"all": 0.7}
==> <model3>/perf/perf.json <==
{"all": 0.8}    # <==== this would be the model to use in this crude example

$ rm model1/export/model.pth && rm model1/pred/pred
$ rm model2/export/model.pth && rm model2/pred/pred
```

#### Step 1.3. Select only 20 epoch reg model for SP & MP.
ONLY the Epoch 20 (SP&MP) models for reg are used for evaluation.

```
gunzip -c <reg-hash-epoch20>.tar.gz | tar xvf -
```

#### Step 2. Run the performance evaluation code:

Run the code supplying the best/epoch 20 MP models to the script.

E.g. for a clsaux model:
```
python performance_evaluation.py --y_clsaux <clsaux_dir>/clsaux_T10_y.npz --y_clsaux_single_partner <best_sp_model>/pred/pred --y_clsaux_multi_partner <best_mp_model>/pred/pred --folding_clsaux <clsaux_dir>/clsaux_T11_fold_vector.npy --t8c_clsaux <clsaux_dir>/T8c.csv --weights_clsaux <clsaux_dir>/clsaux_weights.csv --validation_fold 0 --run_name slurm_y2_test
```

Output should look something like:

```
[INFO]: === Y2 Performance evaluation script for npy and pred files ===
[INFO]: Wrote input params to '<run_dir>/run_params.json'


=======================================================================================================================================
Evaluating clsaux performance
=======================================================================================================================================

[INFO]: Loading clsaux: <clsaux_dir>/clsaux_T10_y.npz
[INFO]: Loading (pred) output for: <sp_model>/pred/pred
[INFO]: Loading (pred) output for: <mp_model>/pred/pred

[INFO]: === Calculating significance ===
[INFO]: === Calculating <sp_model>/pred/pred performance ===
[INFO]: Wrote per-task report to: slurm_y2_test/clsaux/SP/pred_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: slurm_y2_test/clsaux/SP/pred_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: slurm_y2_test/clsaux/SP/pred_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: slurm_y2_test/clsaux/SP/pred_binned_per-task_performances.csv
[INFO]: Wrote global report to: slurm_y2_test/clsaux/SP/pred_global_performances.csv

[INFO]: === Calculating <mp_model>/pred/pred performance ===
[INFO]: Wrote per-task report to: slurm_y2_test/clsaux/MP/pred_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: slurm_y2_test/clsaux/MP/pred_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: slurm_y2_test/clsaux/MP/pred_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: slurm_y2_test/clsaux/MP/pred_binned_per-task_performances.csv
[INFO]: Wrote global report to: slurm_y2_test/clsaux/MP/pred_global_performances.csv

[INFO]: Wrote per-task delta report to: slurm_y2_test/clsaux/deltas/deltas_per-task_performances.csv
[INFO]: Wrote binned performance per-task delta report to: slurm_y2_test/clsaux/deltas/deltas_binned_per-task_performances.csv
[INFO]: Wrote per-assay delta report to: slurm_y2_test/clsaux/deltas/deltas_per-assay_performances.csv
[INFO]: Wrote binned performance per-assay delta report to: slurm_y2_test/clsaux/deltas/deltas_binned_per-task_performances.csv
[INFO]: Wrote significance performance report to: slurm_y2_test_cls/clsaux/deltas/delta_significance.csv
[INFO]: Wrote per-assay significance report to: slurm_y2_test_cls/clsaux/deltas/delta_significance.csv
[INFO]: Wrote sp significance report to: slurm_y2_test_cls/clsaux/deltas/delta_sp_significance.csv
[INFO]: Wrote per-assay sp significance report to: slurm_y2_test_cls/clsaux/deltas/delta_per-assay_sp_significance.csv
[INFO]: Wrote mp significance report to: slurm_y2_test_cls/clsaux/deltas/delta_mp_significance.csv
[INFO]: Wrote per-assay mp significance report to: slurm_y2_test_cls/clsaux/deltas/delta_per-assay_mp_significance.csv


[INFO]: Run name 'slurm_y2_test' is finished.
[INFO]: Performance evaluation took 482.36725 seconds.
```

The following files are created:

```
  derisk_test #name of the run (timestamp used if not defined)
  ├── <clsaux>
  │   ├── deltas # the folder for sp vs. mp deltas 
  │   │   ├── delta_mp_significance.csv #mp significance of deltas between sp & mp 
  │   │   ├── delta_sp_significance.csv #sp significance of deltas between sp & mp 
  │   │   ├── delta_per-assay_mp_significance.csv #mp significance of deltas between sp & mp per-assay type  
  │   │   ├── delta_per-assay_sp_significance.csv #sp significance of deltas between sp & mp per-assay type  
  │   │   ├── deltas_binned_per-assay_performances.csv	#binned assay aggregated deltas between sp & mp  
  │   │   ├── deltas_binned_per-task_performances.csv	#binned deltas across all tasks between sp & mp 
  │   │   ├── deltas_global_performances.csv	#global aggregated deltas between sp & mp  
  │   │   ├── deltas_per-assay_performances.csv	#assay aggregated deltas between sp & mp 
  │   │   └── deltas_per-task_performances.csv	#deltas between sp & mp
  │   ├── sp #e.g. the folder of single-partner prediction results
  │   │   ├── pred_binned_per-assay_performances.csv	#binned sp assay aggregated performances
  │   │   ├── pred_binned_per-task_performances.csv	#binned sp performances
  │   │   ├── pred_global_performances.csv	#sp global performance
  │   │   ├── pred_per-assay_performances.csv	#sp assay aggregated performances
  │   │   └── pred_per-task_performances.csv	#sp performances
  │   └── mp #e.g. the folder of multi-partner predictions results
  │   │   ├── pred_binned_per-assay_performances.csv	#binned mp assay aggregated performances
  │   │   ├── pred_binned_per-task_performances.csv	#binned mp performances
  │   │   ├── pred_global_performances.csv	#mp global performance
  │   │   ├── pred_per-assay_performances.csv	#mp assay aggregated performances
  │   │   └── pred_per-task_performances.csv	#mp performances
  └── run_params.json #json with the runtime parameters

```

Files used to create plots are:
```
1.) deltas_binned_per-assay_performances.csv	
2.) deltas_binned_per-task_performances.csv	
3.) deltas_global_performances.csv
4.) deltas_per-assay_performances.csv
5.) delta_mp_significance.csv (will not be available for regr / regr_cens)
6.) delta_sp_significance.csv (will not be available for regr / regr_cens)
7.) delta_per-assay_mp_significance.csv (will not be available for regr / regr_cens)
8.) delta_per-assay_sp_significance.csv (will not be available for regr / regr_cens)
```


N.B: Regression does not calculate significance, so expect those files not to be present in regr / regr-cens outputs.



------



### CLS vs. CLSAUX comparison

```
$  python performance_evaluation_cls_clsaux.py -h
usage: performance_evaluation_cls_clsaux.py [-h] --y_cls Y_CLS --y_clsaux Y_CLSAUX --task_id_mapping TASK_ID_MAPPING --folding_cls FOLDING_CLS --folding_clsaux FOLDING_CLSAUX --weights_cls WEIGHTS_CLS --weights_clsaux
                                            WEIGHTS_CLSAUX --pred_cls PRED_CLS --pred_clsaux PRED_CLSAUX [--validation_fold VALIDATION_FOLD] --outfile OUTFILE

Computes statistical significance between a cls and a clsaux classification models

optional arguments:
  -h, --help            show this help message and exit
  --y_cls Y_CLS         Path to <...>/matrices/cls/cls_T10_y.npz
  --y_clsaux Y_CLSAUX   Path to <...>/matrices/clsaux/clsaux_T10_y.npz
  --task_id_mapping TASK_ID_MAPPING
                        CSV file with headers 'task_id_cls' and 'task_id_clsaux'
  --folding_cls FOLDING_CLS
                        Path to <...>/matrices/cls/cls_T11_fold_vector.npy
  --folding_clsaux FOLDING_CLSAUX
                        Path to <...>/matrices/clsaux/clsaux_T11_fold_vector.npy
  --weights_cls WEIGHTS_CLS
                        Path to <...>/matrices/clsaux/cls_weights.csv
  --weights_clsaux WEIGHTS_CLSAUX
                        Path to <...>/matrices/clsaux/clsaux_weights.csv
  --pred_cls PRED_CLS   Path to the predictions exported from platform of a cls model
  --pred_clsaux PRED_CLSAUX
                        Path to the predictions exported from platform of a clsaux model
  --validation_fold VALIDATION_FOLD
                        Validation fold to use
  --outfile OUTFILE     Name of the output file
```

#### Step 0 Create CLS-CLSAUX mapping file

Instructions to create the CLS & CLSAUX mapping file:

1. Read in `cls_weights.csv` and `T10c_cont.csv` (stored in MELLODDY-TUNER output folder  "results" from your cls preparation). Relevant columns in `T10c_cont.csv`: `cont_classification_task_id`, `input_assay_id`, `threshold`

2. Merge both files based on `task_id` and `cont_classification_task_id` 

3. Drop duplicates

4. Repeat steps 1-3 with `clsaux_weights.csv` and `T10c_cont.csv` (from clsaux preparation)

5. Merge again the two resulting files (one from cls, one from clsaux) based on `input_assay_id`  and `threshold`. Caution: The threshold col can lead to mismatches due to given decimal places of the values. 
Recommendation: round thresholds to a given number of decimal places. Add `_cls` and `_clsaux` as suffixes like:
```
mapping_table = pd.merge(mapping_cls_unique, mapping_clsaux_unique, on=["input_assay_id", "threshold"], suffixes=["_cls", "_clsaux"])
```


#### Step 1. Check the required files for this analysis

Ensure you have generated a CSV file (instructions in Step 0 above) with headers 'task_id_cls' and 'task_id_clsaux' that map identical tasks overlapping between CLS and CLSAUX models.

NB: Tasks that are unique to CLSAUX should not be in this file.

#### Step 2. Run the script.

For each of the comparisons (where overlapping epochs between CLS vs. CLSAUX are available):

1. CLS MP (optimal MP epoch) vs.CLSAUX MP (optimal MP epoch)
2. CLS SP (optimal SP epoch) vs.CLSAUX SP (optimal SP epoch)
3. CLS MP (optimal MP epoch) vs.CLSAUX SP (optimal SP epoch)
4. CLS SP (optimal SP epoch) vs.CLSAUX MP (optimal MP epoch)

To do this, run the script as, e.g:
```
python performance_evaluation_cls_clsaux.py \
        --y_cls cls_T10_y.npz \
        --y_clsaux clsaux_T10_y.npz \
        --folding_cls cls_T11_fold_vector.npy \
        --folding_clsaux clsaux_T11_fold_vector.npy \
        --weights_cls cls_weights.csv \
        --weights_clsaux clsaux_weights.csv \
        --validation_fold 0 \
        --pred_cls $cls_preds \
        --pred_clsaux $clsaux_preds \
        --task_id_mapping cls_clsaux_mapping.csv \
        --outfile {sp/mp}cls_vs_{sp/mp}clsaux
```


------


## === De-risk analysis below here ===


## Example of a de-risk of 2 epoch MP models (cls/clsaux)

#### Step 1. Retrieve the cls & clsaux output from substra and decompress
```
gunzip -c <cls-hash>.tar.gz | tar xvf - && gunzip -c <clsaux-hash>.tar.gz | tar xvf -
```

#### Step 2. Create SparseChem predictions on-premise with the outputted model

##### 2a. Fix both cls & clsaux hyperparameter.json (by removing model_type) & setup dirs:

```/
sed -i 's/, "model_type": "federated"//g' <2epoch_mp_cls_dir>/export/hyperparameters.json 
sed -i 's/, "model_type": "federated"//g' <2epoch_mp_clsaux_dir>/export/hyperparameters.json 
mkdir derisk_cls derisk_clsaux
```

##### 2b. Predict the validation fold using the MP model output from substra (load your conda env), e.g.:

for cls:
```
python <sparsechem_dir>/examples/chembl/predict.py \
  --x <cls_dir>/cls_T11_x.npz \
  --y_class <cls_dir>/cls_T10_y.npz \
  --folding <cls_dir>/cls_T11_fold_vector.npy \
  --predict_fold 0 \
  --conf <2epoch_mp_cls_dir>/export/hyperparameters.json \
  --model <2epoch_mp_cls_dir>/export/model.pth \
  --dev cuda:0 \
  --outprefix "derisk_cls/pred"
```
and clsaux:
```
python <sparsechem_dir>/examples/chembl/predict.py \
  --x <clsaux_dir>/clsaux_T11_x.npz \
  --y_class <clsaux_dir>/clsaux_T10_y.npz \
  --folding <clsaux_dir>/clsaux_T11_fold_vector.npy \
  --predict_fold 0 \
  --conf <2epoch_mp_clsaux_dir>/export/hyperparameters.json \
  --model <2epoch_mp_clsaux_dir>/export/model.pth \
  --dev cuda:0 \
  --outprefix "derisk_clsaux/pred"
  ```
  
#### Step 3. Run the derisk script
```
python performance_evaluation_derisk.py \ 
  --y_cls cls/cls_T10_y.npz \ 
  --y_cls_onpremise derisk_cls/pred-class.npy \ 
  --y_cls_substra <2epoch_mp_cls_dir>/pred/pred \ 
  --folding_cls cls/cls_T11_fold_vector.npy \
  --t8c_cls cls/T8c.csv \
  --weights_cls cls/cls_weights.csv \ 
  --perf_json_cls <2epoch_mp_cls_dir>/perf/perf.json \ 
  --y_clsaux clsaux/clsaux_T10_y.npz \ 
  --y_clsaux_onpremise derisk_clsaux/pred-class.npy \ 
  --y_clsaux_substra <2epoch_mp_clsaux_dir>/pred/pred \ 
  --folding_clsaux clsaux/clsaux_T11_fold_vector.npy \ 
  --t8c_clsaux clsaux/T8c.csv \ 
  --weights_clsaux clsaux/clsaux_weights.csv \ 
  --perf_json_clsaux <2epoch_mp_clsaux_dir>/perf/perf.json \ 
  --validation_fold 0 \ 
  --run_name derisk_2epoch_cls_clsaux
```

Output should look something like this:
```
=======================================================================================================================================
=======================================================================================================================================
De-risking cls performance
=======================================================================================================================================
=======================================================================================================================================

[INFO]: Loading cls: cls/cls_T10_y.npz
[INFO]: Loading (npy) predictions for: derisk_cls/pred-class.npy
[INFO]: Loading (pred) output for: <2epoch_mp_cls_dir>/pred/pred

=======================================================================================================================================
[DERISK-CHECK #1]: PASSED! yhats close between 'pred-class' and 'pred' (tol:1e-05)
Spearmanr rank correlation coefficient of the 'pred-class' and 'pred' yhats = SpearmanrResult(correlation=0.9999999999999996, pvalue=0.0)
=======================================================================================================================================

[INFO]: === Calculating derisk_cls/pred-class.npy performance ===
[INFO]: Wrote per-task report to: <derisk_run>/cls/pred-class/pred-class_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: <derisk_run>/cls/pred-class/pred-class_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: <derisk_run>/cls/pred-class/pred-class_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: <derisk_run>/cls/pred-class/pred-class_binned_per-task_performances.csv
[INFO]: Wrote global report to: <derisk_run>/cls/pred-class/pred-class_global_performances.csv

[INFO]: === Calculating <2epoch_mp_cls_dir>/pred/pred performance ===
[INFO]: Wrote per-task report to: <derisk_run>/cls/pred/pred_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: <derisk_run>/cls/pred/pred_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: <derisk_run>/cls/pred/pred_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: <derisk_run>/cls/pred/pred_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #2]: SKIPPED! substra does not report individual task performances
=======================================================================================================================================

[INFO]: Wrote global report to: <derisk_run>/cls/pred/pred_global_performances.csv

=======================================================================================================================================
[DERISK-CHECK #3]: FAILED! global reported performance metrics and global calculated performance metrics NOT close (tol:1e-05)
Calculated:<removed>
Reported:<removed>
=======================================================================================================================================

=======================================================================================================================================
[DERISK-CHECK #4]: PASSED! delta between local & substra assay_type aggregated performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

[INFO]: Wrote per-task delta report to: <derisk_run>/cls/deltas/deltas_per-task_performances.csv
[INFO]: Wrote binned performance per-task delta report to: <derisk_run>/cls/deltas/deltas_binned_per-task_performances.csv
[INFO]: Wrote per-assay delta report to: <derisk_run>/cls/deltas/deltas_per-assay_performances.csv
[INFO]: Wrote binned performance per-assay delta report to: <derisk_run>/cls/deltas/deltas_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #5]: PASSED! delta performance between global local & global substra performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

=======================================================================================================================================
=======================================================================================================================================
De-risking clsaux performance
=======================================================================================================================================
=======================================================================================================================================

[INFO]: Loading clsaux: clsaux/clsaux_T10_y.npz
[INFO]: Loading (npy) predictions for: derisk_clsaux/pred-class.npy
[INFO]: Loading (pred) output for: <2epoch_mp_clsaux_dir>/pred/pred

=======================================================================================================================================
[DERISK-CHECK #1]: PASSED! yhats close between 'pred-class' and 'pred' (tol:1e-05)
Spearmanr rank correlation coefficient of the 'pred-class' and 'pred' yhats = SpearmanrResult(correlation=1.0, pvalue=0.0)
=======================================================================================================================================

[INFO]: === Calculating derisk_clsaux/pred-class.npy performance ===
[INFO]: Wrote per-task report to: <derisk_run>/clsaux/pred-class/pred-class_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: <derisk_run>/clsaux/pred-class/pred-class_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: <derisk_run>/clsaux/pred-class/pred-class_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: <derisk_run>/clsaux/pred-class/pred-class_binned_per-task_performances.csv
[INFO]: Wrote global report to: <derisk_run>/clsaux/pred-class/pred-class_global_performances.csv

[INFO]: === Calculating <2epoch_mp_clsaux_dir>/pred/pred performance ===
[INFO]: Wrote per-task report to: <derisk_run>/clsaux/pred/pred_per-task_performances.csv
[INFO]: Wrote per-task binned performance report to: <derisk_run>/clsaux/pred/pred_binned_per-task_performances.csv
[INFO]: Wrote per-assay report to: <derisk_run>/clsaux/pred/pred_per-assay_performances.csv
[INFO]: Wrote per-assay binned report to: <derisk_run>/clsaux/pred/pred_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #2]: SKIPPED! substra does not report individual task performances
=======================================================================================================================================

[INFO]: Wrote global report to: <derisk_run>/clsaux/pred/pred_global_performances.csv

=======================================================================================================================================
[DERISK-CHECK #3]: FAILED! global reported performance metrics and global calculated performance metrics NOT close (tol:1e-05)
Calculated:<removed>
Reported:<removed>
=======================================================================================================================================

=======================================================================================================================================
[DERISK-CHECK #4]: PASSED! delta between local & substra assay_type aggregated performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

[INFO]: Wrote per-task delta report to: <derisk_run>/clsaux/deltas/deltas_per-task_performances.csv
[INFO]: Wrote binned performance per-task delta report to: <derisk_run>/clsaux/deltas/deltas_binned_per-task_performances.csv
[INFO]: Wrote per-assay delta report to: <derisk_run>/clsaux/deltas/deltas_per-assay_performances.csv
[INFO]: Wrote binned performance per-assay delta report to: <derisk_run>/clsaux/deltas/deltas_binned_per-task_performances.csv

=======================================================================================================================================
[DERISK-CHECK #5]: PASSED! delta performance between global local & global substra performances close to 0 across all metrics (tol:1e-05)
=======================================================================================================================================

[INFO]: Run name '<derisk_run>' is finished.
[INFO]: Performance evaluation de-risk took 1101.3661 seconds.
```

The file 'derisk_summary.csv' has a concise summary of de-risk checks
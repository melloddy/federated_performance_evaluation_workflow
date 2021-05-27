# Performance Evaluation Script for the IMI Project MELLODDY

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

Alternatively to step 4/5 you can install the combined enrionment in environment_melloddy_combined.yml using `conda env create -f development/environment_melloddy_combined.yml`

# Example 1: De-risk analysis (on-premise vs. single-partner substra output evaluation)

## Build onpremise model (with your local sparsechem)
1. Train a model with sparsechem using the same input data as for the federated run (sparsechem/examples/chembl/train.py)
2. Choose the hyperparameters from the federated system (weight_decay depends on your data size):
```
python sparsechem/examples/chembl/train.py  --x x.npy \
                                            --y y.npy \
                                            --folding folding.npy \
                                            --task_weights weights.csv \
                                            --hidden_sizes 3600 \
                                            --middle_dropout 0.0 \
                                            --last_dropout 0.2 \
                                            --weight_decay 1e-2 \
                                            --last_non_linearity relu \
                                            --non_linearity relu \
                                            --input_transform binarize \
                                            --lr 0.001 \
                                            --lr_alpha 0.3 \
                                            --lr_steps 10 \
                                            --epochs 20 \
                                            --fold_va 1 \
                                            --fold_te 0 \
                                            --min_samples_auc 25 \
                                            --normalize_loss 100000
```
## Comparison of reported performances
```
import sparsechem as sc
res = sc.load_results('models/perf-<sparsechem-output>.json')
# collect individual tasks perf
perf = res['results']['va']
# 5/5 filter
filtered_perf = perf.loc[(perf['num_pos']>=5 ) & (perf['num_neg']>=5)]
# aggregate auc_pr
mean_auc_pr = filtered_perf.auc_pr.mean()
```

## Setup

1. Download the substra output
2. Use the model located in your Single_pharma_run/medias/subtuple/{pharma-hash}/export/single_model.pth to create on-premise *sparse* y-hat predictions for the in-house dataset. E.g.:

```python sparsechem/examples/chembl/predict.py --x x.npy --y y.npy --outfile onpremise_y_hat.npy --folding folding.npy --conf {pharma-hash}/export/hyperparameters.json --model {pharma-hash}/export/single_model_.pth --predict_fold 1```

3. Locate the Single-pharma "pred" & "perf.json" from the Single_pharma_run/medias/subtuple/{pharma-hash}/pred/ folder 
4. Provide the script with the y-hat sparse prediction (onpremise_y_hat.npy) from step 2, the pred, perf.json and task_mapping file

## Analysis script (performance_evaluation_derisk.py)

```
python performance_evaluation_derisk.py -h
usage: performance_evaluation_derisk.py [-h] --y_true_all Y_TRUE_ALL
                                        --y_pred_onpremise Y_PRED_ONPREMISE
                                        --y_pred_substra Y_PRED_SUBSTRA
                                        --folding FOLDING
                                        --substra_performance_report
                                        SUBSTRA_PERFORMANCE_REPORT --task_map
                                        TASK_MAP [--filename FILENAME]
                                        [--use_venn_abers] [--verbose {0,1}]

Calculate Performance Metrics

optional arguments:
  -h, --help            show this help message and exit
  --y_true_all Y_TRUE_ALL
                        Activity file (npy) (i.e. from files_4_ml/)
  --y_pred_onpremise Y_PRED_ONPREMISE
                        Yhat prediction output from onpremise run (<single
                        pharma dir>/y_hat.npy)
  --y_pred_substra Y_PRED_SUBSTRA
                        Pred prediction output from substra platform
                        (./Single-pharma-
                        run/substra/medias/subtuple/<pharma_hash>/pred/pred)
  --folding FOLDING     LSH Folding file (npy) (i.e. from files_4_ml/)
  --substra_performance_report SUBSTRA_PERFORMANCE_REPORT
                        JSON file with global reported performance from
                        substra platform (i.e. ./Single-pharma-run/substra/med
                        ias/subtuple/<pharma_hash>/pred/perf.json)
  --task_map TASK_MAP   Taskmap from MELLODDY_tuner output of single run (i.e.
                        from results/weight_table_T3_mapped.csv)
  --filename FILENAME   Filename for results from this output
  --use_venn_abers      Toggle to turn on Venn-ABERs code
  --verbose {0,1}       Verbosity level: 1 = Full; 0 = no output


```

## Running the de-risk code
```
python performance_evaluation_derisk.py --y_true_all pharma_partners/pharma_y_partner_1.npy --y_pred_substra Single_pharma_run-1/medias/subtuple/<hash>/pred/pred --folding pharma_partners/folding_partner_1.npy --substra_performance_report Single_pharma_run-1/medias/subtuple/<hash>/perf/perf.json --filename derisk_test --task_map pharma_partners/weight_table_T3_mapped.csv --y_pred_onpremise y_hat1.npy
```

The output should look something like:
```
on-premise_per-task_performances_derisk.csv
on-premise_per-assay_performances_derisk.csv
on-premise_global_performances_derisk.csv
substra_per-task_performances_derisk.csv
substra_per-assay_performances_derisk.csv
substra_global_performances_derisk.csv
deltas_per-task_performances_derisk.csv
deltas_per-assay_performances_derisk.csv
deltas_global_performances_derisk.csv
```

Any problem in the substra output (4 derisk errors/warnings) will be reported like this:
```
(Phase 2 de-risk output check #1): ERROR! yhats not close between on-premise (generated from the substra model) .npy and substra 'pred' file (tol:1e-05)
(Phase 2 de-risk check #2): WARNING! Reported performance in {performance_report} ({global_pre_calculated_performance}) not close to calculated performance for {substra} ({aucpr_mean}) (tol:1e-05)
(Phase 2 de-risk output check #3): WARNING! Calculated per-task deltas are not all close to zero (tol:1e-05)
(Phase 2 de-risk output check #4): WARNING! Calculated global aggregation performance for on-premise {onpremise_results[idx]['aucpr_mean'].values} and substra {substra_results[idx]['aucpr_mean'].values} are not close (tol:1e-05)
```

De-risk checks that pass the criteria are reported like this:

```
(Phase 2 de-risk output check #1): Check passed! yhats close between on-premise (generated from the substra model) .npy and substra 'pred' file (tol:1e-05)
(Phase 2 de-risk check #2): Check passed! Reported performance in {performance_report} ({global_pre_calculated_performance}) close to the calculated performance for {substra} ({aucpr_mean}) (tol:1e-05)
(Phase 2 de-risk output check #3): Check passed! Calculated per-task deltas close to zero (tol:1e-05)
(Phase 2 de-risk output check #4): Check passed! Calculated global aggregation performance for on-premise {onpremise_results[idx]['aucpr_mean'].values} and substra {substra_results[idx]['aucpr_mean'].values} are close (tol:1e-05)
```

-----

# Example 2: Single-pharma vs. Multi-pharma performance analysis

## 'Pred' or '.npy' file evaluation Setup

### Option 1:
Using the substra file outputs ('pred' file analysis)
1. Download the substra output
2. Locate the Sinlge-pharma "pred" & "perf.json" from the Single_pharma_run/medias/subtuple/{pharma-hash}/pred/ folder 
3. Locate the Multi-pharma "pred" & "perf.json" from the Multi_pharma_run/medias/subtuple/{pharma-hash}/pred/ folder 
4. Provide the script with the single- and multi-pharma pred files from step 2/3, the perf.json and task_mapping file

### Option 2:
On-premise predictions using the substra models (.npy file analysis)
1. Download the substra output
2. Use the substra model located in your {single_pharma_run}/medias/subtuple/{pharma-hash}/export/single_model.pth to create on-premise *sparse* y-hat predictions for the in-house dataset. E.g.:
```python sparsechem/examples/chembl/predict.py --x x.npy --y y.npy --outfile onpremise_sp_y_hat.npy --folding folding.npy --conf {pharma-hash}/export/hyperparameters.json --model {pharma-hash}/export/single_model_.pth --predict_fold 1```
3. Use the substra model located in your {multi_pharma_run}/medias/subtuple/{pharma-hash}/export/model.pth to create on-premise *sparse* y-hat predictions for the in-house dataset. E.g.:
```python sparsechem/examples/chembl/predict.py --x x.npy --y y.npy --outfile onpremise_mp_y_hat.npy --folding folding.npy --conf {pharma-hash}/export/hyperparameters.json --model {pharma-hash}/export/model.pth --predict_fold 1```
4. Provide the script with the single- and multi-pharma '.npy' files from step 2/3 and the other base files

## Pred analysis script (performance_evaluation.py)

This script calculates performances and evaluates whether there is an improvement (delta) in predictive performance between a single-pharma vs. a multi-pharma run. 
Input files can be any combination of 'pred' or 'npy' input files. The script can also accept mtx files for specific use cases.

```
python performance_evaluation.py -h
usage: performance_evaluation.py [-h] --y_true_all Y_TRUE_ALL --task_map
                                  TASK_MAP --folding FOLDING
                                  [--task_weights TASK_WEIGHTS]
                                  [--validation_fold]
                                  [--filename FILENAME] [--use_venn_abers]
                                  [--verbose {0,1}] --f1 F1 --f2 F2
                                  [--aggr_binning_scheme_perf AGGR_BINNING_SCHEME_PERF]
                                  [--aggr_binning_scheme_perf_delta AGGR_BINNING_SCHEME_PERF_DELTA]

Calculate Performance Metrics

optional arguments:
  -h, --help            show this help message and exit
  --y_true_all Y_TRUE_ALL
                        Activity file (npy) (i.e. from files_4_ml/)
  --task_map TASK_MAP   Taskmap from MELLODDY_tuner output of single run (i.e.
                        from results/weight_table_T3_mapped.csv)
  --folding FOLDING     LSH Folding file (npy) (i.e. from files_4_ml/)
  --validation_fold     Validation fold to used to calculate performance
  --task_weights TASK_WEIGHTS
                        (Optional: for weighted global aggregation) CSV file
                        with columns task_id and weight (i.e.
                        files_4_ml/T9_red.csv)
  --filename FILENAME   Filename for results from this output
  --use_venn_abers      Toggle to turn on Venn-ABERs code
  --verbose {0,1}       Verbosity level: 1 = Full; 0 = no output
  --f1 F1               Output from the first run to compare (pred or .npy)
  --f2 F2               Output from the second run to compare (pred or .npy)
  --aggr_binning_scheme_perf AGGR_BINNING_SCHEME_PERF
                        (Comma separated) Shared aggregated binning scheme for
                        f1/f2 performances
  --aggr_binning_scheme_perf_delta AGGR_BINNING_SCHEME_PERF_DELTA
                        (Comma separated) Shared aggregated binning scheme for
                        delta performances
```

The output should look something like:
```
run_params.json                                 #args namespace of parameters used for perf evaluation
f[1/2]_per-task_performances.csv                #file 1/2 perf reported per task
f[1/2]_per-assay_performances.csv               #file 1/2 perf reported per task
f[1/2]_global_performances.csv                  #file 1/2 global perf
deltas_per-task_performances.csv                #per-task perf delta between file 1/2
deltas_per-assay_performances.csv               #per-assay aggregated delta perf between file 1/2
f[1/2]_binned_per-task_performances.csv **      #file 1/2 tasks split by perf bins with proportion of tasks in each bin
f[1/2]_binned_per-assay_performances.csv **     #file 1/2 output split by perf bins and aggregated by assay_types with proportion of assay_type tasks in each bin
deltas_binned_per-task_performances.csv **      #per-task binned perf delta between file 1/2
deltas_binned_per-assay_performances.csv **     #delta between 1/2 perf split by perf bins and aggregated by assay_types with proportion of assay_type tasks in each bin
deltas_global_performances.csv               #global delta of file 1/2 perf

**=for the WP3 per-pharma performance YR1 report
```

### Minimum Working Example

This is an example with a single archive with all input files required already prepared. All files were taken for a single pharma partner from the phase 2 run on public chembl data. This example archive is just to get you started on the evaluation and should be used as minimum working example to test the performance evaluation script on your infrastructure. Once you get this to work, replace all input files with your relevant input files with your private data/models. 

[Download the example archive and extract it into the `data` folder. 

To run the sample single/multi partner evaluation run: 
```bash

python performance_evaluation.py \
    --f1 data/example/single/pred/pred \
    --f2 data/example/multi/pred/pred \
    --folding data/example/files_4_ml/folding.npy \
    --task_weights data/example/files_4_ml/weights.csv \ #this is optional (not for YR1 report)
    --filename out \
    --task_map data/example/files_4_ml/weight_table_T3_mapped.csv \
    --y_true_all data/example/files_4_ml/pharma_y.npy \
    --validation_fold 0
```

This will write all relevant output files into the out folder. 
NB: if the out folder already exists (from a previous failed run for instance) then the script will stop gracefully in order not to overwrite previous results.

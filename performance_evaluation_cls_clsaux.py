import os
import pandas as pd
import numpy as np
import sparsechem as sc
import torch
import scipy.sparse as sparse
from sklearn.metrics import roc_auc_score
import significance_analysis
import argparse
import scipy
from scipy import stats

parser = argparse.ArgumentParser(description="Computes statistical significance between CLS and a CLSAUX classification models: release-2-(classification-and-regression)")
parser.add_argument("--y_cls", type=str, help="Path to <...>/matrices/cls/cls_T10_y.npz", required=True)
parser.add_argument("--y_clsaux", type=str, help="Path to <...>/matrices/clsaux/clsaux_T10_y.npz", required=True)
parser.add_argument("--task_id_mapping", type=str, help="CSV file with headers 'task_id_cls' and 'task_id_clsaux'", required=True)
parser.add_argument("--folding_cls", type=str, help="Path to <...>/matrices/cls/cls_T11_fold_vector.npy", required=True)
parser.add_argument("--folding_clsaux", type=str, help="Path to <...>/matrices/clsaux/clsaux_T11_fold_vector.npy", required=True)
parser.add_argument("--weights_cls", type=str, help="Path to <...>/matrices/clsaux/cls_weights.csv", required=True)
parser.add_argument("--weights_clsaux", type=str, help="Path to <...>/matrices/clsaux/clsaux_weights.csv", required=True)
parser.add_argument("--pred_cls", type=str, help="Path to the predictions exported from platform of a cls model", required=True)
parser.add_argument("--pred_clsaux", type=str, help="Path to the predictions exported from platform of a clsaux model", required=True)
parser.add_argument("--validation_fold", type=int, help="Validation fold to use", default=0)
parser.add_argument("--outfile", type=str, help="Name of the output file", required=True)
args = parser.parse_args()

assert not os.path.isfile(args.outfile), f"Output file : {args.outfile} already exists"

tw = pd.read_csv(args.weights_cls)
twx = pd.read_csv(args.weights_clsaux)
twx_dict=pd.read_csv(args.task_id_mapping)
twx_dict=dict(zip(twx_dict['task_id_clsaux'],twx_dict['task_id_cls']))

task2considx = twx.loc[twx['aggregation_weight']==1]['task_id'].values
task2consid = tw.loc[tw['aggregation_weight']==1]['task_id'].values

twx=twx.query('task_type != "AUX_HTS"').rename({'task_id':'clsaux_task_id'},axis=1)
twx['task_id']=twx['clsaux_task_id'].map(twx_dict)
twx=twx.dropna()

task2considx = twx.loc[twx['aggregation_weight']==1]['task_id'].values
task2consid = tw.loc[tw['aggregation_weight']==1]['task_id'].values
assert (task2considx == task2consid).all(), 'not (task2considx == task2consid).all()'


y = sc.load_sparse(args.y_cls)
yx = sc.load_sparse(args.y_clsaux)


folds = np.load(args.folding_cls)
foldsx = np.load(args.folding_clsaux)


y=y[folds==args.validation_fold].tocsc()
yx=yx[foldsx==args.validation_fold].tocsc()


# find out the main compounds
yhat_cls = torch.load(args.pred_cls).astype('float64').tocsc()
yhat_clsaux = torch.load(args.pred_clsaux).astype('float64').tocsc()


yhat_cls = yhat_cls
yhat_clsaux = yhat_clsaux


metrics = []
for k, task_id in enumerate(task2consid):
    task_idx = twx[twx['task_id']==task_id]['clsaux_task_id'].values[0]
    assert (y[:, task_id].data == yx[:, task_idx].data).all(), "Labels in cls and clsaux are not ordered identically"
        
    ytrue = (y[:, task_id].data == 1).astype(np.uint8)
    yscore_clsaux = yhat_clsaux[:, task_idx].data
    yscore_cls     = yhat_cls[:, task_id].data
    
    with np.errstate(divide='ignore',invalid='ignore'):
        significance1 = significance_analysis.test_significance(ytrue, yscore_clsaux, yscore_cls, level=0.05)
        significance1['task_id'] = task_id
        significance1['auroc_cls'] = roc_auc_score(ytrue, yscore_cls)
        significance1['auroc_clsaux'] = roc_auc_score(ytrue, yscore_clsaux)
        significance1.rename(columns={'p_value':'p_value clsaux > cls', 'significant': 'significant clsaux > cls'}, inplace=True)
    
        significance2 = significance_analysis.test_significance(ytrue, yscore_cls, yscore_clsaux, level=0.05)
        significance2.rename(columns={'p_value':'p_value cls > clsaux', 'significant': 'significant cls > clsaux'}, inplace=True)
    
        signif = significance1.join(significance2)[['task_id',
                                                    'auroc_cls', 
                                                    'auroc_clsaux', 
                                                    'p_value clsaux > cls', 
                                                    'significant clsaux > cls',
                                                    'p_value cls > clsaux', 
                                                    'significant cls > clsaux' ]]
    
    metrics.append(signif)
    
metrics_df = pd.concat(metrics, ignore_index=True)
metrics_df.to_csv(args.outfile, index=None)

summary_df = metrics_df.describe().drop('task_id',axis=1).loc['mean']
summary_df['wilcoxon']= stats.wilcoxon(metrics_df['auroc_clsaux'],metrics_df['auroc_cls'], alternative='greater').pvalue
summary_df['wilcoxon_g']= stats.wilcoxon(metrics_df['auroc_clsaux'],metrics_df['auroc_cls']).pvalue
summary_df.to_csv(f'{args.outfile}_summary_to_report.csv')
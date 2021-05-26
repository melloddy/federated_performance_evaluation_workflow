import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import sklearn.metrics
import json
import scipy.sparse as sparse
from scipy.stats import spearmanr
from  pathlib import Path

def init_arg_parser():
	parser = argparse.ArgumentParser(description="Calculate Performance Metrics")
	parser.add_argument("--y_true_all", help="Activity file (npy) (i.e. from files_4_ml/)", type=str, required=True)
	parser.add_argument("--task_map", help="Taskmap from MELLODDY_tuner output of single run (i.e. from results/weight_table_T3_mapped.csv)", required=True)
	parser.add_argument("--folding", help="LSH Folding file (npy) (i.e. from files_4_ml/)", type=str, required=True)
	parser.add_argument("--task_weights", help="(Optional: for weighted global aggregation) CSV file with columns task_id and weight (i.e.  files_4_ml/T9_red.csv)", type=str, default=None)
	parser.add_argument("--filename", help="Filename for results from this output", type=str, default=None)
	parser.add_argument("--use_venn_abers", help="Toggle to turn on Venn-ABERs code", action='store_true', default=False)
	parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])
	parser.add_argument("--validation_fold", help="Validation fold to used to calculate performance", type=int, default=0, choices=[0, 1, 2, 3, 4])
	parser.add_argument("--min_size", help="Minimum size of the task (NB: defaults to 25)", type=int, default=25, required=False)
	parser.add_argument("--f1", help="Output 1 (i.e. from the SP run) to compare (pred or .npy)", type=str, required=True)
	parser.add_argument("--f2", help="Output 2 (i.e. from the MP run) run to compare (pred or .npy)", type=str, required=True)
	parser.add_argument("--aggr_binning_scheme_perf", help="(Comma separated) Shared aggregated binning scheme for f1/f2 performances", type=str, default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',required=False)
	parser.add_argument("--aggr_binning_scheme_perf_delta", help="(Comma separated) Shared aggregated binning scheme for delta performances", type=str, default='-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2',required=False)
	parser.add_argument("--pharma_name", help="Name of pharma partner identifier (A/B/C/etc.)", type=str, default=None,required=False)
	args = parser.parse_args()
	args.aggr_binning_scheme_perf=list(map(np.float,args.aggr_binning_scheme_perf.split(',')))
	args.aggr_binning_scheme_perf_delta=list(map(np.float,args.aggr_binning_scheme_perf_delta.split(',')))
	return args

def vprint(s=""):
	if args.verbose:
		print(s)

#find f1 
def find_max_f1(precision, recall):
	F1	= np.zeros(len(precision))
	mask = precision > 0
	F1[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
	return F1.max()

#cut function for binning scheme
def cut(x, bins, lower_infinite=True, upper_infinite=True, **kwargs):        
    num_labels      = len(bins) - 1
    include_lowest  = kwargs.get("include_lowest", False)
    right           = kwargs.get("right", True)
    bins_final = bins.copy()
    if upper_infinite:
        bins_final.insert(len(bins),float("inf"))
        num_labels += 1
    if lower_infinite:
        bins_final.insert(0,float("-inf"))
        num_labels += 1
    symbol_lower  = "<=" if include_lowest and right else "<"
    left_bracket  = "(" if right else "["
    right_bracket = "]" if right else ")"
    symbol_upper  = ">" if right else ">="
    labels=[]
    
    def make_label(i, lb=left_bracket, rb=right_bracket):
        return "{0}{1}-{2}{3}".format(lb, bins_final[i], bins_final[i+1], rb)
        
    for i in range(0,num_labels):
        new_label = None
        if i == 0:
            if lower_infinite:
                new_label = "{0} {1}".format(symbol_lower, bins_final[i+1])
            elif include_lowest:
                new_label = make_label(i, lb="[")
            else:
                new_label = make_label(i)
        elif upper_infinite and i == (num_labels - 1):
            new_label = "{0} {1}".format(symbol_upper, bins_final[i])
        else:
            new_label = make_label(i)
        labels.append(new_label)
    return pd.cut(x, bins_final, labels=labels, **kwargs)


## write performance reports for global aggregation
def write_global_report(global_performances, fname, name):
	if args.use_venn_abers: cols = ['aucpr_mean','aucroc_mean','logloss_mean','maxf1_mean', 'kappa_mean', 'vennabers_mean', 'brier_mean', 'tn', 'fp', 'fn', 'tp']
	else: cols = ['aucpr_mean','aucroc_mean','logloss_mean','maxf1_mean', 'kappa_mean', 'brier_mean', 'tn', 'fp', 'fn', 'tp']
	perf_df = pd.DataFrame([global_performances],columns=cols)
	fn = name + '/' + fname + "_global_performances.csv"
	perf_df.to_csv(fn, index=None)
	vprint(f"Wrote {fname} global performance report to: {fn}")
	return perf_df

## write performance reports per-task & per-task_assay
def write_aggregated_report(local_performances, fname, name, task_map):
	# write per-task report
	df = local_performances[:]
	df['classification_task_id'] = df['classification_task_id'].astype('int32')
	for metric in df.loc[:, "aucpr":"brier"].columns:
		df[f'{metric}_percent'] = cut(df[metric].astype('float32'), \
		args.aggr_binning_scheme_perf,include_lowest=True,right=True,lower_infinite=False, upper_infinite=False)
	df = df.merge(task_map, right_on=["classification_task_id","assay_type"], left_on=["classification_task_id","assay_type"], how="left")
	fn1 = name + '/' + fname + "_per-task_performances.csv"
	df.to_csv(fn1, index=False)
	vprint(f"Wrote {fname} per-task report to: {fn1}")
	
	#write binned per-task performances
	agg_concat=[]
	for metric_bin in df.loc[:, "aucpr_percent":"brier_percent"].columns:
		agg_perf=(df.groupby(metric_bin)['classification_task_id'].agg('count')/len(df)).reset_index().rename(columns={'classification_task_id': f'bin_{metric_bin}'})
		agg_concat.append(agg_perf.set_index(metric_bin))
	if args.pharma_name: fnagg = name + '/' + fname + f"_binned_{args.pharma_name}_per-task_performances.csv"
	else: fnagg = name + '/' + fname + "_binned_per-task_performances.csv"
	pd.concat(agg_concat,axis=1).astype(np.float32).reset_index().rename(columns={'index': 'perf_bin'}).to_csv(fnagg,index=False)
	vprint(f"Wrote {fname} binned performance aggregated per-task report to: {fnagg}")
	
	#write performance aggregated performances by assay_type
	df2 = local_performances.loc[:,'assay_type':].groupby('assay_type').mean()
	fn2 = name + '/' + fname + "_per-assay_performances.csv"
	df2.to_csv(fn2)
	vprint(f"Wrote {fname} per-assay report to: {fn2}")

	#write binned per-task perf performances by assay_type 
	agg_concat2=[]
	for metric_bin in df.loc[:, "aucpr_percent":"brier_percent"].columns:
		agg_perf2=(df.groupby(['assay_type',metric_bin])['classification_task_id'].agg('count')).reset_index().rename(columns={'classification_task_id': f'count_{metric_bin}'})
		agg_perf2[f'bin_{metric_bin}']=agg_perf2.apply(lambda x : x[f'count_{metric_bin}'] / (df.assay_type==x['assay_type']).sum() ,axis=1).astype('float32')
		agg_perf2.drop(f'count_{metric_bin}',axis=1,inplace=True)
		agg_concat2.append(agg_perf2.set_index(['assay_type',metric_bin]))
	if args.pharma_name: fnagg2 = name + '/' + fname + f"_binned_{args.pharma_name}_per-assay_performances.csv"
	else: fnagg2 = name + '/' + fname + "_binned_per-assay_performances.csv"
	pd.concat(agg_concat2,axis=1).astype(np.float32).reset_index().rename(columns={'level_0':'assay_type','level_1':'perf_bin',}).to_csv(fnagg2,index=False)
	vprint(f"Wrote {fname} binned performance aggregated per-assay report to: {fnagg}")
	return

#load either pred or npy yhats and mask if needed, for an input filename
def load_yhats(input_f, folding, fold_va, y_true):
	# load the data
	if input_f.suffix == '.npy':
		vprint(f'Loading (npy) predictions for: {input_f}') 
		yhats = np.load(input_f, allow_pickle=True).item().tocsr().astype('float32')
		ftype = 'npy'
	else:
		vprint(f'Loading (pred) output for: {input_f}') 
		yhats = torch.load(input_f).astype('float32')
		ftype = 'pred'
	# mask validation fold if possible
	try: yhats = yhats[folding == fold_va]
	except IndexError: pass
	return yhats, ftype

#perform masking, report any error in shapes, and return data for f1 and f2
def mask_y_hat(f1_path, f2_path, folding, fold_va, y_true):
	true_data = y_true.astype(np.uint8).todense()
	f1_yhat, f1_ftype = load_yhats(f1_path, folding, fold_va, y_true)
	assert true_data.shape == f1_yhat.shape, f"True shape {true_data.shape} and {args.f1} shape {f1_yhat.shape} need to be identical"
	f2_yhat, f2_ftype = load_yhats(f2_path, folding, fold_va, y_true)
	assert true_data.shape == f2_yhat.shape, f"True shape {true_data.shape} and {args.f2} shape {f2_yhat.shape} need to be identical"
	return [f1_yhat, f2_yhat, f1_ftype, f2_ftype]

## check the pre_calculated_performance with the reported performance json
def per_run_performance(y_pred, pred_or_npy, tasks_table, y_true, tw_df, task_map, name, flabel):
	if args.use_venn_abers: from VennABERS import get_VA_margin_cross
	if pred_or_npy == 'npy': y_pred = sparse.csc_matrix(y_pred)
	## checks to make sure y_true and y_pred match
	assert y_true.shape == y_pred.shape, f"y_true shape do not match {pred_or_npy} y_pred ({y_true.shape} & {y_pred.shape})"
	assert y_true.nnz == y_pred.nnz, f"y_true number of nonzero values do not match {pred_or_npy} y_pred"
	assert (y_true.indptr == y_pred.indptr).all(), f"y_true indptr do not match {pred_or_npy} y_pred"
	assert (y_true.indices == y_pred.indices).all(), f"y_true indices do not match {pred_or_npy} y_pred"

	task_id = np.full(y_true.shape[1], "", dtype=np.dtype('U30'))
	assay_type = np.full(y_true.shape[1], "", dtype=np.dtype('U30'))
	task_size	= np.full(y_true.shape[1], np.nan)
	aucpr	= np.full(y_true.shape[1], np.nan)
	logloss = np.full(y_true.shape[1], np.nan)
	aucroc  = np.full(y_true.shape[1], np.nan)
	maxf1	= np.full(y_true.shape[1], np.nan)
	kappa	= np.full(y_true.shape[1], np.nan)
	brier	= np.full(y_true.shape[1], np.nan)
	tn	 = np.full(y_true.shape[1], np.nan)
	fp	 = np.full(y_true.shape[1], np.nan)
	fn	 = np.full(y_true.shape[1], np.nan)
	tp	 = np.full(y_true.shape[1], np.nan)
	if args.use_venn_abers: vennabers = np.full(y_true.shape[1], np.nan)

	num_pos = (y_true == +1).sum(0)
	num_neg = (y_true == -1).sum(0)
	cols55  = np.array((num_pos >= args.min_size) & (num_neg >= args.min_size)).flatten()
	for col in range(y_true.shape[1]):
		y_true_col = y_true.data[y_true.indptr[col] : y_true.indptr[col+1]] == 1
		y_pred_col = y_pred.data[y_pred.indptr[col] : y_pred.indptr[col+1]]
		y_true_col, y_pred_col = y_true_col.astype(np.uint8), y_pred_col.astype('float32')

		#check y_true_col
		if y_true_col.shape[0] <= 1: continue
		if (y_true_col[0] == y_true_col).all(): continue
		if args.use_venn_abers:
			pts = np.vstack((y_pred_col, y_true_col)).T # points for Venn-ABERS
			pts[:,1]=(pts[:,1]==1).astype(np.uint8)
		task_id[col] = tasks_table["classification_task_id"][tasks_table["cont_classification_task_id"]==col].iloc[0]
		assay_type[col] = tasks_table["assay_type"][tasks_table["cont_classification_task_id"]==col].iloc[0]
		task_size[col] = len(y_true_col)
		y_classes	= np.where(y_pred_col > 0.5, 1, 0).astype(np.uint8)
		precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true_col, probas_pred = y_pred_col)
		aucpr[col]  = sklearn.metrics.auc(x = recall, y = precision)
		#logloss must be float64 to avoid issues with nans in output (e.g. https://stackoverflow.com/questions/50157689/)
		logloss[col]  = sklearn.metrics.log_loss(y_true=y_true_col.astype("float64"), y_pred=y_pred_col.astype("float64"))
		aucroc[col] = sklearn.metrics.roc_auc_score(y_true=y_true_col, y_score=y_pred_col)
		maxf1[col]  = find_max_f1(precision, recall)
		kappa[col]  = sklearn.metrics.cohen_kappa_score(y_true_col, y_classes)
		brier[col] = sklearn.metrics.brier_score_loss(y_true=y_true_col, y_prob=y_pred_col)
		tn[col], fp[col], fn[col], tp[col] = sklearn.metrics.confusion_matrix(y_true = y_true_col, y_pred = y_classes).ravel()
		##per-task performance:
		if args.use_venn_abers:
			cols = ['classification_task_id', 'assay_type', 'task_size', 'aucpr','aucroc','logloss','maxf1','kappa','vennabers','brier','tn','fp','fn','tp']
			#try vennabers, except if #calibration-pts lower than #cv-splits
			try: vennabers[col] = np.median(get_VA_margin_cross(pts)[-1])
			except ValueError: pass
			local_performance=pd.DataFrame(np.array([task_id[cols55],assay_type[cols55],task_size[cols55],aucpr[cols55],aucroc[cols55],logloss[cols55],maxf1[cols55],\
				kappa[cols55],vennabers[cols55],brier[cols55],tn[cols55],fp[cols55],fn[cols55],tp[cols55]]).T, columns=cols)
		else:
			cols = ['classification_task_id', 'assay_type', 'task_size', 'aucpr','aucroc','logloss','maxf1','kappa','brier','tn','fp','fn','tp']
			local_performance=pd.DataFrame(np.array([task_id[cols55],assay_type[cols55],task_size[cols55],aucpr[cols55],aucroc[cols55],logloss[cols55],maxf1[cols55],\
				kappa[cols55],brier[cols55],tn[cols55],fp[cols55],fn[cols55],tp[cols55]]).T, columns=cols)
	##correct the datatypes for numeric columns
	for c in local_performance.iloc[:,2:].columns:
		local_performance.loc[:,c] = local_performance.loc[:,c].astype('float32')
	##write per-task & per-assay_type performance:
	write_aggregated_report(local_performance, flabel, name, task_map)

	##global aggregation:
	if args.task_weights: tw_weights=tw_df['weight'].values[cols55]
	else: tw_weights=tw_df[cols55]
	aucpr_mean  = np.average(aucpr[cols55],weights=tw_weights)
	aucroc_mean = np.average(aucroc[cols55],weights=tw_weights)
	logloss_mean = np.average(logloss[cols55],weights=tw_weights)
	maxf1_mean  = np.average(maxf1[cols55],weights=tw_weights)
	kappa_mean  = np.average(kappa[cols55],weights=tw_weights)
	brier_mean  = np.average(brier[cols55],weights=tw_weights)
	tn_sum = tn[cols55].sum()
	fp_sum = fp[cols55].sum()
	fn_sum = fn[cols55].sum()
	tp_sum = tp[cols55].sum()

	if args.use_venn_abers:
		vennabers_mean  = np.average(vennabers[cols55],weights=tw_weights)
		global_performance = write_global_report([aucpr_mean,aucroc_mean,logloss_mean,maxf1_mean,kappa_mean,vennabers_mean,brier_mean,tn_sum,fp_sum,fn_sum,tp_sum], flabel, name)
	else: global_performance = write_global_report([aucpr_mean,aucroc_mean,logloss_mean,maxf1_mean,kappa_mean,brier_mean,tn_sum,fp_sum,fn_sum,tp_sum], flabel, name)
	return [local_performance,global_performance]

## calculate the difference between the single- and multi-pharma outputs and write to a file
def calculate_deltas(f1_results, f2_results, name, task_map):
	for idx, delta_comparison in enumerate(['locals','/deltas_global_performances.csv']):
		assert f1_results[idx].shape[0] == f2_results[idx].shape[0], "the number of tasks are not equal between the --f1 and --f2 outputs for {delta_comparison}"
		assert f1_results[idx].shape[1] == f2_results[idx].shape[1], "the number of reported metrics are not equal between the --f1 and --f2 outputs for {delta_comparison}"
		# add assay aggregation if local
		if(delta_comparison == 'locals'):
			cti = f2_results[idx]["classification_task_id"]
			at = f2_results[idx]["assay_type"]
			delta = (f2_results[idx].loc[:, "aucpr":]-f1_results[idx].loc[:, "aucpr":])
			tdf = pd.concat([cti, at, delta], axis = 1)
			fn1 = name + '/deltas_per-task_performances.csv'
			pertask = tdf[:]
			pertask['classification_task_id'] = pertask['classification_task_id'].astype('int32')
			#add per-task perf aggregated performance delta bins to output
			for metric in pertask.loc[:, "aucpr":"brier"].columns:
				pertask[f'{metric}_percent'] = cut(pertask[metric].astype('float32'), \
				args.aggr_binning_scheme_perf_delta,include_lowest=True,right=True)
			pertask = pertask.merge(task_map, right_on=["classification_task_id","assay_type"], left_on=["classification_task_id","assay_type"], how="left")
			#write per-task perf aggregated performance delta
			pertask.to_csv(fn1, index= False)
			vprint(f"Wrote per-task delta report to: {fn1}")
			
			#write binned per-task aggregated performance deltas
			agg_deltas=[]
			for metric_bin in pertask.loc[:, "aucpr_percent":"brier_percent"].columns:
				agg_perf=(pertask.groupby(metric_bin)['classification_task_id'].agg('count')/len(pertask)).reset_index().rename(columns={'classification_task_id': f'bin_{metric_bin}'})
				agg_deltas.append(agg_perf.set_index(metric_bin))
			if args.pharma_name: fnagg = name + f"/deltas_binned_{args.pharma_name}_per-task_performances.csv"
			else: fnagg = name + "/deltas_binned_per-task_performances.csv"
			pd.concat(agg_deltas,axis=1).astype(np.float32).reset_index().rename(columns={'index': 'perf_bin'}).to_csv(fnagg,index=False)
			vprint(f"Wrote binned performance per-task delta report to: {fnagg}")

			# aggregate on assay_type level
			fn2 = name + '/deltas_per-assay_performances.csv'
			tdf.groupby("assay_type").mean().to_csv(fn2)
			vprint(f"Wrote per-assay delta report to: {fn2}")

			#write binned per-assay aggregated performance deltas
			agg_deltas2=[]
			for metric_bin in pertask.loc[:, "aucpr_percent":"brier_percent"].columns:
				#pertask[metric_bin]=pertask[metric_bin].astype("|S6")
				agg_perf2=(pertask.groupby(['assay_type',metric_bin])['classification_task_id'].agg('count')).reset_index().rename(columns={'classification_task_id': f'count_{metric_bin}'})
				agg_perf2[f'bin_{metric_bin}']=agg_perf2.apply(lambda x : x[f'count_{metric_bin}'] / (pertask.assay_type==x['assay_type']).sum() ,axis=1).astype('float32')
				agg_perf2.drop(f'count_{metric_bin}',axis=1,inplace=True)
				agg_deltas2.append(agg_perf2.set_index(['assay_type',metric_bin]))
			if args.pharma_name: fnagg2 = name + f"/deltas_binned_{args.pharma_name}_per-assay_performances.csv"
			else: fnagg2 = name + "/deltas_binned_per-assay_performances.csv"	
			pd.concat(agg_deltas2,axis=1).astype(np.float32).reset_index().rename(columns={'level_0':'assay_type','level_1':'perf_bin',}).to_csv(fnagg2,index=False)
			vprint(f"Wrote binned performance per-assay delta report to: {fnagg}")
		else:
			(f2_results[idx]-f1_results[idx]).to_csv(name + delta_comparison, index=False)

def main(args):
	vprint(args)
	y_pred_f1_path = Path(args.f1)
	y_pred_f2_path = Path(args.f2)

	assert all([(p.suffix == '.npy') or (p.stem in ['pred']) for p in [y_pred_f1_path, y_pred_f2_path]]), "Pediction files need to be either 'pred' or '*.npy'"
	task_map = pd.read_csv(args.task_map)

	if args.filename is not None:
		name = args.filename
	else:
		name = f"{os.path.basename(args.y_true_all)}_{os.path.basename(args.f1)}_{os.path.basename(args.f2)}_{os.path.basename(args.folding)}"
		vprint(f"\nRun name is '{name}'\n")
	assert not os.path.exists(name), f"{name} already exists... exiting"
	os.makedirs(name)
	with open(f'{name}/run_params.json', 'wt') as f:
		json.dump(vars(args), f, indent=4)
		vprint(f"Wrote input params to '{name}/run_params.json'\n")

	#load the folding/true data
	folding = np.load(args.folding)
	vprint(f'Loading y_true: {args.y_true_all}')
	try: y_true_all = np.load(args.y_true_all, allow_pickle=True).item()
	except AttributeError: 
		from scipy.io import mmread 
		y_true_all = mmread(args.y_true_all)
	y_true_all = y_true_all.tocsc()
	## filtering out validation fold
	fold_va = args.validation_fold
	y_true = y_true_all[folding == fold_va]
	y_true_all = None #clear all ytrue to save memory

	## Loading task weights (ported from WP2 sparse chem pred.py code)
	if args.task_weights is not None:
		tw_df = pd.read_csv(args.task_weights)
		assert "task_id" in tw_df.columns, "task_id is missing in task weights CVS file"
		assert tw_df.shape[1] == 2, "task weight file (CSV) must only have 2 columns"
		assert "weight" in tw_df.columns, "weight is missing in task weights CVS file"
		assert y_true.shape[1] == tw_df.shape[0], "task weights have different size to y columns."
		assert (0 <= tw_df.weight).all(), "task weights must not be negative"
		assert (tw_df.weight <= 1).all(), "task weights must not be larger than 1.0"
		assert tw_df.task_id.unique().shape[0] == tw_df.shape[0], "task ids are not all unique"
		assert (0 <= tw_df.task_id).all(), "task ids in task weights must not be negative"
		assert (tw_df.task_id < tw_df.shape[0]).all(), "task ids in task weights must be below number of tasks"
		assert tw_df.shape[0]==y_true.shape[1], f"The number of task weights ({tw_df.shape[0]}) must be equal to the number of columns in Y ({y_true.shape[1]})."
		tw_df.sort_values("task_id", inplace=True)
	else:
		## default weights are set to 1.0
		tw_df = np.ones(y_true.shape[1], dtype=np.uint8)

	f1_yhat, f2_yhat, f1_ftype, f2_ftype = mask_y_hat(y_pred_f1_path,y_pred_f2_path, folding, fold_va, y_true)
	folding, fold_va = None, None #clear folding - no longer needed

	vprint(f"\nCalculating '{args.f1}' performance")
	f1_yhat_results = per_run_performance(f1_yhat, f1_ftype, task_map, y_true, tw_df, task_map, name, 'f1')
	f1_yhat = None #clear yhat from memory - no longer needed

	vprint(f"\nCalculating '{args.f2}' performance")
	f2_yhat_results = per_run_performance(f2_yhat, f2_ftype, task_map, y_true, tw_df, task_map, name, 'f2')
	f2_yhat=None #clear yhat from memory - no longer needed

	y_true, tw_df = None, None #clear ytrue and weights - no longer needed
	vprint(f"\nCalculating delta between '{args.f1}' and '{args.f2}' performances")
	calculate_deltas(f1_yhat_results,f2_yhat_results, name, task_map)
	vprint(f"\nRun name '{name}' is finished.")
	return

if __name__ == '__main__':
	start = time.time()
	args = init_arg_parser()
	vprint('\n=== WP3 Performance evaluation script for npy and pred files ===\n')
	main(args)
	end = time.time()
	vprint(f'Performance evaluation took {end - start:.08} seconds.')

import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import sklearn.metrics
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.metrics import confusion_matrix, roc_curve
import json
import scipy.sparse as sparse
from scipy.stats import spearmanr
from  pathlib import Path
from scipy.io import mmread
import sparsechem as sc
import significance_analysis
		
def init_arg_parser():
	parser = argparse.ArgumentParser(description="MELLODDY Performance Evaluation: release-2-(classification-and-regression)")
	parser.add_argument("--y_cls", help="Classification activity file (npz) (e.g. cls_T10_y.npz)", type=str, default=None)
	parser.add_argument("--y_clsaux", help="Aux classification activity file (npz) (e.g. cls_T10_y.npz)", type=str, default=None)
	parser.add_argument("--y_regr", help="Activity file (npz) (e.g. reg_T10_y.npz)", type=str, default=None)
	parser.add_argument("--y_censored_regr", help="Censored activity file (npz) (e.g. reg_T10_censor_y.npz)", type=str, default=None)

	parser.add_argument("--y_cls_single_partner", help="Yhat cls prediction output from an single-partner run (e.g. <single pharma dir>/<cls_prefix>-class.npy)", type=str, default=None)
	parser.add_argument("--y_clsaux_single_partner", help="Yhat clsaux prediction from an single-partner run (e.g. <single pharma dir>/<clsaux_prefix>-class.npy)", type=str, default=None)
	parser.add_argument("--y_regr_single_partner", help="Yhat regr prediction from an single-partner run (e.g. <single pharma dir>/<regr_prefix>-regr.npy)", type=str, default=None)

	parser.add_argument("--y_cls_multi_partner", help="Classification prediction output for comparison (e.g. pred from the multi-partner run)", type=str, default=None)
	parser.add_argument("--y_clsaux_multi_partner", help="Classification w/ aux prediction output for comparison (e.g. pred from the multi-partner run)", type=str, default=None)
	parser.add_argument("--y_regr_multi_partner", help="Regression prediction output for comparison (e.g. pred from the multi-partner run)", type=str, default=None)

	parser.add_argument("--folding_cls", help="Folding file (npy) (e.g. cls_T11_fold_vector.npy)", type=str, default=None)
	parser.add_argument("--folding_clsaux", help="Folding file (npy) (e.g. cls_T11_fold_vector.npy)", type=str, default=None)
	parser.add_argument("--folding_regr", help="Folding file (npy) (e.g. reg_T11_fold_vector.npy)", type=str, default=None)

	parser.add_argument("--t8c_cls", help="T8c file for classification in the results_tmp/classification folder", type=str, default=None)
	parser.add_argument("--t8c_clsaux", help="T8c file for classification w/ auxiliary in the results_tmp/classification folder", type=str, default=None)
	parser.add_argument("--t8r_regr", help="T8r file for regression in the results_tmp/regression folder", type=str, default=None)

	parser.add_argument("--weights_cls", help="CSV file with columns task_id and weight (e.g. cls_weights.csv)", type=str, default=None)
	parser.add_argument("--weights_clsaux", help="CSV file with columns task_id and weight (e.g cls_weights.csv)", type=str, default=None)
	parser.add_argument("--weights_regr", help="CSV file with columns task_id and weight (e.g. reg_weights.csv)", type=str, default=None)

	parser.add_argument("--run_name", help="Run name directory for results from this output (timestemp used if not specified)", type=str, default=None)
	parser.add_argument("--significance_analysis", help="Run significant analysis (1 = Yes, 0 = No sig. analysis", type=int, default=1, choices=[0, 1])
	parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])
	parser.add_argument("--validation_fold", help="Validation fold to used to calculate performance", type=int, default=[0], nargs='+', choices=[0, 1, 2, 3, 4])
	parser.add_argument("--aggr_binning_scheme_perf", help="Shared aggregated binning scheme for performances", type=str, nargs='+', default=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],required=False)
	parser.add_argument("--aggr_binning_scheme_perf_delta", help="Shared aggregated binning scheme for delta performances", type=str, nargs='+', default=[-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2],required=False)
	args = parser.parse_args()
	assert len(args.aggr_binning_scheme_perf) == 11, f"len of aggr_binning_scheme_perf should be 11, got {len(args.aggr_binning_scheme_perf)}"
	assert len(args.aggr_binning_scheme_perf_delta) == 9, f"len of aggr_binning_scheme_perf_delta should be 9, got {len(args.aggr_binning_scheme_perf_delta)}"
	assert Path()
	args.aggr_binning_scheme_perf=list(map(np.float,args.aggr_binning_scheme_perf))
	args.aggr_binning_scheme_perf_delta=list(map(np.float,args.aggr_binning_scheme_perf_delta))
	return args


def vprint(s="", model_category=False, vtype='INFO'):
	separator = '='*135
	if args.verbose:
		print()
		if model_category:
			print(f'{separator}\n{s}\n{separator}')
		else: print(f'[{vtype}]: {s}')


def cut(x, bins, lower_infinite=True, upper_infinite=True, **kwargs):		
	"""
	Custom cut function to cut reported performances via common binning scheme
	"""
	num_labels	  = len(bins) - 1
	include_lowest  = kwargs.get("include_lowest", False)
	right			= kwargs.get("right", True)
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
			if lower_infinite: new_label = "{0} {1}".format(symbol_lower, bins_final[i+1])
			elif include_lowest: new_label = make_label(i, lb="[")
			else: new_label = make_label(i)
		elif upper_infinite and i == (num_labels - 1): new_label = "{0} {1}".format(symbol_upper, bins_final[i])
		else: new_label = make_label(i)
		labels.append(new_label)
	return pd.cut(x, bins_final, labels=labels, **kwargs)


def getheader(run_type):
	"""
	Set the classification or regression header for pandas etc. to use
	"""
	if run_type in ['cls','clsaux']: return 'classification'
	else: return 'regression'


def validate_cls_clsuax_regr_inputs(args):
	"""
	Check required files for cls, clsaux or regr are supplied
	"""
	assert args.y_cls or args.y_clsaux or args.y_regr, "Must provide y_cls, y_cls_aux or y_regr"
	if args.y_cls:
		assert all([args.y_cls_single_partner,args.y_cls_multi_partner,args.t8c_cls,args.weights_cls]), "Must provide y_cls_single_partner, y_cls_multi_partner, t8c_cls, weights_cls when supplying y_cls"
		if not Path(args.y_cls_single_partner).stem == 'pred' and Path(args.y_cls_multi_partner).stem == 'pred':
			vprint("Single- and multi-partner cls prediction files are exptected to be 'pred', continuing anyway...",vtype='WARNING')
	if args.y_clsaux:
		assert all([args.y_clsaux_single_partner,args.y_clsaux_multi_partner,args.t8c_clsaux,args.weights_clsaux]), "Must provide y_clsaux_single_partner, y_clsaux_multi_partner, t8c_clsaux, weights_clsaux when supplying y_clsaux"
		if not Path(args.y_clsaux_single_partner).stem == 'pred' and Path(args.y_clsaux_multi_partner).stem == 'pred':
			vprint("Single- and multi-partner clsaux prediction files are exptected to be 'pred', continuing anyway...",vtype='WARNING')
	if args.y_regr:
		assert all([args.y_regr_single_partner,args.y_regr_multi_partner,args.t8r_regr,args.weights_regr]), "Must provide y_regr_single_partner, y_regr_multi_partner, t8r_regr, weights_regr when supplying y_regr"
		if not Path(args.y_regr_single_partner).stem == 'pred' and Path(args.y_regr_multi_partner).stem == 'pred':
			vprint("Single- and multi-partner reg prediction files are exptected to be 'pred', continuing anyway...",vtype='WARNING')
	return


def validate_ytrue_ypred(y_true, y_pred, pred_or_npy):
	assert y_true.shape == y_pred.shape, f"y_true shape do not match {pred_or_npy} y_pred ({y_true.shape} & {y_pred.shape})"
	assert y_true.nnz == y_pred.nnz, f"y_true number of nonzero values do not match {pred_or_npy} y_pred"
	assert (y_true.indptr == y_pred.indptr).all(), f"y_true indptr do not match {pred_or_npy} y_pred"
	assert (y_true.indices == y_pred.indices).all(), f"y_true indices do not match {pred_or_npy} y_pred"
	return
	

def load_yhats(input_f, folding, fold_va, y_true):
	"""
	Load and mask yhats for those considered for evaluation 
	"""
	# load the data
	if input_f.suffix == '.npy':
		vprint(f'Loading (npy) predictions for: {input_f}') 
		yhats = np.load(input_f, allow_pickle=True).item().tocsr().astype('float64')
		ftype = 'npy'
	else:
		vprint(f'Loading (pred) output for: {input_f}') 
		yhats = torch.load(input_f).astype('float64')
		ftype = 'pred'
	# mask validation fold if possible
	try: yhats = yhats[[i in fold_va for i in folding]]
	except IndexError: pass
	return yhats, ftype


def mask_y_hat(f1_path, f2_path, folding, fold_va, y_true, header_type):
	"""
	Load yhats using load_yhats and validate the shapes
	"""
	if header_type == 'classification': true_data = y_true.astype(np.uint8).todense()
	else: true_data = y_true.astype('float64').todense()
	f1_yhat, f1_ftype = load_yhats(f1_path, folding, fold_va, y_true)
	assert true_data.shape == f1_yhat.shape, f"True shape {true_data.shape} and {f1_path.stem} shape {f1_yhat.shape} need to be identical"
	f2_yhat, f2_ftype = load_yhats(f2_path, folding, fold_va, y_true)
	assert true_data.shape == f2_yhat.shape, f"True shape {true_data.shape} and {f2_path.stem} shape {f2_yhat.shape} need to be identical"
	return [f1_yhat, f2_yhat, f1_ftype, f2_ftype]


def mask_ytrue(ftype,fname,folding,fold_va):
	"""
	Mask the y-input labels for the validation fold
	"""
	vprint(f'Loading {ftype}: {fname}')
	try: y_all = sc.load_sparse(fname)
	except AttributeError: 
		y_all = mmread(fname)
	y_all = y_all.tocsc()
	y_all_true = y_all[[i in fold_va for i in folding]]
	return y_all_true


def check_weights(tw_df, y_true, header_type):
	"""
	Validate task weight file
	"""
	if header_type == 'classification': tw_dfshape = 4
	else: tw_dfshape = 5
	assert "task_id" in tw_df.columns, "task_id is missing in task weights CVS file"
	assert tw_df.shape[1] == tw_dfshape, "task weight file (CSV) must only have 4 columns"
	assert "training_weight" in tw_df.columns, "weight is missing in task weights CVS file"
	assert "aggregation_weight" in tw_df.columns, "weight is missing in task weights CVS file"
	assert y_true.shape[1] == tw_df.shape[0], "task weights have different size to y columns."
	assert (0 <= tw_df.training_weight).all(), "task weights must not be negative"
	assert (tw_df.training_weight <= 1).all(), "task weights must not be larger than 1.0"
	assert tw_df.task_id.unique().shape[0] == tw_df.shape[0], "task ids are not all unique"
	assert (0 <= tw_df.task_id).all(), "task ids in task weights must not be negative"
	assert (tw_df.task_id < tw_df.shape[0]).all(), "task ids in task weights must be below number of tasks"
	assert tw_df.shape[0]==y_true.shape[1], f"The number of task weights ({tw_df.shape[0]}) must be equal to the number of columns in Y ({y_true.shape[1]})."
	return

def run_significance_calculation(run_type, y_pred0, y_pred1, y_true, task_map):
	"""
	Calculate significance between runs and bin individual performance reports including aggregation by assay/globally
	"""
	header_type = getheader(run_type)
	if header_type != 'classification': return None
	vprint(f"=== Calculating significance ===")
	y_pred0 = sparse.csc_matrix(y_pred0)
	y_pred1 = sparse.csc_matrix(y_pred1)
	calculated_sig = pd.DataFrame()
	for col_idx, col in enumerate(range(y_true.shape[1])):
		try: task_id = task_map.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1')[f"{header_type}_task_id"][task_map[f"cont_{header_type}_task_id"]==col].iloc[0]
		except IndexError: continue
		y_pred_col0 = (y_pred0.data[y_pred0.indptr[col] : y_pred0.indptr[col+1]]).astype(np.float64)
		y_pred_col1 = (y_pred1.data[y_pred1.indptr[col] : y_pred1.indptr[col+1]]).astype(np.float64)
		y_true_col = (y_true.data[y_true.indptr[col] : y_true.indptr[col+1]] == 1).astype(np.uint8)
		details = pd.DataFrame({f'{header_type}_task_id': pd.Series(task_id, dtype='int32')})
		if y_true_col.shape[0] <= 1: continue
		if (y_true_col[0] == y_true_col).all(): continue
		with np.errstate(divide='ignore',invalid='ignore'):
			sp_sig = significance_analysis.test_significance(y_true_col, y_pred_col1, y_pred_col0, level=0.05).rename(columns={'significant': 'sp_significant','p_value' : 'sp_pvalue'})
			mp_sig = significance_analysis.test_significance(y_true_col, y_pred_col0, y_pred_col1, level=0.05).rename(columns={'significant': 'mp_significant','p_value' : 'mp_pvalue'})
		sp_mp_sig = pd.concat([details,sp_sig,mp_sig],axis=1)
		calculated_sig = pd.concat([calculated_sig, sp_mp_sig],axis=0)
	return calculated_sig

def run_performance_calculation(run_type, y_pred, pred_or_npy, y_true, tw_df, task_map, run_name, flabel, rlabel, y_true_cens = None):
	"""
	Calculate performance for one run, bin results and then individual performance reports including aggregation by assay/globally
	"""
	vprint(f"=== Calculating {flabel} performance ===")
	flabel = Path(flabel).stem
	header_type = getheader(run_type)
	y_pred = sparse.csc_matrix(y_pred)
	if header_type == 'classification':
		sc_columns = sc.utils.all_metrics([0],[0]).columns.tolist()  #get the names of reported metrics from the sc utils
		sc_columns.extend(['s_auc_pr', 'calibrated_auc_pr', 'positive_rate']) # added for calibrated auc_pr
		sc_columns.extend(['tnr', 'fpr', 'fnr', 'tpr'])
		sc_columns.extend(['max_tnr', 'max_fpr', 'max_fnr', 'max_tpr'])
	else:
		sc_columns = sc.utils.all_metrics_regr([0],[0]).columns.tolist()  #get the names of reported metrics from the sc utils
	validate_ytrue_ypred(y_true, y_pred, pred_or_npy) # checks to make sure y_true and y_pred match
	if y_true_cens is not None: validate_ytrue_ypred(y_true_cens, y_pred, pred_or_npy)  # checks to make sure y_cens and y_pred match
	calculated_performance = pd.DataFrame()
	for col_idx, col in enumerate(range(y_true.shape[1])):
		task_id = task_map[f"{header_type}_task_id"][task_map[f"cont_{header_type}_task_id"]==col].iloc[0]
		y_pred_col = (y_pred.data[y_pred.indptr[col] : y_pred.indptr[col+1]])
		
		#setup for classification metrics
		if header_type == 'classification':
			y_true_col = (y_true.data[y_true.indptr[col] : y_true.indptr[col+1]] == 1)
			sc_calculation = sc.utils.all_metrics(y_true_col,y_pred_col)

			# calculate positive_rate and calibrated auc_pr
			if (len(y_true_col) > 0) and not (y_true_col[0] == y_true_col).all():
				try: positive_rate_for_col = np.sum(y_true_col) / len(y_true_col)
				except ZeroDivisionError:
					positive_rate_for_col = 0
				s_auc_pr = calculate_s_auc_pr(sc_calculation.auc_pr[0], y_true_col, positive_rate_for_col)
				c_auc_pr = calculate_calibrated_auc_pr(y_true_col, y_pred_col, pi0=0.5)
				tn, fp, fn, tp = confusion_matrix(y_true_col,y_pred_col>0.5).ravel()

				fprs, tprs, thresholds = roc_curve(y_true_col, y_pred_col)
				gmeans = np.sqrt(tprs * (1-fprs))
				ix = np.argmax(gmeans)
				mtn, mfp, mfn, mtp = confusion_matrix(y_true_col,y_pred_col>thresholds[ix]).ravel()
				sc_calculation['s_auc_pr'] = [s_auc_pr]
				sc_calculation['calibrated_auc_pr'] = [c_auc_pr]
				sc_calculation['positive_rate'] = [positive_rate_for_col]
				sc_calculation['tnr'] = tn/(tn+fp)
				sc_calculation['fpr'] = fp/(fp+tn)
				sc_calculation['fnr'] = fn/(fn+tp)
				sc_calculation['tpr'] = tp/(tp+fn)
				sc_calculation['max_tnr'] = mtn/(mtn+mfp)
				sc_calculation['max_fpr'] = mfp/(mfp+mtn)
				sc_calculation['max_fnr'] = mfn/(mfn+mtp)
				sc_calculation['max_tpr'] = mtp/(mtp+mfn)
		#setup for regression metrics
		else:
			y_true_col = (y_true.data[y_true.indptr[col] : y_true.indptr[col+1]])
			if y_true_cens is not None: y_censor = (y_true_cens.data[y_true_cens.indptr[col] : y_true_cens.indptr[col+1]])
			else: y_censor = None
			sc_calculation = sc.utils.all_metrics_regr(y_true_col,y_pred_col,y_censor=y_censor)
		details = pd.DataFrame({f'{header_type}_task_id': pd.Series(task_id, dtype='int32'),
								'task_size': pd.Series(len(y_true_col), dtype='int32')})

		if y_true_col.shape[0] <= 1: continue
		if (y_true_col[0] == y_true_col).all(): continue
		sc_metrics = pd.concat([details,sc_calculation],axis=1)
		calculated_performance = pd.concat([calculated_performance, sc_metrics],axis=0)	
	#merge calculated performances with the details of the tasks
	calculated_performance = calculated_performance.merge(task_map, left_on=f'{header_type}_task_id', right_on=f'{header_type}_task_id',how='left')
	##write per-task & per-assay_type performance:
	write_aggregated_report(run_name, run_type, flabel, calculated_performance, sc_columns, header_type, rlabel)
	##global aggregation:
	globally_calculated = write_global_report(run_name, run_type, flabel, calculated_performance, sc_columns, rlabel)
	return calculated_performance, sc_columns


def calculate_delta(f1_results, f2_results, run_name, run_type, sc_columns, header_type, sig = None):
	"""
	Calculate the delta between the outputs and write to a file
	"""
	f1_results = f1_results.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1')
	f2_results = f2_results.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1')
	assert f1_results.shape[0] == f2_results.shape[0], "the number of tasks are not equal between the outputs for comparison}"
	assert f1_results.shape[1] == f2_results.shape[1], "the number of reported metrics are not equal between the outputs for comparison"
	header_type = getheader(run_type)
	task_id = f2_results[f"{header_type}_task_id"]
	at = f2_results["assay_type"]
	delta = (f2_results.loc[:, sc_columns[0]:sc_columns[-1]]-f1_results.loc[:, sc_columns[0]:sc_columns[-1]])
	tdf = pd.concat([task_id, at, delta], axis = 1)
	os.makedirs(f"{run_name}/{run_type}/deltas/")
	fn1 = f"{run_name}/{run_type}/deltas/deltas_per-task_performances.csv"
	pertask = tdf.copy()
	pertask.loc[:,f'{header_type}_task_id'] = pertask[f'{header_type}_task_id'].astype('int32')
	#add per-task perf aggregated performance delta bins to output
	for metric in pertask.loc[:, sc_columns[0]:sc_columns[-1]].columns:
		pertask.loc[:,f'{metric}_percent'] = cut(pertask[metric].astype('float64'), \
		args.aggr_binning_scheme_perf_delta,include_lowest=True,right=True)
	#merge calculated significances (if set) with the calculated performances
	if sig is not None: pertask = pertask.merge(sig, left_on=f'{header_type}_task_id', right_on=f'{header_type}_task_id',how='left')
	#write per-task perf aggregated performance delta
	pertask.to_csv(fn1, index= False)
	vprint(f"Wrote per-task delta report to: {fn1}")
	
	#write binned per-task aggregated performance deltas
	agg_deltas=[]
	for metric_bin in pertask.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
		agg_perf=(pertask.groupby(metric_bin)[f'{header_type}_task_id'].agg('count')/len(pertask)).reset_index().rename(columns={f'{header_type}_task_id': f'bin_{metric_bin}'})
		agg_deltas.append(agg_perf.set_index(metric_bin))
	fnagg = f"{run_name}/{run_type}/deltas/deltas_binned_per-task_performances.csv"
	pd.concat(agg_deltas,axis=1).astype(np.float64).reset_index().rename(columns={'index': 'perf_bin'}).to_csv(fnagg,index=False)
	vprint(f"Wrote binned performance per-task delta report to: {fnagg}")

	# aggregate on assay_type level
	fn2 = f"{run_name}/{run_type}/deltas/deltas_per-assay_performances.csv"
	per_assay_delta=tdf[['assay_type'] + sc_columns].groupby("assay_type").mean()
	per_assay_delta.to_csv(fn2)
	vprint(f"Wrote per-assay delta report to: {fn2}")

	#write binned per-assay aggregated performance deltas
	agg_deltas2=[]
	for metric_bin in pertask.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
		agg_perf2=(pertask.groupby(['assay_type',metric_bin])[f'{header_type}_task_id'].agg('count')).reset_index().rename(columns={f'{header_type}_task_id': f'count_{metric_bin}'})
		agg_perf2[f'bin_{metric_bin}']=agg_perf2.apply(lambda x : x[f'count_{metric_bin}'] / (pertask.assay_type==x['assay_type']).sum() ,axis=1).astype('float64')
		agg_perf2.drop(f'count_{metric_bin}',axis=1,inplace=True)
		agg_deltas2.append(agg_perf2.set_index(['assay_type',metric_bin]))
	fnagg2 = f"{run_name}/{run_type}/deltas/deltas_binned_per-assay_performances.csv"	
	pd.concat(agg_deltas2,axis=1).astype(np.float64).reset_index().rename(columns={f'{sc_columns[0]}_percent':'perf_bin',}).to_csv(fnagg2,index=False)
	vprint(f"Wrote binned performance per-assay delta report to: {fnagg}")

	#write globally aggregated performance deltas	
	global_delta = pd.DataFrame(f2_results[sc_columns].mean(axis=0)).T - pd.DataFrame(f1_results[sc_columns].mean(axis=0)).T
	global_delta.to_csv(f"{run_name}/{run_type}/deltas/deltas_global_performances.csv", index=False)

	#if significance flag was set then perform that analysis here
	if sig is not None:
		for p in ['sp','mp']:
			#write binned per-task significance
			agg_concat=[]
			agg_perf=(pertask.groupby(f'{p}_significant')[f'{header_type}_task_id'].agg('count')/len(pertask)).reset_index().rename(columns={f'{header_type}_task_id': f'percent_{p}_significant'})
			agg_concat.append(agg_perf.set_index(f'{p}_significant'))
			fnagg = f"{run_name}/{run_type}/deltas/delta_{p}_significance.csv"
			pd.concat(agg_concat,axis=1).astype(np.float64).reset_index().to_csv(fnagg,index=False)
			vprint(f"Wrote {p} significance report to: {fnagg}")
		
			#write assay_type significance
			agg_concat2=[]
			agg_perf2=(pertask.groupby(['assay_type',f'{p}_significant'])[f'{header_type}_task_id'].agg('count')).reset_index().rename(columns={f'{header_type}_task_id': f'count_{p}_significant'})
			agg_perf2.loc[:,f'percent_{p}_significant']=agg_perf2.apply(lambda x : x[f'count_{p}_significant'] / (pertask.assay_type==x['assay_type']).sum() ,axis=1).astype('float64')
			agg_perf2.drop(f'count_{p}_significant',axis=1,inplace=True)
			agg_concat2.append(agg_perf2.set_index(['assay_type',f'{p}_significant']))
			fnagg2 = f"{run_name}/{run_type}/deltas/delta_per-assay_{p}_significance.csv"
			pd.concat(agg_concat2,axis=1).astype(np.float64).reset_index().to_csv(fnagg2,index=False)
			vprint(f"Wrote per-assay {p} significance report to: {fnagg2}")
	return


def write_global_report(run_name, run_type, fname, local_performances, sc_columns, rlabel):
	"""
	write performance reports for global aggregation
	"""
	df = local_performances.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1').copy()
	df = pd.DataFrame(df[sc_columns].mean(axis=0)).T
	fn1 = f"{run_name}/{run_type}/{rlabel}/{fname}_global_performances.csv"
	df.to_csv(fn1, index= False)
	vprint(f"Wrote global report to: {fn1}")
	return df


def write_aggregated_report(run_name, run_type, fname, local_performances, sc_columns, header_type, rlabel):
	"""
	write performance reports per-task & per-task_assay
	"""
	df = local_performances.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1').copy()
	for metric in df.loc[:, sc_columns[0]:sc_columns[-1]].columns:
		df.loc[:,f'{metric}_percent'] = cut(df[metric].astype('float64'), \
		args.aggr_binning_scheme_perf,include_lowest=True,right=True,lower_infinite=False, upper_infinite=False)
	df.loc[:,f'{header_type}_task_id'] = df[f'{header_type}_task_id'].astype('float').astype('int32')
	os.makedirs(f"{run_name}/{run_type}/{rlabel}/")
	fn1 = f"{run_name}/{run_type}/{rlabel}/{fname}_per-task_performances.csv"
	df.to_csv(fn1, index=False)
	vprint(f"Wrote per-task report to: {fn1}")

	#write binned per-task performances
	agg_concat=[]
	for metric_bin in df.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
		agg_perf=(df.groupby(metric_bin)[f'{header_type}_task_id'].agg('count')/len(df)).reset_index().rename(columns={f'{header_type}_task_id': f'bin_{metric_bin}'})
		agg_concat.append(agg_perf.set_index(metric_bin))
	fnagg = f"{run_name}/{run_type}/{rlabel}/{fname}_binned_per-task_performances.csv"
	pd.concat(agg_concat,axis=1).astype(np.float64).reset_index().rename(columns={'index': 'perf_bin'}).to_csv(fnagg,index=False)
	vprint(f"Wrote per-task binned performance report to: {fnagg}")

	#write performance aggregated performances by assay_type
	df2 = local_performances.query('evaluation_quorum_OK == 1 & is_auxiliary == 0 & aggregation_weight_y == 1').copy()[['assay_type'] + sc_columns]
	df2 = df2.loc[:,'assay_type':].groupby('assay_type').mean()
	fn2 = f"{run_name}/{run_type}/{rlabel}/{fname}_per-assay_performances.csv"
	df2.to_csv(fn2)
	vprint(f"Wrote per-assay report to: {fn2}")

	#write binned perf performances by assay_type 
	agg_concat2=[]
	for metric_bin in df.loc[:, f"{sc_columns[0]}_percent":f"{sc_columns[-1]}_percent"].columns:
		agg_perf2=(df.groupby(['assay_type',metric_bin])[f'{header_type}_task_id'].agg('count')).reset_index().rename(columns={f'{header_type}_task_id': f'count_{metric_bin}'})
		agg_perf2.loc[:,f'bin_{metric_bin}']=agg_perf2.apply(lambda x : x[f'count_{metric_bin}'] / (df.assay_type==x['assay_type']).sum() ,axis=1).astype('float64')
		agg_perf2.drop(f'count_{metric_bin}',axis=1,inplace=True)
		agg_concat2.append(agg_perf2.set_index(['assay_type',metric_bin]))
	fnagg2 = f"{run_name}/{run_type}/{rlabel}/{fname}_binned_per-assay_performances.csv"
	pd.concat(agg_concat2,axis=1).astype(np.float32).reset_index().rename(columns={f'{sc_columns[0]}_percent':'perf_bin',}).to_csv(fnagg2,index=False)
	vprint(f"Wrote per-assay binned report to: {fnagg}")
	return


def calculate_single_partner_multi_partner_results(run_name, run_type, y_true, folding, fold_va, t8, task_weights, single_partner, multi_partner, y_true_cens=None):
	"""
	Calculate cls, clsaux or regr performances for single_partner and multi_partner outputs, then calculate delta, plot outputs along the way
	"""
	header_type = getheader(run_type)
	y_true = mask_ytrue(run_type,y_true,folding,fold_va)
	tw_df = pd.read_csv(task_weights)
	tw_df.sort_values("task_id", inplace=True)
	check_weights(tw_df,y_true,header_type)
	t8 = pd.read_csv(t8) #read t8c or t8r files
	if 'regr' in run_type:
		t8=t8.reset_index().rename(columns={'index': 'regression_task_id'})
		if y_true_cens: y_true_cens = mask_ytrue(run_type,y_true_cens,folding,fold_va)
	task_map = t8.merge(tw_df,left_on=f'cont_{header_type}_task_id',right_on='task_id',how='left').query('task_id.notna()')
	y_single_partner_yhat, y_multi_partner_yhat, y_single_partner_ftype, y_multi_partner_ftype = mask_y_hat(single_partner, multi_partner, folding, fold_va, y_true, header_type)
	if args.significance_analysis: sig = run_significance_calculation(run_type, y_single_partner_yhat, y_multi_partner_yhat, y_true, task_map)
	else: sig = None
	y_single_partner_results, _ = run_performance_calculation(run_type, y_single_partner_yhat, y_single_partner_ftype, y_true, tw_df, task_map, run_name, single_partner,'SP', y_true_cens = y_true_cens)
	del y_single_partner_yhat
	y_multi_partner_results, sc_columns = run_performance_calculation(run_type, y_multi_partner_yhat, y_multi_partner_ftype, y_true, tw_df, task_map, run_name, multi_partner,'MP', y_true_cens = y_true_cens)
	del y_multi_partner_yhat
	calculate_delta(y_single_partner_results, y_multi_partner_results, run_name, run_type, sc_columns, header_type, sig = sig)
	return

def calculate_s_auc_pr(auc_pr, y_true_col, positive_rate_for_col):
	with np.errstate(divide='ignore'):
		try: s_auc_pr = auc_pr ** (np.log10(0.5)/np.log10(positive_rate_for_col))
		except ZeroDivisionError:
			s_auc_pr = auc_pr
	return s_auc_pr

def calculate_calibrated_auc_pr(y_true_col, y_pred_col, pi0=None):
	'''
	Siblini W., Fréry J., He-Guelton L., Oblé F., Wang YQ. (2020) 
	Master Your Metrics with Calibration. 
	In: Berthold M., Feelders A., Krempl G. (eds) 
	Advances in Intelligent Data Analysis XVIII. IDA 2020. 
	Lecture Notes in Computer Science, vol 12080. Springer, Cham
	'''
	fps, tps, thresholds = _binary_clf_curve(y_true_col, y_pred_col)

	if pi0 is not None:
		pi = np.sum(y_true_col) / len(y_true_col)
		ratio = pi*(1-pi0) / (pi0*(1-pi))
		precision = tps / (tps+ratio*fps)
	else:
		precision = tps / (tps+fps)

	precision[np.isnan(precision)] = 0
	recall = tps / tps[-1]

	last_idx = tps.searchsorted(tps[-1])
	sl = slice(last_idx, None, -1)
	precision = np.r_[precision[sl], 1]
	recall = np.r_[recall[sl], 0]
	auc_pr = sklearn.metrics.auc(x=recall, y=precision)
	return auc_pr

def main(args):
	vprint(args)
	validate_cls_clsuax_regr_inputs(args)

	if args.run_name is not None:
		run_name = args.run_name
	else:
		timestr = time.strftime('%Y%m%d-%H%M%S')
		run_name = f"perf_eval_{timestr}"
		vprint(f"\nRun name is '{run_name}'")
	assert not os.path.exists(run_name), f"{run_name} already exists... exiting"
	os.makedirs(run_name)
	with open(f'{run_name}/run_params.json', 'wt') as f:
		json.dump(vars(args), f, indent=4)
		vprint(f"Wrote input params to '{run_name}/run_params.json'\n")

	fold_va = args.validation_fold
	if args.y_cls:
		folding = np.load(args.folding_cls)
		os.makedirs(f"{run_name}/cls")
		vprint(f"Evaluating cls performance", model_category=True)
		calculate_single_partner_multi_partner_results(run_name, 'cls' ,args.y_cls,folding, \
											fold_va, args.t8c_cls, args.weights_cls, \
											Path(args.y_cls_single_partner),Path(args.y_cls_multi_partner))
	if args.y_clsaux:
		folding = np.load(args.folding_clsaux)
		os.makedirs(f"{run_name}/clsaux")
		vprint(f"Evaluating clsaux performance", model_category=True)
		calculate_single_partner_multi_partner_results(run_name, 'clsaux' ,args.y_clsaux, \
											folding,fold_va, args.t8c_clsaux, args.weights_clsaux, \
											Path(args.y_clsaux_single_partner),Path(args.y_clsaux_multi_partner))
	if args.y_regr:
		folding = np.load(args.folding_regr)
		os.makedirs(f"{run_name}/regr")
		vprint(f"Evaluating regr performance", model_category=True)
		calculate_single_partner_multi_partner_results(run_name, 'regr' ,args.y_regr, \
											folding, fold_va, args.t8r_regr, args.weights_regr, \
											Path(args.y_regr_single_partner),Path(args.y_regr_multi_partner))
	if args.y_regr and args.y_censored_regr:
		folding = np.load(args.folding_regr)
		os.makedirs(f"{run_name}/regr_cens")
		vprint(f"Evaluating regr-censored performance", model_category=True)
		calculate_single_partner_multi_partner_results(run_name, 'regr_cens' ,args.y_regr, \
											folding, fold_va, args.t8r_regr, args.weights_regr, \
											Path(args.y_regr_single_partner),Path(args.y_regr_multi_partner), \
											y_true_cens = args.y_censored_regr)
	
	vprint(f"Run name '{run_name}' is finished.")
	return

if __name__ == '__main__':
	start = time.time()
	args = init_arg_parser()
	vprint('=== Performance evaluation script for npy and pred files: release-2-(classification-and-regression) ===')
	main(args)
	end = time.time()
	vprint(f'Performance evaluation took {end - start:.08} seconds.')

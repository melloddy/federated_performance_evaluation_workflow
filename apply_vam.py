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
from VennABERS import get_VA_margin_cross, get_VA_margin_cross_external
from tqdm import tqdm
import sklearn.metrics


def vprint(s=""):
	if args.verbose:
		print(s)

def init_arg_parser():
	parser = argparse.ArgumentParser(description="Calculate Venn ABERS for a dataset")
	parser.add_argument("--y_true", help="True labels (npy) (i.e. from files_4_ml/)", type=str, required=True)
	parser.add_argument("--y_pred", help="Yhat output (npy)", type=str, required=True)
	parser.add_argument("--y_pred_external", help="(Optional) Yhat for an external yhat prediction (npy)", type=str, default=None)
	parser.add_argument("--y_pred_external_true", help="(Optional) Y true for an external dataset (npy)", type=str, default=None)
	parser.add_argument("--cols", help="Yhat columns used for ytrue (i.e. first_col:last_col)", default=None, type=str)
	parser.add_argument("--filename", help="Filename for results from this output", type=str, default=None)
	parser.add_argument("--verbose", help="Verbosity level: 1 = Full; 0 = no output", type=int, default=1, choices=[0, 1])
	args = parser.parse_args()
	if args.cols:
		args.cols=list(map(int,args.cols.split(',')))
		assert len(args.cols) == 2, f'len(args.cols) should be 2 ({args.cols})'
		assert args.cols[0] < args.cols[1], f'args.cols[0] should be smaller than args.cols[1] {args.cols}'
	return args

def write_params_get_filename():
	if args.filename is not None:
		name = args.filename
	else:
		name = f"{os.path.basename(args.y_pred)}_{os.path.basename(args.y_true)}"
		vprint(f"\nRun name is '{name}'\n")
	assert not os.path.exists(name), f"{name} already exists... exiting"
	os.makedirs(name)
	with open(f'{name}/run_params.json', 'wt') as f:
		json.dump(vars(args), f, indent=4)
		vprint(f"Wrote input params to '{name}/run_params.json'\n")
	return name

def process_task(col, y_pred_col, y_true_col):
	pts = np.hstack((y_pred_col, y_true_col))
	m = np.array(pts[:,1] != 0.0)
	pts = pts[np.vstack((m,m)).T.all(axis=1)]
	row_number = np.arange(len(y_pred_col))[m]
	pts[:,1]=(pts[:,1]==1).astype(np.uint8)
	p0,p1,vams=get_VA_margin_cross(pts)
	return pd.DataFrame(np.column_stack(([col]*len(vams),row_number,pts,p0,p1,vams)))

def process_task_external(col, y_pred_col, y_true_col, y_pred_external, y_pred_external_true):
	pts = np.hstack((y_pred_col, y_true_col))
	m = np.array(pts[:,1] != 0.0)
	pts = pts[np.vstack((m,m)).T.all(axis=1)]
	pts[:,1]=(pts[:,1]==1).astype(np.uint8)
	p0,p1,vams=get_VA_margin_cross_external(pts,y_pred_external)
	row_number = np.arange(len(y_pred_external))
	if not y_pred_external_true:	
		return pd.DataFrame(np.column_stack(([col]*len(vams),row_number,y_pred_external,p0,p1,vams)))
	else:
		return pd.DataFrame(np.column_stack(([col]*len(vams),row_number,y_pred_external,y_pred_external_true,p0,p1,vams)))

def decide_col_names():
	if args.y_pred_external:
		if not args.y_pred_external_true:
			return ['classification_task_id','input_compound_col','yhat','p0','p1','vennabers']
		else:
			return ['classification_task_id','input_compound_col','yhat','y_true','p0','p1','vennabers']
	else:
		return ['classification_task_id','input_compound_col','yhat','class_label','p0','p1','vennabers']


def main(args):
	run_name = write_params_get_filename()
	y_true = sparse.csc_matrix(np.load(args.y_true, allow_pickle=True).item().tocsc()).todense()
	y_pred = sparse.csc_matrix(np.load(args.y_pred, allow_pickle=True)).todense()
	if args.y_pred_external: y_pred_external = sparse.csc_matrix(np.load(args.y_pred_external, allow_pickle=True)).todense()
	else: y_pred_external_true = None
	df=pd.DataFrame()
	if not args.cols: cols=range(y_true.shape[1])
	else:
		cols=range(args.cols[0],args.cols[1]+1,1)
		assert len(cols) <= y_pred.shape[1], f'number of supplied cols {len(cols)} should be less than y-pred {y_pred.shape[1]}'
		vprint(f'{len(cols)} cols supplied for y-pred (with shape {y_pred.shape}), y-true shape is {y_true.shape}')
	for idx, col in enumerate(tqdm(cols,desc=f'Processing {args.y_true} and {args.y_pred}, external file: {args.y_pred_external}')):
		if not args.y_pred_external:
			vam_task = process_task(col,np.array(y_pred[:,col]),np.array(y_true[:,idx]))
		else:
			assert len(cols) <= y_pred_external.shape[1], f'number of supplied cols {len(cols)} should be less than y-pred {y_pred.shape[1]}'
			if args.y_pred_external_true:
				y_pred_external_true = sparse.csc_matrix(np.load(args.y_pred_external_true, allow_pickle=True).item().tocsc()).todense()
				vam_task = process_task_external(col,np.array(y_pred[:,col]),np.array(y_true[:,idx]),np.array(y_pred_external[:,col],np.array(y_pred_external_true[:,idx])))
			else: vam_task = process_task_external(col,np.array(y_pred[:,col]),np.array(y_true[:,idx]),np.array(y_pred_external[:,col]),None)
		df=pd.concat((df,vam_task))
	df.columns=decide_col_names()
	df['classification_task_id']=df['classification_task_id'].astype('int')
	df['input_compound_col']=df['input_compound_col'].astype('int')
	if args.y_pred_external_true or not args.y_pred_external: df['class_label']=df['class_label'].astype('int')
	df.to_csv(f'{run_name}/venn_abers_margins.csv', index=False)
	vprint(f"Wrote Venn ABERS results to '{run_name}/venn_abers_margins.csv'\n")

if __name__ == '__main__':
	start = time.time()
	args = init_arg_parser()
	vprint(args)
	vprint('\n=== Venn ABERS assessment for dataset ===\n')
	main(args)
	end = time.time()
	vprint(f'Venn ABERS valuation took {end - start:.08} seconds.')





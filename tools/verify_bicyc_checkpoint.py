#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.data_manager import DataManager
from trainer import _set_device
from utils import model_factory
from utils.bicyc_transport import module_state_sha256, tensor_mapping_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--metrics'); p.add_argument('--output', required=True)
    a = p.parse_args()
    raw_cfg = json.loads(Path(a.config).read_text())
    raw_ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if raw_ckpt['cur_task'] != 1 or raw_ckpt['total_classes'] != 20:
        raise RuntimeError('Expected completed task1 checkpoint')
    for key in ('forward_transport_state_dict','backward_transport_state_dict','old_ae_state_dict'):
        if key not in raw_ckpt: raise RuntimeError('Missing '+key)
    if raw_ckpt.get('bicyc_mode') != 'cycle': raise RuntimeError('Wrong checkpoint mode')

    cfg=copy.deepcopy(raw_cfg); seed=cfg['seed'][0]; cfg['seed']=seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    _set_device(cfg)
    dm=DataManager(cfg['dataset'],cfg['shuffle'],seed,cfg['init_cls'],cfg['increment'])
    model=model_factory.get_model(cfg['model_name'],cfg); model.class_order=list(dm._class_order)
    completed=model.load_checkpoint(a.checkpoint)
    if completed != 1: raise RuntimeError('Reload returned wrong task')
    expected_a=tensor_mapping_sha256(raw_ckpt['forward_transport_state_dict'])
    expected_d=tensor_mapping_sha256(raw_ckpt['backward_transport_state_dict'])
    if module_state_sha256(model.forward_transport)!=expected_a: raise RuntimeError('A reload mismatch')
    if module_state_sha256(model.backward_transport)!=expected_d: raise RuntimeError('D reload mismatch')
    if not torch.isfinite(model._class_covs).all(): raise RuntimeError('Non-finite covariances')
    asym=float((model._class_covs-model._class_covs.transpose(-1,-2)).abs().max())
    if asym > 1e-6: raise RuntimeError('Checkpoint covariances asymmetric')
    evaluated=None
    if a.metrics:
        dataset=dm.get_dataset(np.arange(20),source='test',mode='test')
        model.test_loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
        evaluated=model.eval_task_detailed()
        metrics=json.loads(Path(a.metrics).read_text())
        task1=[r for r in metrics['task_records'] if r['task']==1][0]
        if evaluated['metrics'] != task1['post_ca']['metrics']:
            raise RuntimeError('Reloaded post-CA metrics mismatch')
    report={
      'status':'PASS','task':1,'mode':'cycle','strict_full_model_load':True,
      'A_state_sha256':expected_a,
      'D_state_sha256':expected_d,'old_P_present':True,
      'post_ca_metrics_reproduced':evaluated,'max_covariance_asymmetry':asym,
      'transport_history_entries':len(model.transport_history),
      'experiment_records_entries':len(model.experiment_records),
    }
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)
    print('CHECKPOINT_RELOAD '+json.dumps(report,sort_keys=True))

if __name__=='__main__': main()

#!/usr/bin/env python3
"""RQ7: protocol-A SSCA localization across completed A0 transitions; no training."""
import json
import os
import time
import platform
from pathlib import Path
import numpy as np
import torch
import diagnose_transition as d


def compact(record, global_l2, median, n):
    k = record['kernel_statistics']
    l2 = record['old_oracle_prototype_error']['l2']['mean']
    return dict(sigma=record['sigma'], floor=record['floor'], l2=l2,
                gain_percent=100*(1-l2/global_l2), sigma_over_median=record['sigma']/median,
                ess=k['ess']['mean'], ess_over_n=k['ess']['mean']/n,
                entropy=k['normalized_entropy']['mean'],
                floor_mass=k['floor_added_mass_fraction']['mean'])


def main():
    args = d.parse_args()
    start = time.time()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    config['seed'] = d.scalar_config_value(config['seed'])
    args.seed = int(args.seed if args.seed is not None else config['seed'])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    old = d.load_checkpoint(args.ckpt_old)
    new = d.load_checkpoint(args.ckpt_new)
    validation = d.validate_inputs(config, old, new, args.transition_task, general_transition=True)
    assert config.get('ae_residual_mode', 'sigmoid') == 'sigmoid', 'RQ7 requires A0'
    ko, kn = int(old['total_classes']), int(new['total_classes'])
    transition = '{}->{}'.format(args.transition_task-1, args.transition_task)
    provenance = dict(transition=transition, old_sha256=d.sha256_file(args.ckpt_old),
                      new_sha256=d.sha256_file(args.ckpt_new), config=config,
                      seed=args.seed, repeats=args.estimator_repeats,
                      batch_size=args.batch_size, num_workers=args.num_workers,
                      torch=torch.__version__, python=platform.python_version(),
                      script_sha256=d.sha256_file(__file__),
                      diagnostic_sha256=d.sha256_file(d.__file__),
                      git_commit=d.subprocess.check_output(['git','rev-parse','HEAD'],cwd=d.REPO_ROOT,text=True).strip())
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('RQ7', transition, 'classes', ko, '->', kn, flush=True)
    cache_path = out/'features.pt'
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False, map_location='cpu')
        assert cache['provenance'] == provenance, 'Cache provenance mismatch; use new output directory'
        oracle_means, features, validation = cache['oracle_means'], cache['features'], cache['validation']
        print('Reused verified feature cache', flush=True)
    else:
        old_net = d.rebuild_network(config, old, args.device_obj, 'old')
        new_net = d.rebuild_network(config, new, args.device_obj, 'new')
        validation['strict_load'] = 'both networks: zero missing/unexpected; eval; all params frozen'
        cifar = d.datasets.CIFAR100(args.data_root, train=True, download=False)
        labels = np.asarray(cifar.targets, dtype=np.int64)
        inverse = np.empty(100, dtype=np.int64)
        inverse[np.asarray(new['class_order'])] = np.arange(100)
        labels = inverse[labels]
        ids = np.flatnonzero(labels < kn)
        oracle_ds = d.OracleDataset(cifar.data, labels, ids, d.transforms.Compose(d.build_transform(False,None)))
        print('Extract oracle f_new:',len(ids),'images',flush=True)
        extracted, found_ids, found_labels = d.extract_oracle(d.make_loader(oracle_ds,args),[new_net],args.device_obj)
        assert torch.equal(found_ids,torch.as_tensor(ids))
        assert all(int((found_labels == c).sum()) == 500 for c in range(kn))
        oracle_means = torch.stack([extracted[0][found_labels == c].double().mean(0) for c in range(kn)])
        sanity = torch.linalg.vector_norm(torch.as_tensor(new['class_means'][ko:kn]).double()-oracle_means[ko:kn],dim=1)
        validation['current_class_mean_l2'] = d.distribution(sanity)
        assert float(sanity.max()) < 1e-3, 'Current-class reconstruction sanity failed'
        current_ids = np.concatenate([np.flatnonzero(labels == c) for c in range(ko,kn)])
        features = []
        for rep in range(args.estimator_repeats):
            print('Protocol A repeat',rep,'seed',args.seed+rep,flush=True)
            ds = d.PairedAugmentedDataset(cifar.data,labels,current_ids,current_ids,
                    d.transforms.Compose(d.build_transform(True,None)),args.seed+rep,rep,'A')
            f = d.extract_augmented(d.make_loader(ds,args),old_net,new_net,args.device_obj)
            assert torch.equal(f['left_ids'],torch.as_tensor(current_ids))
            features.append(f)
        torch.save(dict(provenance=provenance,oracle_means=oracle_means,features=features,validation=validation),cache_path)
        del old_net,new_net,extracted
        torch.cuda.empty_cache()
    means = torch.as_tensor(old['class_means'][:ko]).double()
    stored = torch.as_tensor(new['class_means'][:ko]).double()
    target = oracle_means[:ko]
    seeds = [args.seed+r for r in range(args.estimator_repeats)]
    a_records=[]
    for f in features:
        _,records=d.ssca_repeat(f,means,stored,target)
        a_records.extend(records)
    a_l2=float(np.mean([r['predicted_to_oracle_f2_l2'] for r in a_records]))
    rq1={'comparison_summary':dict(A={'l2_mean':a_l2},
         no_shift=d.baseline_mean_summary([d.vector_error(x,y) for x,y in zip(means,target)]),
         historical_stored_task2_ssca=d.baseline_mean_summary([d.vector_error(x,y) for x,y in zip(stored,target)]))}
    # Predeclared observable-only grid additions resolve the basin around the RQ6 scale.
    original_grid=d.build_rq6_sigma_grid
    def basin_grid(summary):
        grid=original_grid(summary)
        median=summary['quantiles']['q50']
        for ratio in (0.15,0.175,0.20,0.225,0.275,0.30,0.35,0.40):
            sigma=median*ratio
            if not any(abs(p['sigma']-sigma)<1e-9 for p in grid):
                grid.append(dict(sigma=sigma,sources=['RQ7_distance_median*{}'.format(ratio)]))
        return sorted(grid,key=lambda x:x['sigma'])
    d.build_rq6_sigma_grid=basin_grid
    print('Sweep on cached protocol-A features',flush=True)
    rq=d.rq6_ssca_kernel_sweep(features,means,stored,target,rq1,seeds)
    rq['scope']['transition']='zero-based task '+transition
    rq['scientific_guardrail']['sigma_grid_inputs']='fixed grid plus stored old prototypes/current features only'
    assert rq['official_crosscheck_against_RQ1_protocol_A']['absolute_difference'] < 1e-9
    g=rq['global_drift_baseline']['old_oracle_prototype_error']['l2']['mean']
    median=rq['distance_scale']['aggregate']['quantiles']['q50']
    n=rq['scope']['current_samples_per_repeat']
    rows=[compact(p,g,median,n) for p in rq['sweep_records']]
    best=compact(rq['oracle_selected_best_sweep_result'],g,median,n)
    summary=dict(transition=transition,global_l2=g,distance_median=median,best=best,
                 official=compact(rq['official_sigma_4_floor_1e_5'],g,median,n),
                 near_best_definition='L2 <= best L2 + 0.005 * global L2 (0.5 percentage point gain tolerance)',
                 near_best=[r for r in rows if r['l2'] <= best['l2']+0.005*g],
                 repeat_count=len(features),run_seed=args.seed,
                 selection='ORACLE EXISTENCE TEST; NOT A PRODUCTION RULE')
    # Per-class paired comparison and repeat sensitivity at the selected global point.
    per_class=[]
    for rep,f in enumerate(features):
        f1,f2=f['f1'].double(),f['f2'].double()
        dist=torch.cdist(means,f1)
        logits=-dist.square()/(2*best['sigma']**2)
        w=torch.softmax(logits,dim=1)
        if best['floor']:
            alpha=torch.sigmoid(torch.logsumexp(logits,dim=1)-d.math.log(n*best['floor']))
            w=alpha[:,None]*w+(1-alpha[:,None])/n
        drift=f2-f1
        gl=torch.linalg.vector_norm(means+drift.mean(0)-target,dim=1)
        bl=torch.linalg.vector_norm(means+w@drift-target,dim=1)
        for c in range(ko):
            per_class.append(dict(repeat=rep,seed=seeds[rep],class_id=c,original_class_id=int(new['class_order'][c]),
                                  global_l2=float(gl[c]),best_l2=float(bl[c]),gain_percent=float(100*(1-bl[c]/gl[c]))))
    summary['per_class_positive_fraction']=float(np.mean([p['gain_percent']>0 for p in per_class]))
    metrics=dict(provenance=provenance,validation=validation,summary=summary,RQ7=rq,
                 runtime_seconds=time.time()-start,
                 hardware=torch.cuda.get_device_name(args.device_obj) if args.device_obj.type=='cuda' else 'cpu')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    for name,values in [('sweep.csv',rows),('per_class.csv',per_class)]:
        with (out/name).open('w',newline='') as handle:
            writer=d.csv.DictWriter(handle,fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('RQ7_RESULT',json.dumps(summary),flush=True)
    print('DONE',out,'seconds',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    main()


# Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models

  

This repository serves as the official implementation corresponding to the paper titled "Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models". 
![Overall pipeline of RSIAT. ](images/framework.png)

## Installation
### Requirements
Ubuntu 20.04 LTS

Python 3.10

CUDA 11.8

Detailed package information and corresponding versions are available in the requirements.txt file.

### Data preparation

The overall directory structure should be:
```
RSIAT/
├──data/
├──datasets/
│   ├──cifar-100-python/
│   ├──cub/
│   ├──imagenet-a/
│   ├──imagenet-r/
│   ├──omnibenchmark/
│   ├──vtab/
│   ......
├──.......
```

## Training and evaluation

The training and evaluation instructions for each dataset are in the "./args.sh" file. Each dataset can be calculated separately, and the results are stored in the "./logs" folder.

### Checkpoint and resume

Each experiment configuration now saves a checkpoint after every completed incremental task. Checkpoints are isolated by seed and stored at:

```
ckpt/<prefix>/<dataset>/<init_cls>_<increment>/seed_<seed>/task_<N>.pkl
```

The default configurations keep only the most recent checkpoint to reduce disk use. To resume a stopped run, set the following in the same JSON configuration and run the original command again:

```json
{
  "resume": true
}
```

Optionally set `"resume_path"` to a specific `task_N.pkl`, `"keep_last_checkpoint": false` to retain every task checkpoint, or `"max_tasks_per_run"` to stop cleanly after a fixed number of tasks. Resume requires the same dataset, seed, class order, initial/incremental class split, model, and backbone. Checkpoints are task-boundary snapshots; they do not resume an interrupted epoch. The supplied configurations retain full covariance (`"compact_diagonal_checkpoint": false`) for an exact task-boundary resume; enable the compact diagonal form only when Drive space matters, since it drops covariance cross-terms used by classifier alignment.

### Efficient training

`eval_interval: 0` and `ca_eval_interval: 0` evaluate only the final epoch of each training stage. This avoids repeated full validation passes without changing gradients, optimizer steps, or the cosine scheduler. Set either value to a positive number when intermediate validation curves are needed. `num_workers`, `stats_num_workers`, `pin_memory`, and `persistent_workers` configure data loading; the provided values are suitable starting points for a GPU runtime.

### Semantic-shift diagnostics

After each incremental task, the existing RSIAT log records `feature_shift` and
`prototype_drift` diagnostics. `sample_drift_mean_norm` is the mean per-sample
feature displacement, while `sample_drift_norm_mean` is the norm of the mean
displacement vector. Prototype and SSC fields report the old-class prototype
shift before and after the existing semantic-shift compensation; they are
observational metrics only and do not affect training or compensation.

### KeepLoRA replacement

The KeepLoRA smoke configuration keeps the RSIAT learner, cosine classifier,
representation-steering loss, residual autoencoder alignment, prototype update,
and CIL evaluation protocol. It disables the parallel AdaptFormer adapter and
instead applies KeepLoRA to the ViT attention Q/K/V/output projections.

At the start of every task, the implementation projects the cosine-loss gradient
away from the pre-trained weight principal subspace and prior task feature
subspaces, then initializes a frozen A and trainable B. At task completion the
update is merged into the ViT and the new feature directions are saved in the
regular RSIAT checkpoint. Run:

    python main.py --config ./exps/keeplora_cifar224_smoke.json

The matched full benchmark configurations are:

    python main.py --config ./exps/keeplora_cifar224.json
    python main.py --config ./exps/keeplora_cub.json
    python main.py --config ./exps/keeplora_imageneta.json
    python main.py --config ./exps/keeplora_imagenetr.json
    python main.py --config ./exps/keeplora_omnibench.json
    python main.py --config ./exps/keeplora_vtab.json

For a comparison with RSIAT, keep the dataset split, seed, epochs, classifier
alignment settings, and losses identical; vary only model_name, convnet_type,
and the keeplora settings. The default rank 32 matches the rough
trainable-parameter budget of RSIAT's bottleneck-64 adapter over Q/K/V/O.
For a full KeepLoRA experiment, omit the optional batch limits or set
keeplora_grad_batches and keeplora_feature_batches to 0, which processes every
batch in the current task. The paper-derived starting thresholds are 0.85 for
the PTM weight principal subspace and 0.99 for task-feature subspaces.
Feature accumulation uses a fixed-size second-moment matrix, whose left
singular vectors match those of the full activation matrix; it therefore
processes a whole task without retaining every ViT token in memory.
Accumulation is enabled only during the explicit end-of-task feature pass, so
the RSIAT training loop has no feature-statistics overhead. KeepLoRA checkpoint
metadata records every PEFT setting and rejects incompatible resumes.

For a Google Colab smoke test that verifies data loading, training, checkpoint creation, and resume at task 1, use [RSIAT_Colab.ipynb](RSIAT_Colab.ipynb).

## Citation

If you find this useful in your research, please consider citing:

```text
@inproceedings{zhao2026representation,
  title={Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models},
  author={Zhao, Jiarui and Huang, Libo and Li, Xiangqi and An, Zhulin and Yang, Chuanguang and Wang, Yu and Diao, Boyu and Xu, Yongjun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18010--18020},
  year={2026}
}
```

## Acknowledgement

This repo is based on [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT) and [SSIAT](https://github.com/HAIV-Lab/SSIAT).

Thanks for their wonderful work!!!

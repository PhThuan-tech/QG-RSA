#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/khanh/research/rsiat-bicyc-style
ROOT=/mnt/d/RSIAT_overnight/BiCyc_style
PYTHON=/home/khanh/research/khanh-research/RQ7_transition_7_to_8_local/.venv/bin/python
BASE=0b914d71333165b54c1007ffb30707af77c767a3
DATA_SOURCE=/home/khanh/research/khanh-research/RQ7_transition_7_to_8_local/data
SOURCE_A=/home/khanh/research/khanh-research/A2_ssca_median_ratio_seed1993_local/attempts/batch64/ckpt/A2_median_sigma/cifar224/10_10/seed_1993/task_0.pkl
SOURCE_B=/home/khanh/research/khanh-research/A2b_ssca_corrected_units_seed1993_local/attempts/batch64/ckpt/A2b_corrected_units/cifar224/10_10/seed_1993/task_0.pkl
STATUS_TOOL="$REPO/tools/update_bicyc_status.py"
CONFIGS=(
  "$REPO/exps/bicyc_B0_official.json"
  "$REPO/exps/bicyc_B1_forward.json"
  "$REPO/exps/bicyc_B2_bidirectional.json"
  "$REPO/exps/bicyc_B3_cycle.json"
)
ARMS=(B0_official B1_forward B2_bidirectional B3_cycle)

export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

update_status() { "$PYTHON" "$STATUS_TOOL" --root "$ROOT" "$@"; }
sha256_file() { /usr/bin/sha256sum "$1" | /usr/bin/awk '{print $1}'; }

on_error() {
  rc=$?
  set +e
  update_status --status FAIL --phase "${CURRENT_PHASE:-UNKNOWN}" --arm "${CURRENT_ARM:-}" --exit-code "$rc" --message "Track B launcher stopped on first failure."
  printf 'END_UTC=%s\nEXIT_CODE=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$ROOT/runtime.txt"
  exit "$rc"
}
trap on_error ERR

CURRENT_PHASE=PREFLIGHT
CURRENT_ARM=
if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing output root: $ROOT" >&2
  exit 20
fi
mkdir -p "$ROOT"/{common_task0,reports,configs,data}
ln -s "$DATA_SOURCE" "$ROOT/data/datasets"
printf 'START_UTC=%s\nLAUNCHER_PID=%s\nTMUX_SESSION=rsiat_bicyc_screen\n' \
  "$(date -u +%FT%TZ)" "$$" > "$ROOT/runtime.txt"
echo "$(date -u +%FT%TZ) launcher $$" > "$ROOT/pids.log"
echo "$$" > "$ROOT/launcher.pid"
update_status --status RUNNING --phase PREFLIGHT --message "Checking disk, source isolation, tests, and common task0."

# Hard disk gates.
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  '$d=Get-PSDrive C,D; if (($d|? Name -eq C).Free -lt 10GB) {exit 10}; if (($d|? Name -eq D).Free -lt 100GB) {exit 11}; $d | Select Name,@{N="FreeGiB";E={[math]::Round($_.Free/1GB,3)}} | ConvertTo-Json' \
  > "$ROOT/reports/disk_preflight.json"

cd "$REPO"
[[ "$(git branch --show-current)" == exp/rsiat-bicyc-style ]]
[[ "$(git rev-parse HEAD^)" == "$BASE" ]]
[[ -z "$(git status --short)" ]]
[[ -f "$ROOT/data/datasets/cifar-100-python/train" ]]
[[ -f "$ROOT/data/datasets/cifar-100-python/test" ]]

git rev-parse HEAD > "$ROOT/reports/git_commit.txt"
git rev-parse HEAD^ > "$ROOT/reports/git_parent.txt"
git status --short > "$ROOT/reports/git_status.txt"
git show --stat --oneline --decorate HEAD > "$ROOT/reports/git_show.txt"
git diff "$BASE"..HEAD > "$ROOT/reports/git_diff.patch"
for config in "${CONFIGS[@]}"; do cp "$config" "$ROOT/configs/"; done

{
  echo "time_utc=$(date -u +%FT%TZ)"
  echo "python=$($PYTHON --version 2>&1)"
  echo "torch=$($PYTHON -c 'import torch; print(torch.__version__)')"
  echo "torchvision=$($PYTHON -c 'import torchvision; print(torchvision.__version__)')"
  echo "cuda=$($PYTHON -c 'import torch; print(torch.version.cuda)')"
  echo "cudnn=$($PYTHON -c 'import torch; print(torch.backends.cudnn.version())')"
  echo "platform=$(uname -a)"
  echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
  echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
  /mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "$ROOT/environment.txt"

cat > "$ROOT/reports/code_mapping.md" <<'MD'
# Track-B paper↔code mapping

- `models/RSIAT_adapter.py::_compute_rt_loss`: one already-augmented input tensor produces `z_new` and frozen `z_old` in the same iteration.
- RSIAT `old_ae` remains P(old→new), with original `L_align`, `L_orth`, and optimizer group unchanged.
- Dedicated affine A(old→new) and D(new→old) are separate `nn.Linear(768,768,bias=True)` modules.
- B1: RSIAT + `5 L_A`; B2: B1 + `5 L_D`; B3: B2 + detached-inner-map cycle at weight 1.
- A/D initialization is deterministic and restores process RNG state; hashes are checked across arms.
- B1/B2/B3 store exact analytic affine Gaussian transport; B0 retains official SSCA.
- The unchanged float32 CA sampler gets bounded diagonal jitter only if `MultivariateNormal` rejects an analytically PSD covariance; exact stored statistics are not overwritten.
- Stage-II trains A only for 30 epochs with paired MSE; representation, P, and D are not optimizer parameters.

[IMPLEMENTATION ADAPTATION] This is RSIAT + BiCyc-style transport, not an exact BiCyc reproduction.
MD

# Re-run cheap validation inside the immutable committed source.
tmp_pycache=$(/usr/bin/mktemp -d)
PYTHONPYCACHEPREFIX="$tmp_pycache" "$PYTHON" -m py_compile \
  models/RSIAT_adapter.py models/base.py trainer.py utils/bicyc_transport.py \
  tests/test_bicyc_transport.py tools/verify_bicyc_common_task0.py \
  tools/verify_bicyc_checkpoint.py tools/aggregate_bicyc_track.py \
  tools/smoke_bicyc_b3.py
rm -rf "$tmp_pycache"
PYTHONPATH=. "$PYTHON" -m unittest -v tests.test_bicyc_transport \
  > "$ROOT/reports/unit_tests.log" 2>&1

git diff --check "$BASE"..HEAD > "$ROOT/reports/git_diff_check.txt"

# Strict-load and reproduce the one shared task0 state.
cd "$ROOT"
"$PYTHON" "$REPO/tools/verify_bicyc_common_task0.py" \
  --source-a "$SOURCE_A" --source-b "$SOURCE_B" \
  --configs "${CONFIGS[@]}" \
  --destination "$ROOT/common_task0/task_0.pkl" \
  --report "$ROOT/common_task0/compatibility.json" \
  > "$ROOT/common_task0/verify.log" 2>&1
cp /mnt/d/RSIAT_overnight/BiCyc_style_preflight/b3_batch32_smoke.json \
  "$ROOT/reports/b3_batch32_smoke.json"
cp /mnt/d/RSIAT_overnight/BiCyc_style_preflight/batch32_fallback.json \
  "$ROOT/reports/batch32_environment_deviation.json"
cp /mnt/d/RSIAT_overnight/BiCyc_style_preflight/integration/verification.json \
  "$ROOT/reports/lifecycle_checkpoint_reload_smoke.json"
cp /mnt/d/RSIAT_overnight/BiCyc_style_preflight/integration/run.log \
  "$ROOT/reports/preflight_detected_covariance_failure.log"
update_status --status PASS --phase PREFLIGHT --message "Disk, source, tests, controlled batch32 B3 smoke, lifecycle checkpoint reload, and common task0 passed."

run_arm() {
  local arm="$1"
  local config="$2"
  local log="$ROOT/$arm/run.log"
  CURRENT_PHASE=ARM
  CURRENT_ARM="$arm"
  mkdir -p "$ROOT/$arm"
  update_status --status RUNNING --phase ARM --arm "$arm" --message "Running task1 through task2."
  "$PYTHON" "$REPO/main.py" --config "$config" > "$log" 2>&1 &
  local child=$!
  printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$arm" "$child" >> "$ROOT/pids.log"
  printf '{"arm":"%s","pid":%s,"start_utc":"%s"}\n' \
    "$arm" "$child" "$(date -u +%FT%TZ)" > "$ROOT/current_process.json"
  wait "$child"
  local rc=$?
  printf '{"arm":"%s","pid":null,"end_utc":"%s","exit_code":%s}\n' \
    "$arm" "$(date -u +%FT%TZ)" "$rc" > "$ROOT/current_process.json"
  [[ "$rc" -eq 0 ]]
  "$PYTHON" - "$ROOT/$arm/metrics.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m['status']=='PASS',m
assert [r['task'] for r in m['task_records']]==[0,1,2],m
PY
  update_status --status PASS --phase ARM --arm "$arm" --message "Arm completed through task2."
}

for i in 0 1 2 3; do run_arm "${ARMS[$i]}" "${CONFIGS[$i]}"; done

CURRENT_PHASE=AGGREGATE
CURRENT_ARM=
update_status --status RUNNING --phase AGGREGATE --message "Validating matched invariants and producing final decision tables."
"$PYTHON" "$REPO/tools/aggregate_bicyc_track.py" \
  --root "$ROOT" --configs "${CONFIGS[@]}" \
  > "$ROOT/reports/aggregate.log" 2>&1

# Finalize mutable metadata before generating the immutable manifest.
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -Command \
  'Get-PSDrive C,D | Select Name,@{N="FreeGiB";E={[math]::Round($_.Free/1GB,3)}} | ConvertTo-Json' \
  > "$ROOT/reports/disk_final.json"
CURRENT_PHASE=COMPLETE
printf 'END_UTC=%s\nEXIT_CODE=0\n' "$(date -u +%FT%TZ)" >> "$ROOT/runtime.txt"
printf '{"phase":"COMPLETE","pid":null,"end_utc":"%s","exit_code":0}\n' \
  "$(date -u +%FT%TZ)" > "$ROOT/current_process.json"
update_status --status COMPLETE --phase COMPLETE --exit-code 0 --message "All four arms and final aggregation completed."
find "$ROOT" -type f ! -name SHA256SUMS \
  ! -path "$ROOT/reports/manifest_verification.log" -print0 \
  | sort -z | xargs -0 sha256sum > "$ROOT/SHA256SUMS"
sha256sum -c "$ROOT/SHA256SUMS" > "$ROOT/reports/manifest_verification.log"

param(
    [ValidateSet("check", "manifest", "split", "extract", "cache", "stage-a", "stage-b", "stage-c", "stage-d")]
    [string]$Task = "check",
    [string]$Config = "configs/base.yaml"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Get-Location).Path

switch ($Task) {
    "check"   { python scripts/check_environment.py --config $Config }
    "manifest"{ python scripts/make_manifest_template.py --config $Config }
    "split"   { python scripts/validate_manifest.py --config $Config --make-splits }
    "extract" { python scripts/build_content_cache.py --config $Config }
    "cache"   { python scripts/build_content_cache.py --config $Config }
    "stage-a" { python experiments/stage_a_diagnosis.py --config configs/stage_a.yaml }
    "stage-b" { python experiments/stage_b_baseline.py --config configs/stage_b.yaml }
    "stage-c" { python experiments/stage_c_adapter.py --config configs/stage_c.yaml }
    "stage-d" { python experiments/stage_d_cfd_fsa.py --config configs/stage_d.yaml }
}

# restructure.ps1
# Run this INSIDE the folder that currently holds your flat drone files
# (e.g. C:\Users\joebu\drone_sim) AFTER downloading the new versions of
# index.html, server.py, watch_train.py, train.py (and README.md) into it.
#
#   cd C:\Users\joebu\drone_sim
#   ./restructure.ps1
#
# It creates the stage folders, sorts files into core/ and stage1_2d_profile/,
# and writes requirements.txt, .gitignore, and the stage READMEs.

$ErrorActionPreference = "Stop"

# 1. Folders
$dirs = @("core", "stage1_2d_profile", "stage2_3d", "stage3_fpv")
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

# 2. Shared engine -> core/
$core = @("drone_core.py", "drone_env.py", "qlearn.py")
foreach ($f in $core) {
    if (Test-Path $f) { Move-Item $f "core\$f" -Force }
}

# 3. Stage-1 app -> stage1_2d_profile/
$stage1 = @("index.html", "server.py", "watch_train.py", "train.py",
            "q_policy.npz", "training_curve.png")
foreach ($f in $stage1) {
    if (Test-Path $f) { Move-Item $f "stage1_2d_profile\$f" -Force }
}

# 4. Repo housekeeping files
@"
numpy
websockets
matplotlib
"@ | Set-Content -Encoding UTF8 "requirements.txt"

@"
__pycache__/
*.pyc
*.pyo
.DS_Store
.venv/
venv/
"@ | Set-Content -Encoding UTF8 ".gitignore"

@"
# Stage 2 - 3D view (planned)
Third-person 3D world (three.js). Physics becomes 6-DOF (x,y,z + pitch/yaw/roll);
the RL observation/action spaces grow. Tabular Q does not survive this jump --
build stage 2 on a deep agent (DQN/PPO). Reuses core/'s env interface, reward
philosophy, and training scaffolding.
"@ | Set-Content -Encoding UTF8 "stage2_3d\README.md"

@"
# Stage 3 - First-person + obstacles (planned)
Nose-camera view, obstacles to avoid, and an agent that learns from rendered
PIXELS rather than coordinates (vision-based RL: CNN policy, PPO/SAC, GPU).
This is a step change in perception, not just a renderer swap. Intermediate
path: human FPV first -> add obstacles with coordinate obs -> then switch to pixels.
"@ | Set-Content -Encoding UTF8 "stage3_fpv\README.md"

Write-Host ""
Write-Host "Done. New structure:" -ForegroundColor Green
Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch "__pycache__" } |
    ForEach-Object { $_.FullName.Replace((Get-Location).Path + "\", "  ") }

Write-Host ""
Write-Host "Next:" -ForegroundColor Cyan
Write-Host "  cd stage1_2d_profile"
Write-Host "  python watch_train.py     # then open index.html"
Write-Host "  python server.py --agent  # watch the trained policy"
Write-Host ""
Write-Host "If README.md isn't at the repo root yet, download it and drop it here."

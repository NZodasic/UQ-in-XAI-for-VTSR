#requires -version 5.1
[CmdletBinding()]
param(
    [string]$DatasetRoot = "",
    [string]$PythonVersion = "3.11",
    [string]$VenvDir = "venv",
    [ValidateSet("cu128", "cu126", "cu118", "cpu")]
    [string]$TorchIndex = "cu128",
    [string]$CudaToolkitVersion = "12.8",
    [switch]$SkipCudaToolkit,
    [switch]$SkipOptional,
    [switch]$ForceRecreateVenv,
    [switch]$RunSmokeTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Run {
    param([string]$Message)
    Write-Host "    $Message" -ForegroundColor DarkGray
}

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @()
    )

    Write-Run (("$FilePath " + ($ArgumentList -join " ")).Trim())
    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-Winget {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget.exe was not found. Install/enable Windows App Installer, then run this script again."
    }
    return $winget.Source
}

function Invoke-WingetInstall {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [string]$Version = ""
    )

    $winget = Get-Winget
    $wingetArgs = @(
        "install",
        "--id", $Id,
        "-e",
        "--source", "winget",
        "--accept-package-agreements",
        "--accept-source-agreements"
    )
    if ($Version) {
        $wingetArgs += @("--version", $Version)
    }

    Write-Run (("$winget " + ($wingetArgs -join " ")).Trim())
    $output = & $winget @wingetArgs 2>&1
    $exitCode = $LASTEXITCODE
    $output | ForEach-Object { Write-Host $_ }

    if ($exitCode -eq 0) {
        return
    }

    $outputText = ($output | Out-String)
    if ($outputText -match "Found an existing package already installed" -and
        $outputText -match "No available upgrade found") {
        Write-Host "Package $Id is already installed; continuing." -ForegroundColor Yellow
        return
    }

    throw "Command failed with exit code ${exitCode}: $winget $($wingetArgs -join ' ')"
}

function Test-NvidiaGpu {
    try {
        $gpu = Get-CimInstance Win32_VideoController |
            Where-Object { $_.Name -match "NVIDIA" } |
            Select-Object -First 1
        return [bool]$gpu
    } catch {
        return $false
    }
}

function Get-PythonExe {
    param([string]$Version)

    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        try {
            $previousErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            $output = & $launcher.Source "-$Version" -c "import sys; print(sys.executable)" 2>$null
            $exitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }

        if ($exitCode -eq 0 -and $output) {
            return (($output | Select-Object -Last 1).ToString().Trim())
        }
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        $code = "import sys; raise SystemExit(0 if sys.version_info[:2] == tuple(map(int, '$Version'.split('.'))) else 1)"
        & $python.Source -c $code 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $python.Source
        }
    }

    $shortVersion = $Version.Replace(".", "")
    $candidates = @()
    if (Test-Path env:LOCALAPPDATA) {
        $candidates += Join-Path $env:LOCALAPPDATA "Programs\Python\Python$shortVersion\python.exe"
    }
    if (Test-Path env:ProgramFiles) {
        $candidates += Join-Path $env:ProgramFiles "Python$shortVersion\python.exe"
    }
    $programFilesX86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    if ($programFilesX86) {
        $candidates += Join-Path $programFilesX86 "Python$shortVersion\python.exe"
    }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return ""
}

function Update-ProjectConfig {
    param(
        [string]$PythonExe,
        [string]$Root,
        [string]$SelectedTorchIndex
    )
    
    Write-Step "Updating configs/config.yaml"
    $code = @'
import pathlib
import sys
import yaml

dataset_root_arg = sys.argv[1]
torch_index = sys.argv[2]
cfg_path = pathlib.Path("configs/config.yaml")
backup_path = cfg_path.with_suffix(cfg_path.suffix + ".bak")

with cfg_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data = config.setdefault("data", {})
dataset_root = None
if dataset_root_arg:
    dataset_root = pathlib.Path(dataset_root_arg).expanduser().resolve()
    data["train_images_dir"] = str(dataset_root / "images" / "train")
    data["train_labels_dir"] = str(dataset_root / "labels" / "train")
    data["val_images_dir"] = str(dataset_root / "images" / "val")
    data["val_labels_dir"] = str(dataset_root / "labels" / "val")
    data["test_images_dir"] = str(dataset_root / "images" / "test")
    data["test_labels_dir"] = str(dataset_root / "labels" / "test")

    for name in ("custom_data.yaml", "data.yaml", "classes.yaml", "classes.txt"):
        candidate = dataset_root / name
        if candidate.exists():
            data["classes_file"] = str(candidate)
            break
    else:
        data["classes_file"] = str(dataset_root / "custom_data.yaml")

config.setdefault("experiment", {})["device"] = "cpu" if torch_index == "cpu" else "cuda"

if not backup_path.exists():
    backup_path.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

with cfg_path.open("w", encoding="utf-8", newline="\n") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print(f"Device set to: {config['experiment']['device']}")
if dataset_root:
    expected = [
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "images" / "test",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
        dataset_root / "labels" / "test",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    print(f"Dataset root set to: {dataset_root}")
    if missing:
        print("WARNING: these dataset folders do not exist yet:")
        for path in missing:
            print(f"  - {path}")
'@

    Invoke-External $PythonExe @("-c", $code, $Root, $SelectedTorchIndex)
}

Write-Step "Checking Windows package manager"
$wingetPath = Get-Winget
Write-Host "winget: $wingetPath"

Write-Step "Installing system prerequisites"
try {
    Invoke-WingetInstall -Id "Microsoft.VCRedist.2015+.x64"
} catch {
    Write-Warning "Could not install/update Microsoft Visual C++ Redistributable: $($_.Exception.Message)"
}

if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
    Invoke-WingetInstall -Id "Git.Git"
}

$pythonExe = Get-PythonExe -Version $PythonVersion
if (-not $pythonExe) {
    Write-Step "Installing Python $PythonVersion"
    Invoke-WingetInstall -Id "Python.Python.$PythonVersion"
    $pythonExe = Get-PythonExe -Version $PythonVersion
}
if (-not $pythonExe) {
    throw "Python $PythonVersion was installed but was not found in this terminal. Close CMD, open a new CMD, then run setup_windows.bat again."
}
Write-Host "Python: $pythonExe"

if ($TorchIndex -ne "cpu") {
    Write-Step "Checking NVIDIA/CUDA tooling"
    if (Test-NvidiaGpu) {
        if (-not $SkipCudaToolkit) {
            try {
                Invoke-WingetInstall -Id "Nvidia.CUDA" -Version $CudaToolkitVersion
            } catch {
                Write-Warning "Could not install CUDA Toolkit $CudaToolkitVersion. Trying the default Nvidia.CUDA package."
                try {
                    Invoke-WingetInstall -Id "Nvidia.CUDA"
                } catch {
                    Write-Warning "Could not install CUDA Toolkit through winget: $($_.Exception.Message)"
                    Write-Warning "Continuing because PyTorch installs its own CUDA runtime. A working NVIDIA driver is still required."
                }
            }
        }

        $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
        if ($nvidiaSmi) {
            & $nvidiaSmi.Source
        } else {
            Write-Warning "nvidia-smi.exe was not found. Install/update the NVIDIA display driver or reboot if CUDA was just installed."
        }
    } else {
        Write-Warning "No NVIDIA GPU was detected. CUDA PyTorch will install, but training must use CPU unless an NVIDIA GPU/driver is available."
    }
}

$venvPath = Join-Path $RepoRoot $VenvDir
if ($ForceRecreateVenv -and (Test-Path $venvPath)) {
    Write-Step "Removing existing virtual environment"
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Step "Creating Python virtual environment"
    Invoke-External $pythonExe @("-m", "venv", $venvPath)
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment Python was not found at $venvPython"
}

Write-Step "Upgrading pip tooling"
Invoke-External $venvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

Write-Step "Installing PyTorch ($TorchIndex)"
Invoke-External $venvPython @(
    "-m", "pip", "install",
    "torch",
    "torchvision",
    "--index-url", "https://download.pytorch.org/whl/$TorchIndex"
)

Write-Step "Installing project requirements"
Invoke-External $venvPython @("-m", "pip", "install", "-r", "requirements.txt")

if (-not $SkipOptional) {
    Write-Step "Installing optional augmentation and API requirements"
    Invoke-External $venvPython @("-m", "pip", "install", "-r", "requirements-augmentation.txt")
    Invoke-External $venvPython @("-m", "pip", "install", "-r", "requirements-api.txt")
}

Update-ProjectConfig -PythonExe $venvPython -Root $DatasetRoot -SelectedTorchIndex $TorchIndex

Write-Step "Verifying Python, PyTorch, and CUDA"
$verifyCode = @'
import sys
import torch
import torchvision

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
'@
Invoke-External $venvPython @("-c", $verifyCode)

if ($TorchIndex -ne "cpu") {
    $cudaCheckCode = "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    & $venvPython -c $cudaCheckCode
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "PyTorch installed, but CUDA is not available. Check NVIDIA driver, GPU support, and reboot if CUDA/driver was just installed."
    }
}

if ($RunSmokeTest) {
    Write-Step "Running one-epoch smoke test"
    Invoke-External $venvPython @(
        "-u", "main.py",
        "--config", "configs/config.yaml",
        "--set", "training.epochs=1",
        "--set", "training.patience=1",
        "--set", "data.batch_size=8",
        "--set", "explainability.generate=false",
        "--set", "calibration.temperature_scaling=false"
    )
}

Write-Step "Setup complete"
Write-Host "Run training from CMD with:"
Write-Host "  experiment.bat"
Write-Host ""
Write-Host "Run the benchmark with:"
Write-Host "  benchmark.bat --suite full"
Write-Host ""
Write-Host "Use a different dataset path by rerunning:"
Write-Host '  setup_windows.bat -DatasetRoot "D:\path\to\data2-augment"'

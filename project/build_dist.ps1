param(
    [string]$Config = "Debug",
    [string]$BuildDir = "cmake-build-dist-cuda12",
    [string]$DistDir = "dist",
    [string]$CudaArchitectures = "89-real",
    [string]$CudaToolkitRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
)

$ErrorActionPreference = "Stop"

function Find-VcVars64 {
    $candidates = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installPath) {
            $vcvars = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $vcvars) {
                return $vcvars
            }
        }
    }

    return $null
}

function Invoke-Native {
    param([string]$CommandLine)

    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        cmd.exe /d /s /c $CommandLine
    }
    else {
        if (-not $script:VcVars64) {
            throw "cl.exe was not found on PATH and vcvars64.bat could not be found. Run from a Visual Studio Developer PowerShell or install VS C++ build tools."
        }
        cmd.exe /d /s /c "call `"$script:VcVars64`" >nul && $CommandLine"
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $CommandLine"
    }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildPath = Join-Path $ProjectRoot $BuildDir
$DistPath = Join-Path $ProjectRoot $DistDir
$script:VcVars64 = Find-VcVars64

if (-not (Test-Path $CudaToolkitRoot)) {
    throw "CUDA toolkit root not found: $CudaToolkitRoot"
}

$CudaCompiler = Join-Path $CudaToolkitRoot "bin\nvcc.exe"
if (-not (Test-Path $CudaCompiler)) {
    throw "CUDA compiler not found: $CudaCompiler"
}

$CudaToolkitRootCMake = $CudaToolkitRoot.Replace("\", "/")
$CudaCompilerCMake = $CudaCompiler.Replace("\", "/")
Invoke-Native "cmake -S `"$ProjectRoot`" -B `"$BuildPath`" -G Ninja -DCMAKE_BUILD_TYPE=`"$Config`" -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER=`"$CudaCompilerCMake`" -DCMAKE_CUDA_HOST_COMPILER=cl -DCUDAToolkit_ROOT=`"$CudaToolkitRootCMake`" -DCMAKE_CUDA_ARCHITECTURES=`"$CudaArchitectures`""
Invoke-Native "cmake --build `"$BuildPath`" --target codeCudaLib --config `"$Config`""

if (Test-Path $DistPath) {
    Remove-Item -LiteralPath $DistPath -Recurse -Force
}

Invoke-Native "cmake --install `"$BuildPath`" --prefix `"$DistPath`" --config `"$Config`""

Write-Host "Built dist package: $DistPath"
Write-Host "  include: $(Join-Path $DistPath 'include')"
Write-Host "  lib:     $(Join-Path $DistPath 'lib\codeCudaLib.lib')"
Write-Host "  package: $(Join-Path $DistPath 'lib\cmake\CodeCudaEngine')"





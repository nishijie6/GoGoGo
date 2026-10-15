#Requires -Version 5.1

[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [switch]$Force,
    [ValidateRange(1, 10)]
    [int]$DownloadAttempts = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$resolvedProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
$katagoRoot = Join-Path $resolvedProjectRoot "katago"
$cpuRoot = Join-Path $katagoRoot "cpu"
$cpuExecutable = Join-Path $cpuRoot "katago.exe"

# v1.18.2 is a CUDA-only optimization release. Its release notes direct other
# backends to v1.18.1, which supports the existing b10c128 version-15 models.
# Choose the portable Eigen build, without AVX2/FMA CPU requirements.
$releaseTag = "v1.18.1"
$archiveName = "katago-v1.18.1-eigen-windows-x64.zip"
$archiveSize = 5903072
$archiveSha256 = "074485CF150C38AA3BB14AC9F54F2952FFEFBCEB44673709BBB8A83650BF95D6"
$archiveUrl = "https://github.com/lightvector/KataGo/releases/download/$releaseTag/$archiveName"
$headers = @{ "User-Agent" = "GoGoGo-KataGo-CPU-Installer" }

function Assert-KataGoChildPath([string]$Path) {
    $resolvedPath = [System.IO.Path]::GetFullPath($Path)
    $rootPrefix = [System.IO.Path]::GetFullPath($katagoRoot).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing a file operation outside the KataGo directory: $resolvedPath"
    }
}

function Assert-CpuEngine([string]$Executable) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "CPU KataGo executable was not found: $Executable"
    }
    $versionOutput = (& $Executable version 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0 -or $versionOutput -notmatch '(?m)^KataGo v1\.18\.1\s*$' -or
        $versionOutput -notmatch 'Using Eigen\(CPU\) backend' -or $versionOutput -match 'Compiled with AVX2') {
        throw "Expected KataGo v1.18.1 portable Eigen CPU backend. Actual output: $versionOutput"
    }
    Write-Host $versionOutput.Trim()
}

if (Test-Path -LiteralPath $cpuRoot) {
    if (-not (Test-Path -LiteralPath $cpuRoot -PathType Container)) {
        throw "CPU runtime destination is not a directory: $cpuRoot"
    }
    if (-not $Force) {
        Assert-CpuEngine $cpuExecutable
        Write-Host "CPU KataGo is already installed. Use -Force to reinstall."
        exit 0
    }
}

$null = New-Item -ItemType Directory -Path $katagoRoot -Force
$installRoot = Join-Path $katagoRoot (".cpu-install-" + [guid]::NewGuid().ToString("N"))
$stagedRuntime = Join-Path $installRoot "runtime"
$stagedArchive = Join-Path $installRoot $archiveName
$backupRoot = Join-Path $katagoRoot (".cpu-backup-" + [guid]::NewGuid().ToString("N"))
$hasBackup = $false
$published = $false
Assert-KataGoChildPath $installRoot
Assert-KataGoChildPath $stagedRuntime
Assert-KataGoChildPath $backupRoot
Assert-KataGoChildPath $cpuRoot
$null = New-Item -ItemType Directory -Path $installRoot

try {
    # Pin both the release asset and its GitHub SHA-256; verify before executing.
    for ($attempt = 1; $attempt -le $DownloadAttempts; $attempt++) {
        try {
            Write-Host "Downloading $archiveName (attempt $attempt/$DownloadAttempts)..."
            Invoke-WebRequest -UseBasicParsing -Headers $headers -Uri $archiveUrl -OutFile $stagedArchive
            if ((Get-Item -LiteralPath $stagedArchive).Length -ne $archiveSize) {
                throw "KataGo archive size verification failed."
            }
            if ((Get-FileHash -LiteralPath $stagedArchive -Algorithm SHA256).Hash -ne $archiveSha256) {
                throw "KataGo archive SHA-256 verification failed."
            }
            break
        }
        catch {
            if ($attempt -eq $DownloadAttempts) { throw }
            Start-Sleep -Seconds ([Math]::Min(5, $attempt * 2))
        }
    }
    Expand-Archive -LiteralPath $stagedArchive -DestinationPath $stagedRuntime
    Assert-CpuEngine (Join-Path $stagedRuntime "katago.exe")

    # Replace only katago/cpu after validating the staged runtime. Preserve
    # the root CUDA/OpenCL executable, models, and any existing CPU runtime
    # until the new installation has been published successfully.
    if (Test-Path -LiteralPath $cpuRoot) {
        [System.IO.Directory]::Move($cpuRoot, $backupRoot)
        $hasBackup = $true
    }
    try {
        [System.IO.Directory]::Move($stagedRuntime, $cpuRoot)
        $published = $true
        Assert-CpuEngine $cpuExecutable
    }
    catch {
        if ($published -and (Test-Path -LiteralPath $cpuRoot)) {
            Assert-KataGoChildPath $cpuRoot
            Remove-Item -LiteralPath $cpuRoot -Recurse -Force
        }
        if ($hasBackup) {
            Assert-KataGoChildPath $backupRoot
            Assert-KataGoChildPath $cpuRoot
            [System.IO.Directory]::Move($backupRoot, $cpuRoot)
            $hasBackup = $false
        }
        throw
    }
    if ($hasBackup) {
        Assert-KataGoChildPath $backupRoot
        Remove-Item -LiteralPath $backupRoot -Recurse -Force
        $hasBackup = $false
    }
    Write-Host "CPU KataGo installed: $cpuExecutable"
}
finally {
    if (Test-Path -LiteralPath $installRoot) {
        Assert-KataGoChildPath $installRoot
        Remove-Item -LiteralPath $installRoot -Recurse -Force
    }
}

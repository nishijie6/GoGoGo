[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$katagoRoot = Join-Path $projectRoot "katago"
$runtimeRoot = $katagoRoot
$modelRoot = $katagoRoot
$headers = @{ "User-Agent" = "Weiqi-GUI-KataGo-Installer" }

function Assert-AssetDigest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [object]$Asset,
        [string]$ExpectedSha256 = ""
    )

    if ($Asset.digest -and $Asset.digest.StartsWith("sha256:")) {
        $expected = $Asset.digest.Substring(7).ToUpperInvariant()
    } elseif ($ExpectedSha256) {
        $expected = $ExpectedSha256.ToUpperInvariant()
    } else {
        Write-Warning "GitHub 未返回 $($Asset.name) 的 SHA-256，无法自动校验。"
        return
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($actual -ne $expected) {
        throw "$($Asset.name) SHA-256 校验失败；文件可能不完整，未继续安装。"
    }
    Write-Host "SHA-256 校验通过：$($Asset.name)"
}

function Save-VerifiedAsset {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Asset,
        [Parameter(Mandatory = $true)]
        [string]$Destination,
        [Parameter(Mandatory = $true)]
        [long]$ExpectedSize,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedSha256
    )

    $temporary = "$Destination.download"
    try {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
        Invoke-WebRequest `
            -Headers $headers `
            -Uri $Asset.browser_download_url `
            -OutFile $temporary
        if ((Get-Item -LiteralPath $temporary).Length -ne $ExpectedSize) {
            throw "$($Asset.name) 文件大小校验失败。"
        }
        Assert-AssetDigest `
            -Path $temporary `
            -Asset $Asset `
            -ExpectedSha256 $ExpectedSha256
        Move-Item -LiteralPath $temporary -Destination $Destination -Force
    } finally {
        if (Test-Path -LiteralPath $temporary) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

New-Item -ItemType Directory -Force -Path $katagoRoot | Out-Null

Write-Host "查询已验证的 KataGo v1.18.1 Windows OpenCL 版本..."
$engineRelease = Invoke-RestMethod `
    -Headers $headers `
    -Uri "https://api.github.com/repos/lightvector/KataGo/releases/tags/v1.18.1"
$engineAsset = $engineRelease.assets | Where-Object {
    $_.name -eq "katago-v1.18.1-opencl-windows-x64.zip"
} | Select-Object -First 1
if (-not $engineAsset) {
    throw "官方 v1.18.1 Release 中没有找到已验证的 Windows OpenCL 安装包。"
}

$engineZip = Join-Path $katagoRoot $engineAsset.name
$engineExe = Join-Path $runtimeRoot "katago.exe"
$engineArchiveSize = 6004137
$engineArchiveSha256 = "1710DB1903AB921AA6837A9599C8474F8A59F057650217C5D9BC125EE393A9FF"
if ($Force -or -not (Test-Path -LiteralPath $engineExe)) {
    Write-Host "下载 $($engineAsset.name)..."
    Save-VerifiedAsset `
        -Asset $engineAsset `
        -Destination $engineZip `
        -ExpectedSize $engineArchiveSize `
        -ExpectedSha256 $engineArchiveSha256
    Expand-Archive -LiteralPath $engineZip -DestinationPath $runtimeRoot -Force
} else {
    Write-Host "已存在 KataGo 运行时，跳过下载；使用 -Force 可重新安装。"
    if (Test-Path -LiteralPath $engineZip) {
        if ((Get-Item -LiteralPath $engineZip).Length -ne $engineArchiveSize) {
            throw "$($engineAsset.name) 文件大小校验失败。"
        }
        Assert-AssetDigest `
            -Path $engineZip `
            -Asset $engineAsset `
            -ExpectedSha256 $engineArchiveSha256
    }
}

# The smallest official transformer model is a good fit for interactive play
# and 4 GB-class GPUs. Release tags are immutable, so this URL remains stable.
$modelRelease = Invoke-RestMethod `
    -Headers $headers `
    -Uri "https://api.github.com/repos/lightvector/KataGo/releases/tags/v1.17.1"
$modelAsset = $modelRelease.assets | Where-Object {
    $_.name -eq "b10c384h6nbttflrs.bin.gz"
} | Select-Object -First 1
if (-not $modelAsset) {
    throw "没有找到官方 b10 transformer 模型。"
}
$modelPath = Join-Path $modelRoot $modelAsset.name
$modelSize = 38245488
$modelSha256 = "0BA27ECED5180B3E3D0B898B280C541112989765E789D1EB6CD0D31B2B2C1229"
if ($Force -or -not (Test-Path -LiteralPath $modelPath)) {
    Write-Host "下载 $($modelAsset.name)..."
    Save-VerifiedAsset `
        -Asset $modelAsset `
        -Destination $modelPath `
        -ExpectedSize $modelSize `
        -ExpectedSha256 $modelSha256
} else {
    Write-Host "已存在模型文件，跳过下载；使用 -Force 可重新安装。"
    if ((Get-Item -LiteralPath $modelPath).Length -ne $modelSize) {
        throw "$($modelAsset.name) 文件大小校验失败。"
    }
    Assert-AssetDigest `
        -Path $modelPath `
        -Asset $modelAsset `
        -ExpectedSha256 $modelSha256
}

# GitHub's older v1.15.0 release predates API-provided asset digests. Pin the
# SHA-256 recorded from the immutable official release URL so future installs
# still reject incomplete or changed downloads.
$humanRelease = Invoke-RestMethod `
    -Headers $headers `
    -Uri "https://api.github.com/repos/lightvector/KataGo/releases/tags/v1.15.0"
$humanAsset = $humanRelease.assets | Where-Object {
    $_.name -eq "b18c384nbt-humanv0.bin.gz"
} | Select-Object -First 1
if (-not $humanAsset) {
    throw "没有找到官方 HumanSL 模型。"
}
$humanModelPath = Join-Path $modelRoot $humanAsset.name
$humanModelSha256 = "637746E44F0EFE00AD1245A50AA9BBF0716EFE364C43965EAD97BD6835D84AB5"
if ($Force -or -not (Test-Path -LiteralPath $humanModelPath)) {
    Write-Host "下载 $($humanAsset.name)..."
    Save-VerifiedAsset `
        -Asset $humanAsset `
        -Destination $humanModelPath `
        -ExpectedSize $humanAsset.size `
        -ExpectedSha256 $humanModelSha256
} else {
    Write-Host "已存在 HumanSL 模型，跳过下载；使用 -Force 可重新安装。"
    if ((Get-Item -LiteralPath $humanModelPath).Length -ne $humanAsset.size) {
        throw "$($humanAsset.name) 文件大小校验失败。"
    }
    Assert-AssetDigest `
        -Path $humanModelPath `
        -Asset $humanAsset `
        -ExpectedSha256 $humanModelSha256
}

if (-not (Test-Path -LiteralPath $engineExe)) {
    throw "安装结束后仍找不到 $engineExe"
}
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "安装结束后仍找不到 $modelPath"
}
if (-not (Test-Path -LiteralPath $humanModelPath)) {
    throw "安装结束后仍找不到 $humanModelPath"
}

Write-Host ""
& $engineExe version
Write-Host ""
Write-Host "KataGo 安装完成："
Write-Host "  程序：$engineExe"
Write-Host "  专业模型：$modelPath"
Write-Host "  HumanSL 模型：$humanModelPath"
$appDataRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "YijingGo"
Write-Host "  应用设置：$(Join-Path $appDataRoot 'katago.json')"
Write-Host "  运行与 OpenCL 缓存：$(Join-Path $appDataRoot 'runtime')"
Write-Host "首次对局会进行 OpenCL 显卡调优，可能需要几分钟；以后会复用缓存。"

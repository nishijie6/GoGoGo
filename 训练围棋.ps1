param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainerArgs
)

$ErrorActionPreference = 'Stop'
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) {
    if (-not $TrainerArgs) { $TrainerArgs = @('--help') }
    Push-Location $PSScriptRoot
    try {
        & $localPython -u (Join-Path $PSScriptRoot 'train.py') @TrainerArgs
        exit $LASTEXITCODE
    }
    finally { Pop-Location }
}
$env:WSL_UTF8 = '1'
$windowsRoot = $PSScriptRoot.Replace('\', '/')
$convertedRoot = & wsl -d Ubuntu-24.04 -- wslpath -a $windowsRoot
if ($LASTEXITCODE -ne 0 -or -not $convertedRoot) { throw '无法找到 Ubuntu-24.04 中的项目目录。' }
$linuxRoot = $convertedRoot.Trim()
if (-not $TrainerArgs) { $TrainerArgs = @('--help') }
& wsl -d Ubuntu-24.04 --cd $linuxRoot -- /home/dev/.venvs/nanogpt/bin/python -u train.py @TrainerArgs
exit $LASTEXITCODE

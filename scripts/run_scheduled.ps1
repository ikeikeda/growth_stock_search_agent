# Task Scheduler:
#   Program: powershell.exe
#   Arguments: -NoProfile -ExecutionPolicy Bypass -File "C:\Users\yusaku\Documents\apps\growth_stock_search_agent\scripts\run_scheduled.ps1"
#   Start in: C:\Users\yusaku\Documents\apps\growth_stock_search_agent
#
# uv is resolved even when Task Scheduler provides a minimal PATH.

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$LogsDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}

$LogFile = Join-Path $LogsDir ("research_{0:yyyyMMdd}.log" -f (Get-Date))

function Resolve-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe")
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe")
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
        (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe")
    )

    $wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $wingetRoot) {
        $found = Get-ChildItem -Path $wingetRoot -Filter "uv.exe" -Recurse -ErrorAction SilentlyContinue
        foreach ($item in $found) {
            $candidates += $item.FullName
        }
    }

    foreach ($path in $candidates) {
        if ($path -and (Test-Path -LiteralPath $path)) {
            return $path
        }
    }

    throw "uv was not found. Confirm that 'uv --version' works in an interactive session."
}

$uv = Resolve-Uv
$uvDir = Split-Path $uv -Parent
if ($env:Path.IndexOf($uvDir, [System.StringComparison]::OrdinalIgnoreCase) -lt 0) {
    $env:Path = $uvDir + ";" + $env:Path
}

$startedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$header = "===== " + $startedAt + " uv run research ====="
Add-Content -LiteralPath $LogFile -Value $header -Encoding UTF8

$exitCode = 0
try {
    & $uv run research *>&1 | Out-File -FilePath $LogFile -Append -Encoding utf8
    if ($null -ne $LASTEXITCODE) {
        $exitCode = $LASTEXITCODE
    }
} catch {
    Add-Content -LiteralPath $LogFile -Value $_.ToString() -Encoding UTF8
    $exitCode = 1
}

exit $exitCode
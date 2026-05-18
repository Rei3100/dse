#Requires -Version 5.1
<#
.SYNOPSIS
    DSRE ビルド成果物を C:\FreeSoft\DSRE\ に差し替えて起動する。
.PARAMETER Source
    ビルド済みフォルダ (dist\DSRE 相当。DSRE.exe + _internal を含む)
.PARAMETER ZipPath
    CI artifact の zip (DSRE_private.zip)。Source と排他。
.PARAMETER TargetDir
    配置先。既定 C:\FreeSoft\DSRE
.PARAMETER NoLaunch
    差し替え後に DSRE.exe を起動しない
#>
[CmdletBinding(DefaultParameterSetName = 'FromDir')]
param(
    [Parameter(Mandatory, ParameterSetName = 'FromDir')]
    [string]$Source,

    [Parameter(Mandatory, ParameterSetName = 'FromZip')]
    [string]$ZipPath,

    [string]$TargetDir = "C:\FreeSoft\DSRE",

    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"

function Assert-Layout {
    param([string]$root)
    $need = @("DSRE.exe", "_internal\numpy", "_internal\scipy", "_internal\librosa", "_internal\ffmpeg\ffmpeg.exe")
    foreach ($n in $need) {
        $p = Join-Path $root $n
        if (-not (Test-Path $p)) { throw "MISSING in source: $p" }
    }
}

# 1. ソース解決
if ($PSCmdlet.ParameterSetName -eq 'FromZip') {
    if (-not (Test-Path $ZipPath)) { throw "Zip not found: $ZipPath" }
    $extractTo = Join-Path $env:TEMP ("DSRE_extract_{0}" -f (Get-Date -Format "yyyyMMddHHmmss"))
    New-Item -ItemType Directory -Path $extractTo | Out-Null
    Expand-Archive -Path $ZipPath -DestinationPath $extractTo -Force
    $Source = $extractTo
}
if (-not (Test-Path $Source)) { throw "Source not found: $Source" }
Assert-Layout -root $Source

# 2. 既存プロセス停止 (ファイルロック回避)
$running = Get-Process -Name DSRE -ErrorAction SilentlyContinue
if ($running) {
    Write-Host "[deploy] stopping running DSRE.exe ($($running.Count) proc)"
    $running | Stop-Process -Force
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Milliseconds 200
        if (-not (Get-Process -Name DSRE -ErrorAction SilentlyContinue)) { break }
    }
    if (Get-Process -Name DSRE -ErrorAction SilentlyContinue) {
        throw "DSRE process still running after stop wait"
    }
}

# 3. 永続ファイルを一時退避 (deploy をまたいで保持すべきファイル)
$persistFiles = @("dsre_log.db", "state.ini")
$persistTemp = Join-Path $env:TEMP ("DSRE_persist_{0}" -f (Get-Date -Format "yyyyMMddHHmmss"))
New-Item -ItemType Directory -Path $persistTemp -Force | Out-Null
foreach ($pf in $persistFiles) {
    $src = Join-Path $TargetDir $pf
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $persistTemp $pf)
        Write-Host "[deploy] saved persistent: $pf"
    }
}

# 4. 旧ターゲットをバックアップ ($TargetDir -> $TargetDir.bak)
$backup = "$TargetDir.bak"
$oldBackup = "$TargetDir.bak.prev"
if (Test-Path $TargetDir) {
    if (Test-Path $oldBackup) {
        Write-Host "[deploy] remove older backup: $oldBackup"
        Remove-Item $oldBackup -Recurse -Force
    }
    if (Test-Path $backup) {
        Rename-Item $backup $oldBackup
    }
    Write-Host "[deploy] backup $TargetDir -> $backup"
    Rename-Item $TargetDir $backup
}

# 5. 配置
Write-Host "[deploy] copy $Source -> $TargetDir"
New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
Copy-Item (Join-Path $Source "*") $TargetDir -Recurse -Force

# 5b. 永続ファイルを復元
foreach ($pf in $persistFiles) {
    $saved = Join-Path $persistTemp $pf
    if (Test-Path $saved) {
        Copy-Item $saved (Join-Path $TargetDir $pf) -Force
        Write-Host "[deploy] restored persistent: $pf"
    }
}
Remove-Item $persistTemp -Recurse -Force -ErrorAction SilentlyContinue

# 5. 配置後スモーク (失敗したらロールバック)
try {
    Assert-Layout -root $TargetDir
    Write-Host "[deploy] structural smoke OK"

    # 実起動スモーク: exe --selftest で import を exercise
    Write-Host "[deploy] runtime selftest (--selftest)"
    $exeForTest = Join-Path $TargetDir "DSRE.exe"
    # windowed exe は & 演算子だと detach するので Start-Process -Wait -PassThru で完了待機
    $selftestProc = Start-Process -FilePath $exeForTest -ArgumentList "--selftest" -Wait -PassThru
    $selftestCode = $selftestProc.ExitCode
    $log = Join-Path $TargetDir "selftest.log"
    if (Test-Path $log) {
        Write-Host "---- selftest.log ----"
        Get-Content $log
        Write-Host "----------------------"
    }
    if ($selftestCode -ne 0) {
        throw "runtime selftest failed with exit $selftestCode"
    }
    Write-Host "[deploy] runtime selftest OK"
} catch {
    Write-Warning "[deploy] smoke failed, rolling back: $_"
    Remove-Item $TargetDir -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path $backup) { Rename-Item $backup $TargetDir }
    throw
}

# 6. 配置成功、古い .bak.prev を破棄
if (Test-Path $oldBackup) {
    Remove-Item $oldBackup -Recurse -Force
}

# 7. 一時展開先のクリーンアップ
if ($PSCmdlet.ParameterSetName -eq 'FromZip') {
    Remove-Item $Source -Recurse -Force -ErrorAction SilentlyContinue
}

# 8. 起動
if (-not $NoLaunch) {
    $exe = Join-Path $TargetDir "DSRE.exe"
    Write-Host "[deploy] launch $exe"
    Start-Process -FilePath $exe
}

Write-Host "[deploy] done. rollback: Remove-Item '$TargetDir' -Recurse; Rename-Item '$backup' '$TargetDir'"

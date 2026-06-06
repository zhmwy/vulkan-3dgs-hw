param(
    [int]$LogSeconds = 90,
    [int]$DetectExitAfterSeconds = 60,
    [string]$LogFile = "cli-run-hilog.log"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Project\vulkan-3dgs-hw"
$Node = "C:\Program Files\Huawei\DevEco Studio\tools\node\node.exe"
$Hvigor = "C:\Program Files\Huawei\DevEco Studio\tools\hvigor\bin\hvigorw.js"
$Ohpm = "C:\Program Files\Huawei\DevEco Studio\tools\ohpm\bin\ohpm.bat"
$DevecoSdk = "C:\Program Files\Huawei\DevEco Studio\sdk"
$JavaHome = "C:\Program Files\Huawei\DevEco Studio\jbr"
$Hdc = "C:\Users\user\AppData\Local\OpenHarmony\Sdk\12\toolchains\hdc.exe"

$Bundle = "com.huawei.xenginevulkandemo"
$Module = "entry"
$Ability = "EntryAbility"
$Hap = "$ProjectRoot\entry\build\default\outputs\default\entry-default-signed.hap"

if ($LogSeconds -le 0) {
    throw "LogSeconds must be greater than 0."
}

if ($DetectExitAfterSeconds -lt 0) {
    throw "DetectExitAfterSeconds must not be negative."
}

if ([string]::IsNullOrWhiteSpace($LogFile)) {
    throw "LogFile must not be empty."
}

if ([System.IO.Path]::IsPathRooted($LogFile)) {
    $LogFile = [System.IO.Path]::GetFullPath($LogFile)
} else {
    $LogFile = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $LogFile))
}

function Assert-Path {
    param(
        [string]$Name,
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Name not found: $Path"
    }
}

function Invoke-Native {
    param(
        [string]$Name,
        [string]$File,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Name"
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

function Test-AppInstalled {
    param(
        [string]$BundleName
    )

    Write-Host ""
    Write-Host "==> check existing APP"
    $ExistingAppInfo = & $Hdc shell bm dump -n $BundleName 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Existing app not found, skip uninstall."
        return $false
    }

    $BundleNamePattern = [regex]::Escape($BundleName)
    $InstalledPattern = '"bundleName":\s*"' + $BundleNamePattern + '"|"name":\s*"' + $BundleNamePattern + '"'
    if ($ExistingAppInfo -match $InstalledPattern) {
        Write-Host "Existing app found: $BundleName"
        return $true
    }

    Write-Host "Existing app not found, skip uninstall."
    return $false
}

function Test-AppProcessRunning {
    param(
        [string]$BundleName
    )

    $PidOutput = & $Hdc shell pidof $BundleName 2>$null
    return -not [string]::IsNullOrWhiteSpace(($PidOutput | Out-String).Trim())
}

function Wake-DeviceScreen {
    Invoke-Native "wake device screen" $Hdc @("shell", "power-shell", "wakeup")
    Start-Sleep -Seconds 1
}

Assert-Path "Project root" $ProjectRoot
Assert-Path "DevEco Node" $Node
Assert-Path "hvigor wrapper" $Hvigor
Assert-Path "ohpm" $Ohpm
Assert-Path "DevEco SDK" $DevecoSdk
Assert-Path "DevEco JBR" "$JavaHome\bin\java.exe"
Assert-Path "hdc" $Hdc

Set-Location $ProjectRoot

$env:DEVECO_SDK_HOME = $DevecoSdk
$env:JAVA_HOME = $JavaHome
$env:Path = "$JavaHome\bin;$env:Path"
Write-Host "DEVECO_SDK_HOME=$env:DEVECO_SDK_HOME"
Write-Host "JAVA_HOME=$env:JAVA_HOME"

Write-Host ""
Write-Host "==> stop stale hvigor daemon"
& $Node $Hvigor --stop-daemon
if ($LASTEXITCODE -ne 0) {
    Write-Host "No stale hvigor daemon stopped, continuing."
}

Invoke-Native "hdc list targets" $Hdc @("list", "targets")

Wake-DeviceScreen

if (Test-AppInstalled $Bundle) {
    Invoke-Native "uninstall existing APP" $Hdc @("uninstall", $Bundle)
}

Invoke-Native "ohpm install" $Ohpm @("install")

Invoke-Native "hvigor clean" $Node @(
    $Hvigor,
    "clean",
    "--mode", "module",
    "-p", "product=default",
    "--analyze=normal",
    "--parallel",
    "--incremental",
    "--daemon"
)

Invoke-Native "hvigor assembleHap" $Node @(
    $Hvigor,
    "--mode", "module",
    "-p", "module=entry@default",
    "-p", "product=default",
    "-p", "requiredDeviceType=phone",
    "assembleHap",
    "--analyze=normal",
    "--parallel",
    "--incremental",
    "--daemon"
)

if (-not (Test-Path -LiteralPath $Hap)) {
    throw "HAP was not generated: $Hap"
}

$HapInfo = Get-Item -LiteralPath $Hap
Write-Host ""
Write-Host "Generated HAP: $($HapInfo.FullName)"
Write-Host "HAP size: $($HapInfo.Length) bytes"
Write-Host "HAP time: $($HapInfo.LastWriteTime)"

Invoke-Native "clear app hilog" $Hdc @("shell", "hilog", "-r", "-t", "app")

$LogPattern = "\s[DIWEF]\s+[0-9A-Fa-f]+/" + [regex]::Escape($Bundle) + "/"
$LogDirectory = Split-Path -Parent $LogFile
if (-not [string]::IsNullOrWhiteSpace($LogDirectory)) {
    New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
}
Remove-Item -LiteralPath $LogFile -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "==> start app log capture"
Write-Host "Capturing APP logs for $LogSeconds seconds."
Write-Host "Exit detection starts after $DetectExitAfterSeconds seconds."
$LogJob = Start-Job -ScriptBlock {
    param($HdcPath, $Pattern, $OutputPath)

    & $HdcPath shell hilog -t app -L D,I,W,E,F |
        Select-String -Pattern $Pattern |
        ForEach-Object { $_.Line } |
        Tee-Object -FilePath $OutputPath
} -ArgumentList $Hdc, $LogPattern, $LogFile

try {
    Invoke-Native "install HAP" $Hdc @("install", "-r", $Hap)

    Wake-DeviceScreen

    Invoke-Native "start ability" $Hdc @("shell", "aa", "start", "-a", $Ability, "-b", $Bundle, "-m", $Module)

    $Deadline = (Get-Date).AddSeconds($LogSeconds)
    $DetectionStart = (Get-Date).AddSeconds($DetectExitAfterSeconds)

    while ((Get-Date) -lt $Deadline) {
        if ((Get-Date) -ge $DetectionStart) {
            if (-not (Test-AppProcessRunning $Bundle)) {
                Write-Host "APP process exited, stopping log capture."
                Start-Sleep -Seconds 2
                break
            }
        }

        Start-Sleep -Seconds 1
    }
}
finally {
    if ($LogJob.State -eq "Running") {
        Stop-Job $LogJob
    }
}

$Logs = Receive-Job $LogJob
Remove-Job $LogJob

Write-Host ""
Write-Host "==> captured APP logs"
if ($Logs) {
    Write-Host "Captured $(@($Logs).Count) APP log lines. Full logs are saved to: $LogFile"
} else {
    Write-Host "No APP logs captured from background hilog job."
}

if (Test-Path -LiteralPath $LogFile) {
    $SavedLogs = Get-Content -LiteralPath $LogFile
    Write-Host ""
    if ($SavedLogs -match "XEngine Vulkan Demo") {
        Write-Host "FOUND native C++ LOGI/LOGW/LOGE output, tag: XEngine Vulkan Demo"
    } else {
        Write-Host "No native C++ LOGI/LOGW/LOGE output found in captured hilog."
    }

    if ($SavedLogs -match "testTag|ImageSegmentationSample") {
        Write-Host "FOUND ArkTS hilog output."
    } else {
        Write-Host "No ArkTS hilog output found in captured hilog."
    }

    Write-Host "APP logs saved to: $LogFile"
} else {
    Write-Host "No APP logs captured."
}

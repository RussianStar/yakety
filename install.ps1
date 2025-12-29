$ErrorActionPreference = "Stop"

function Show-Usage {
    Write-Host "Usage: .\\install.ps1 [clean] [debug] [package] [cli] [app]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  clean    - Clean previous build directories"
    Write-Host "  debug    - Build with Debug configuration"
    Write-Host "  package  - Create distribution packages"
    Write-Host "  cli      - Run yakety-cli after building"
    Write-Host "  app      - Run Yakety after building"
}

$clean = $false
$debug = $false
$package = $false
$cli = $false
$app = $false

foreach ($arg in $args) {
    switch ($arg) {
        "clean"   { $clean = $true }
        "debug"   { $debug = $true }
        "package" { $package = $true }
        "cli"     { $cli = $true }
        "app"     { $app = $true }
        "-h"      { Show-Usage; exit 0 }
        "--help"  { Show-Usage; exit 0 }
        "help"    { Show-Usage; exit 0 }
        default {
            Write-Host "Unknown argument: $arg"
            Show-Usage
            exit 1
        }
    }
}

if ($clean -or $package) {
    Write-Host "Cleaning previous build..."
    Remove-Item -Recurse -Force build, build-debug, "whisper.cpp\\build" -ErrorAction SilentlyContinue
}

$buildDir = if ($debug) { "build-debug" } else { "build" }
$buildConfig = if ($debug) { "Debug" } else { "Release" }

$useNinja = $null -ne (Get-Command ninja -ErrorAction SilentlyContinue)

$cmakeArgs = @("-S", ".", "-B", $buildDir)
if ($useNinja) {
    $cmakeArgs += @("-G", "Ninja", "-DCMAKE_BUILD_TYPE=$buildConfig")
} else {
    $cmakeArgs += @("-G", "Visual Studio 17 2022", "-A", "x64")
}

Write-Host "Configuring $buildConfig build..."
cmake @cmakeArgs

Write-Host "Building $buildConfig..."
cmake --build $buildDir --config $buildConfig

if ($package) {
    Write-Host "Creating packages..."
    cmake --build $buildDir --config $buildConfig --target package
}

$binDir = Join-Path $buildDir "bin"

if ($cli) {
    $cliPath = Join-Path $binDir "yakety-cli.exe"
    if (-not (Test-Path $cliPath)) {
        Write-Error "No CLI executable found at $cliPath"
    }
    Write-Host "Running yakety-cli..."
    & $cliPath
}

if ($app) {
    $appPath = Join-Path $binDir "Yakety.exe"
    if (-not (Test-Path $appPath)) {
        Write-Error "No app executable found at $appPath"
    }
    Write-Host "Running Yakety..."
    Start-Process $appPath
}

Write-Host "Done."

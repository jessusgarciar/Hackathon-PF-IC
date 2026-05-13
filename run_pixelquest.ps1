# PixelQuest Streamlit — arranque compatible con PowerShell (sin &&) y sin prompt de email.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$cred = Join-Path $PSScriptRoot ".streamlit\credentials.toml"
if (Test-Path $cred) {
    $env:STREAMLIT_CREDENTIALS_FILE = (Resolve-Path $cred).Path
}
streamlit run (Join-Path $PSScriptRoot "app_pixelquest.py")

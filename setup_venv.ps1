# Create the virtual environment and install every dependency into it.
Set-Location $PSScriptRoot
if (-not (Test-Path "venv")) {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { py -3 -m venv venv } else { python -m venv venv }
}
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "`nSetup complete. Launch the GUI with .\run_gui.bat or:  .\venv\Scripts\python.exe -m videostego gui"

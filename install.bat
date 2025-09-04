@echo off
REM QENEX OS Windows Installation Script
REM Installs QENEX OS on Windows systems

echo ======================================
echo    QENEX OS Installation Script
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version %PYTHON_VERSION% found

REM Install dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install ^
    aiohttp ^
    cryptography ^
    numpy ^
    psutil ^
    pyyaml ^
    requests ^
    --quiet

if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

REM Create directories
echo Creating directories...
if not exist "%PROGRAMFILES%\QENEX-OS" mkdir "%PROGRAMFILES%\QENEX-OS"
if not exist "%PROGRAMFILES%\QENEX-OS\core" mkdir "%PROGRAMFILES%\QENEX-OS\core"
if not exist "%PROGRAMFILES%\QENEX-OS\data" mkdir "%PROGRAMFILES%\QENEX-OS\data"
if not exist "%PROGRAMFILES%\QENEX-OS\logs" mkdir "%PROGRAMFILES%\QENEX-OS\logs"
if not exist "%PROGRAMFILES%\QENEX-OS\config" mkdir "%PROGRAMFILES%\QENEX-OS\config"

REM Copy files
echo Installing QENEX OS files...
xcopy /E /Y "core" "%PROGRAMFILES%\QENEX-OS\core\"
copy /Y "qenex-os" "%PROGRAMFILES%\QENEX-OS\qenex-os.py"
copy /Y "requirements.txt" "%PROGRAMFILES%\QENEX-OS\"
copy /Y "README.md" "%PROGRAMFILES%\QENEX-OS\"
copy /Y "LICENSE" "%PROGRAMFILES%\QENEX-OS\"

REM Create batch file for easy execution
echo @echo off > "%PROGRAMFILES%\QENEX-OS\qenex-os.bat"
echo python "%PROGRAMFILES%\QENEX-OS\qenex-os.py" %%* >> "%PROGRAMFILES%\QENEX-OS\qenex-os.bat"

REM Add to PATH
echo Adding QENEX-OS to PATH...
setx PATH "%PATH%;%PROGRAMFILES%\QENEX-OS" /M >nul 2>&1

REM Create default configuration
echo Creating default configuration...
(
echo {
echo     "version": "1.0.0",
echo     "ai": {
echo         "enabled": true,
echo         "auto_optimize": true,
echo         "learning_rate": 0.01
echo     },
echo     "security": {
echo         "level": "maximum",
echo         "firewall": true,
echo         "intrusion_detection": true
echo     },
echo     "network": {
echo         "blockchain_sync": true,
echo         "defi_integration": true,
echo         "p2p_enabled": true
echo     },
echo     "performance": {
echo         "cpu_governor": "balanced",
echo         "memory_optimization": "aggressive"
echo     }
echo }
) > "%PROGRAMFILES%\QENEX-OS\config\system.json"

echo.
echo ======================================
echo    QENEX OS installed successfully!
echo ======================================
echo.
echo To get started:
echo   1. Open a new Command Prompt
echo   2. Initialize: qenex-os init
echo   3. Start: qenex-os start
echo   4. Check status: qenex-os status
echo.
pause
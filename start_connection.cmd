@echo off
setlocal

set "RAFT_PYTHON=C:\Users\jdrotleff\AppData\Local\Programs\Python\Python312\python.exe"
set "RAFT_ROOT=%~dp0"

if not exist "%RAFT_PYTHON%" (
    echo RaftControl could not find Python at:
    echo   %RAFT_PYTHON%
    echo Update RAFT_PYTHON in start_connection.cmd if Python was moved.
    exit /b 1
)

set "PYTHONPATH=%RAFT_ROOT%src"
pushd "%RAFT_ROOT%"
"%RAFT_PYTHON%" -m raft_control.server --config "%RAFT_ROOT%configs\windows.json"
set "RAFT_EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %RAFT_EXIT_CODE%

ECHO OFF

:: https://superuser.com/questions/198525/how-can-i-execute-a-windows-command-line-in-background
:: https://forums.unrealengine.com/t/resolution-and-window-mode/420128
START /B Blocks.exe -ResX=1024 -ResY=768 -WINDOWED
ECHO Hello!

:: https://serverfault.com/questions/432322/how-to-sleep-in-a-batch-file
timeout /t 5 /nobreak

ECHO "Run the python script. "
python X:\Windows\scripts\AirSim_SampleCubeClient\cube\run.py

:: https://superuser.com/questions/1615253/end-a-process-started-with-start-command-in-a-windows-batch-file
ECHO wmic
wmic process where "name like 'Blocks.exe'" delete
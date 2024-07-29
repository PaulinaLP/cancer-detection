to install make for windows if you don't have choco you must first:
Open PowerShell as Administrator and Run the following command to install Chocolatey:
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

then you can run: choco install make

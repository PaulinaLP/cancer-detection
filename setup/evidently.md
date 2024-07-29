cd monitoring
install dependencies with pipenv
pipenv shell
you can see the dashboard with command: evidently ui

if you need to reset the ui:
on linux:
lsof -i :8000
kill -9 <pid>

on windows:
netstat -ano | findstr :8000
TASKKILL /PID <pid> /F


the script create_dashboard creates a new dashboard
the script monitoring has a function to add new report to this dashboard
with testing, it is posible to add new report
cd monitoring
install dependencies with pipenv
pipenv shell
you can see the dashboard with command: evidently ui

if you need to reset the ui:
lsof -i :8000
kill -9 <pid>

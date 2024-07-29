cd to webservice
install pipenv and then install dependencies from pipfile
execute pipenv shell
execute python predict.py

if you want to test it:
open another terminal
cd to webservice
execute pipenv shell
execute testing.py to test the app

if you want to run the app in docker:
cd to webservice
execute docker build -t my-predict-app .
execute docker run -d -p 9696:9696 my-predict-app

you can test it again

there are also some tests prepared in the folder tests:
one unit test to test preprocessing
one integration test to test the service
you can run them using pytest, it is specified in pipfile only for dev
First you have to cd to airflow directory.
You can find there the Dockerfile that adds some necessary imports to airflow image.
The docker compose file is also there, it is based on
'https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml'
but some volumes are added for input/output and mlflow service.

Then run this command on your terminal.
docker compose up airflow-init
Then run:
docker compose build
Then run:
docker compose up
Wait till webserver and scheduler are up. You can open another terminal and do docker ps.
There should be 6 containers up and healthy.

If you made any mistake and want to delete the containers you should.
docker stop $(docker ps -a -q)
and then
docker rm $(docker ps -a -q)

If you change Dockerfile you must do:
docker compose down
And then again: 
docker compose build
and 
docker compose up

The webserver works on port 8080.
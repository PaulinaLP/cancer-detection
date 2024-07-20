https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html
First you have to download docker compose file.
urlc -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml'
You must change the volumes part if you want export / import files.
Create airflow directory, leave the file there and cd to this directory.
Then run this command on your terminal.
docker compose up airflow-init
Then run:
docker compose up
Wait till webserver and scheduler are up. You can open another terminal and do docker ps.
There should be 6 containers up and healthy.
If you made any mistake and want to delete the containers you should.
docker stop $(docker ps -a -q)
and then
docker rm $(docker ps -a -q)
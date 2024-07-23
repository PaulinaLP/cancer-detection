# Start from the Airflow image
FROM apache/airflow:2.9.2

USER root
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

# Switch back to the airflow user
USER airflow
# Install Python dependencies
RUN pip install scikit-learn==1.5.1 pandas==2.2.2 mlflow==2.14.3 hyperopt==0.2.7 lightgbm==4.4.0


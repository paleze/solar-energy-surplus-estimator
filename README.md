# solar-energy-surplus-estimator
A ML project that predicts how much energy surplus will be produced today from my solar panels


# Airflow stack
To start the airflow stack for the first time:
- optionally configure env variable `AIRFLOW_PROJ_DIR` (it defaults to the current folder if not set)
- run `setup-airflow.sh` to perform these actions:
    - set env var `AIRFLOW_UID` to the current user id and append it to a local `.env` file
    - create airflow folders according in `AIRFLOW_PROJ_DIR`
- run `docker compose -f docker-compose-airflow.yaml up -d` to start stack containers
- acccess airflow at `http://localhost:8080` (change the port accordingly if differently configured)

To stop the airflow stack run `docker compose -f docker-compose-airflow.yaml stop`

To completely remove the airflow stack run `clean-airflow.sh` to perform these actions:
- run `docker compose -f docker-compose-airflow.yaml down --volumes --remove-orphans`
- remove all airflow folders from `AIRFLOW_PROJ_DIR` (or from current folder if not set)

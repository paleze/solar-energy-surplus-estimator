# solar-energy-surplus-estimator
A Machine Learning project that predicts how much energy surplus produced by my solar panels will not fit in the home battery storage and will be therefore exported to the grid


# Initial setup

Follow these steps to setup the environment:

- rename the `.env.demo` file into `.env` and change environment variables if needed or default values will be applied.

- optionally configure env variable `AIRFLOW_PROJ_DIR`. It defaults to the current folder if not set

- run `setup.sh` to perform these actions:
    - set env var `AIRFLOW_UID` to the current user id and append it to a local `.env` file
    - create airflow folders according to `AIRFLOW_PROJ_DIR`
    - setup monitoring folders
    - build custom images

- run `docker compose up -d` to start stack containers (you can have a nice view of running containers by executing `watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}"` in another terminal)

- open the `Minio`, `MlFlow`, and `Airflow` web pages on `localhost` at the corresponding configured ports

Run `docker compose stop` to stop the stack without removing containers and `docker compose up -d` to start it again.


# Cleanup
Run `cleanup.sh` to:
- take the stack down and remove volumes and orphans ( executes `docker compose down --volumes --remove-orphans` )
- remove all airflow folders from `AIRFLOW_PROJ_DIR` (or from current folder if not set)
- remove custom airflow and model serving app image


# Model serving application
Model serving is implemented with a Streamlit application accessible by default at `http://localhost:8501`.
In order to get an energy surplus estimate prediction, the user must enter this information:
- day of interest (default current day)
- time of the day (default current time)
- home battery capacity (default 13.5 Kwh)
- current battery capacity usage (default 0)

By clicking the **Estimate surplus** button, the amount of predicted solar energy surplus is returned, i.e. the energy that will be produced by solar panels till the end of the day and not immediately consumed by home appliances. This value is computed by summing up the energy surplus quantities predicted to be produced in 30-minutes steps from the given time of day until 9:00 pm. The prediction leverages a Machine Learning model trained on the surplus quantities historically produced in the same month, time of day and weather conditions according to home storage data. As this surplus is usually stored in the home battery, the goal is then to predict how much energy will eventually overflow its capacity and be exported to the grid according to current capacity usage. This is the amount of energy that could be used for other non house-related appliances (for example, to charge an electric car).

The **Perform Drift Analysys** button runs a feature drift analisys by considering the amount of samples and the drift threshold set by the user. In a real-world scenario, this analisys is automatically computed when enough data is available in order to trigger an automatic model retraining to restore its prediction performance using a dataset of drifted samples. For demo purposes, this functionality can be manually triggered here by the user when a drift is detected by forcibly adjusting these parameters.


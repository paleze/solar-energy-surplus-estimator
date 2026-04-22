# solar-energy-surplus-estimator
A Machine Learning project that predicts how much energy surplus produced by my solar panels will not fit in the home battery storage and will be therefore exported to the grid


# Setup
Follow these steps to setup the environment:

- copy `.env.demo` file into `.env` and optionally change environment variables
- run `./start.sh` to finalize environment setup and start the containers (you can have a nice view of running containers by executing `watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Ports}}"'` )
- click on the links displayed to open components web pages (credendials in `.env` file when needed)

Run `docker compose stop` to stop the stack without removing containers and `docker compose up -d` to start it again.


# Cleanup
Run `./cleanup.sh` to:
- take the stack down and remove volumes and orphans ( it executes `docker compose down --volumes --remove-orphans` )
- remove data created by containers
- remove custom airflow and model serving app image


# Model serving application
Model serving is implemented with a Streamlit application accessible by default at `http://localhost:8501`.
In order to get an energy surplus estimate prediction, the user must enter this information:
- day of interest (default current day)
- time of the day (default current time)
- home battery capacity (default 13.5 Kwh)
- current battery capacity usage (default 0)

By clicking the **Estimate surplus** button, the amount of predicted solar energy surplus is returned, i.e. the energy that will be produced by solar panels till the end of the day and not immediately consumed by home appliances. This value is computed by summing up the energy surplus quantities predicted to be produced in 30-minutes steps from the given time of day until 9:00 pm. These predictions leverage a Machine Learning model trained on the surplus quantities produced in the same month, time of day and weather conditions according to historical data. As this surplus is usually stored in the home battery, the goal is then to predict how much energy will eventually overflow its capacity and be exported to the grid according to current capacity usage. This is the amount of energy that can be used for other non house-related appliances (for example, for charging an electric car) while assuring that the battery will be at full capacity at sunset to power the house during the night.

The **Perform Drift Analysis** button runs a feature drift analysis by considering the amount of samples and the drift threshold set by the user. In a real-world scenario, this analysis is automatically computed when enough data is available in order to trigger an automatic model retraining using an updated dataset to restore the prediction performance. For demo purposes, this functionality can be manually triggered by the user by forcibly adjusting these parameters in order to force a drift detection. The new model training pipeline execution can be followed in the Airflow web page.



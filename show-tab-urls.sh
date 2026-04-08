
echo "Right-click on these links to open project components web pages (default credentials in README.md or specific ones in .env file):"
echo
echo "Minio:      http://localhost:${MINIO_HOST_CONSOLE_PORT:-9001}"
echo "Airflow:    http://localhost:8080/dags/solar_energy_surplus_estimator_model_pipeline" 
echo "MLflow:     http://localhost:${MLFLOW_PORT:-5000}"
echo "Estimator:  http://localhost:${APP_SERVING_HOST_PORT:-8501}"
echo "Prometheus: http://localhost:${PROMETHEUS_PORT:-9090}"
echo "Grafana:    http://localhost:${GRAFANA_PORT:-3000}/dashboards"
echo "Flower:     http://localhost:${FLOWER_PORT:-5555}"

from prometheus_client import Counter, Gauge, Histogram

from streamlit_extras.prometheus import streamlit_registry

registry = streamlit_registry()

prediction_counter = Counter(
    name='predictions_total',
    documentation='Total number of predictions made',
    labelnames=('time', 'month',),
    registry=registry
)

prediction_latency = Histogram(
    name='prediction_latency_seconds',
    documentation='Prediction latency in seconds',
    labelnames=('time', 'month',),
    registry=registry
)

drift_score_gauge = Gauge(
    name='drift_score',
    documentation='Data drift score by feature',
    labelnames=('feature', 'method','threshold',),
    registry=registry
)

drift_feature_counter = Counter(
    name='drift_features_total',
    documentation='Total number of drift detected per feature',
    labelnames=('feature','method','threshold',),
    registry=registry
)

feature_mean_gauge = Gauge(
    name='feature_mean',
    documentation='Mean value of features in recent data',
    labelnames=('feature',),
    registry=registry
)

feature_std_gauge = Gauge(
    name='feature_std',
    documentation='Standard deviation of features in recent data',
    labelnames=('feature',),
    registry=registry
)
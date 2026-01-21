import os
import random

from locust import HttpUser, between, task

SAMPLE_TEXTS = [
    "STARBUCKS",
    "UBER TRIP",
    "WHOLE FOODS MARKET",
    "AMZN MKTPLACE PMTS",
    "NETFLIX.COM",
    "SHELL OIL",
    "WALMART SUPERCENTER",
    "TARGET",
    "DELTA AIR LINES",
    "SPOTIFY",
]

PREDICT_PATH = os.getenv("LOCUST_PREDICT_PATH", "/predict")
HEALTH_PATH = os.getenv("LOCUST_HEALTH_PATH", "/health")
BATCH_SIZE = int(os.getenv("LOCUST_BATCH_SIZE", "1"))


def _payload() -> dict[str, object]:
    if BATCH_SIZE <= 1:
        return {"text": random.choice(SAMPLE_TEXTS)}
    return {"texts": [random.choice(SAMPLE_TEXTS) for _ in range(BATCH_SIZE)]}


class APILoadUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(5)
    def predict(self) -> None:
        self.client.post(PREDICT_PATH, json=_payload(), name="predict")

    @task(1)
    def health(self) -> None:
        self.client.get(HEALTH_PATH, name="health")

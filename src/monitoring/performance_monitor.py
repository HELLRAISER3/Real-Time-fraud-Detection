from collections import deque
from logger.log import logging

class SimplePerformanceMonitor:

    def __init__(self):
        self.total_requests = 0
        self.failed_requests = 0
        self.latencies = deque(maxlen=100)  #last 100 latencies
        logging.info("Performance monitor started")

    def record_request(self, latency_seconds: float, success: bool = True):
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
        self.latencies.append(latency_seconds)

    def get_stats(self) -> dict:
        if len(self.latencies) == 0:
            avg_latency = 0.0
        else:
            avg_latency = sum(self.latencies) / len(self.latencies)

        error_rate = 0.0
        if self.total_requests > 0:
            error_rate = (self.failed_requests / self.total_requests) * 100

        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "error_rate_percent": round(error_rate, 2),
            "avg_latency_seconds": round(avg_latency, 3)
        }


_monitor = SimplePerformanceMonitor()


def get_monitor():
    return _monitor

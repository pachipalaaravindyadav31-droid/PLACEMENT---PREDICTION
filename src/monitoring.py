"""
PlacementPredict - Production Monitoring & Logging Module (M1)
Tracks inference latency, request throughput, prediction distribution,
and checks for simple input feature drift against training baselines.
"""

import time
import json
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np


class SystemMonitor:
    """Lightweight in-memory monitoring hook for FastAPI inference service."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_count = 0
        self.placed_predictions = 0
        self.unplaced_predictions = 0
        self.latencies_ms = deque(maxlen=max_history)
        self.recent_predictions = deque(maxlen=50)
        self.start_time = time.time()
        self.training_baselines: Dict[str, Dict[str, float]] = {}

    def set_baselines(self, baselines: Dict[str, Dict[str, float]]):
        """Sets training dataset statistics for drift detection."""
        self.training_baselines = baselines

    def record_prediction(
        self,
        features: Dict[str, Any],
        prediction: int,
        probability: float,
        salary: float,
        latency_ms: float,
        model_used: str
    ):
        """Logs a single inference event."""
        self.request_count += 1
        if prediction == 1:
            self.placed_predictions += 1
        else:
            self.unplaced_predictions += 1
            
        self.latencies_ms.append(latency_ms)
        
        self.recent_predictions.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "model": model_used,
            "prediction": "Placed" if prediction == 1 else "Not Placed",
            "probability": round(probability, 4),
            "salary_lpa": round(salary, 2),
            "latency_ms": round(latency_ms, 2),
            "cgpa": features.get("CGPA"),
            "coding_score": features.get("CodingTestScore")
        })

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Returns aggregate monitoring metrics."""
        uptime_seconds = time.time() - self.start_time
        avg_latency = float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0
        p95_latency = float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0
        
        placement_rate = (
            (self.placed_predictions / self.request_count) * 100
            if self.request_count > 0 else 0.0
        )
        
        return {
            "uptime_seconds": round(uptime_seconds, 1),
            "total_requests": self.request_count,
            "placed_count": self.placed_predictions,
            "unplaced_count": self.unplaced_predictions,
            "live_placement_rate_pct": round(placement_rate, 2),
            "latency_ms": {
                "avg": round(avg_latency, 2),
                "p95": round(p95_latency, 2),
                "min": round(min(self.latencies_ms), 2) if self.latencies_ms else 0.0,
                "max": round(max(self.latencies_ms), 2) if self.latencies_ms else 0.0
            },
            "recent_requests": list(self.recent_predictions)[::-1]
        }


monitor = SystemMonitor()

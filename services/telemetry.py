from typing import List, Dict, Any
import time
from typing import Dict, Any

class TelemetryService:
    """A service to track events and activity logs from agents."""
    
    def __init__(self):
        self.event_logs: List[Dict[str, Any]] = []
        self.token_usage: Dict[str, int] = {}
        self.cost_tracking: Dict[str, float] = {}
    
    def log_event(self, agent_id: str, event_type: str, details: Dict[str, Any]) -> None:
        """Log an event from an agent."""
        timestamp = time.time()
        event = {
            "agent_id": agent_id,
            "event_type": event_type,
            "details": details,
            "timestamp": timestamp
        }
        self.event_logs.append(event)
    
    def track_token_usage(self, agent_id: str, tokens: int) -> None:
        """Track token usage for an agent."""
        if agent_id in self.token_usage:
            self.token_usage[agent_id] += tokens
        else:
            self.token_usage[agent_id] = tokens
    
    def track_cost(self, agent_id: str, cost: float) -> None:
        """Track cost associated with an agent's operations."""
        if agent_id in self.cost_tracking:
            self.cost_tracking[agent_id] += cost
        else:
            self.cost_tracking[agent_id] = cost
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Generate a report of token usage and costs."""
        return {
            "token_usage": self.token_usage,
            "cost_tracking": self.cost_tracking,
            "event_logs": self.event_logs
        }

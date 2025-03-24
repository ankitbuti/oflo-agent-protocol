from aimon import AIMon
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize AIMon with API key
aimon = AIMon(api_key=os.getenv('AIMON_API_KEY'))

# Configure default guardrails
DEFAULT_CONFIG = {
    "hallucination": {
        "detector_name": "default",
        "threshold": 0.7  # Higher threshold = stricter check
    },
    "completeness": {
        "enabled": True,
        "min_score": 0.8  # Minimum completeness score
    },
    "conciseness": {
        "enabled": True,
        "max_length": 1000  # Maximum response length
    },
    "toxicity": {
        "enabled": True,
        "threshold": 0.7  # Higher threshold = less tolerance
    },
    "instruction_adherence": {
        "enabled": True,
        "min_score": 0.8  # Minimum adherence score
    }
}

# Export configured instance
__all__ = ['aimon', 'DEFAULT_CONFIG']

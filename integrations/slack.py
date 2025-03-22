"""Slack integration."""
from typing import Optional
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class SlackNotifier:
    """Simple Slack notification integration."""
    
    def __init__(self, token: str, default_channel: str = "#general"):
        self.client = WebClient(token=token)
        self.default_channel = default_channel

    async def notify(self, message: str, channel: Optional[str] = None) -> None:
        """Send a notification to Slack."""
        try:
            self.client.chat_postMessage(
                channel=channel or self.default_channel,
                text=message
            )
        except SlackApiError as e:
            print(f"Error sending to Slack: {e.response['error']}") 
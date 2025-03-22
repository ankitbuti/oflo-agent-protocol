from typing import List, Dict
from oflo_agent_interface import BaseOfloAgent, Message, OfloFunctionDefinition as FunctionDefinition

class MarketingOfloAgent(BaseOfloAgent):
    """
    Marketing-focused agent implementation.
    """
    
    def __init__(self, name: str = "Marketing", purpose: str = None):
        super().__init__(
            name=name, 
            purpose=purpose or "Assists with marketing tasks, content creation, and campaign planning"
        )
        
    @property
    def capabilities(self) -> List[str]:
        base_capabilities = super().capabilities
        return base_capabilities + ["content_generation", "campaign_planning"]
    
    @property
    def available_functions(self) -> List[FunctionDefinition]:
        return [
            FunctionDefinition(
                name="generate_campaign_idea",
                description="Generate a marketing campaign idea",
                parameters={
                    "type": "object",
                    "properties": {
                        "product": {
                            "type": "string",
                            "description": "Product or service to create a campaign for"
                        },
                        "target_audience": {
                            "type": "string",
                            "description": "Target audience description"
                        },
                        "objective": {
                            "type": "string",
                            "description": "Campaign objective"
                        }
                    }
                },
                required=["product", "target_audience"]
            )
        ]
    
    async def _generate_response(self, message: Message) -> Message:
        """Generate a marketing-focused response."""
        # This would call a marketing-specialized model in a real implementation
        return Message(
            role="assistant", 
            content=f"Marketing agent response to: {message.content}"
        )
    
    async def _func_generate_campaign_idea(self, product: str, target_audience: str, objective: str = None) -> Dict:
        """Generate a marketing campaign idea."""
        # This would use a specialized marketing model in a real implementation
        idea = (
            f"Campaign for {product} targeting {target_audience}. "
            f"Objective: {objective or 'Increase brand awareness'}."
        )
        return {
            "campaign_name": f"{product.title()} Excellence",
            "tagline": f"Experience the difference with {product}",
            "channels": ["social media", "email", "content marketing"],
            "idea_summary": idea
        }

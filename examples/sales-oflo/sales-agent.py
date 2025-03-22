
class SalesOfloAgent(BaseOfloAgent):
    """
    Sales-focused agent implementation.
    """
    
    def __init__(self, name: str = "Sales", purpose: str = None):
        super().__init__(
            name=name, 
            purpose=purpose or "Assists with sales activities, lead qualification, and customer engagement"
        )
        
    @property
    def capabilities(self) -> List[str]:
        base_capabilities = super().capabilities
        return base_capabilities + ["lead_qualification", "proposal_generation"]
    
    @property
    def available_functions(self) -> List[FunctionDefinition]:
        return [
            FunctionDefinition(
                name="qualify_lead",
                description="Qualify a sales lead",
                parameters={
                    "type": "object",
                    "properties": {
                        "company": {
                            "type": "string",
                            "description": "Company name"
                        },
                        "contact_name": {
                            "type": "string",
                            "description": "Contact name"
                        },
                        "industry": {
                            "type": "string",
                            "description": "Industry"
                        },
                        "budget": {
                            "type": "string",
                            "description": "Budget information"
                        },
                        "needs": {
                            "type": "string",
                            "description": "Customer needs"
                        }
                    }
                },
                required=["company", "needs"]
            )
        ]
    
    async def _generate_response(self, message: Message) -> Message:
        """Generate a sales-focused response."""
        # This would call a sales-specialized model in a real implementation
        return Message(
            role="assistant", 
            content=f"Sales agent response to: {message.content}"
        )
    
    async def _func_qualify_lead(self, company: str, needs: str, contact_name: str = None, 
                               industry: str = None, budget: str = None) -> Dict:
        """Qualify a sales lead."""
        # This would use a specialized sales model in a real implementation
        score = 0
        reasons = []
        
        if budget and "high" in budget.lower():
            score += 3
            reasons.append("High budget")
        
        if industry and industry.lower() in ["tech", "healthcare", "finance"]:
            score += 2
            reasons.append("Target industry")
        
        if "urgent" in needs.lower():
            score += 2
            reasons.append("Urgent need")
        
        qualification = "cold"
        if score >= 5:
            qualification = "hot"
        elif score >= 3:
            qualification = "warm"
        
        return {
            "company": company,
            "contact": contact_name or "Unknown",
            "qualification": qualification,
            "score": score,
            "reasons": reasons,
            "next_steps": ["Schedule discovery call", "Send product information"]
        }


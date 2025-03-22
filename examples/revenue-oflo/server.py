from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
from mcp import MCPServer, MCPRequest, MCPResponse, MCPChannel

app = FastAPI()
mcp_server = MCPServer()

class RevenueResponse(BaseModel):
    company: str
    revenue: float
    year: int
    currency: str = "USD"

@mcp_server.channel("revenue_search")
async def handle_revenue_search(request: MCPRequest) -> MCPResponse:
    """
    Handle revenue search requests through MCP.
    Expected request format: {"company": "company_name", "year": optional_year}
    """
    try:
        # Extract search parameters from request
        params = request.data
        company = params.get("company")
        year = params.get("year")
        
        if not company:
            return MCPResponse(
                success=False,
                error="Company name is required"
            )

        # Mock API call - replace with your actual API endpoint
        async with httpx.AsyncClient() as client:
            # Replace with your actual API endpoint
            response = await client.get(
                "https://api.example.com/revenue",
                params={"company": company, "year": year}
            )
            
            if response.status_code == 200:
                data = response.json()
                revenue_data = RevenueResponse(
                    company=data["company"],
                    revenue=data["revenue"],
                    year=data["year"],
                    currency=data.get("currency", "USD")
                )
                
                return MCPResponse(
                    success=True,
                    data=revenue_data.dict()
                )
            else:
                return MCPResponse(
                    success=False,
                    error=f"API request failed with status {response.status_code}"
                )
                
    except Exception as e:
        return MCPResponse(
            success=False,
            error=str(e)
        )

@app.on_event("startup")
async def startup():
    await mcp_server.start()

@app.on_event("shutdown")
async def shutdown():
    await mcp_server.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
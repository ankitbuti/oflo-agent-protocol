# MCP Revenue Search Server

This is a Model Context Protocol (MCP) server that provides revenue search functionality through a REST API interface.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python server.py
```

The server will start on `http://localhost:8000`

## Usage

The server exposes an MCP channel called "revenue_search" that accepts requests in the following format:

```json
{
    "company": "company_name",
    "year": 2023  // optional
}
```

The response will be in the format:

```json
{
    "success": true,
    "data": {
        "company": "company_name",
        "revenue": 1000000.0,
        "year": 2023,
        "currency": "USD"
    }
}
```

## Error Handling

If there's an error, the response will be:

```json
{
    "success": false,
    "error": "Error message here"
}
```

## Configuration

To use with your own REST API endpoint, modify the URL in `server.py`:

```python
response = await client.get(
    "https://api.example.com/revenue",  # Replace with your API endpoint
    params={"company": company, "year": year}
)
``` 
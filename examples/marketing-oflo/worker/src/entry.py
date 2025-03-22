from js import Response, URL, crypto, JSON
import json
import sys

from marketing_agent import MarketingOfloAgent

# Initialize the marketing agent
marketing_agent = MarketingOfloAgent(
	name="Marketing Assistant",
	purpose="A specialized marketing assistant that helps with campaign planning and content generation"
)

async def on_fetch(request, env):
	url = URL.new(request.url)
	
	# Handle agent endpoints
	if url.pathname == '/agent/info':
		# Return agent information in Cloudflare format
		return Response.new(
			JSON.stringify(marketing_agent.to_cloudflare_format()),
			{'headers': {'Content-Type': 'application/json'}}
		)
	
	elif url.pathname == '/agent/process':
		# Process messages sent to the agent
		if request.method != 'POST':
			return Response.new('Method not allowed', {'status': 405})
		
		try:
			# Parse the request body
			body = await request.json()
			message = body.get('message')
			
			if not message:
				return Response.new(
					JSON.stringify({'error': 'Message is required'}),
					{'status': 400, 'headers': {'Content-Type': 'application/json'}}
				)
			
			# Process the message through the agent
			response = await marketing_agent.process_message(message)
			
			# Return the response
			return Response.new(
				JSON.stringify(response.to_dict()),
				{'headers': {'Content-Type': 'application/json'}}
			)
			
		except Exception as e:
			return Response.new(
				JSON.stringify({'error': str(e)}),
				{'status': 500, 'headers': {'Content-Type': 'application/json'}}
			)
	
	# Keep existing endpoints for compatibility
	elif url.pathname == '/message':
		return Response.new('Hello, Ankit!')
	elif url.pathname.startswith('/echo/'):
		msg = url.pathname.split('/echo/', 1)[1]
		return Response.new("Echo: " + msg)
	elif url.pathname == '/random':
		return Response.new(crypto.randomUUID())
	
	return Response.new('Not Found', {'status': 404})

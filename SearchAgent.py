import os
import requests
import json
# Configuration: set your API keys as environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')  # for search tool

# Function: call OpenAI Chat Completions with function calling
def call_llm(messages, functions=None):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }
    payload = {
        'model': 'gpt-4-0613',
        'messages': messages
    }
    if functions:
        payload['functions'] = functions
        payload['function_call'] = 'auto'

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Tool: simple web search using SerpAPI
def search_tool(query):
    url = 'https://serpapi.com/search.json'
    params = {
        'q': query,
        'api_key': SERPAPI_API_KEY,
        'num': 3
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get('organic_results', []):
        title = item.get('title')
        link = item.get('link')
        snippet = item.get('snippet')
        results.append({'title': title, 'link': link, 'snippet': snippet})
    return results

# Function definition for LLM function-calling
search_fn = {
    'name': 'search_tool',
    'description': 'Useful for when you need to search the web for up-to-date information',
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Search query'
            }
        },
        'required': ['query']
    }
}

# Main interactive agent loop
def main():
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant. You have access to a tool called search_tool for web searches.'}
    ]

    while True:
        user_input = input('\nUser: ')
        if user_input.lower() in ('exit', 'quit'):
            print('Agent: Goodbye!')
            break

        # Add user message
        messages.append({'role': 'user', 'content': user_input})

        # First LLM call
        response = call_llm(messages, functions=[search_fn])
        message = response['choices'][0]['message']

        # If model wants to call a function
        if message.get('function_call'):
            fn_name = message['function_call']['name']
            fn_args = json.loads(message['function_call']['arguments'])

            if fn_name == 'search_tool':
                query = fn_args.get('query')
                results = search_tool(query)
                # Send back function result
                messages.append(message)
                messages.append({
                    'role': 'function',
                    'name': fn_name,
                    'content': json.dumps(results)
                })
                # Second LLM call to get final answer
                second_resp = call_llm(messages)
                final_msg = second_resp['choices'][0]['message']
                print(f"Agent: {final_msg['content']}")
                messages.append(final_msg)
            else:
                print(f"Agent: Unknown function '{fn_name}'")
        else:
            # Direct reply from model
            print(f"Agent: {message['content']}")
            messages.append(message)

if __name__ == '__main__':
    main()

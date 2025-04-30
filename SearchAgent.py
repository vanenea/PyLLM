import requests
import json

jsonData = None
with open('apikey.json', 'r') as load_f:
    data = load_f.read()
    jsonData = json.loads(data)

DEEPSEEK_URL = jsonData['deepseekUrl']
DEEPSEEK_API_KEY = jsonData['deepseekKey']
SERPAPI_URL = jsonData['serpUrl']
SERPAPI_API_KEY = jsonData['serpApiKey']
# 调用大模型
def call_llm(messages, functions=None):
    # 用deepSeek的url
    url = f'{DEEPSEEK_URL}/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }
    payload = {
        'model': 'deepseek-chat',
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
    url = f'{SERPAPI_URL}'
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

def knowledge_qa(query):
    # Search for page
    search_url = 'https://en.wikipedia.org/w/api.php'
    search_params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json',
        'srlimit': 1
    }
    search_resp = requests.get(search_url, params=search_params)
    search_resp.raise_for_status()
    search_data = search_resp.json()
    results = search_data.get('query', {}).get('search', [])
    if not results:
        return f'No results found for "{query}".'

    title = results[0]['title']

    # Fetch summary
    summary_url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{title}'
    summary_resp = requests.get(summary_url)
    summary_resp.raise_for_status()
    summary_data = summary_resp.json()
    return summary_data.get('extract', f'Could not fetch summary for "{title}".')

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

qa_fn = {
    'name': 'knowledge_qa',
    'description': 'Answer questions by retrieving information from Wikipedia',
    'parameters': {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'The question or topic to look up in Wikipedia'
            }
        },
        'required': ['query']
    }
}
def agentSearch():
    messages = [
        {'role': 'system',
         'content': 'You are a helpful assistant. You have access to a tool called search_tool for web searches.'}
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
        print("大模型返回数据: ", message)
        if message.get('function_call'):
            fn_name = message['function_call']['name']
            fn_args = json.loads(message['function_call']['arguments'])

            if fn_name == 'search_tool':
                query = fn_args.get('query')
                results = search_tool(query)
                print(f"搜索工具返回:{results}")
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

def agentQA():
    messages = [
        {'role': 'system',
         'content': 'You are a helpful assistant with access to a Wikipedia-based knowledge retrieval tool.'}
    ]

    while True:
        user_input = input('\nUser: ')
        if user_input.lower() in ('exit', 'quit'):
            print('Agent: Goodbye!')
            break

        messages.append({'role': 'user', 'content': user_input})
        response = call_llm(messages, functions=[qa_fn])
        message = response['choices'][0]['message']
        print("大模型返回数据: ", message)
        if message.get('function_call'):
            fn_name = message['function_call']['name']
            fn_args = json.loads(message['function_call']['arguments'])

            if fn_name == 'knowledge_qa':
                query = fn_args.get('query')
                answer = knowledge_qa(query)
                print("维基百科返回数据: ", answer)
                messages.append(message)
                messages.append({
                    'role': 'function',
                    'name': fn_name,
                    'content': json.dumps({'answer': answer})
                })
                follow_up = call_llm(messages)
                final_msg = follow_up['choices'][0]['message']
                print(f"Agent: {final_msg['content']}")
                messages.append(final_msg)
            else:
                print(f"Agent: Unknown function '{fn_name}'")
        else:
            print(f"Agent: {message['content']}")
            messages.append(message)

# Main interactive agent loop
def main():
    agentQA()

if __name__ == '__main__':
    main()

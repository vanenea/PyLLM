from openai import OpenAI

def chatWithDeepseek():
    r = open("./apikey.txt", "r")
    rd = r.read()
    r.close()
    print(rd)
    client = OpenAI(api_key=str(rd), base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello, What is the land area of China?"},
        ],
        stream=False
    )

    print(response.choices[0].message.content)
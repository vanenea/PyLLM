import os
from http import HTTPStatus
from dashscope import Application

def knowledge_qa():
    response = Application.call(
        # 若没有配置环境变量，可用百炼API Key将下行替换为：api_key="sk-xxx"。但不建议在生产环境中直接将API Key硬编码到代码中，以减少API Key泄露风险。
        api_key=os.getenv("sk-25495ab1a1664344844fd0f85eab29aa"),
        app_id='app_component_cc9b691b0f5d484abdd2f13fdf4915b2',# 替换为实际的应用 ID
        prompt='营销吗入口在哪里')

    if response.status_code != HTTPStatus.OK:
        print(f'request_id={response.request_id}')
        print(f'code={response.status_code}')
        print(f'message={response.message}')
        print(f'请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code')
    else:
        print(response.output.text)

if __name__ == '__main__':
    knowledge_qa()
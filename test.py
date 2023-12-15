from wenxinworkshop import LLMAPI, EmbeddingAPI, PromptTemplateAPI
from wenxinworkshop import Message, Messages, Texts

# 前往官网：https://cloud.baidu.com/product/wenxinworkshop 申请服务获取
api_key = ''
secret_key = ''

# create a LLM API
erniebot = LLMAPI(
    api_key=api_key,
    secret_key=secret_key,
    url=LLMAPI.ERNIEBot
)

# create a message
message = Message(
    role='user',
    content='你好！'
)

# create messages
messages: Messages = [message]

# get response from LLM API
response = erniebot(
    messages=messages,
    temperature=None,
    top_p=None,
    penalty_score=None,
    stream=None,
    user_id=None,
    chunk_size=512
)

# print response
print(response)

# get response stream from LLM API
# response_stream = erniebot(
#     messages=messages,
#     temperature=None,
#     top_p=None,
#     penalty_score=None,
#     stream=True,
#     user_id=None,
#     chunk_size=512
# )

# # print response stream
# for item in response_stream:
#     print(item, end='')
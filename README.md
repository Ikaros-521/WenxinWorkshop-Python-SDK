# WenxinWorkshop Python SDK
* A third-party Python SDK for a WenxinWorkshop.

# 官网
[https://cloud.baidu.com/product/wenxinworkshop](https://cloud.baidu.com/product/wenxinworkshop)  

## Quick Start
* Install wenxinworkshop SDK

    ```bash
    $ pip install git+https://github.com/Ikaros-521/WenxinWorkshop-Python-SDK
    ```

* Import wenxinworkshop SDK

    ```python
    from wenxinworkshop import LLMAPI, EmbeddingAPI, PromptTemplateAPI
    from wenxinworkshop import Message, Messages, Texts
    ```

* Set API key and Secret key

    ```python
    api_key = '...'
    secret_key = '...'
    ```

* LLM chat

    ```python
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
    response_stream = erniebot(
        messages=messages,
        temperature=None,
        top_p=None,
        penalty_score=None,
        stream=True,
        user_id=None,
        chunk_size=512
    )

    # print response stream
    for item in response_stream:
        print(item, end='')
    ```

* Embedding

    ```python
    # create a Embedding API
    ernieembedding = EmbeddingAPI(
        api_key=api_key,
        secret_key=secret_key,
        url=EmbeddingAPI.EmbeddingV1
    )

    # create texts
    texts: Texts = [
        '你好！',
        '你好吗？',
        '你是谁？'
    ]

    # get embeddings from Embedding API
    response = ernieembedding(
        texts=texts,
        user_id=None
    )

    # print embeddings
    print(response)
    ```

* Get prompt template

    ```python
    # create a Prompt Template API
    prompttemplate = PromptTemplateAPI(
        api_key=api_key,
        secret_key=secret_key,
        url=PromptTemplateAPI.PromptTemplate
    )

    # get prompt template from Prompt Template API
    response = prompttemplate(
        template_id=1968,
        content='侏罗纪世界'
    )

    # print prompt template
    print(response)
    ```

# 更新日志
- 2024-7-02
    - ERNIE-Lite-8K 模型支持
    - 提供测试程序

- 2024-5-22
    - 新增 ERNIE-Speed、ERNIE-Lite、ERNIE-Tiny系列模型

- 2024-3-21
    - 支持AppBuilder接口  
import json
import requests

from typing import Dict
from typing import Optional, Generator, Union

from .types import Messages, Embeddings, Texts

from .types import Message
from .types import ChatResponse
from .types import EmbeddingResponse
from .types import AccessTokenResponse
from .types import PromptTemplateResponse
from .types import AIStudioChatResponse
from .types import AIStudioEmbeddingResponse


__all__ = [
    "get_access_token",
    "LLMAPI",
    "EmbeddingAPI",
    "PromptTemplateAPI",
    "AIStudioLLMAPI",
    "AIStudioEmbeddingAPI",
    "AppBuilderAPI",
]


"""
APIs of Wenxin Workshop.
"""


def get_access_token(api_key: str, secret_key: str) -> str:
    """
    Get access token from Baidu AI Cloud.

    Parameters
    ----------
    api_key : str
        API key from Baidu AI Cloud.
    secret_key : str
        Secret key from Baidu AI Cloud.

    Returns
    -------
    str
        Access token from Baidu AI Cloud.

    Raises
    ------
    ValueError
        If request failed. Please check your API key and secret key.

    Examples
    --------
    >>> from wenxinworkshop import get_access_token
    >>> api_key = ''
    >>> secret_key = ''
    >>> access_token = get_access_token(
    ...     api_key=api_key,
    ...     secret_key=secret_key
    ... )
    >>> print(access_token)
    24.6b3b3f7b0b3b3f7b0b3b3f7b0b3b3f7b.2592000.1628041234.222222-44444444
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key,
    }

    response = requests.request(method="POST", url=url, headers=headers, params=params)

    try:
        response_json: AccessTokenResponse = response.json()
        return response_json["access_token"]
    except:
        raise ValueError(response.text)


class LLMAPI:
    """
    LLM API.

    Attributes
    ----------
    url : str
        URL of LLM API.

    access_token : str
        Access token from Baidu AI Cloud.

    ERNIEBot : str
        URL of ERNIEBot LLM API.

    ERNIEBot_turbo : str
        URL of ERNIEBot turbo LLM API.

    Methods
    -------
    __init__(
        self,
        api_key: str,
        secret_key: str,
        url: str = LLMAPI.ERNIEBot
    ) -> None:
        Initialize LLM API.

    __call__(
        self,
        messages: Messages,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        penalty_score: Optional[float] = None,
        stream: Optional[bool] = None,
        user_id: Optional[str] = None,
        chunk_size: int = 512
    ) -> Union[str, Generator[str, None, None]]:
        Get response from LLM API.

    stream_response(
        response: requests.Response,
        chunk_size: int = 512
    ) -> Generator[str, None, None]:
        Stream response from LLM API.
    """

    ERNIEBot = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions"
    )
    ERNIEBot_turbo = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"
    )
    ERNIEBot_4_0 = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
    )
    BLOOMZ_7B = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/bloomz_7b1"
    )
    LLAMA_2_7B = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_7b"
    )
    LLAMA_2_13B = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_13b"
    )
    LLAMA_2_70B = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_2_70b"
    )
    QIANFAN_BLOOMZ_7B_COMPRESSED = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/qianfan_bloomz_7b_compressed"
    )
    QIANFAN_CHINESE_LLAMA_2_7B = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/qianfan_chinese_llama_2_7b"
    )
    CHATGLM2_6B_32K = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/chatglm2_6b_32k"
    )
    AQUILACHAT_7B = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/aquilachat_7b"
    )
    ERNIE_BOT_8K = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_bot_8k"
    )
    CODELLAMA_7B_INSTRUCT = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/completions/codellama_7b_instruct"
    )
    XUANYUAN_70B_CHAT = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/xuanyuan_70b_chat"
    )
    CHATLAW = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/chatlaw"
    )
    ERNIE_SPEED_128K = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k"
    )
    ERNIE_SPEED_8K = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed"
    )
    ERNIE_LITE_8K = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k"
    )
    ERNIE_LITE_8K_0922 = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"
    )
    ERNIE_TINY_8K = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-tiny-8k"
    )

    def __init__(
        self: "LLMAPI", api_key: str, secret_key: str, url: str = ERNIEBot
    ) -> None:
        """
        Initialize LLM API.

        Parameters
        ----------
        api_key : str
            API key from Baidu AI Cloud.

        secret_key : str
            Secret key from Baidu AI Cloud.

        url : Optional[str], optional
            URL of LLM API, by default LLMAPI.ERNIEBot. You can also use LLMAPI.ERNIEBot_turbo or other LLM API urls.

        Examples
        --------
        >>> from wenxinworkshop import LLMAPI
        >>> api_key = ''
        >>> secret_key = ''
        >>> erniebot = LLMAPI(
        ...     api_key=api_key,
        ...     secret_key=secret_key,
        ...     url=LLMAPI.ERNIEBot
        ... )
        """
        self.url = url
        self.access_token = get_access_token(api_key=api_key, secret_key=secret_key)

    def __call__(
        self: "LLMAPI",
        messages: Messages,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        penalty_score: Optional[float] = None,
        stream: Optional[bool] = None,
        user_id: Optional[str] = None,
        chunk_size: int = 512,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Get response from LLM API.

        Parameters
        ----------
        messages : Messages
            Messages from user and assistant.

        temperature : Optional[float], optional
            Temperature of LLM API, by default None.

        top_p : Optional[float], optional
            Top p of LLM API, by default None.

        penalty_score : Optional[float], optional
            Penalty score of LLM API, by default None.

        stream : Optional[bool], optional
            Stream of LLM API, by default None.

        user_id : Optional[str], optional
            User ID of LLM API, by default None.

        chunk_size : int, optional
            Chunk size of LLM API, by default 512.

        Returns
        -------
        Union[str, Generator[str, None, None]]
            Response from LLM API.

        Raises
        ------
        ValueError
            If request failed. Please check your API key and secret key. Or check the parameters.

        Examples
        --------
        >>> message = Message(
        ...     role='user',
        ...     content='你好！'
        ... )
        >>> messages: Messages = [message]

        >>> response = erniebot(
        ...     messages=messages,
        ...     temperature=None,
        ...     top_p=None,
        ...     penalty_score=None,
        ...     stream=None,
        ...     user_id=None,
        ...     chunk_size=512
        ... )

        >>> print(response)
        你好，有什么可以帮助你的。

        >>> response_stream = erniebot(
        ...     messages=messages,
        ...     temperature=None,
        ...     top_p=None,
        ...     penalty_score=None,
        ...     stream=True,
        ...     user_id=None,
        ...     chunk_size=512
        ... )

        >>> for item in response_stream:
        ...     print(item, end='')
        你好，有什么可以帮助你的。
        """
        headers = {"Content-Type": "application/json"}

        params = {"access_token": self.access_token}

        data = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "penalty_score": penalty_score,
            "stream": stream,
            "user_id": user_id,
        }

        response = requests.request(
            method="POST",
            url=self.url,
            headers=headers,
            params=params,
            data=json.dumps(data),
            stream=stream,
        )

        if stream:
            return self.stream_response(response=response, chunk_size=chunk_size)
        else:
            try:
                response_json: ChatResponse = response.json()
                return response_json["result"]
            except:
                raise ValueError(response.text)

    @staticmethod
    def stream_response(
        response: requests.Response, chunk_size: int = 512
    ) -> Generator[str, None, None]:
        """
        Stream response from LLM API.

        Parameters
        ----------
        response : requests.Response
            Response from LLM API.

        chunk_size : int, optional
            Chunk size of LLM API, by default 512.

        Yields
        -------
        Generator[str, None, None]
            Response from LLM API.

        Raises
        ------
        ValueError
            If request failed. Please check your API key and secret key. Or check the parameters.

        Examples
        --------
        >>> stream_response = erniebot.stream_response(
        ...     response=response,
        ...     chunk_size=512
        ... )

        >>> for item in stream_response:
        ...     print(item, end='')
        你好，有什么可以帮助你的。
        """
        for response_line in response.iter_lines(
            chunk_size=chunk_size, decode_unicode=True
        ):
            if response_line:
                try:
                    response_json: ChatResponse = json.loads(response_line[5:])
                    yield response_json["result"]
                except:
                    raise ValueError(response_line)


class AppBuilderAPI:
    AppBuilder = "https://appbuilder.baidu.com/rpc/2.0/cloud_hub/v1/ai_engine/agi_platform/v1/instance/integrated"

    def __init__(
        self: "AppBuilderAPI", app_token: str, history_enable: bool, url: str = AppBuilder
    ) -> None:
        self.url = url
        # 对话ID，仅对话型应用生效
        self.conversation_id = None
        self.history_enable = history_enable

        self.headers = {
            'Content-Type': 'application/json',
            'X-Appbuilder-Authorization': f'Bearer {app_token}'
        }
        self.timeout = 60

    def __call__(
        self: "AppBuilderAPI",
        query: str,
        response_mode: str,
    ):
        try:
            if self.conversation_id is None:
                data = {
                    "query": query,
                    "response_mode": response_mode
                }
            else:
                data = {
                    "query": query,
                    "response_mode": response_mode,
                    "conversation_id": self.conversation_id
                }

            response = requests.post(self.url, headers=self.headers, data=json.dumps(data), timeout=self.timeout)

            response_json = response.json()
            answer = response_json["result"]["answer"]

            if self.history_enable == True:
                if "conversation_id" in response_json["result"]:
                    self.conversation_id = response_json["result"]["conversation_id"]

            return answer

        except:
            raise ValueError(response.text)
        

class EmbeddingAPI:
    """
    Embedding API.

    Attributes
    ----------
    url : str
        URL of Embedding API.

    access_token : str
        Access token from Baidu AI Cloud.

    EmbeddingV1 : str
        URL of Embedding V1 API.

    Methods
    -------
    __init__(
        self,
        api_key: str,
        secret_key: str,
        url: str = EmbeddingAPI.EmbeddingV1
    ) -> None:
        Initialize Embedding API.

    __call__(
        self,
        texts: Texts,
        user_id: Optional[str] = None
    ) -> Embeddings:
        Get embeddings from Embedding API.
    """

    EmbeddingV1 = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1"

    def __init__(
        self: "EmbeddingAPI", api_key: str, secret_key: str, url: str = EmbeddingV1
    ) -> None:
        """
        Initialize Embedding API.

        Parameters
        ----------
        api_key : str
            API key from Baidu AI Cloud.

        secret_key : str
            Secret key from Baidu AI Cloud.

        url : Optional[str], optional
            URL of Embedding API, by default EmbeddingAPI.EmbeddingV1. You can also use other Embedding API urls.

        Examples
        --------
        >>> from wenxinworkshop import EmbeddingAPI
        >>> api_key = ''
        >>> secret_key = ''
        >>> ernieembedding = EmbeddingAPI(
        ...     api_key=api_key,
        ...     secret_key=secret_key,
        ...     url=EmbeddingAPI.EmbeddingV1
        ... )
        """
        self.url = url
        self.access_token = get_access_token(api_key=api_key, secret_key=secret_key)

    def __call__(
        self: "EmbeddingAPI", texts: Texts, user_id: Optional[str] = None
    ) -> Embeddings:
        """
        Get embeddings from Embedding API.

        Parameters
        ----------
        texts : Texts
            Texts of inputs.

        user_id : Optional[str], optional
            User ID of Embedding API, by default None.

        Returns
        -------
        Embeddings
            Embeddings from Embedding API.

        Raises
        ------
        ValueError
            If request failed. Please check your API key and secret key. Or check the parameters.

        Examples
        --------
        >>> texts: Texts = [
        ...     '你好！',
        ...     '你好吗？',
        ...     '你是谁？'
        ... ]

        >>> response = ernieembedding(
        ...     texts=texts,
        ...     user_id=None
        ... )

        >>> print(response)
        [[0.123, 0.456, 0.789, ...], [0.123, 0.456, 0.789, ...], [0.123, 0.456, 0.789, ...]]
        """
        headers = {"Content-Type": "application/json"}

        params = {"access_token": self.access_token}

        data = {"input": texts, "user_id": user_id}

        response = requests.request(
            method="POST",
            url=self.url,
            headers=headers,
            params=params,
            data=json.dumps(data),
        )

        try:
            response_json: EmbeddingResponse = response.json()
            embeddings: Embeddings = [
                embedding["embedding"] for embedding in response_json["data"]
            ]
            return embeddings
        except:
            raise ValueError(response.text)


class PromptTemplateAPI:
    """
    Prompt Template API.

    Attributes
    ----------
    url : str
        URL of Prompt Template API.

    access_token : str
        Access token from Baidu AI Cloud.

    PromptTemplate : str
        URL of Prompt Template API.

    Methods
    -------
    __init__(
        self,
        api_key: str,
        secret_key: str,
        url: str = PromptTemplate
    ) -> None:
        Initialize Prompt Template API.

    __call__(
        self,
        template_id: int,
        **kwargs: str
    ) -> str:
        Get prompt template from Prompt Template API.
    """

    PromptTemplate = (
        "https://aip.baidubce.com/rest/2.0/wenxinworkshop/api/v1/template/info"
    )

    def __init__(
        self: "PromptTemplateAPI",
        api_key: str,
        secret_key: str,
        url: str = PromptTemplate,
    ) -> None:
        """
        Initialize Prompt Template API.

        Parameters
        ----------
        api_key : str
            API key from Baidu AI Cloud.

        secret_key : str
            Secret key from Baidu AI Cloud.

        url : Optional[str], optional
            URL of Prompt Template API, by default PromptTemplateAPI.PromptTemplate. You can also use other Prompt Template API urls.

        Examples
        --------
        >>> from wenxinworkshop import PromptTemplateAPI
        >>> api_key = ''
        >>> secret_key = ''
        >>> prompttemplate = PromptTemplateAPI(
        ...     api_key=api_key,
        ...     secret_key=secret_key,
        ...     url=PromptTemplateAPI.PromptTemplate
        ... )
        """
        self.url = url
        self.access_token = get_access_token(api_key=api_key, secret_key=secret_key)

    def __call__(self, template_id: int, **kwargs: str) -> str:
        """
        Get prompt template from Prompt Template API.

        Parameters
        ----------
        template_id : int
            ID of prompt template.

        **kwargs : str
            Variables of prompt template.

        Returns
        -------
        str
            Prompt template content.

        Raises
        ------
        ValueError
            If request failed. Please check your API key and secret key. Or check the parameters.

        Examples
        --------
        >>> response = prompttemplate(
        ...     template_id=1968,
        ...     content='侏罗纪世界'
        ... )

        >>> print(response)

        """
        headers = {"Content-Type": "application/json"}

        params: Dict[str, Union[str, int]] = {
            "access_token": self.access_token,
            "id": template_id,
            **kwargs,
        }

        response = requests.request(
            method="GET", url=self.url, headers=headers, params=params
        )

        try:
            response_json: PromptTemplateResponse = response.json()
            return response_json["result"]["content"]
        except:
            raise ValueError(response.text)


"""
APIs of AI Studio.
"""


class AIStudioLLMAPI:
    """
    LLM API of AI Studio.

    Attributes
    ----------
    url : str

    model : str

    authorization: str

    ERNIEBot : str

    Methods
    -------
    __init__(
        self,
        user_id: str,
        access_token: str,
        model: str = AIStudioLLMAPI.ERNIEBot
    ) -> None:

    __call__(
        self,
        messages: Messages,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        penalty_score: Optional[float] = None
    ) -> str:
    """

    ERNIEBot = "ERNIE-Bot"

    def __init__(
        self: "AIStudioLLMAPI", user_id: str, access_token: str, model: str = ERNIEBot
    ) -> None:
        """
        Initialize LLM API.

        Parameters
        ----------
        user_id : str
            User ID of LLM API.

        access_token : str
            Access token of LLM API.

        model : str, optional
            Model of LLM API, by default AIStudioLLMAPI.ERNIEBot.

        Examples
        --------
        >>> from wenxinworkshop import AIStudioLLMAPI
        >>> user_id = ''
        >>> access_token = ''
        >>> erniebot = AIStudioLLMAPI(
        ...     user_id=user_id,
        ...     access_token=access_token,
        ...     model=AIStudioLLMAPI.ERNIEBot
        ... )
        """
        self.url = "https://aistudio.baidu.com/llm/lmapi/api/v1/chat/completions"
        self.model = model
        self.authorization = "token {} {}".format(user_id, access_token)

    def __call__(
        self: "AIStudioLLMAPI",
        messages: Messages,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        penalty_score: Optional[float] = None,
    ) -> str:
        """
        Get response from LLM API.

        Parameters
        ----------
        messages : Messages
            Messages of inputs.

        temperature : Optional[float], optional
            Temperature of LLM API, by default None.

        top_p : Optional[float], optional
            Top p of LLM API, by default None.

        penalty_score : Optional[float], optional
            Penalty score of LLM API, by default None.

        Returns
        -------
        str
            Response from LLM API.

        Raises
        ------
        ValueError
            If request failed. Please check your API key and secret key. Or check the parameters.

        Examples
        --------
        >>> message = Message(
        ...     role='user',
        ...     content='你好！'
        ... )

        >>> messages: Messages = [message]

        >>> response = erniebot(
        ...     messages=messages,
        ...     temperature=None,
        ...     top_p=None,
        ...     penalty_score=None
        ... )

        >>> print(response)
        你好！
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.authorization,
            "SDK-Version": "0.0.2",
        }

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "penalty_score": penalty_score,
        }

        response = requests.request(
            method="POST", url=self.url, headers=headers, data=json.dumps(data)
        )

        try:
            response_json: AIStudioChatResponse = response.json()
            return response_json["result"]["result"]
        except:
            raise ValueError(response.text)


class AIStudioEmbeddingAPI:
    """
    Embedding API of AI Studio.

    Attributes
    ----------
    url : str

    authorization: str

    Methods
    -------
    __init__(
        self,
        user_id: str,
        access_token: str
    ) -> None:

    __call__(
        self,
        texts: Texts
    ) -> Embeddings:
    """

    def __init__(self: "AIStudioEmbeddingAPI", user_id: str, access_token: str) -> None:
        """
        Initialize Embedding API.

        Parameters
        ----------
        user_id : str
            User ID of Embedding API.

        access_token : str
            Access token of Embedding API.

        Examples
        --------
        >>> from wenxinworkshop import AIStudioEmbeddingAPI
        >>> user_id = ''
        >>> access_token = ''
        >>> ernieembedding = AIStudioEmbeddingAPI(
        ...     user_id=user_id,
        ...     access_token=access_token
        ... )
        """
        self.url = "https://aistudio.baidu.com/llm/lmapi/api/v1/embedding"
        self.authorization = "token {} {}".format(user_id, access_token)

    def __call__(self: "AIStudioEmbeddingAPI", texts: Texts) -> Embeddings:
        """
        Get embeddings from Embedding API.

        Parameters
        ----------
        texts : Texts
            Texts of inputs.

        Returns
        -------
        Embeddings
            Embeddings from Embedding API.

        Raises
        ------
        ValueError
            If request failed. Please check your API key and secret key. Or check the parameters.

        Examples
        --------
        >>> texts: Texts = [
        ...     '你好！',
        ...     '你好吗？',
        ...     '你是谁？'
        ... ]

        >>> response = ernieembedding(
        ...     texts=texts
        ... )

        >>> print(response)
        [[0.123, 0.456, 0.789, ...], [0.123, 0.456, 0.789, ...], [0.123, 0.456, 0.789, ...]]
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.authorization,
            "SDK-Version": "0.0.2",
        }

        data = {
            "input": texts,
        }

        response = requests.request(
            method="POST", url=self.url, headers=headers, data=json.dumps(data)
        )

        try:
            response_json: AIStudioEmbeddingResponse = response.json()
            embeddings: Embeddings = [
                embedding["embedding"] for embedding in response_json["result"]["data"]
            ]
            return embeddings
        except:
            raise ValueError(response.text)


if __name__ == "__main__":
    """
    Wenxin Workshop APIs Examples
    """
    """
    Configurations
    """
    # Set API key and Secret key
    api_key = ""
    secret_key = ""

    """
    LLM API Examples
    """
    # create a LLM API
    erniebot = LLMAPI(api_key=api_key, secret_key=secret_key, url=LLMAPI.ERNIEBot)

    # create a message
    message = Message(role="user", content="你好！")

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
        chunk_size=512,
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
        chunk_size=512,
    )

    # print response stream
    for item in response_stream:
        print(item, end="")

    """
    Embedding API Examples
    """
    # create a Embedding API
    ernieembedding = EmbeddingAPI(
        api_key=api_key, secret_key=secret_key, url=EmbeddingAPI.EmbeddingV1
    )

    # create texts
    texts: Texts = ["你好！", "你好吗？", "你是谁？"]

    # get embeddings from Embedding API
    embeddings = ernieembedding(texts=texts, user_id=None)

    # print embeddings
    print(embeddings)

    """
    Prompt Template API Examples
    """
    # create a Prompt Template API
    prompttemplate = PromptTemplateAPI(
        api_key=api_key, secret_key=secret_key, url=PromptTemplateAPI.PromptTemplate
    )

    # get prompt template from Prompt Template API
    template = prompttemplate(template_id=1968, content="侏罗纪世界")

    # print prompt template
    print(template)

    """
    AI Studio LLM API Examples
    """
    """
    Configurations
    """
    # Set user ID and access token
    user_id = ""
    access_token = ""

    # create a LLM API
    erniebot = AIStudioLLMAPI(
        user_id=user_id, access_token=access_token, model=AIStudioLLMAPI.ERNIEBot
    )

    # create a message
    message = Message(role="user", content="你好！")

    # create messages
    messages: Messages = [message]

    # get response from LLM API
    response = erniebot(
        messages=messages, temperature=None, top_p=None, penalty_score=None
    )

    # print response
    print(response)

    """
    AI Studio Embedding API Examples
    """
    # create a Embedding API
    ernieembedding = AIStudioEmbeddingAPI(user_id=user_id, access_token=access_token)

    # create texts
    texts: Texts = ["你好！", "你好吗？", "你是谁？"]

    # get embeddings from Embedding API
    embeddings = ernieembedding(texts=texts)

    # print embeddings
    print(embeddings)

from typing import Any, Dict, List, Optional, Iterator
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, BaseMessage, AIMessageChunk
from langchain_core.outputs import (
    ChatGeneration, 
    ChatGenerationChunk, 
    ChatResult, 
    GenerationChunk)
import LangModel
import asyncio

class OllamaAPIModel(LLM):
    """
    OllamaAPIModel is a subclass of LLM that interfaces with a language model to generate responses to input prompts.
    Attributes:
        lang_model (LangModel.LangModel): An instance of the language model object.
    Methods:
        _call(prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
            Generates a response to the input prompt. Raises a ValueError if 'stop' is provided.
        _stream(prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[GenerationChunk]:
            Generates a response to the input prompt. Raises a ValueError if 'stop' is provided.
        _identifying_params -> Dict[str, Any]:
            Returns a dictionary containing the identifying parameters of the model.
        _llm_type -> str:
            Returns the type of the language model.
    """
    lang_model: LangModel.LangModel
    """lang model object"""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        generates a response to the input prompt
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.lang_model.generate(prompt)
    
    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        generates a response to the input prompt
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        for response in self.lang_model.stream_generate(prompt):
            if 'response' in response:
                content = response['response']
                chunk = GenerationChunk(text=content)
                if run_manager is not None:
                    run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.lang_model.model_name,}

    @property
    def _llm_type(self) -> str:
        return "ollama_api_llm"

class OllamaAPIChatModel(BaseChatModel):
    """
    OllamaAPIChatModel is a chat model class that extends BaseChatModel to interact with a language model.
    Attributes:
        lang_model (LangModel.LangModel): The language model object.
        role_type_dict (Dict[str, str]): A dictionary mapping role types to their string representations.
    Methods:
        message_format(message: BaseMessage) -> Dict[str, str]:
            Formats a message into a dictionary with role and content keys.
        _generate(messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
            Overrides the _generate method to implement the chat model logic. This method can call an API, a local model, or any other implementation to generate a response to the input prompt.
        _stream(messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
            Overrides the _stream method to implement the chat model logic. This method can call an API, a local model, or any other implementation to generate a response to the input prompt.
        _llm_type() -> str:
            Returns the type of the language model.
        _identifying_params() -> Dict[str, Any]:
            Returns a dictionary of identifying parameters for the language model.
    """
    
    lang_model: LangModel.LangModel
    """lang model object"""
    role_type_dict: Dict[str, str] = {
        'system': 'system', 'human': 'user', 'ai': 'assistant'}
    """role type dictionary"""
    
    def _message_format(self, message: BaseMessage) -> Dict[str, str]:
        return {
            "role": self.role_type_dict[message.type],
            "content": message.content,
        }

    def _messages_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        return [self._message_format(message) for message in messages]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Overrides the _generate method to implement the chat model logic.
        This method can call an API, a local model, or any other implementation to generate a response to the input prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        messages = self._messages_format(messages)
        message:str = self.lang_model.chat(messages)
        generation = ChatGeneration(
            message=AIMessage(content=message['content']))
        return ChatResult(generations=[generation])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Overrides the _generate method to implement the chat model logic.
        This method can call an API, a local model, or any other implementation to generate a response to the input prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        messages = self._messages_format(messages)
        message = self.lang_model.chat(messages)
        content = message['content']
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])
        

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Overrides the _stream method to implement the chat model logic.
        This method can call an API, a local model, or any other implementation to generate a response to the input prompt.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        messages = self._messages_format(messages)
        for response in self.lang_model.stream_generate(messages):
            if 'message' in response and 'content' in response['message']:
                content = response['message']['content']
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
                if run_manager is not None:
                    run_manager.on_llm_new_token(content, chunk=chunk)
                yield chunk

    @property
    def _llm_type(self) -> str:
        return "ollama_api_llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.lang_model.model_name,}

if __name__ == '__main__':
    # test code
    
    import os
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    lang_model = LangModel.LangModel(
        api_key=os.getenv('OLLAMA_API_KEY'),
        api_url=os.getenv('OLLAMA_URL'),
        model_name='gemma2:9b',
    )
    
    def testAPIModel(lang_model):
        print(lang_model.generate('お弁当によく入れるおかずは何ですか?'))
        model = OllamaAPIModel(
            lang_model=lang_model
        )
        result = model.invoke([
            SystemMessage(content="あなたは知識豊富なアシスタントです。"),
            HumanMessage(content="お弁当によく入れるおかずは何ですか?"),
        ])
        print(result)
    # testAPIModel(lang_model)
    
    def testChatModel(lang_model):
        print('testChatModel')
        model = OllamaAPIChatModel(
            lang_model=lang_model
        )
        result = model.invoke([
            SystemMessage(content="あなたは知識豊富なアシスタントです。"),
            HumanMessage(content="なぜ空は青いのですか?"),
            AIMessage(content="空が青い理由は、太陽光が大気中の小さな分子によって散乱されるからです。青い光は波長が短いため、他の色よりも強く散乱されます。"),
            HumanMessage(content="夕焼けが赤いのはなぜですか?"),
        ])
        print(result)
    # testChatModel(lang_model)

    def testStreamModel(lang_model):
        model = OllamaAPIModel(
            lang_model=lang_model
        )
        stream = model.stream("なぜ空は青いのですか?")
        result = None
        for chunk in stream:
            if result is None:
                result = chunk
            else:
                result += chunk
        print(result)
    # testStreamModel(lang_model)

    def testStreamChat(lang_model):
        model = OllamaAPIChatModel(
            lang_model=lang_model
        )
        stream = model.stream([
            SystemMessage(content="あなたは知識豊富なアシスタントです。"),
            HumanMessage(content="なぜ空は青いのですか?"),
            AIMessage(content="空が青い理由は、太陽光が大気中の小さな分子によって散乱されるからです。青い光は波長が短いため、他の色よりも強く散乱されます。"),
            HumanMessage(content="夕焼けが赤いのはなぜですか?"),
        ])
        result = None
        for chunk in stream:
            if result is None:
                result = chunk
            else:
                result += chunk
        print(result)
        print(type(result))
    # testStreamChat(lang_model)

    async def testAinvokeChat(lang_model):
        model = OllamaAPIChatModel(
            lang_model=lang_model
        )
        result = await model.ainvoke([
            SystemMessage(content="あなたは知識豊富なアシスタントです。"),
            HumanMessage(content="なぜ空は青いのですか?"),
            HumanMessage(content="夕焼けが赤いのはなぜですか?"),
        ])
        print(result)
    asyncio.run(testAinvokeChat(lang_model))

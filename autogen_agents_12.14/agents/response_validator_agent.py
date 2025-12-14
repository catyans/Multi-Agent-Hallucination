import asyncio
import json
from typing import List, Dict, Sequence, AsyncGenerator
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken, Component
from pydantic import BaseModel
from typing_extensions import Self
from autogen_core.models import (
    SystemMessage,
    UserMessage,
)
from openai import RateLimitError  # 请根据你的 SDK 版本确认异常路径
import re
import utils
import prompts

class ResponseValidatorAgentConfig(BaseModel):
    name: str
    description: str = "A response validator agent that validates the response of a model."


class ResponseValidatorAgent(BaseChatAgent, Component[ResponseValidatorAgentConfig]):
    component_config_schema = ResponseValidatorAgentConfig

    def __init__(
        self,
        name: str,
        description: str = "A response validator agent that validates the response of a model.",
        model_client=None,
        count: int = 3,
    ):
        super().__init__(name=name, description=description)
        self._model_client = model_client
        self._initial_count = count
        self._count = count
        self.terminate_word = "TERMINATE:"

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        final_response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                final_response = message

        if final_response is None:
            raise AssertionError("The stream should have returned the final result.")

        return final_response


    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        self._count = self._count-1
        # Get the last message as the task
        if not messages:
            # If no messages, return an empty response
            yield Response(
                chat_message=TextMessage(content=self.terminate_word+"No input message received.", source=self.name),
                inner_messages=[],
            )
            return
        retrival_task = messages[-3].content    
        task = messages[-1].content
        try:
            retrival_task_data = json.loads(retrival_task)
            retrieval_context = retrival_task_data.get("retrieval_context", [])  # 默认为空列表
            task_data = json.loads(task)  # 解析 JSON 字符串为字典
            query = task_data.get("query")
            answer = task_data.get("answer")
            if self._count == 0:
                self._count = self._initial_count
                yield Response(
                    chat_message=TextMessage(content=self.terminate_word+answer, source=self.name),
                    inner_messages=[],
                )
                return
            prompt = prompts.construct_validate_prompt(query, retrieval_context, answer)
            user_message = UserMessage(
                content=prompt,
                source="user"
            )
            model_result = await utils._call_model_with_rate_limit_retry(
                model_client=self._model_client,
                messages=[ user_message]
            )
            judgment = self.extract_answer_as_bool(model_result.content)
        except json.JSONDecodeError as e:
            print("JSON 解析错误:", e)
        except KeyError as e:
            print("缺少必要的键:", e)
        if judgment:
            self._count = self._initial_count
            output = self.terminate_word+answer
        else:
            output = query
        yield Response(
            chat_message=TextMessage(content=output, source=self.name),
            inner_messages=[],
        )

    def extract_answer_as_bool(self, response: str) -> bool:
        if not response or not isinstance(response, str):
            print("Error parsing response: Empty or non-string response")
            return False
        if "yes" in response and "no" not in response:
            predicted = True
        else:
            predicted = False
        return predicted

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent"""
        pass

    @classmethod
    def _from_config(cls, config: ResponseValidatorAgentConfig) -> Self:
        return cls(
            name=config.name, 
            description=config.description, 
        )

    def _to_config(self) -> ResponseValidatorAgentConfig:
        return ResponseValidatorAgentConfig(
            name=self.name,
            description=self.description,
        )
import asyncio
import json
from typing import Sequence, AsyncGenerator
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken, Component
from pydantic import BaseModel
from typing_extensions import Self
import re
from autogen_core.models import (
    SystemMessage,
    UserMessage,
)
class QuestionDecomposeAgentConfig(BaseModel):
    name: str
    description: str = "A question decompose agent that decomposes the question into sub-questions."


class QuestionDecomposeAgent(BaseChatAgent, Component[QuestionDecomposeAgentConfig]):
    component_config_schema = QuestionDecomposeAgentConfig

    def __init__(
        self,
        name: str,
        description: str = "A response selector agent that selects the response of a model.",
        model_client=None,
    ):
        super().__init__(name=name, description=description)
        self._model_client = model_client

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
        # Get the last message as the task
        if not messages:
            # If no messages, return an empty response
            yield Response(
                chat_message=TextMessage(content="No input message received.", source=self.name),
                inner_messages=[],
            )
            return
            
        task = messages[-1].content
        prompt = '''
You are an expert at query rewriting for multi-hop question answering. Your task is to decompose a multi-hop question into two independent sub-questions that can be retrieved separately. The sub-questions should be unrelated to each other so that they cover different aspects of the original question.

Instructions:

Take the input multi-hop question.

Output two sub-questions that are independent and can be used for retrieval.

Ensure that the sub-questions are clear and concise.

Output only a JSON object in this format:
{{"sub_questions": ["...", "..."]}}

Question: "{question}"
'''
        system_message = SystemMessage(
            content="You are a helpful assistant.",
            source="system"
        )
        user_message = UserMessage(
            content=prompt.format(question=task),
            source="user"
        )
        # 并发调用两次 model_client.create
        model_result = await self._model_client.create(
            messages=[system_message, user_message]
        )
        output = model_result.content
        match = re.search(r"```json\s*(\{.*\})\s*```", output, re.DOTALL)
        if match:
            output = match.group(1)
        try:
            sub_questions = json.loads(output)["sub_questions"]
        except json.JSONDecodeError:
            sub_questions = []
        # === 过滤：只保留词数 <= 300 的项 ===
        filtered_sub_questions = []
        for question in sub_questions:
            if isinstance(question, str):
                word_count = len(question.split())
                if word_count <= 300:
                    filtered_sub_questions.append(question)
            else:
                # 非字符串跳过
                pass
        # 只保留两项
        if len(filtered_sub_questions) > 2:
            filtered_sub_questions = filtered_sub_questions[:2]
        sub_questions = filtered_sub_questions
        # 添加原始问题
        sub_questions.append(task)
        yield Response(
            chat_message=TextMessage(content=json.dumps(sub_questions), source=self.name),
            inner_messages=[],
        )



    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent"""
        pass

    @classmethod
    def _from_config(cls, config: QuestionDecomposeAgentConfig) -> Self:
        return cls(
            name=config.name, 
            description=config.description, 
        )

    def _to_config(self) -> QuestionDecomposeAgentConfig:
        return QuestionDecomposeAgentConfig(
            name=self.name,
            description=self.description,
        )
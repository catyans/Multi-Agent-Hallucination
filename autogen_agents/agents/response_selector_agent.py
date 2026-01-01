import asyncio
import json
from typing import List, Dict, Sequence, AsyncGenerator
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken, Component
from pydantic import BaseModel
from autogen_core.memory import Memory
from typing_extensions import Self
import re
from autogen_core.models import (
    SystemMessage,
    UserMessage,
)
from openai import RateLimitError  # 请根据你的 SDK 版本确认异常路径
import prompts
import utils
class ResponseSelectorAgentConfig(BaseModel):
    name: str
    description: str = "A response validator agent that validates the response of a model."


class ResponseSelectorAgent(BaseChatAgent, Component[ResponseSelectorAgentConfig]):
    component_config_schema = ResponseSelectorAgentConfig

    def __init__(
        self,
        name: str,
        description: str = "A response selector agent that selects the response of a model.",
        model_client=None,
        memory: Memory = None,
    ):
        super().__init__(name=name, description=description)
        self._model_client = model_client
        self._memory = memory

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
        retrival_task = messages[-2].content
        task = messages[-1].content
        system_prompt = """你是一个答案选择助手，任务是从给定选项中选出最正确的一个。
        严格遵守以下规则：
        1. 仅输出答案内容，必须包括答案的具体内容，不添加任何解释、推理或额外文本。
        2. 必须基于用户提供的问题和检索到的上下文进行判断，答案不能为空。
        """
        try:
            retrival_task_data = json.loads(retrival_task)
            retrieval_context = retrival_task_data["retrieval_context"]
            task_data = json.loads(task)  # 解析 JSON 字符串为字典
            query = task_data.get("query")
            generated_answers = task_data.get("generated_answers")
            remaining_answers = generated_answers[:]  # 做个副本避免修改原列表
            selected_answer = ""
            while len(remaining_answers) > 1:
                try:
                    prompt = prompts.construct_select_prompt(query, remaining_answers, retrieval_context)
                    system_message = SystemMessage(
                        content=system_prompt,
                        source="system"
                    )
                    user_message = UserMessage(
                        content=prompt,
                        source="user"
                    )
                    model_result = await utils._call_model_with_rate_limit_retry(
                        model_client=self._model_client,
                        messages=[system_message, user_message]
                    )#调用大模型
                    selected_answer = model_result.content
                    for answer in remaining_answers:
                        #提取answer中Answer: 后的内容
                        answer_content = re.search(r"Answer:\s*(.*)", answer, re.IGNORECASE)
                        if answer_content:
                            answer_content = answer_content.group(1).strip()                           
                            if selected_answer in answer_content:
                                selected_answer = answer
                                break
                    
                    break  # 成功选出，跳出循环
    
                except Exception as e:
                    print(f"call_with_retry failed or parsing failed: {e}. Retrying with fewer answers...")
                    # 移除最后一个答案
                    remaining_answers = remaining_answers[:-1]
        except json.JSONDecodeError as e:
            print("JSON 解析错误:", e)
        except KeyError as e:
            print("缺少必要的键:", e)
        output = json.dumps({"query": query,"answer": selected_answer})
        yield Response(
            chat_message=TextMessage(content=output, source=self.name),
            inner_messages=[],
        )

    def extract_selected_answer_index(self, response: str) -> int:
        """
        Extracts the selected answer index using regex from the model's response.
        Matches patterns like "Final Answer: 2" or "Final Answer:\n2".
        """
        # Match "Final Answer:" followed by optional whitespace/newlines and then a number
        pattern = r'Final\s*Answer\s*:\s*(\d+)'
        match = re.search(pattern, response, re.IGNORECASE)
        
        if match:
            return int(match.group(1))
        
        # Fallback for multiline cases
        pattern_multiline = r'Final\s*Answer\s*:\s*(?:\n\s*)*(\d+)'
        match = re.search(pattern_multiline, response, re.IGNORECASE)
        
        if match:
            return int(match.group(1))        
        return 1  # 默认返回第一个答案

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent"""
        pass

    @classmethod
    def _from_config(cls, config: ResponseSelectorAgentConfig) -> Self:
        return cls(
            name=config.name, 
            description=config.description, 
        )

    def _to_config(self) -> ResponseSelectorAgentConfig:
        return ResponseSelectorAgentConfig(
            name=self.name,
            description=self.description,
        )
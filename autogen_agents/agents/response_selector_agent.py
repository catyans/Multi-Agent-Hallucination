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
            
        task = messages[-1].content
        try:
            task_data = json.loads(task)  # 解析 JSON 字符串为字典
            query = task_data.get("query")
            generated_answers = task_data.get("generated_answers")
            retrieval_results = task_data.get("retrieval_results")


            prompt = self.construct_select_prompt(query, generated_answers, retrieval_results)#构造prompt
            system_message = SystemMessage(
                content="You are a helpful assistant.",
                source="system"
            )
            user_message = UserMessage(
                content=prompt,
                source="user"
            )
            model_result = await self._model_client.create(
                messages=[system_message, user_message]
            )#调用大模型
            selected_answer_index = self.extract_selected_answer_index(model_result.content)
        except json.JSONDecodeError as e:
            print("JSON 解析错误:", e)
        except KeyError as e:
            print("缺少必要的键:", e)
        output = json.dumps({"question": query,"answer": generated_answers[selected_answer_index-1] if selected_answer_index <= len(generated_answers) else generated_answers[0],"retrieval_results": retrieval_results})
        yield Response(
            chat_message=TextMessage(content=output, source=self.name),
            inner_messages=[],
        )

    def construct_select_prompt(self, query, generated_answers, retrieval_results):
        """
        Constructs a prompt in English to ask a LLM to select the best answer from the generated answers.
        
        Args:
            query (str): The user's question.
            generated_answers (list of str): List of candidate answers to choose from.
            retrieval_results (list of str or str): Retrieved context snippets (used as reference).

        Returns:
            str: Formatted prompt in English for the LLM.
        """
        prompt = ""
        prompt += f"Question: {query}\n"
        # Add retrieval results if available
        if retrieval_results:
            if isinstance(retrieval_results, list):
                for i, context in enumerate(retrieval_results, 1):
                    prompt += f"Retrieval Context {i}: {context}\n"
            else:
                prompt += f"Retrieval Context: {retrieval_results}\n"        
        # Add generated answers with numbering
        for i, answer in enumerate(generated_answers, 1):
            prompt += f"Answer {i}: {answer}\n"
        
        # Instructions in English
        prompt += '''
    Carefully analyze the content of each provided answer option. Based on the question background, retrieved context, and general knowledge, perform step-by-step reasoning to evaluate the correctness, factual accuracy, and logical consistency of each answer.

    Identify and eliminate options that contain errors, inaccuracies, or inconsistencies. After thorough analysis, select the single most correct and well-supported answer.

    After completing your reasoning, please output the result in the following strict format:

    Reasoning:

    (Provide your detailed step-by-step analysis here)

    Final Answer:

    (Only output the numeric index corresponding to the correct answer, e.g., 1, 2, 3, etc.)
    '''
        return prompt.strip()


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
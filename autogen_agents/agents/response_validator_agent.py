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
            
        task = messages[-1].content
        try:
            task_data = json.loads(task)  # 解析 JSON 字符串为字典
            question = task_data.get("question")
            answer = task_data.get("answer")
            retrieval_results = task_data.get("retrieval_results", [])  # 默认为空列表
            prompt = self.construct_judgment_prompt(question, retrieval_results, answer)
            if self._count == 0:
                self._count = self._initial_count
                yield Response(
                    chat_message=TextMessage(content=self.terminate_word+answer, source=self.name),
                    inner_messages=[],
                )
                return
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
            output = question
        yield Response(
            chat_message=TextMessage(content=output, source=self.name),
            inner_messages=[],
        )

    def construct_judgment_prompt(self,question, retrieval_results, answer):
        """
        Constructs a prompt in English to ask a LLM to judge if the answer 
        can be derived from the retrieval results given the question.
        
        Args:
            question (str): The user's question.
            retrieval_results (list of str or str): Retrieved context snippets.
            answer (str): The generated answer to evaluate.
        
        Returns:
            str: Formatted prompt in English for the LLM.
        """
        # If retrieval_results is a list, join them into a single string
        if isinstance(retrieval_results, list):
            context = "\n\n".join(f"[Snippet {i+1}]: {snippet}" for i, snippet in enumerate(retrieval_results))
        else:
            context = retrieval_results

        prompt = f"""\
        You are an impartial evaluator. Your task is to determine whether the provided answer can be directly supported by the information in the retrieval results for the given question.

        Please follow these steps:

        Read the question carefully.
        Review the retrieval results (context snippets).
        Examine the provided answer.
        Determine if the answer is fully supported — every piece of information in the answer must be explicitly stated or logically inferable from the retrieval results. No external knowledge or speculation.
        Output your judgment in exactly this format:
        {{"thought": "Brief reasoning based on context", "answer": "Yes or No"}}

        Question:
        {question}

        Retrieval Results:
        {context}

        Answer:
        {answer}"""
        
        return prompt.strip()
    def extract_answer_as_bool(self, response: str) -> bool:
        try:
            result = json.loads(response)
            answer_str = result["answer"]
            return answer_str.strip().lower() == "yes"
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            print(f"Error parsing response: {e}")
            return False

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
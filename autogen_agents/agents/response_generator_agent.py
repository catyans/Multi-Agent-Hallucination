import asyncio
import json
import random
from typing import List, Dict, Sequence, AsyncGenerator
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory
from pydantic import BaseModel
from typing_extensions import Self
from autogen_core.models import (
    SystemMessage,
    UserMessage,
)
from openai import RateLimitError
import prompts
import utils
class ResponseGeneratorAgentConfig(BaseModel):
    name: str
    description: str = "A multi-version RAG agent that generates multiple responses based on shuffled context."
    num_versions: int = 5
    retrival_num: int = 5


class ResponseGeneratorAgent(BaseChatAgent, Component[ResponseGeneratorAgentConfig]):
    component_config_schema = ResponseGeneratorAgentConfig

    def __init__(
        self,
        name: str,
        description: str = "A multi-version RAG agent that generates multiple responses based on shuffled context.",
        model_client=None,
        memory: Memory = None,
        num_versions: int = 5,
        retrival_num: int = 5,
    ):
        super().__init__(name=name, description=description)
        self._model_client = model_client
        self._memory = memory
        self._num_versions = num_versions
        self.retrival_num = retrival_num

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
            task_data = json.loads(task)
        except json.JSONDecodeError:
            yield Response(
                chat_message=TextMessage(content="Invalid JSON format in task.", source=self.name),
                inner_messages=[],
            )
            return
        query = task_data["query"]
        retrieval_context = task_data["retrieval_context"]        
        # Process versions concurrently
        results = await self._process_versions_concurrently(retrieval_context, query)
        
        # Extract answers
        answers = []
        for result in results:
            if hasattr(result["response"], 'content'):
                # Direct model response
                answers.append(result["response"].content)
            elif hasattr(result["response"], 'choices') and len(result["response"].choices) > 0:
                # OpenAI-style response
                answers.append(result["response"].choices[0].message.content)
            else:
                # Fallback
                answers.append(str(result["response"]))
        
        # Create a JSON object containing both arrays
        output_json = json.dumps({
            "query": query,
            "generated_answers": answers
        }, ensure_ascii=False, indent=2)
        
        # Yield the final response with the JSON object
        yield Response(
            chat_message=TextMessage(content=output_json, source=self.name),
            inner_messages=[],
        )
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent"""
        pass


    async def _process_versions_concurrently(self, retrieval_context: str, task: str) -> List[Dict]:
        """Process multiple versions concurrently"""
        tasks = []
        results = []
        prompt = prompts.construct_generate_prompt(task, retrieval_context)
        for i in range(self._num_versions):
            async def call_model(prompt):
                user_message = UserMessage(
                    content=prompt,
                    source="user"
                )
                model_result = await utils._call_model_with_rate_limit_retry(
                    model_client=self._model_client,
                    messages=[user_message]
                )
                return model_result
            # Wrap async call into task
            task_coro = call_model(prompt)
            tasks.append(task_coro)

            # Save context info for later assembly
            results.append({
                "version": i+1,
                "response": None  # Placeholder
            })
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks)
        # Fill responses into results
        for i, response in enumerate(responses):
            results[i]["response"] = response
        return results
    @classmethod
    def _from_config(cls, config: ResponseGeneratorAgentConfig) -> Self:
        return cls(
            name=config.name, 
            description=config.description, 
            num_versions=config.num_versions
        )

    def _to_config(self) -> ResponseGeneratorAgentConfig:
        return ResponseGeneratorAgentConfig(
            name=self.name,
            description=self.description,
            num_versions=self._num_versions,
        )
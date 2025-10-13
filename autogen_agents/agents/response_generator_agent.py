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
class ResponseGeneratorAgentConfig(BaseModel):
    name: str
    description: str = "A multi-version RAG agent that generates multiple responses based on shuffled context."
    num_versions: int = 3


class ResponseGeneratorAgent(BaseChatAgent, Component[ResponseGeneratorAgentConfig]):
    component_config_schema = ResponseGeneratorAgentConfig

    def __init__(
        self,
        name: str,
        description: str = "A multi-version RAG agent that generates multiple responses based on shuffled context.",
        model_client=None,
        memory: Memory = None,
        num_versions: int = 3,
    ):
        super().__init__(name=name, description=description)
        self._model_client = model_client
        self._memory = memory
        self._num_versions = num_versions

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

    async def _query_and_process(self, prompts: List[str]) -> tuple[List[str], List[List[str]]]:
        """Query memory and generate multiple shuffled versions of the context"""
        k = 5  # 目标结果数量
    
        # 并发查询
        query_results_list = await asyncio.gather(*[self._memory.query(prompt) for prompt in prompts])
        
        seen_contents = set()  # 用于去重 content
        merged_results = []    # 存储最终选中的结果对象
        index = 0              # 当前轮询的层级（第 index 个元素）
        
        while len(merged_results) < k:
            for query_result in query_results_list:
                if not query_result.results or index >= len(query_result.results):
                    continue
        
                item = query_result.results[index]
                content = item.content
        
                # 确保 content 是字符串
                if not isinstance(content, str):
                    content = str(content)
        
                # 如果 content 已存在，跳过（不计数）
                if content in seen_contents:
                    continue
        
                # 添加唯一内容
                seen_contents.add(content)
                merged_results.append(item)  
        
                # 达到 k 个就立即退出
                if len(merged_results) >= k:
                    break
        
            index += 1
            if index==len(query_results_list[0].results):
                break
        
        contents = list(seen_contents)  

        
        # Generate multiple versions
        versions = []
        for i in range(self._num_versions):
            # Create shuffled version
            shuffled_contents = contents.copy()
            random.shuffle(shuffled_contents)
            versions.append(shuffled_contents)

        return contents, versions

    async def _process_versions_concurrently(self, versions: List[List[str]], task: str) -> List[Dict]:
        """Process multiple versions concurrently"""
        tasks = []
        results = []
        prefix = '''Below is a question followed by context from different sources. Please answer the question based on the provided context.

Think step by step:

First, identify what the question is asking for (e.g., a person, date, location, or specific fact).
Next, extract relevant information from each source that relates to the question.
Then, Based on the relevant information, logically derive the answer. Explain how you reached your conclusion.
Finally, based on your reasoning, determine the answer. The answer should be a single word or entity.
If the information is insufficient to reach a confident conclusion after this analysis, respond with 'Insufficient Information.'.

Output format:

Thought: [Your step-by-step reasoning here]

Answer: [Final answer]'''

        for i, version in enumerate(versions):
            retrieval_context = '--------------'.join(e for e in version)
            prompt = f"{prefix}\n\nQuestion:{task}\n\nContext:\n\n{retrieval_context}"

            # Create async task for model client call
            async def call_model(prompt):
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
                return model_result
            # Wrap async call into task
            task_coro = call_model(prompt)
            tasks.append(task_coro)

            # Save context info for later assembly
            results.append({
                "version": i+1,
                "context_order": [content[:50] + "..." for content in version],
                "response": None  # Placeholder
            })

        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks)

        # Fill responses into results
        for i, response in enumerate(responses):
            results[i]["response"] = response

        return results

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
        task = task_data[-1]
        # Query and generate multiple versions
        original_contents, versions = await self._query_and_process(task_data)
        
        # Check if we have any versions
        if not versions:
            yield Response(
                chat_message=TextMessage(content="No relevant information found for the query.", source=self.name),
                inner_messages=[],
            )
            return
        
        # Process versions concurrently
        results = await self._process_versions_concurrently(versions, task)
        
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
        # 过滤掉长度超过2000的答案，避免选择答案时超出最大token限制
        filtered_answers = [ans for ans in answers if len(ans) < 2000]
        # Prepare the retrieval results (versions) and generated answers
        # Return the original retrieval result and all shuffled versions
        retrieval_results = original_contents
        generated_answers = filtered_answers
        
        # Create a JSON object containing both arrays
        output_json = json.dumps({
            "query": task,
            "retrieval_results": retrieval_results,
            "generated_answers": generated_answers
        }, ensure_ascii=False, indent=2)
        
        # Yield the final response with the JSON object
        yield Response(
            chat_message=TextMessage(content=output_json, source=self.name),
            inner_messages=[],
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent"""
        pass

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
import asyncio
import json
from typing import List, Dict, Sequence, AsyncGenerator
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken, Component
from autogen_core.memory import Memory
from pydantic import BaseModel
from typing_extensions import Self
import re
import pickle
from autogen_core.models import (
    SystemMessage,
    UserMessage,
)
from openai import RateLimitError
import prompts
import utils
class RetrievalAgentConfig(BaseModel):
    name: str
    description: str = "A retrieval agent that retrives the information from the knowledge base."



class RetrievalAgent(BaseChatAgent, Component[RetrievalAgentConfig]):
    component_config_schema = RetrievalAgentConfig

    def __init__(
        self,
        name: str,
        description: str = "A retrieval agent that retrives the information from the knowledge base.",
        model_client=None,
        memory: Memory = None,
        bm25_retriever=None,
        num_versions: int = 5,
        retrieval_num: int = 20,
        index_path="../bm25.pkl",
    ):
        super().__init__(name=name, description=description)
        self._model_client = model_client
        self._memory = memory
        self._bm25_retriever = bm25_retriever
        self._num_versions = num_versions
        self.retrieval_num = retrieval_num
        with open(index_path, "rb") as f:
            bm25_data = pickle.load(f)
            self._bm25_retriever = bm25_data["bm25"]
            self._bm25_chunks = bm25_data["chunks"]

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
        prompt = prompts.construct_decompose_prompt(task)
        system_message = SystemMessage(
            content="You are a helpful assistant.",
            source="system"
        )
        user_message = UserMessage(
            content=prompt,
            source="user"
        )
        model_result = await utils._call_model_with_rate_limit_retry(
            model_client=self._model_client,
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
        retrieval_context = await self._query_and_process(sub_questions)
        output_json = json.dumps({
            "query": task,
            "retrieval_context": retrieval_context
        }, ensure_ascii=False, indent=2)
        yield Response(
            chat_message=TextMessage(content=output_json, source=self.name),
            inner_messages=[],
        )

    async def bm25_retrieve(self, query: str, k: int = 20) -> List[Dict]:
        """Retrieve documents using BM25"""
            # 查询转小写并分词
        tokenized_query = query.lower().split()

        # 计算分数
        scores = self._bm25_retriever.get_scores(tokenized_query)

        # 获取 top-k 结果
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 过滤无相关性结果
                results.append(self._bm25_chunks[idx]["original"])
        return results

    async def _query_and_process(self, questions: List[str]) -> str:
        """Query memory and generate multiple shuffled versions of the context"""
        query_results_list = await asyncio.gather(*[self._memory.query(query) for query in questions])
        #query_results_list转换成str数组[[str]]
        for i, query_result in enumerate(query_results_list):
            if query_result.results:
                query_results_list[i] = [item.content for item in query_result.results]
        bm25_results_list = await asyncio.gather(*[self.bm25_retrieve(query, self.retrieval_num) for query in questions])
        query_results_list.extend(bm25_results_list)
        seen_contents = set()  # 用于去重 content
        index = 0              # 当前轮询的层级（第 index 个元素）
        while len(seen_contents) < self.retrieval_num:
            for query_result in query_results_list:
                if not query_result or index >= len(query_result):
                    continue       
                content = query_result[index]        
                # 确保 content 是字符串
                if not isinstance(content, str):
                    content = str(content)       
                # 如果 content 已存在，跳过（不计数）
                if content in seen_contents:
                    continue       
                # 添加唯一内容
                seen_contents.add(content)      
                # 达到 k 个就立即退出
                if len(seen_contents) >= self.retrieval_num:
                    break       
            index += 1
            if index==len(query_results_list[0]):
                break
        contents = list(seen_contents) 
        retrieval_context = "\n".join(
            json.dumps({
                "content": text
            }, ensure_ascii=False)
            for i, text in enumerate(contents)
        )

        return retrieval_context


    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset the agent"""
        pass

    @classmethod
    def _from_config(cls, config: RetrievalAgentConfig) -> Self:
        return cls(
            name=config.name, 
            description=config.description, 
        )

    def _to_config(self) -> RetrievalAgentConfig:
        return RetrievalAgentConfig(
            name=self.name,
            description=self.description,
        )
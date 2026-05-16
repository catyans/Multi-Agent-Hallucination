import asyncio
import json
from typing import List, Dict, Sequence, AsyncGenerator, Tuple
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
    UserMessage,
)
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
        summary_file_path: str = "chunks_with_summary.json",
        num_versions: int = 5,
        retrieval_num: int = 20,
        index_path="../bm25.pkl",
        enable_bm25 = False,
        enable_decompose = False
    ):
        super().__init__(name=name, description=description)
        self._model_client = model_client
        self._memory = memory
        self._bm25_retriever = bm25_retriever
        with open(summary_file_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
            #这是一个json数组文件，比如“chunks_with_summary.json”，里面每行有一个json对象，每个json对象有"original"和"summary"两个字段
            self._summary_dict = {item["original"]: item["summary"] for item in summary_data}
        self._num_versions = num_versions
        self.retrieval_num = retrieval_num
        with open(index_path, "rb") as f:
            bm25_data = pickle.load(f)
            self._bm25_retriever = bm25_data["bm25"]
            self._bm25_chunks = bm25_data["chunks"]
        self.enable_bm25 = enable_bm25
        self.enable_decompose = enable_decompose

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
        sub_questions = []
        usage = {}
        
        if self.enable_decompose:
            prompt = prompts.construct_decompose_prompt(task)
            user_message = UserMessage(
                content=prompt,
                source="user"
            )
            model_result = await utils._call_model_with_rate_limit_retry(
                model_client=self._model_client,
                messages=[user_message]
            )
            output = model_result.content
            usage = {
                "prompt_tokens": model_result.usage.prompt_tokens,
                "completion_tokens": model_result.usage.completion_tokens,
                "total_tokens": model_result.usage.prompt_tokens + model_result.usage.completion_tokens,
            }
            match = re.search(r"```json\s*(\{.*\})\s*```", output, re.DOTALL)
            if match:
                output = match.group(1)
            try:
                data = json.loads(output)
                sub_questions = data.get("sub_questions", [])
            except json.JSONDecodeError:
                sub_questions = []
            filtered_sub_questions = []
            for question in sub_questions:
                if isinstance(question, str):
                    word_count = len(question.split())
                    if word_count <= 300:
                        filtered_sub_questions.append(question)
            if len(filtered_sub_questions) > 2:
                filtered_sub_questions = filtered_sub_questions[:2]
            sub_questions = filtered_sub_questions
        sub_questions.append(task)
        usage["sub_questions"] = sub_questions
        retrieval_context, usage = await self._query_and_process(sub_questions, usage)
        output_json = json.dumps({
            "query": task,
            "retrieval_context": retrieval_context,
            "usage": usage
        }, ensure_ascii=False, indent=2)
        yield Response(
            chat_message=TextMessage(content=output_json, source=self.name),
            inner_messages=[],
        )

    async def _extract_keywords(self, query: str) -> Tuple[str, Dict]:
        """Extract keywords from the query using the model"""
        prompt = prompts.construct_bm25_keyword_prompt(query)
        user_message = UserMessage(
            content=prompt,
            source="user"
        )
        model_result = await utils._call_model_with_rate_limit_retry(
            model_client=self._model_client,
            messages=[user_message]
        )
        return model_result.content, {
            "prompt_tokens": model_result.usage.prompt_tokens,
            "completion_tokens": model_result.usage.completion_tokens,
            "total_tokens": model_result.usage.prompt_tokens + model_result.usage.completion_tokens,
        }

    async def bm25_retrieve(self, query: str, k: int = 20) -> List[Dict]:
        """Retrieve documents using BM25"""
        tokenized_query = query.lower().split()
        scores = self._bm25_retriever.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(self._bm25_chunks[idx])
        return results
    
    async def _query_and_process(self, questions: List[str], usage: Dict) -> Tuple[str, Dict]:
        """Query memory and generate multiple shuffled versions of the context"""
        query_results_list = await asyncio.gather(*[self._memory.query(query) for query in questions])
        for i, query_result in enumerate(query_results_list):
            if query_result.results:
                query_results_list[i] = [item.content for item in query_result.results]
        query_list = []
        if self.enable_bm25:
            gather_results = await asyncio.gather(*[self._extract_keywords(query) for query in questions])
            query_list = []
            bm25_usage_list = []
            for kw, bm25_usage in gather_results:
                query_list.append(kw)
                bm25_usage_list.append(bm25_usage)
            for bm25_usage in bm25_usage_list:
                usage["prompt_tokens"] += bm25_usage["prompt_tokens"]
                usage["completion_tokens"] += bm25_usage["completion_tokens"]
                usage["total_tokens"] += bm25_usage["total_tokens"]
            query_list = [
                f"{q} {k}" if k and str(k).strip() else q
                for q, k in zip(questions, query_list)
            ]
            usage["query_list"] = query_list
            bm25_results_list = await asyncio.gather(*[self.bm25_retrieve(query, self.retrieval_num) for query in query_list])
            query_results_list.extend(bm25_results_list)
        seen_contents = set()
        index = 0
        while len(seen_contents) < self.retrieval_num:
            for query_result in query_results_list:
                if not query_result or index >= len(query_result):
                    continue       
                content = query_result[index]        
                if not isinstance(content, str):
                    content = str(content)       
                if content in seen_contents:
                    continue       
                seen_contents.add(content)      
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

        return retrieval_context, usage


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
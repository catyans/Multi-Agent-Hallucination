import json
from typing import List, Dict, Any, Optional, Union, Tuple
from openai import AsyncOpenAI
from autogen_core.models import (
    SystemMessage,
    UserMessage,
)
class CustomModelResponse:
    def __init__(self, content: str):
        self.content = content

class CustomModelClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        temperature: float = 0.7,
        top_p: float = 0.8,
        n: int = 1,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    async def create(self, messages: List[Dict[str, str]]) -> Tuple[CustomModelResponse, Dict[str, int]]:
        openai_messages: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.content
            
            if isinstance(content, list):
                text_parts = [part for part in content if isinstance(part, str)]
                content = " ".join(text_parts) if text_parts else ""
    
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, UserMessage):
                role = "user"
            else:
                role = "assistant"
    
            openai_messages.append({"role": role, "content": content})
    
        request_kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n
        }

        response = await self.client.chat.completions.create(**request_kwargs)
        usage = {}
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        if self.n == 1:
            content = response.choices[0].message.content or ""
        else:
            contents_list = [
                choice.message.content or ""
                for choice in response.choices
            ]
            content = json.dumps({"contents": contents_list}, ensure_ascii=False)
        return CustomModelResponse(content=content), usage
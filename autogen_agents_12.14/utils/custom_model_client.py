from typing import List, Dict, Any, Optional, Union
from openai import AsyncOpenAI
from autogen_core.models import (
    SystemMessage,
    UserMessage,
)
class CustomModelResponse:
    """包装模型响应，提供 .content 属性（str 类型）"""
    def __init__(self, content: str):
        self.content = content

class CustomModelClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        lora_name: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = -1,
    ):
        self.model = model
        self.lora_name = lora_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    async def create(self, messages: List[Dict[str, str]]) -> CustomModelResponse:
        openai_messages: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                text_parts = [part for part in content if isinstance(part, str)]
                content = " ".join(text_parts) if text_parts else ""
            # 判断角色
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, UserMessage):
                role = "user"
            else:
                # 其他消息（如模型生成的回复）视为 assistant
                role = "assistant"
    
            openai_messages.append({"role": role, "content": content})
    
        request_kwargs = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens > 0:
            request_kwargs["max_tokens"] = self.max_tokens

        # 如果指定了 LoRA，通过 extra_body 透传（vLLM 支持）
        if self.lora_name is not None and self.lora_name.strip() != "":
            request_kwargs["extra_body"] = {"lora_name": self.lora_name}
        response = await self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        return CustomModelResponse(content=content) 
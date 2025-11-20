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
    ):
        self.model = model
        self.lora_name = lora_name
        self.temperature = temperature
        self.top_p = top_p
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    async def create(self, messages: List[Dict[str, str]]) -> CustomModelResponse:
        """
        调用模型生成回复
        
        Args:
            messages: OpenAI 格式的消息列表
            
        Returns:
            CustomModelResponse 对象，可通过 .content 获取字符串结果
        """
        openai_messages: List[Dict[str, Any]] = []
        for msg in messages:
            # 提取 content（UserMessage.content 可能是 list，需转为 str 或保留原样？）
            content = msg.content
            
            # 处理多模态 content（如 [str, Image]）—— 如果你的模型不支持，可强制转 str
            if isinstance(content, list):
                # 简单策略：只保留文本部分，或转为字符串（根据后端能力调整）
                # 这里保守处理：拼接所有字符串元素，忽略 Image（或报错）
                text_parts = [part for part in content if isinstance(part, str)]
                content = " ".join(text_parts) if text_parts else ""
                # 或者 raise ValueError("Multimodal input not supported")
    
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
        # 如果指定了 LoRA，通过 extra_body 透传（vLLM 支持）
        if self.lora_name is not None and self.lora_name.strip() != "":
            request_kwargs["extra_body"] = {"lora_name": self.lora_name}

        response = await self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        return CustomModelResponse(content=content) 
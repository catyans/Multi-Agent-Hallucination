import asyncio
from openai import RateLimitError
async def _call_model_with_rate_limit_retry(model_client, messages, max_retries=12, delay=5):
    for attempt in range(max_retries + 1):
        try:
            return await model_client.create(messages=messages)
        except RateLimitError as e:
            if attempt < max_retries:
                print(f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print(f"Max retries ({max_retries}) exceeded for rate limit. Raising error.")
                raise
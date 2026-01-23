import os
import yaml
import asyncio
import json
import sys
from typing import Any, Dict
from asyncio import Queue
from pathlib import Path
from datetime import datetime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_ext.memory.chromadb import ChromaDBVectorMemory, PersistentChromaDBVectorMemoryConfig,CustomEmbeddingFunctionConfig
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from agents.retrieval_agent import RetrievalAgent
from agents.response_generator_agent import ResponseGeneratorAgent
from agents.response_validator_agent import ResponseValidatorAgent
from agents.response_selector_agent import ResponseSelectorAgent

now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

with open('settings.yaml', 'r') as f:
    config = yaml.safe_load(f)
input_file = config.get("input_file")
output_file = config.get("output_file")
llm_config = config['llm']
model_client = OpenAIChatCompletionClient(
        model=llm_config['model'],
        model_info=ModelInfo(vision=False, function_calling=True, json_output=True, family="unknown", structured_output=True),
        api_key=os.environ.get('OPENAI_API_KEY', llm_config['api_key']),
        base_url=llm_config['base_url'],
)
selector_model_client = OpenAIChatCompletionClient(
        model=llm_config['lora_name_selector'],
        model_info=ModelInfo(vision=False, function_calling=True, json_output=True, family="unknown", structured_output=True),
        api_key=os.environ.get('OPENAI_API_KEY', llm_config['api_key']),
        base_url=llm_config['base_url_lora'],
)
validator_model_client = OpenAIChatCompletionClient(
        model=llm_config['lora_name_validator'],
        model_info=ModelInfo(vision=False, function_calling=True, json_output=True, family="unknown", structured_output=True),
        api_key=os.environ.get('OPENAI_API_KEY', llm_config['api_key']),
        base_url=llm_config['base_url_lora'],
        temperature=0.0,
        max_tokens=10
)
# Initialize vector memory
vector_store_config = PersistentChromaDBVectorMemoryConfig(
        collection_name=config['vector_store']['collection_name'],
        persistence_path=os.path.expandvars(config['vector_store']['persistence_path'].replace('${HOME}', str(Path.home()))),
        k=config['vector_store']['k'],
        score_threshold=config['vector_store']['score_threshold'],
)
    
if config['vector_store'].get('embedding', {}).get('use_custom', False):
        def create_openai_embedding_function(api_key, model, api_base):
            from chromadb.utils import embedding_functions
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model,
                api_base=api_base
            )
        
        embedding_config = config['vector_store']['embedding']
        params = {
            "api_key": embedding_config['api_key'],
            "model": embedding_config['model'],
            "api_base": embedding_config['api_base']
        }
        
        vector_store_config.embedding_function_config = CustomEmbeddingFunctionConfig(
            function=create_openai_embedding_function,
            params=params
        )
    
memory = ChromaDBVectorMemory(config=vector_store_config)


async def create_team_pool(size: int) -> Queue:
    pool = Queue()
    for i in range(size):
        text_termination = TextMentionTermination("TERMINATE:")
        retrieval_agent = RetrievalAgent(
            name='retrieval_agent',
            model_client=model_client,
            memory=memory,
            retrieval_num=config['vector_store']['k'],
            index_path=config['retrieval_agent']['bm25_index_path'],
            enable_bm25=config['retrieval_agent']['enable_bm25'],
            enable_decompose=config['retrieval_agent']['enable_decompose']
        )
        generator_agent = ResponseGeneratorAgent(
            name='generator_agent',
            model_client=model_client,
            memory=memory,
            num_versions=config['generator_agent']['num_versions']
        )
        validator_agent = ResponseValidatorAgent(
            name='validator_agent',
            model_client=validator_model_client,
            max_retries=config['validator_agent']['max_retries']
        )
        selector_agent = ResponseSelectorAgent(
            name='selector_agent',
            model_client=selector_model_client,
            memory=memory,
        )
        team = RoundRobinGroupChat(
            [retrieval_agent, generator_agent, selector_agent, validator_agent],
            termination_condition=text_termination
        )
        pool.put_nowait(team)
    return pool

async def process_item_with_pool(
    data: Dict[str, Any],
    line_num: int,
    team_pool: Queue
) -> Dict[str, Any]:
    team = None
    try:
        team = await team_pool.get()
        query = data.get("question")
        if not query:
            return {"line_num": line_num, "error": "lack 'query'", "status": "skipped"}

        result = await team.run(task=query)
        messages = result.messages
        retrival_contexts = []
        generated_answers_list=[]
        selected_answers=[]
        validator_answer=""
        selector_response=[]
        for message in messages:
            if message.source == 'retrieval_agent':
                retrival_task = message.content
                retrival_task_data = json.loads(retrival_task)
                retrieval_context = retrival_task_data["retrieval_context"]
                retrival_contexts.append(retrieval_context)
            elif message.source == 'generator_agent':
                generator_task = message.content
                generator_task_data = json.loads(generator_task)
                generated_answers = generator_task_data["generated_answers"]
                generated_answers_list.append(generated_answers)
            elif message.source == 'selector_agent':
                selector_task = message.content
                selector_task_data = json.loads(selector_task)
                selected_answer = selector_task_data["answer"]
                selected_answers.append(selected_answer)
                selector_response.append(selector_task_data["model_result"])
            elif message.source == 'validator_agent':
                validator_answer = message.content[10:]
        return {
            "query": query,
            "answer": data.get("answer", ""),
            "question_type": data.get("question_type", ""),
            "evidence_list": data.get("evidence_list", []),
            "retrival_contexts": retrival_contexts,
            "generated_answers_list": generated_answers_list,
            "selected_answers": selected_answers,
            "selector_response": selector_response,
            "validator_answer": validator_answer,
            "line_num": line_num,
            "status": "success"
        }

    except Exception as e:
        return {
            "line_num": line_num,
            "error": str(e),
            "query": data.get("query"),
            "status": "error"
        }
    finally:
        if team:
            await team_pool.put(team)

async def main():
    try:
        with open(input_file, "r", encoding="utf-8") as fin:
            data_list = json.load(fin)
    except Exception as e:
        print(f"❌ read input file error: {e}")
        sys.exit(1)

    processed_line_nums = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            line_num = item.get("line_num")
                            if line_num is not None:
                                processed_line_nums.add(line_num)
                        except json.JSONDecodeError:
                            continue
            print(f"🟡 read {len(processed_line_nums)} processed line nums from {output_file}")
        except Exception as e:
            print(f"⚠️ read output file error: {e}")

    team_pool = await create_team_pool(3)

    tasks = [
        process_item_with_pool(data, line_num, team_pool)
        for line_num, data in enumerate(data_list, start=1)
        if data and line_num not in processed_line_nums
    ]

    print(f"🚀 Total {len(data_list)} items, {len(processed_line_nums)} processed, {len(tasks)} remaining...")

    mode = "a" if processed_line_nums else "w"
    with open(output_file, mode, encoding="utf-8") as fout:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            try:
                fout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                fout.flush()

                line_num = result["line_num"]
                if result["status"] == "error":
                    print(f"❌ Error on line {line_num}: {result['error']}")
            except Exception as e:
                print(f"❌ Failed to write result: {e}")

    print(f"🎉 All tasks completed. Results saved to {output_file}")
# 🚀 运行
if __name__ == "__main__":
    asyncio.run(main())
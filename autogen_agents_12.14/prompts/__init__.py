import json
from typing import List, Union
def construct_decompose_prompt(task: str) -> str:
    prompt = '''
You are an expert at query rewriting for multi-hop question answering. Your task is to decompose a multi-hop question into two independent sub-questions that can be retrieved separately. The sub-questions should be unrelated to each other so that they cover different aspects of the original question.

Instructions:

Take the input multi-hop question.

Output two sub-questions that are independent and can be used for retrieval.

Ensure that the sub-questions are clear and concise.

Output only a JSON object in this format:
{{"sub_questions": ["...", "..."]}}

Question: "{question}"
'''
    prompt = prompt.format(question=task)
    return prompt

def construct_generate_prompt(task: str, retrieval_context: str) -> str:
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
    prompt = f"{prefix}\n\nQuestion:{task}\n\nContext:\n\n{retrieval_context}"
    return prompt

def construct_select_prompt(query, generated_answers, retrieval_context):
    """
    Constructs a prompt in English to ask a LLM to select the best answer from the generated answers.
    
    Args:
        query (str): The user's question.
        generated_answers (list of str): List of candidate answers to choose from.
        retrieval_results (list of str or str): Retrieved context snippets (used as reference).

    Returns:
        str: Formatted prompt in English for the LLM.
    """
    m = ""
    for i, answer in enumerate(generated_answers, 1):
        m += f"答案 {i}:{answer}\n"
    user_prompt = f"""用户的问题是：{query}
            
            请从以下答案中选择最正确的一个：{m}"""
    return user_prompt.strip()

def construct_validate_prompt(query, retrieval_context, answer):
    """
    Constructs a prompt in English to ask a LLM to judge if the answer 
    can be derived from the retrieval results given the question.
    
    Args:
        query (str): The user's query.
        retrieval_context (list of str or str): Retrieved context snippets.
        answer (str): The generated answer to evaluate.
    
    Returns:
        str: Formatted prompt in English for the LLM.
    """
    instruction = "Judge whether the given selected answer (containing both thought process and the final deduced answer) is correct based on the query. Respond strictly with either yes or no."
    input_content = f"Query: {query}\n\nSelected answer: {answer}"
    prompt = f"{instruction}\n\n{input_content}"
    
    return prompt.strip()

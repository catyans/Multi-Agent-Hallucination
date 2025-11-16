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
    prompt = f"Question: {query}\n"
    # Add retrieval results if available
    if retrieval_context:
        if isinstance(retrieval_context, list):
            for i, context in enumerate(retrieval_context, 1):
                prompt += f"Retrieval Context {i}: {context}\n"
        else:
            prompt += f"Retrieval Context: {retrieval_context}\n"        
    # Add generated answers with numbering
    for i, answer in enumerate(generated_answers, 1):
        prompt += json.dumps({f"answer{i}": answer})
    
    # Instructions in English
    prompt += '''
Carefully analyze the content of each provided answer option. Based on the question background, retrieved context, and general knowledge, perform step-by-step reasoning to evaluate the correctness, factual accuracy, and logical consistency of each answer.

Identify and eliminate options that contain errors, inaccuracies, or inconsistencies. After thorough analysis, select the single most correct and well-supported answer.

After completing your reasoning, please output the result in the following strict format:

Reasoning:

(Provide your detailed step-by-step analysis here)

Final Answer:

(Only output the numeric index corresponding to the correct answer, e.g., 1, 2, 3, etc.)
'''
    return prompt.strip()

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
    # If retrieval_results is a list, join them into a single string
    if isinstance(retrieval_context, list):
        context = "\n\n".join(f"[Snippet {i+1}]: {snippet}" for i, snippet in enumerate(retrieval_context))
    else:
        context = retrieval_context

    prompt = f"""\
    You are an impartial evaluator. Your task is to determine whether the provided answer can be directly supported by the information in the retrieval context for the given question.

    Please follow these steps:

    Read the question carefully.
    Review the retrieval context (context snippets).
    Examine the provided answer.
    Determine if the answer is fully supported — every piece of information in the answer must be explicitly stated or logically inferable from the retrieval context. No external knowledge or speculation.
    Output your judgment in exactly this format:
    {{"thought": "Brief reasoning based on context", "answer": "Yes or No"}}

    Question:
    {query}

    Retrieval Context:  
    {context}

    Answer:
    {answer}"""
    
    return prompt.strip()

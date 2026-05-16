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

def construct_bm25_keyword_prompt(task: str) -> str:
    prompt = '''You are an expert at query understanding and keyword extraction. Given a user's question, your task is to extract the most important and informative keywords or short phrases that best represent the core intent and key entities of the question. These keywords will be used as a query for a sparse retrieval system (e.g., BM25), so prioritize content words (nouns, verbs, named entities) and avoid stop words, questions words (who, what, when, etc.), and overly generic terms unless they are essential.
Output only the extracted keywords as a single line of space-separated terms—no explanations, no punctuation, no numbering.
Question: {question}
Keywords:'''
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
    prompt = f"Question: {query}\n\n"
    prompt += "Candidate answers (each includes reasoning):\n"
    for i, answer in enumerate(generated_answers, 1):
        prompt += f"{i}. {answer}\n"
    prompt += "\n"
    prompt += '''
Evaluate each answer based on:
- How well it addresses the question,
- The clarity and logical coherence of its reasoning,
- Whether the conclusion follows from the reasoning,
- Absence of contradictions or unsupported claims.

First, analyze the reasoning and conclusion of each candidate.
Then, compare them and explain why one is superior.

After completing your reasoning, please output the result in the following strict format:

Reasoning:

(Provide your detailed step-by-step analysis here)

Final Answer:

(Only output the numeric index corresponding to the correct answer, e.g., 1, 2, 3, etc.)
'''
    return prompt.strip()

def construct_validate_prompt(query, retrieval_context, answer):
    instruction = """You are an expert evaluator. Your task is to judge whether the provided answer is correct based on the query and the given retrieval context.

Please follow these steps strictly:
1. **Analyze the Query**: Understand exactly what information is being requested.
2. **Verify Reasoning**: Check if the thought process in the answer aligns logically with the facts found in the retrieval context.
3. **Fact-Check**: Ensure every claim in the final answer is supported by the retrieval context. Identify any hallucinations or logical gaps.
4. **Conclusion**: Based on your analysis, determine if the answer is fully correct.

Output Format:
- First, provide your detailed analysis and reasoning process.
- Finally, on a new line, output strictly "### Verdict: yes" or "### Verdict: no"."""

    input_content = f"""Context:
{retrieval_context}

Query: {query}

Selected Answer:
{answer}"""

    prompt = f"{instruction}\n\n{input_content}"
    return prompt.strip()

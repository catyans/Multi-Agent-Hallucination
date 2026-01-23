input_file = "multihop-rag_results.jsonl"
import json
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
split_index = int(len(lines) * 0.8)
test_data = lines[split_index:]
output_file = "MultiHopRAG.json"
data = []
for line in test_data:
    line = json.loads(line)
    query = line["query"]
    answer = line["answer"]
    question_type = line["question_type"]
    evidence_list = line["evidence_list"]
    data.append({
        "query": query,
        "answer": answer,
        "question_type": question_type,
        "evidence_list": evidence_list
    })
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
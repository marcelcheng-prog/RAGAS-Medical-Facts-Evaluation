# Clients Module

## Purpose

HTTP clients for external APIs: RAGFlow agent API and RAGAS evaluation framework.

## `medical_facts.py` - RAGFlow API Client

### `MedicalFactsClient`

- HTTP client with retry logic for RAGFlow agents
- `extract_facts(transcript)` - POST transcript to agent, returns `ApiResponse`
- `from_settings(agent_id, settings)` - Factory from settings
- Agent URL format: `{base_url}/{agent_id}/chat/completions`
- Auth: Bearer token from `MEDICAL_FACTS_AUTH_TOKEN`

### `parse_medical_facts(output)`

Parses JSON response handling multiple formats:
- Flat structures (Merger agent)
- Nested structures (4-agent parallel approach)
- Double-quote escaping issues
- Array responses

## `ragas_client.py` - RAGAS Evaluation

### `RagasEvaluator`

- `setup_ragas()` - Initialize LLM + embeddings with OpenAI
- Handles GPT-5.x models (`max_completion_tokens` vs `max_tokens`)
- `evaluate()` - Faithfulness, Context Recall, Answer Relevancy on SingleTurnSample
- `evaluate_family_history()` - Semantic family history recall
- `evaluate_semantic_list_recall()` - Generic list comparison for SOAP sections
  - Uses RAGAS `context_recall` metric
  - Converts item lists to text for semantic matching
  - Returns float score or None on error

"""
RAGAS Evaluation Client.

Handles RAGAS framework setup and evaluation for Medical Facts.
"""

import json
from typing import Optional, Tuple, Any
from dataclasses import dataclass

from openai import OpenAI

from ..config.settings import Settings, get_settings
from ..models.ground_truth import GroundTruth


@dataclass
class RagasScores:
    """RAGAS evaluation scores."""
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevancy: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None
    
    @property
    def all_available(self) -> bool:
        return all([
            self.faithfulness is not None,
            self.context_recall is not None,
            self.answer_relevancy is not None,
        ])


def setup_ragas(
    openai_client: OpenAI,
    model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-ada-002",
    verbose: bool = False,
) -> Tuple[Any, Any]:
    """
    Initialize RAGAS with LLM and embeddings.
    
    Args:
        openai_client: Configured OpenAI client
        model: LLM model name (supports gpt-4.x, gpt-5.x, o1, o3, etc.)
        embedding_model: Embedding model name
        verbose: Whether to print status messages
        
    Returns:
        Tuple of (llm, embeddings) for RAGAS evaluation
    """
    from ragas.llms import llm_factory
    from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings
    
    
    if verbose:
        print("  Creating LLM via llm_factory...")
    
    # Check if model is GPT-5.x or o-series (reasoning models)
    model_lower = model.lower()
    is_reasoning_model = (
        model_lower.startswith('gpt-5') or 
        model_lower.startswith('gpt-6') or
        (model_lower.startswith('o') and len(model_lower) >= 2 and model_lower[1].isdigit())
    )
    
    if is_reasoning_model:
        # For GPT-5.x and o-series: use max_completion_tokens directly
        llm = llm_factory(
            model=model,
            provider="openai",
            client=openai_client,
            max_completion_tokens=10000
        )
    else:
        # For GPT-4.x and older: use max_tokens
        llm = llm_factory(
            model=model,
            provider="openai",
            client=openai_client,
            max_tokens=10000
        )
    
    if verbose:
        print(f"  ✅ LLM ready: {llm}")
        print("  Creating embeddings...")
    
    # Use LangChain's OpenAIEmbeddings which has the embed_query method
    # required by RAGAS answer_relevancy metric
    embeddings = LangChainOpenAIEmbeddings(
        model=embedding_model,
        # LangChain will use OPENAI_API_KEY from environment
    )
    
    if verbose:
        print(f"  ✅ Embeddings ready: {embedding_model}")
    
    return llm, embeddings


class RagasEvaluator:
    """Evaluator using RAGAS framework."""
    
    def __init__(
        self,
        llm: Any,
        embeddings: Any,
        verbose: bool = False,
    ):
        """
        Initialize RAGAS evaluator.
        
        Args:
            llm: RAGAS LLM instance
            embeddings: RAGAS embeddings instance
            verbose: Whether to print status messages
        """
        self.llm = llm
        self.embeddings = embeddings
        self.verbose = verbose
    
    def evaluate(
        self,
        transcript: str,
        agent_output: str,
        ground_truth: GroundTruth,
    ) -> RagasScores:
        """
        Run RAGAS evaluation on agent output.
        
        Args:
            transcript: Original medical transcript
            agent_output: Raw output from the Medical Facts agent
            ground_truth: Expected ground truth data
            
        Returns:
            RagasScores with evaluation results
        """
        try:
            import pandas as pd
            from ragas import evaluate
            from ragas.metrics import faithfulness, context_recall, answer_relevancy
            from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
            
            if self.verbose:
                print("  Creating evaluation sample...")
            
            # Build reference from ground truth
            reference = json.dumps({
                "medications": ground_truth.all_medication_names,
                "symptoms": ground_truth.symptoms,
                "vital_signs": [
                    f"{v.parameter}: {v.value}" 
                    for v in ground_truth.vital_measurements
                ]
            }, ensure_ascii=False)
            
            sample = SingleTurnSample(
                user_input=transcript,
                response=agent_output,
                retrieved_contexts=[transcript],
                reference=reference
            )
            
            dataset = EvaluationDataset(samples=[sample])
            
            if self.verbose:
                print("  Running evaluate() with RAGAS metrics...")
                print(f"    Dataset features: {dataset.features()}")
            
            results = evaluate(
                dataset=dataset,
                metrics=[faithfulness, context_recall, answer_relevancy],
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            if self.verbose:
                print("  Extracting scores...")
            
            df = results.to_pandas()
            
            if self.verbose:
                print(f"    Result columns: {list(df.columns)}")
            
            scores = RagasScores()
            
            if 'faithfulness' in df.columns:
                val = df['faithfulness'].iloc[0]
                scores.faithfulness = float(val) if pd.notna(val) else None
                
            if 'context_recall' in df.columns:
                val = df['context_recall'].iloc[0]
                scores.context_recall = float(val) if pd.notna(val) else None
                
            if 'answer_relevancy' in df.columns:
                val = df['answer_relevancy'].iloc[0]
                scores.answer_relevancy = float(val) if pd.notna(val) else None
            
            if self.verbose:
                if scores.faithfulness is not None:
                    print(f"    ✅ Faithfulness: {scores.faithfulness:.1%}")
                if scores.context_recall is not None:
                    print(f"    ✅ Context Recall: {scores.context_recall:.1%}")
                if scores.answer_relevancy is not None:
                    print(f"    ✅ Answer Relevancy: {scores.answer_relevancy:.1%}")
            
            return scores
            
        except Exception as e:
            if self.verbose:
                print(f"  ❌ RAGAS failed: {e}")
                import traceback
                traceback.print_exc()
            
            return RagasScores(error=str(e))
    
    @classmethod
    def from_settings(
        cls,
        openai_client: OpenAI,
        settings: Optional[Settings] = None,
        verbose: bool = False,
    ) -> "RagasEvaluator":
        """
        Create evaluator from application settings.
        
        Args:
            openai_client: Configured OpenAI client
            settings: Settings object (uses default if not provided)
            verbose: Whether to print status messages
            
        Returns:
            Configured RagasEvaluator
        """
        if settings is None:
            settings = get_settings()
        
        llm, embeddings = setup_ragas(
            openai_client=openai_client,
            model=settings.openai_model,
            embedding_model=settings.openai_embedding_model,
            verbose=verbose,
        )
        
        return cls(
            llm=llm, 
            embeddings=embeddings, 
            verbose=verbose,
        )

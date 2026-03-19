# behavioral_scanner.py
"""
Behavioral risk scanner for detecting hallucinations, toxicity, and safety violations.

Four-layer architecture:
  Layer 1 - Fast heuristics: empty response, refusals, prompt repetition, math errors (zero-latency)
  Layer 2 - Intra-response NLI: checks if sentences within the response contradict each other
  Layer 3 - Toxicity classifier: detects unsafe/harmful content
  Layer 4 - Self-consistency sampling (optional deep scan): SelfCheckGPT-style hallucination
             detection via multiple LLM samples at temperature > 0

All local inference happens on-device — no external API calls except optional LLM sampling in
deep scan mode (requires llm_fn callback).
"""

import re
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Callable, Dict, Any, List, Optional


class BehavioralScanner:
    """
    Local behavioral scanner for LLM outputs using ML models.

    Args:
        llm_fn: Optional callable for deep scan mode (self-consistency sampling).
                Signature: llm_fn(prompt: str, temperature: float) -> str
                If not provided, deep=True in scan() will warn and fall back.
    """

    def __init__(self, llm_fn: Optional[Callable] = None):
        print("Loading behavioral scanner models...")

        self.llm_fn = llm_fn

        # NLI model for intra-response consistency checking
        self.nli_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

        # Toxicity model for unsafe content detection
        self.tox_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
        self.tox_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

        print("Behavioral scanner models loaded successfully.")

    # ------------------------------------------------------------------
    # Layer 1: Fast heuristics (zero-latency, no model inference)
    # ------------------------------------------------------------------

    def fast_heuristics(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Detect obvious response failures without model inference.

        Returns:
            Dict of boolean flags and a heuristic_risk score (0-1)
        """
        flags = {
            "is_empty": False,
            "is_refusal": False,
            "is_repetition": False,
            "has_math_error": False,
        }

        # Empty or near-empty response
        if not response or len(response.strip()) < 5:
            flags["is_empty"] = True

        # Common refusal patterns (unexpected refusals inflate uncertainty)
        refusal_patterns = [
            "i cannot", "i can't", "i am not able", "as an ai",
            "i'm not able to", "i don't have access", "i am unable to"
        ]
        if any(p in response.lower() for p in refusal_patterns):
            flags["is_refusal"] = True

        # Response is mostly a repetition of the prompt
        prompt_words = set(prompt.lower().split())
        response_words = response.lower().split()
        if len(prompt_words) > 3 and len(response_words) > 0:
            overlap = len(prompt_words & set(response_words)) / (len(response_words) + 1e-6)
            if overlap > 0.8:
                flags["is_repetition"] = True

        # Arithmetic claim verification (e.g. "1+1=3")
        math_error_score = self._check_math_claims(response)
        if math_error_score > 0:
            flags["has_math_error"] = True

        # Combine flags into a heuristic risk score
        heuristic_risk = 0.0
        if flags["is_empty"]:
            heuristic_risk = 1.0
        elif flags["has_math_error"]:
            heuristic_risk = max(heuristic_risk, math_error_score)
        if flags["is_repetition"]:
            heuristic_risk = max(heuristic_risk, 0.7)

        flags["heuristic_risk"] = round(heuristic_risk, 3)
        flags["math_error_score"] = round(math_error_score, 3)
        return flags

    def _check_math_claims(self, text: str) -> float:
        """
        Verify simple arithmetic expressions in text (e.g. 1+1=3 → error).

        Returns:
            Fraction of detected arithmetic claims that are wrong (0-1).
        """
        pattern = r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)"
        matches = re.findall(pattern, text)
        if not matches:
            return 0.0

        errors = 0
        for a, op, b, claimed in matches:
            a, b, claimed = int(a), int(b), int(claimed)
            if op == "+":
                correct = a + b
            elif op == "-":
                correct = a - b
            elif op == "*":
                correct = a * b
            elif op == "/" and b != 0:
                correct = a // b
            else:
                continue
            if correct != claimed:
                errors += 1

        return errors / len(matches)

    # ------------------------------------------------------------------
    # Layer 2: Intra-response NLI contradiction detection
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str):
        """Split text into sentences, filtering fragments under 4 words."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if len(s.split()) >= 4]

    def detect_internal_contradiction(self, response: str) -> float:
        """
        Check whether sentences within the response contradict each other.
        Uses NLI on sentence pairs — high score means internal inconsistency.

        Previously: NLI(prompt, response) — wrong framing for Q&A.
        Now: NLI(sentence_i, sentence_j) for pairs within the response.

        Returns:
            Max contradiction probability across sentence pairs (0-1).
        """
        sentences = self._split_sentences(response)
        if len(sentences) < 2:
            return 0.0

        # Build pairs, capped at 10 to avoid O(n^2) latency blow-up
        pairs = [
            (sentences[i], sentences[j])
            for i in range(len(sentences))
            for j in range(i + 1, len(sentences))
        ][:10]

        max_contradiction = 0.0
        for premise, hypothesis in pairs:
            inputs = self.nli_tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                logits = self.nli_model(**inputs).logits
                probs = logits.softmax(dim=1)
                # BART-MNLI: index 0 = contradiction, 1 = neutral, 2 = entailment
                contradiction = probs[0][0].item()
            max_contradiction = max(max_contradiction, contradiction)

        return max_contradiction

    # ------------------------------------------------------------------
    # Layer 3: Toxicity detection (unchanged, directionally correct)
    # ------------------------------------------------------------------

    def detect_toxicity(self, text: str) -> float:
        """
        Use toxicity model to detect unsafe or harmful content.

        Returns:
            Toxicity probability (0-1)
        """
        inputs = self.tox_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self.tox_model(**inputs).logits
            # toxic-bert is multi-label: sigmoid gives independent probability per class
            # labels: {0: toxic, 1: severe_toxic, 2: obscene, 3: threat, 4: insult, 5: identity_hate}
            probs = logits.sigmoid()
            toxicity_score = probs[0].max().item()

        return toxicity_score

    # ------------------------------------------------------------------
    # Uncertainty heuristic (fixed: hedging language, not response length)
    # ------------------------------------------------------------------

    def uncertainty_score(self, response: str) -> float:
        """
        Detect uncertainty via hedging language and word repetition.

        Previously penalized long responses — wrong for long-form prompts.
        Now detects hedging phrases ('I think', 'possibly', etc.) and
        high word repetition, which are actual signals of low-confidence output.

        Returns:
            Uncertainty score (0-1)
        """
        if not response or len(response.strip()) == 0:
            return 1.0

        hedging_phrases = [
            "i think", "i believe", "i'm not sure", "i am not sure",
            "possibly", "perhaps", "it may be", "it might be",
            "approximately", "roughly", "unclear", "uncertain",
            "not entirely sure", "could be", "might be",
            "seems like", "appears to be", "i'm not certain",
            "i am not certain", "not sure", "I think maybe"
        ]
        response_lower = response.lower()
        hedge_count = sum(1 for phrase in hedging_phrases if phrase in response_lower)
        hedging_score = min(hedge_count / 3.0, 1.0)

        words = response.split()
        unique_words = len(set(words))
        repetition_score = 1 - (unique_words / (len(words) + 1e-6))

        return min(0.5 * hedging_score + 0.5 * repetition_score, 1.0)

    # ------------------------------------------------------------------
    # Layer 4: Self-consistency deep scan (optional, requires llm_fn)
    # ------------------------------------------------------------------

    def self_consistency_scan(
        self,
        prompt: str,
        response: str,
        n_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        SelfCheckGPT-style hallucination detection via repeated sampling.

        Generates n_samples additional responses at temperature=0.7, then for each
        sentence in the original response checks whether the samples contradict it.
        Sentences that are consistently contradicted by the samples are likely hallucinated.

        Requires llm_fn to be set on the scanner.

        Args:
            prompt: Original user prompt
            response: Original LLM response (temperature=0)
            n_samples: Number of stochastic samples to generate

        Returns:
            Dict with per-sentence scores and aggregate consistency_score (0-1,
            higher = more likely hallucinated)
        """
        if self.llm_fn is None:
            raise ValueError(
                "llm_fn must be provided at BehavioralScanner init to use deep scan. "
                "Pass llm_fn=<callable(prompt, temperature) -> str>."
            )

        # Generate stochastic samples
        samples: List[str] = []
        for _ in range(n_samples):
            try:
                samples.append(self.llm_fn(prompt, temperature=0.7))
            except Exception as e:
                warnings.warn(f"llm_fn sample failed: {e}")

        if not samples:
            return {"consistency_score": 0.0, "sentence_scores": []}

        sentences = self._split_sentences(response)
        if not sentences:
            return {"consistency_score": 0.0, "sentence_scores": []}

        sentence_scores = []
        for sentence in sentences:
            per_sample_contradictions = []
            for sample in samples:
                # Premise = the sampled response; hypothesis = sentence from original.
                # High contradiction means the sample disagrees with the original sentence.
                inputs = self.nli_tokenizer(
                    sample,
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    logits = self.nli_model(**inputs).logits
                    probs = logits.softmax(dim=1)
                    contradiction = probs[0][0].item()
                per_sample_contradictions.append(contradiction)

            avg = sum(per_sample_contradictions) / len(per_sample_contradictions)
            sentence_scores.append({"sentence": sentence, "score": round(avg, 3)})

        aggregate = sum(s["score"] for s in sentence_scores) / len(sentence_scores)
        return {
            "consistency_score": round(aggregate, 3),
            "sentence_scores": sentence_scores,
        }

    # ------------------------------------------------------------------
    # Main scan entry point
    # ------------------------------------------------------------------

    def scan(self, prompt: str, response: str, deep: bool = False) -> Dict[str, Any]:
        """
        Scan prompt and response for behavioral risks.

        Pipeline:
          1. Fast heuristics (zero-latency: empty, math errors, refusals, repetition)
          2. Intra-response NLI contradiction detection
          3. Toxicity classification
          4. (Optional) Self-consistency deep scan — pass deep=True; requires llm_fn at init

        Args:
            prompt: User input prompt
            response: LLM response text
            deep: If True, run self-consistency sampling (requires llm_fn set at init).
                  Slower but catches hallucinations that are internally consistent.

        Returns:
            Dictionary with behavioral risk scores, flags, and optional deep scan results.
        """
        # Layer 1: Fast heuristics
        flags = self.fast_heuristics(prompt, response)

        # Short-circuit: empty response
        if flags["is_empty"]:
            return {
                "hallucination_score": 1.0,
                "contradiction_score": 0.0,
                "consistency_score": None,
                "toxicity_score": 0.0,
                "uncertainty_score": 1.0,
                "safety_violation_score": 0.0,
                "behavioral_risk": 0.8,
                "has_behavioral_issues": True,
                "deep_scan": False,
                "flags": flags,
            }

        # Layer 2: Intra-response NLI
        contradiction_score = self.detect_internal_contradiction(response)

        # Layer 4: Self-consistency deep scan (optional)
        consistency_score = None
        deep_scan_detail = None
        if deep:
            if self.llm_fn is None:
                warnings.warn(
                    "deep=True passed but llm_fn is not set. "
                    "Skipping self-consistency scan. "
                    "Pass llm_fn=<callable(prompt, temperature) -> str> at init."
                )
            else:
                deep_result = self.self_consistency_scan(prompt, response)
                consistency_score = deep_result["consistency_score"]
                deep_scan_detail = deep_result["sentence_scores"]

        # Hallucination = max signal across all hallucination detectors
        hallucination_signal = max(
            contradiction_score,
            flags["math_error_score"],
            consistency_score if consistency_score is not None else 0.0,
        )
        print("Contradiction score: ", contradiction_score)
        
        if flags["is_repetition"]:
            hallucination_signal = max(hallucination_signal, 0.7)

        # Layer 3: Toxicity
        toxicity_score = self.detect_toxicity(response)

        # Uncertainty (hedging-based)
        uncertainty = self.uncertainty_score(response)

        # Overall behavioral risk — weights tuned for NLI + toxicity as primary signals
        behavioral_risk = (
            0.4 * hallucination_signal +
            0.4 * toxicity_score +
            0.2 * uncertainty
        )

        result = {
            "hallucination_score": round(hallucination_signal, 3),
            "contradiction_score": round(contradiction_score, 3),
            "consistency_score": round(consistency_score, 3) if consistency_score is not None else None,
            "toxicity_score": round(toxicity_score, 3),
            "uncertainty_score": round(uncertainty, 3),
            "safety_violation_score": round(toxicity_score, 3),
            "behavioral_risk": round(behavioral_risk, 3),
            "has_behavioral_issues": behavioral_risk > 0.5 or toxicity_score > 0.8 or hallucination_signal > 0.8,
            "deep_scan": deep,
            "flags": flags,
        }
        if deep_scan_detail is not None:
            result["sentence_scores"] = deep_scan_detail
        return result

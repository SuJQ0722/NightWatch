"""AOI's rubric and scoring, with an optional OpenAI-compatible judge transport."""

from nightwatch.config import env


def make_evaluator(model=None):
    from dynacubench.llm_evaluator import LLMEvaluator

    model = model or env("DYNACU_JUDGE", "gemini-2.0-flash")
    base = env("DYNACU_JUDGE_BASE_URL")
    if not base:
        return LLMEvaluator(model)

    class CompatibleEvaluator(LLMEvaluator):
        def _init_client(self):
            if self._client is None:
                import openai

                key = env("DYNACU_JUDGE_API_KEY")
                if not key and "openrouter.ai" in base:
                    key = env("OPENROUTER_API_KEY") or env("AGENT_API_KEY")
                if not key:
                    raise RuntimeError("DYNACU_JUDGE_API_KEY is required for this judge endpoint")
                self._client = openai.OpenAI(base_url=base, api_key=key, timeout=60, max_retries=2)

        def _call_model(self, prompt):
            response = self._client.chat.completions.create(
                model=self.judge_model, messages=[{"role": "user", "content": prompt}],
                max_tokens=1024, temperature=0,
            )
            text = response.choices[0].message.content
            if not text:
                raise RuntimeError("Judge returned no answer")
            return text

    return CompatibleEvaluator(model)


def checked_outcome(outcome):
    """A failed judge is an infrastructure error, never a benchmark failure/pass."""
    result = outcome.llm_result
    if result is not None and result.error:
        raise RuntimeError(f"DynaCU judge failed: {result.error}")
    evidence = outcome.to_dict()
    if result is not None:
        evidence["llm_raw_response"] = result.raw_response
    return evidence

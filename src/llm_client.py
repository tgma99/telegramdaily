import json
import time
import re
from typing import Dict, Any, Optional, List
from openai import OpenAI
from src.config import Secrets

class LLMClient:
    def __init__(self, secrets: Secrets):
        self.secrets = secrets
        # Initialize OpenAI Client (for remote)
        self.oa_client = OpenAI(api_key=secrets.openai_api_key, base_url=secrets.openai_base_url)
        
        # Initialize Local Client (for LM Studio)
        # Note: LM Studio creates an OpenAI-compatible endpoint
        self.local_client = None
        if secrets.lmstudio_base_url:
            self.local_client = OpenAI(
                api_key="lm-studio", 
                base_url=secrets.lmstudio_base_url
            )

    def _get_client_and_model(self, prefer_local: bool):
        if prefer_local and self.local_client:
            return self.local_client, self.secrets.lmstudio_model or "local-model"
        return self.oa_client, self.secrets.openai_model

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        prefer_local: bool = False,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_retries: int = 3
    ) -> str:
        client, model = self._get_client_and_model(prefer_local)
        
        # Determine if we're likely pointing to a local instance
        base_url = str(client.base_url).lower()
        is_likely_local = "localhost" in base_url or "127.0.0.1" in base_url or "0.0.0.0" in base_url

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Only apply response_format if we requested JSON mode
        # AND we are not obviously running against a local LLM that might choke on it.
        # (Though some local servers support it, many don't. We'll rely on retry logic too.)
        if json_mode:
            # If it's a known local client object, or the URL looks local, be conservative initially
            is_using_local_client = prefer_local and self.local_client is not None
            
            if not is_using_local_client and not is_likely_local:
                 kwargs["response_format"] = {"type": "json_object"}

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e)
                # Check for the specific error about response_format
                if "response_format" in err_str and "json_object" in str(kwargs.get("response_format", "")):
                    print(f"  ⚠️ LLM doesn't support json_object (Error: {e}). Retrying without it...")
                    kwargs.pop("response_format", None)
                    # Retry immediately in the next loop iteration (or same if we want, but loop is fine)
                    continue
                    
                last_err = e
                print(f"  ❌ LLM Error (Attempt {attempt}/{max_retries}, Model: {model}): {e}")
                time.sleep(min(2 ** attempt, 10))
        
        raise last_err or RuntimeError("Unknown LLM error")

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        """Robust parsing of potentially malformed JSON from LLMs."""
        text = text.strip()
        
        # Remove <think>...</think> blocks common in reasoning models
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        # Strip code fences
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
        
        # Try finding the first brace pair
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)
            
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def classify_and_translate(self, prompt: str, prefer_local: bool = False) -> Dict[str, Any]:
        """
        Classify and translate a text.
        Returns a dict. Expects JSON output from the LLM.
        """
        messages = [
            {"role": "system", "content": "Return ONLY a valid JSON object."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            resp = self.chat_completion(messages, prefer_local=prefer_local, json_mode=True)
            parsed = self._safe_parse_json(resp)
            if not parsed:
                print(f"  ⚠️ LLM Output Parsing Failed. Raw: {resp[:100]}...")
            return parsed
        except Exception as e:
            print(f"  ❌ LLM Error: {e}")
            return {"skip": True, "error": str(e)}

    def deduplicate_pair(self, prompt: str, prefer_local: bool = False) -> Dict[str, Any]:
        """
        Compare two items and decide if they are duplicates.
        """
        messages = [
            {"role": "system", "content": "You are a deduplication judge. Return ONLY JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            resp = self.chat_completion(messages, prefer_local=prefer_local, json_mode=True)
            return self._safe_parse_json(resp)
        except Exception:
            return {"decision": "error", "confidence": 0.0}

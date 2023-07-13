from typing import Any, Dict, List
from uuid import UUID
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import LLMResult

class StreamingWebCallbackHandler(BaseCallbackHandler):
    tokens: List[str] = []
    is_responding: bool = False
    response_id: UUID
    
    def on_llm_new_token(self, token: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens.append(token)  

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
        self.is_responding = True
        self.response_id = run_id

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
       self.is_responding = False

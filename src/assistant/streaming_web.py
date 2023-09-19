import sys
from uuid import UUID
from typing import Any, Dict, List
from langchain.callbacks.base import BaseCallbackHandler

class StreamingWebCallbackHandler(BaseCallbackHandler):
    tokens: List[str] = []
    is_responding: bool = False
    response_id: str
    response: str = None
    
    def on_llm_new_token(self, token: str, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()
        self.tokens.append(token)

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
        self.is_responding = True
        self.response_id = run_id
        self.response = None

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.is_responding = False
        self.response = outputs['response']
        print("END: "+self.response)

    def get_response(self) -> str:
        response_result = self.response
        self.response = None
        
        return response_result
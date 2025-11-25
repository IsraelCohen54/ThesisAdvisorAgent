# External Tools (e.g. Gemini)
from abc import ABC, abstractmethod
from typing import Any, Dict


# Dependency Inversion Principle: Agents depend on this abstraction, not the concrete ADK class.
class AgentInterface(ABC):
    """Abstract contract for all Agents."""

    @abstractmethod
    def run(self, input_data: str, session_id: str) -> str:
        """Processes input and returns a result."""
        pass


class ToolInterface(ABC):
    """Abstract contract for all Tools."""

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Executes the external action (search, database, etc.)."""
        pass

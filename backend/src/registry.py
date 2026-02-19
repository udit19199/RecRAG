from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineDefinition:
    """Definition of a pipeline for registration."""

    pipeline_id: str
    pipeline_type: str
    config: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineRegistry:
    """Registry for pipeline definitions."""

    def __init__(self):
        self._pipelines: dict[str, PipelineDefinition] = {}

    def register(self, definition: PipelineDefinition) -> None:
        self._pipelines[definition.pipeline_id] = definition

    def get(self, pipeline_id: str) -> PipelineDefinition | None:
        return self._pipelines.get(pipeline_id)

    def list_all(self) -> list[PipelineDefinition]:
        return list(self._pipelines.values())

    def clear(self) -> None:
        self._pipelines.clear()

    def remove(self, pipeline_id: str) -> bool:
        if pipeline_id in self._pipelines:
            del self._pipelines[pipeline_id]
            return True
        return False


_registry = PipelineRegistry()


def get_registry() -> PipelineRegistry:
    """Get the global pipeline registry."""
    return _registry

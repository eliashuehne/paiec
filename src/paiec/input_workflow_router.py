"""Version routing without altering any historical, hash-pinned renderer."""
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .collected_inputs_v3 import CollectedInputWorkflowConfig


class GroupedPreparationWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["grouped_enriched_preparation_workflow_v1"]
    recipe: str = "configs/head_finetuning/grouped_enriched_v1.json"
    native_python: Literal[".venv-laya/bin/python"] = ".venv-laya/bin/python"


class InputWorkflowConfig:
    @classmethod
    def model_validate(cls, value):
        if value.get("version") == "grouped_enriched_preparation_workflow_v1":
            return GroupedPreparationWorkflow.model_validate(value)
        return CollectedInputWorkflowConfig.model_validate(value)

    @classmethod
    def model_validate_json(cls, value):
        return cls.model_validate(json.loads(value))

from typing import Any, Dict, Optional

from pydantic import BaseModel


class WorkflowRunRequest(BaseModel):
    config_path: str


class WorkflowRunResponse(BaseModel):
    run_id: str
    status: str


class RunSummary(BaseModel):
    run_id: str
    task: str
    model_type: str
    status: str
    config_path: str
    output_dir: str
    metrics: Optional[Dict[str, Any]] = None
    report_path: Optional[str] = None
    created_at: str
    updated_at: str

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceFrame(BaseModel):
    timestamp_sec: float = Field(ge=0)
    image_base64: str


class Violation(BaseModel):
    type: str
    timecode_sec: float = Field(ge=0)
    description: str
    evidence_frames: list[EvidenceFrame]


class AnalysisResult(BaseModel):
    job_id: str
    status: Literal["processing", "done", "failed"]
    verdict: Literal["no_violation", "violation"] | None = None
    violations: list[Violation] = Field(default_factory=list)
    error: str | None = None
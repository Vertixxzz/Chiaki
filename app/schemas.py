from pydantic import BaseModel

class JobResponse(BaseModel):
    job_id: str
    status: str

class JobResult(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None

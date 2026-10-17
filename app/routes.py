from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.config import PROJECT_DIR
from app.ml.inference import analyze_video

router = APIRouter()
templates = Jinja2Templates(directory=str(PROJECT_DIR / 'templates'))
jobs: dict[str, dict] = {}
UPLOADS = PROJECT_DIR / 'outputs' / 'uploads'

def _run(job_id: str, path: Path) -> None:
    try:
        jobs[job_id] = {'status': 'done', 'result': analyze_video(path)}
    except Exception as error:
        jobs[job_id] = {'status': 'failed', 'error': str(error)}

@router.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@router.post('/api/v1/analyze')
async def analyze(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    if Path(video.filename or '').suffix.lower() != '.mov':
        raise HTTPException(400, 'Only .mov files are accepted')
    UPLOADS.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid4()); target = UPLOADS / f'{job_id}.mov'
    target.write_bytes(await video.read())
    jobs[job_id] = {'status': 'processing'}
    background_tasks.add_task(_run, job_id, target)
    return {'job_id': job_id, 'status': 'processing'}

@router.post('/analyze', response_class=HTMLResponse)
async def analyze_form(request: Request, video: UploadFile = File(...)):
    """No-JavaScript form flow: render the completed result in the next page."""
    if Path(video.filename or '').suffix.lower() != '.mov':
        raise HTTPException(400, 'Only .mov files are accepted')
    UPLOADS.mkdir(parents=True, exist_ok=True)
    target = UPLOADS / f'{uuid4()}.mov'
    target.write_bytes(await video.read())
    try:
        result = analyze_video(target)
    except Exception as error:
        return templates.TemplateResponse('result.html', {'request': request, 'error': str(error)})
    return templates.TemplateResponse('result.html', {'request': request, 'result': result})

@router.get('/api/v1/jobs/{job_id}')
def job(job_id: str):
    if job_id not in jobs: raise HTTPException(404, 'Unknown job')
    return {'job_id': job_id, **jobs[job_id]}

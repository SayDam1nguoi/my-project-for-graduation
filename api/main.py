"""
Interview Analysis API

FastAPI backend để phân tích video phỏng vấn.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import shutil
from pathlib import Path
import uuid
import sys
import logging
from datetime import datetime

# Import core engine
sys.path.append('./')
from src.evaluation.integrated_interview_evaluator import IntegratedInterviewEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Interview Analysis API",
    description="API để phân tích video phỏng vấn và đánh giá ứng viên",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: thay bằng domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
UPLOAD_DIR = Path("api_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory database (Production: dùng Redis hoặc PostgreSQL)
jobs_db: Dict[str, Dict[str, Any]] = {}


# Pydantic models
class JobStatus(BaseModel):
    job_id: str
    status: str  # uploaded, processing, completed, failed
    message: Optional[str] = None


class AnalysisResult(BaseModel):
    job_id: str
    status: str
    filename: str
    scores: Optional[Dict[str, float]] = None
    rating: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


# ===== ENDPOINTS =====

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "Interview Analysis API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Kiểm tra sức khỏe API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_jobs": len(jobs_db)
    }


@app.post("/api/upload", response_model=JobStatus)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video phỏng vấn.
    
    Returns:
        job_id để track tiến trình
    """
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only video files are allowed."
        )
    
    # Tạo job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Lưu file
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Tạo job record
        jobs_db[job_id] = {
            "job_id": job_id,
            "status": "uploaded",
            "filename": file.filename,
            "file_path": str(file_path),
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "scores": None,
            "rating": None,
            "details": None,
            "error": None
        }
        
        logger.info(f"[{job_id}] Video uploaded: {file.filename}")
        
        return JobStatus(
            job_id=job_id,
            status="uploaded",
            message=f"Video uploaded successfully. Use job_id to start analysis."
        )
        
    except Exception as e:
        logger.error(f"[{job_id}] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/{job_id}", response_model=JobStatus)
async def start_analysis(job_id: str, background_tasks: BackgroundTasks):
    """
    Bắt đầu phân tích video (background task).
    
    Args:
        job_id: ID từ endpoint /api/upload
    """
    
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    if job["status"] == "processing":
        return JobStatus(
            job_id=job_id,
            status="processing",
            message="Analysis already in progress"
        )
    
    if job["status"] == "completed":
        return JobStatus(
            job_id=job_id,
            status="completed",
            message="Analysis already completed"
        )
    
    # Update status
    jobs_db[job_id]["status"] = "processing"
    
    # Add background task
    background_tasks.add_task(process_video, job_id)
    
    logger.info(f"[{job_id}] Analysis started")
    
    return JobStatus(
        job_id=job_id,
        status="processing",
        message="Analysis started. Check /api/status/{job_id} for progress."
    )


@app.post("/api/analyze-sync", response_model=AnalysisResult)
async def analyze_video_sync(file: UploadFile = File(...)):
    """
    Upload và phân tích video ngay lập tức (synchronous).
    Phù hợp cho video ngắn hoặc testing.
    
    Warning: Request có thể mất vài phút!
    """
    
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}"
        )
    
    # Tạo job ID
    job_id = str(uuid.uuid4())[:8]
    
    # Lưu file
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"[{job_id}] Analyzing video: {file.filename}")
        
        # Phân tích ngay
        evaluator = IntegratedInterviewEvaluator()
        score, details = evaluator.evaluate_video_interview(
            video_path=str(file_path),
            candidate_id=job_id,
            save_report=False
        )
        
        # Tạo result
        result = {
            "job_id": job_id,
            "status": "completed",
            "filename": file.filename,
            "scores": {
                "emotion": round(score.emotion_score, 2),
                "focus": round(score.focus_score, 2),
                "clarity": round(score.clarity_score, 2),
                "content": round(score.content_score, 2),
                "total": round(score.total_score, 2)
            },
            "rating": score.overall_rating,
            "details": {
                "emotion": details.get("emotion", {}),
                "focus": details.get("focus", {}),
                "clarity": details.get("clarity", {}),
                "content": details.get("content", {})
            },
            "error": None,
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat()
        }
        
        # Lưu vào DB
        jobs_db[job_id] = result
        
        logger.info(f"[{job_id}] Analysis completed: {score.total_score:.1f}/10")
        
        return AnalysisResult(**result)
        
    except Exception as e:
        logger.error(f"[{job_id}] Analysis failed: {e}")
        
        error_result = {
            "job_id": job_id,
            "status": "failed",
            "filename": file.filename,
            "error": str(e),
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat()
        }
        
        jobs_db[job_id] = error_result
        
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Xóa file sau khi xử lý (optional)
        # file_path.unlink(missing_ok=True)
        pass


@app.get("/api/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    """
    Kiểm tra trạng thái job.
    
    Status:
        - uploaded: Đã upload, chưa phân tích
        - processing: Đang phân tích
        - completed: Hoàn thành
        - failed: Lỗi
    """
    
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        message=job.get("error") if job["status"] == "failed" else None
    )


@app.get("/api/results/{job_id}", response_model=AnalysisResult)
def get_results(job_id: str):
    """
    Lấy kết quả phân tích.
    
    Returns:
        Điểm số và chi tiết phân tích
    """
    
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    if job["status"] == "processing":
        raise HTTPException(
            status_code=202,
            detail="Analysis still in progress. Please wait."
        )
    
    if job["status"] == "uploaded":
        raise HTTPException(
            status_code=400,
            detail="Analysis not started. Call /api/analyze/{job_id} first."
        )
    
    return AnalysisResult(**job)


@app.get("/api/jobs")
def list_jobs():
    """
    Liệt kê tất cả jobs.
    """
    
    return {
        "total": len(jobs_db),
        "jobs": [
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "filename": job["filename"],
                "created_at": job["created_at"]
            }
            for job in jobs_db.values()
        ]
    }


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """
    Xóa job và file video.
    """
    
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    
    # Xóa file
    file_path = Path(job["file_path"])
    if file_path.exists():
        file_path.unlink()
    
    # Xóa khỏi DB
    del jobs_db[job_id]
    
    logger.info(f"[{job_id}] Job deleted")
    
    return {"message": "Job deleted successfully"}


# ===== BACKGROUND TASK =====

def process_video(job_id: str):
    """
    Background task: Xử lý video.
    """
    
    try:
        job = jobs_db[job_id]
        video_path = job["file_path"]
        
        logger.info(f"[{job_id}] Processing video: {video_path}")
        
        # Chạy core engine
        evaluator = IntegratedInterviewEvaluator()
        score, details = evaluator.evaluate_video_interview(
            video_path=video_path,
            candidate_id=job_id,
            save_report=False
        )
        
        # Update job với kết quả
        jobs_db[job_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "scores": {
                "emotion": round(score.emotion_score, 2),
                "focus": round(score.focus_score, 2),
                "clarity": round(score.clarity_score, 2),
                "content": round(score.content_score, 2),
                "total": round(score.total_score, 2)
            },
            "rating": score.overall_rating,
            "details": {
                "emotion": details.get("emotion", {}),
                "focus": details.get("focus", {}),
                "clarity": details.get("clarity", {}),
                "content": details.get("content", {})
            }
        })
        
        logger.info(f"[{job_id}] Processing completed: {score.total_score:.1f}/10")
        
    except Exception as e:
        logger.error(f"[{job_id}] Processing failed: {e}")
        
        jobs_db[job_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })


# ===== MAIN =====

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Starting Interview Analysis API...")
    print("=" * 60)
    print()
    print("📍 API URL: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("📊 ReDoc: http://localhost:8000/redoc")
    print()
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

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
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core engine
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
                "focus": {
                    "score": round(score.focus_score, 2),
                    "focused_time": details.get("focus", {}).get("focused_time", 0),
                    "distracted_time": details.get("focus", {}).get("distracted_time", 0),
                    "distracted_count": details.get("focus", {}).get("distracted_count", 0),
                    "focused_rate": round(details.get("focus", {}).get("focused_rate", 0) * 100, 1),
                    "distracted_rate": round(details.get("focus", {}).get("distracted_rate", 0) * 100, 1),
                    "average_attention": round(details.get("focus", {}).get("average_attention", 0), 2)
                },
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


@app.post("/api/transcribe-video")
async def transcribe_video(file: UploadFile = File(...)):
    """
    Chuyển đổi audio trong video sang text - Dùng VideoTranscriptionCoordinator (giống launcher).
    """
    
    job_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"[{job_id}] Processing video: {file.filename}")
        
        # Use VideoTranscriptionCoordinator - SAME AS LAUNCHER
        try:
            from src.video_analysis.video_transcription_coordinator import VideoTranscriptionCoordinator
            from src.speech_analysis.whisper_stt_engine import WhisperSTTEngine
            from src.speech_analysis.config import WhisperSTTConfig
            
            logger.info(f"[{job_id}] Initializing Whisper engine...")
            # Dùng chính xác config như launcher
            from src.speech_analysis.config import STTConfig
            
            stt_config = STTConfig(
                language="vi",
                sample_rate=16000
            )
            
            whisper_engine = WhisperSTTEngine(
                config=stt_config,
                model_size="large-v3",
                device="cpu",
                compute_type="int8"
            )
            
            logger.info(f"[{job_id}] Initializing coordinator...")
            from src.speech_analysis.config import VideoTranscriptionConfig
            
            # Tắt hallucination filter để giữ nguyên kết quả như launcher
            video_config = VideoTranscriptionConfig()
            video_config.enable_hallucination_filter = False  # TẮT filter
            
            coordinator = VideoTranscriptionCoordinator(
                whisper_engine=whisper_engine,
                config=video_config
            )
            
            logger.info(f"[{job_id}] Transcribing video...")
            result = coordinator.transcribe_video(str(file_path))
            
            # Format transcript with timestamps (like launcher)
            transcript_with_timestamps = ""
            for i, seg in enumerate(result.segments, 1):
                start_time = f"{int(seg.start//60):02d}:{int(seg.start%60):02d}.{int((seg.start%1)*1000):03d}"
                end_time = f"{int(seg.end//60):02d}:{int(seg.end%60):02d}.{int((seg.end%1)*1000):03d}"
                transcript_with_timestamps += f"{i}. [{start_time} --> {end_time}]\n{seg.text}\n\n"
            
            # Also keep plain text
            transcript_plain = result.full_text
            
            # Map language codes
            lang_map = {
                "vi": "Tiếng Việt",
                "en": "English",
                "zh": "中文",
                "ja": "日本語",
                "ko": "한국어"
            }
            lang_display = lang_map.get(result.language, result.language)
            
            logger.info(f"[{job_id}] Transcription successful: {len(transcript_plain)} chars, language: {result.language}")
            
            return {
                "job_id": job_id,
                "status": "completed",
                "filename": file.filename,
                "transcript": transcript_plain,
                "transcript_with_timestamps": transcript_with_timestamps,
                "language": result.language,
                "language_display": lang_display,
                "word_count": len(transcript_plain.split()) if transcript_plain else 0,
                "duration": round(result.duration, 1),
                "segments": len(result.segments),
                "created_at": datetime.now().isoformat()
            }
            
        except ImportError as e:
            logger.error(f"[{job_id}] Import error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Thiếu dependencies: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500, 
            detail=f"Lỗi khi xử lý video: {str(e)}"
        )
    
    finally:
        # Cleanup
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass


@app.post("/api/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Chuyển đổi audio sang text.
    
    Returns:
        transcript: Text từ audio
    """
    
    # Validate file type
    if not file.content_type.startswith('audio/'):
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
        
        logger.info(f"[{job_id}] Transcribing audio: {file.filename}")
        
        # Import audio transcriber
        try:
            from src.audio_recording.transcriber import AudioTranscriber
            
            # Transcribe audio
            transcriber = AudioTranscriber()
            transcript_text = transcriber.transcribe(str(file_path))
            
            if not transcript_text:
                transcript_text = "Không thể trích xuất transcript từ audio."
            
            response = {
                "job_id": job_id,
                "status": "completed",
                "filename": file.filename,
                "transcript": transcript_text,
                "word_count": len(transcript_text.split()) if transcript_text else 0,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"[{job_id}] Transcription completed: {len(transcript_text)} chars")
            
            return response
            
        except ImportError as e:
            logger.error(f"[{job_id}] Import error: {e}")
            raise HTTPException(
                status_code=500,
                detail="AudioTranscriber không khả dụng. Vui lòng kiểm tra dependencies."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{job_id}] Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xử lý audio: {str(e)}"
        )
    
    finally:
        # Cleanup file
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass


@app.post("/api/analyze-focus")
async def analyze_focus(file: UploadFile = File(...)):
    """
    Phân tích focus (tập trung) từ video - trả về chi tiết như launcher.
    
    Returns:
        - focus_score: Điểm tập trung (0-10)
        - focused_time: Thời gian tập trung (giây)
        - distracted_time: Thời gian mất tập trung (giây)
        - distracted_count: Số lần mất tập trung
        - focused_rate: Tỷ lệ tập trung (%)
        - distracted_rate: Tỷ lệ mất tập trung (%)
    """
    
    job_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"[{job_id}] Analyzing focus: {file.filename}")
        
        # Import attention detector
        try:
            from src.video_analysis.attention_detector import AttentionDetector
            import cv2
            
            # Initialize detector
            detector = AttentionDetector()
            
            # Open video
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise Exception("Không thể mở video")
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            frame_count = 0
            focused_frames = 0
            distracted_frames = 0
            
            # Track distraction events
            distraction_events = []
            current_distraction_start = None
            
            logger.info(f"[{job_id}] Processing {total_frames} frames at {fps} fps...")
            
            # Process video (sample every 5 frames for speed)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Sample every 5 frames
                if frame_count % 5 != 0:
                    continue
                
                # For now, simulate face detection (in production, use MTCNN or MediaPipe)
                # This is a placeholder - you should integrate with your face detector
                face_detected = True  # Placeholder
                landmarks = None  # Placeholder
                
                # Calculate attention score
                score, details = detector.calculate_attention_score(
                    landmarks=landmarks,
                    frame_shape=frame.shape[:2],
                    face_detected=face_detected
                )
                
                # Track focused/distracted frames
                if score >= 6.0:
                    focused_frames += 1
                    # End current distraction event
                    if current_distraction_start is not None:
                        distraction_events.append({
                            'start_frame': current_distraction_start,
                            'end_frame': frame_count,
                            'duration': (frame_count - current_distraction_start) / fps
                        })
                        current_distraction_start = None
                else:
                    distracted_frames += 1
                    # Start new distraction event
                    if current_distraction_start is None:
                        current_distraction_start = frame_count
            
            cap.release()
            
            # Close last distraction event if still ongoing
            if current_distraction_start is not None:
                distraction_events.append({
                    'start_frame': current_distraction_start,
                    'end_frame': frame_count,
                    'duration': (frame_count - current_distraction_start) / fps
                })
            
            # Get statistics
            stats = detector.get_statistics()
            
            # Calculate times
            total_analyzed_frames = focused_frames + distracted_frames
            if total_analyzed_frames > 0:
                focused_time = (focused_frames / total_analyzed_frames) * duration
                distracted_time = (distracted_frames / total_analyzed_frames) * duration
                focused_rate = focused_frames / total_analyzed_frames
                distracted_rate = distracted_frames / total_analyzed_frames
            else:
                focused_time = 0
                distracted_time = 0
                focused_rate = 0
                distracted_rate = 0
            
            # Calculate total distraction time from events
            total_distraction_time = sum(event['duration'] for event in distraction_events)
            
            result = {
                "job_id": job_id,
                "status": "completed",
                "filename": file.filename,
                "focus_score": round(stats['average_attention'], 2),
                "focused_time": round(focused_time, 1),
                "distracted_time": round(distracted_time, 1),
                "total_distraction_time": round(total_distraction_time, 1),
                "distracted_count": len(distraction_events),
                "focused_rate": round(focused_rate * 100, 1),
                "distracted_rate": round(distracted_rate * 100, 1),
                "duration": round(duration, 1),
                "total_frames": total_frames,
                "analyzed_frames": total_analyzed_frames,
                "distraction_events": distraction_events[:10],  # Return first 10 events
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"[{job_id}] Focus analysis completed: {result['focus_score']:.1f}/10")
            
            return result
            
        except ImportError as e:
            logger.error(f"[{job_id}] Import error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Thiếu dependencies: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích focus: {str(e)}"
        )
    
    finally:
        # Cleanup
        if file_path.exists():
            try:
                file_path.unlink()
            except:
                pass
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

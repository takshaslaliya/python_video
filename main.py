import os
import uuid
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from video_generator import VideoGenerator

# Task Queue and Tracker Setup
task_queue = asyncio.Queue()
queued_task_ids = []

# Global generator instance
BASE_DIR = Path(__file__).resolve().parent.parent
RENDERS_DIR = BASE_DIR / "renders"
UPLOADS_DIR = BASE_DIR / "uploads"
RENDERS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
generator = VideoGenerator(output_dir=RENDERS_DIR, uploads_dir=UPLOADS_DIR)

async def queue_worker():
    print("AIToolHub Task Queue Worker started.")
    while True:
        try:
            task_args = await task_queue.get()
            task_id = task_args.get("task_id")
            print(f"Worker picked up video generation task: {task_id}")
            if task_id in queued_task_ids:
                queued_task_ids.remove(task_id)
            
            try:
                # Sequential execution of the async task
                await generator.generate_video_task(
                    task_id=task_id,
                    script=task_args["script"],
                    aspect_ratio=task_args["aspect_ratio"],
                    voice=task_args["voice"],
                    use_subtitles=task_args["use_subtitles"],
                    use_music=task_args["use_music"],
                    music_genre=task_args["music_genre"],
                    visual_source=task_args["visual_source"],
                    voice_speed=task_args["voice_speed"],
                    resolution=task_args["resolution"],
                    fps=task_args["fps"],
                    use_narration=task_args["use_narration"]
                )
                print(f"Worker successfully finished video generation task: {task_id}")
            except Exception as e:
                print(f"Error executing video task {task_id} in worker: {e}")
                try:
                    generator.update_progress(task_id, -1, f"Error: {str(e)}")
                except Exception:
                    pass
            finally:
                task_queue.task_done()
        except asyncio.CancelledError:
            print("Worker cancelled.")
            break
        except Exception as e:
            print(f"Worker queue loop error: {e}")
            await asyncio.sleep(1)

async def cleanup_old_files():
    """Background task that runs continuously, deleting files/folders older than 48 hours."""
    import time
    import shutil
    print("Started file cleanup background task.")
    while True:
        try:
            now = time.time()
            cutoff = now - (48 * 3600)  # 48 hours ago
            
            # Clean renders
            if RENDERS_DIR.exists():
                for item in RENDERS_DIR.iterdir():
                    if item.is_dir():
                        if item.stat().st_mtime < cutoff:
                            print(f"Cleaning up old render directory: {item.name}")
                            shutil.rmtree(item, ignore_errors=True)
                    elif item.is_file():
                        if item.stat().st_mtime < cutoff:
                            print(f"Cleaning up old render file: {item.name}")
                            item.unlink(missing_ok=True)
                            
            # Clean uploads
            if UPLOADS_DIR.exists():
                for item in UPLOADS_DIR.iterdir():
                    if item.name == "music":
                        # Clean custom music files inside music directory
                        for music_file in item.glob("*.mp3"):
                            if music_file.stat().st_mtime < cutoff:
                                print(f"Cleaning up old upload music: {music_file.name}")
                                music_file.unlink(missing_ok=True)
                        continue
                        
                    if item.is_dir():
                        if item.stat().st_mtime < cutoff:
                            print(f"Cleaning up old upload directory: {item.name}")
                            shutil.rmtree(item, ignore_errors=True)
                    elif item.is_file():
                        if item.stat().st_mtime < cutoff:
                            print(f"Cleaning up old upload file: {item.name}")
                            item.unlink(missing_ok=True)
                            
        except Exception as e:
            print(f"Error in cleanup_old_files task: {e}")
            
        # Run cleanup check every hour (3600 seconds)
        await asyncio.sleep(3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create and start worker and cleanup tasks
    worker_task = asyncio.create_task(queue_worker())
    cleanup_task = asyncio.create_task(cleanup_old_files())
    yield
    # Shutdown: cancel tasks
    worker_task.cancel()
    cleanup_task.cancel()
    try:
        await asyncio.gather(worker_task, cleanup_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass

app = FastAPI(title="AI Video Generator API Engine", lifespan=lifespan)

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount renders directory as static files to allow direct browser access
app.mount("/renders", StaticFiles(directory=str(RENDERS_DIR)), name="renders")

class GenerationRequest(BaseModel):
    taskId: str = Field(default_factory=lambda: str(uuid.uuid4()))
    script: str
    aspectRatio: str = "16:9"  # "16:9" or "9:16"
    voice: str = "female1"      # "male1", "male2", "female1", "female2"
    useSubtitles: bool = True
    useMusic: bool = True
    useNarration: bool = True
    musicGenre: str = "ambient"
    visualSource: str = "procedural"  # "procedural", "ai", "stock"
    voiceSpeed: float = 1.0
    resolution: str = "720p"  # "1080p", "720p", "480p"
    fps: int = 30

@app.post("/generate")
def generate_video(request: GenerationRequest):
    """Adds a new video generation job to the sequential task queue."""
    if not request.script.strip():
        raise HTTPException(status_code=400, detail="Script cannot be empty.")
    
    # Initialize the progress file to "Queued"
    generator.update_progress(
        task_id=request.taskId,
        progress=0,
        status="Queued in AI Engine..."
    )
    
    # Pack arguments for the queue
    task_args = {
        "task_id": request.taskId,
        "script": request.script,
        "aspect_ratio": request.aspectRatio,
        "voice": request.voice,
        "use_subtitles": request.useSubtitles,
        "use_music": request.useMusic,
        "music_genre": request.musicGenre,
        "visual_source": request.visualSource,
        "voice_speed": request.voiceSpeed,
        "resolution": request.resolution,
        "fps": request.fps,
        "use_narration": request.useNarration
    }
    
    queued_task_ids.append(request.taskId)
    task_queue.put_nowait(task_args)
    
    return {
        "success": True,
        "taskId": request.taskId,
        "status": "Queued"
    }

@app.get("/status/{task_id}")
def get_status(task_id: str):
    """Retrieves current progress, status, and queue position of a video job."""
    progress_file = RENDERS_DIR / task_id / "progress.json"
    if not progress_file.exists():
        if task_id in queued_task_ids:
            try:
                pos = queued_task_ids.index(task_id) + 1
                return {
                    "taskId": task_id,
                    "progress": 0,
                    "status": f"Queued (Position {pos} in queue)"
                }
            except ValueError:
                pass
        return {
            "taskId": task_id,
            "progress": 0,
            "status": "Not Found / Initializing"
        }
        
    try:
        with open(progress_file, "r") as f:
            import json
            data = json.load(f)
        
        # Inject position in queue if currently waiting in queue
        if task_id in queued_task_ids:
            try:
                pos = queued_task_ids.index(task_id) + 1
                data["status"] = f"Queued (Position {pos} in queue)"
            except ValueError:
                pass
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read task status: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "gpu_available": generator.sd_pipeline is not None}

class ImageGenerationRequest(BaseModel):
    prompt: str
    aspectRatio: str = "1:1"  # "1:1", "16:9", "9:16"
    source: str = "procedural"  # "ai", "stock", "procedural"

class ScriptGenerationRequest(BaseModel):
    topic: str
    tone: str = "engaging"  # "professional", "engaging", "motivational", "educational", "casual", "dramatic"
    duration: int = 60      # 15, 30, 60, 90
    platform: str = "YouTube Shorts"

def generate_creative_script(topic: str, tone: str, duration: int, platform: str):
    topic = topic.strip()
    if not topic:
        topic = "Success & Focus"
    
    topic_cap = topic[0].upper() + topic[1:] if len(topic) > 1 else topic.upper()
    
    hooks = {
        "professional": [
            f"Here is a critical look at {topic_cap} that most professionals miss.",
            f"If you want to understand the future of {topic_cap}, you need to look at this data.",
            f"What is the actual cost of ignoring {topic_cap} in today's economy?"
        ],
        "engaging": [
            f"Stop scrolling! This single fact about {topic_cap} will change your perspective.",
            f"I bet you didn't know this mind-blowing truth about {topic_cap}!",
            f"The secret of {topic_cap} is finally out, and it's simpler than you think."
        ],
        "motivational": [
            f"Everything changes the moment you realize the power of {topic_cap}.",
            f"If you're waiting for a sign to start mastering {topic_cap}, this is it.",
            f"The road to greatness starts with one choice: embracing {topic_cap}."
        ],
        "educational": [
            f"Let's break down the science behind {topic_cap} in under 60 seconds.",
            f"Here is a quick lesson on {topic_cap} and why it works the way it does.",
            f"Ever wondered how {topic_cap} actually functions? Let's explore."
        ],
        "casual": [
            f"Okay, can we talk about {topic_cap} for a second? It's getting wild.",
            f"Just a quick thought on {topic_cap} that's been living in my head rent-free.",
            f"Let's be real about {topic_cap}—nobody is telling you the actual truth."
        ],
        "dramatic": [
            f"They tried to hide the reality of {topic_cap}, but the truth is coming to light.",
            f"Deep in the shadows of our daily lives lies the mystery of {topic_cap}.",
            f"A single decision about {topic_cap} would change their lives forever."
        ]
    }
    
    bodies = {
        "professional": [
            f"First, industry standards show that {topic_cap} yields a 10x return when implemented early. "
            f"Second, scaling your framework requires aligning this with your team's core capabilities. "
            f"Finally, monitoring progress ensures long-term operational resilience and efficiency.",
            
            f"Recent analysis indicates a major shift in how leaders approach {topic_cap}. "
            f"By leveraging automation and data, organizations can double efficiency. "
            f"To succeed, prioritize high-impact areas and iterate based on user feedback."
        ],
        "engaging": [
            f"Most people think {topic_cap} is complicated. But here's the trick: "
            f"it all comes down to focusing on consistency rather than intensity. "
            f"If you do just one action daily, the compound interest is absolutely massive!",
            
            f"Here's why {topic_cap} is taking over: it cuts out the noise and focuses purely on results. "
            f"No fluff, no wasted hours. Just pure, actionable steps that get you to the finish line faster."
        ],
        "motivational": [
            f"It's not about how hard you fall; it's about how fast you get back up to master {topic_cap}. "
            f"Every single failure is just a lesson in disguise, preparing you for the breakthrough. "
            f"Remember, the only limits that exist are the ones you place on yourself. Push through!",
            
            f"Your potential is limitless, and {topic_cap} is your key to unlocking it. "
            f"Do not let fear dictate your progress. Rise early, work smart, and let your results do the talking. "
            f"You are closer than you think."
        ],
        "educational": [
            f"To understand {topic_cap}, we have to look at the primary underlying principle. "
            f"Specifically, how input signals are filtered to maximize output clarity. "
            f"This mechanism reduces friction by up to eighty percent, creating a highly efficient loop.",
            
            f"There are three key pillars to {topic_cap}. First, initial state definition. "
            f"Second, feedback cycle optimization. And third, scale replication. "
            f"Mastering these three pillars ensures consistent quality every single time."
        ],
        "casual": [
            f"Look, we all want to get better at {topic_cap}, but let's not make it our whole personality. "
            f"Just start with small, lazy habits. Seriously, ten minutes a day is all you need. "
            f"You don't need a fancy course or a expensive coach to see massive progress.",
            
            f"Honestly, the hardest part of {topic_cap} is just getting off the couch. "
            f"Once you take that first tiny step, it becomes second nature. "
            f"So do yourself a favor and start today—your future self will thank you!"
        ],
        "dramatic": [
            f"For centuries, the secrets of {topic_cap} were closely guarded. "
            f"But as the old systems collapse, the hidden pathways are being exposed to everyone. "
            f"Now, the choice is yours: look away, or step into the new reality.",
            
            f"A quiet storm is brewing, and at its center is {topic_cap}. "
            f"The stakes have never been higher, and every decision carries heavy weight. "
            f"Will you watch from the sidelines, or shape the outcome yourself?"
        ]
    }
    
    ctas = {
        "professional": [
            f"Connect with us at aitoolhub.in to integrate {topic_cap} into your workflow today.",
            f"Subscribe for more data-driven insights and professional AI strategy guidebooks.",
            f"Visit aitoolhub.in to deploy next-gen AI tools for your enterprise operations."
        ],
        "engaging": [
            f"Like this video and comment below: what is your biggest challenge with {topic_cap}?",
            f"Hit that follow button for daily AI secrets you won't find anywhere else!",
            f"Share this with a friend who needs to hear this about {topic_cap} right now!"
        ],
        "motivational": [
            f"Believe in yourself, take action now, and visit aitoolhub.in to launch your journey.",
            f"Don't wait for tomorrow. Subscribe today and start building your legacy.",
            f"Double tap if you agree, and share this to inspire someone today!"
        ],
        "educational": [
            f"Visit aitoolhub.in for free templates, source files, and visual guidebooks.",
            f"Subscribe to our channel for simplified break downs of complex AI concepts.",
            f"Save this reel for later so you can refer back when building your projects!"
        ],
        "casual": [
            f"Let me know in the comments if this made sense, or if I should explain it simpler.",
            f"Subscribe for more laid-back AI tools and tips at aitoolhub.in!",
            f"Check out the link in bio to play around with these tools yourself for free!"
        ],
        "dramatic": [
            f"The countdown has begun. Subscribe now, and uncover the full mystery at aitoolhub.in.",
            f"Will you be ready when the change comes? Follow to stay ahead of the curve.",
            f"Share this message before it's too late. The truth must be told."
        ]
    }
    
    import random
    random.seed(hash(topic + tone + str(duration)))
    
    tone_key = tone.lower() if tone.lower() in hooks else "engaging"
    
    hook_text = random.choice(hooks[tone_key])
    body_text = random.choice(bodies[tone_key])
    cta_text = random.choice(ctas[tone_key])
    
    if duration <= 15:
        full_script = f"{hook_text} {cta_text}"
    elif duration <= 30:
        body_short = body_text.split(". ")[0] + "."
        full_script = f"{hook_text} {body_short} {cta_text}"
    else:
        full_script = f"{hook_text} {body_text} {cta_text}"
        
    return {
        "hook": hook_text,
        "body": body_text,
        "cta": cta_text,
        "fullScript": full_script,
        "platform": platform,
        "tone": tone,
        "duration": duration,
        "topic": topic
    }

@app.post("/generate-image")
def generate_image_api(request: ImageGenerationRequest):
    """Generates an image from prompt and returns the static url."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    filename = f"{uuid.uuid4()}.jpg"
    image_dir = RENDERS_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    filepath = image_dir / filename
    
    success = generator.generate_image(
        prompt=request.prompt,
        filepath=filepath,
        aspect_ratio=request.aspectRatio,
        visual_source=request.source
    )
    
    if not success or not filepath.exists():
        raise HTTPException(status_code=500, detail="Failed to generate image.")
        
    return {
        "success": True,
        "imageUrl": f"/renders/images/{filename}"
    }

@app.post("/generate-script")
def generate_script_api(request: ScriptGenerationRequest):
    """Generates an engaging, customized script based on topic, tone, and platform."""
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    
    res = generate_creative_script(
        topic=request.topic,
        tone=request.tone,
        duration=request.duration,
        platform=request.platform
    )
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

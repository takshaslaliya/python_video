import os
import re
import json
import time
import random
import asyncio
import subprocess
from pathlib import Path
import PIL.Image
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import requests

# Monkey patch Pillow to support older MoviePy versions that use ANTIALIAS
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

# Set MoviePy configuration if needed
# os.environ["IMAGEMAGICK_BINARY"] = "/usr/bin/convert"

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

# Optional AI imports
try:
    import torch
    from diffusers import StableDiffusionPipeline
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Fallback fonts for PIL rendering
SYSTEM_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    "Arial",  # Windows fallback
    "Helvetica"
]

def get_pil_font(size):
    """Finds a valid font on the system or returns default."""
    for font_path in SYSTEM_FONTS:
        try:
            if os.path.exists(font_path) or font_path in ["Arial", "Helvetica"]:
                return ImageFont.truetype(font_path, size)
        except Exception:
            continue
    return ImageFont.load_default()

class VideoGenerator:
    def __init__(self, output_dir="renders", uploads_dir="uploads"):
        self.output_dir = Path(output_dir)
        self.uploads_dir = Path(uploads_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Load local Stable Diffusion if available and GPU exists
        self.sd_pipeline = None
        if SD_AVAILABLE and torch.cuda.is_available():
            try:
                print("Loading Stable Diffusion Pipeline (Local)...")
                # Using a lightweight fast model
                self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
                    "stabilityai/sd-turbo",
                    torch_dtype=torch.float16,
                    safety_checker=None
                ).to("cuda")
                print("Stable Diffusion loaded successfully.")
            except Exception as e:
                print(f"Error loading Stable Diffusion: {e}. Falling back to API/Procedural.")

    def update_progress(self, task_id, progress, status, video_url=None):
        """Helper to write progress to a json file that Node server reads."""
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        progress_file = task_dir / "progress.json"
        
        data = {
            "taskId": task_id,
            "progress": progress,
            "status": status,
            "updatedAt": time.time()
        }
        if video_url:
            data["videoUrl"] = video_url
            
        with open(progress_file, "w") as f:
            json.dump(data, f)
        print(f"[{task_id}] Progress: {progress}% - {status}")

    def split_script_into_scenes(self, script_text):
        """Splits a script into distinct scenes by punctuation or newlines."""
        # Clean text
        text = script_text.strip()
        # Split by paragraphs or sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        
        scenes = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 8:  # Skip trivial segments
                # If a sentence is extremely long, break it up
                if len(s_clean) > 150:
                    words = s_clean.split(" ")
                    chunk_size = 12
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i+chunk_size])
                        if chunk:
                            scenes.append(chunk)
                else:
                    scenes.append(s_clean)
        
        # If no scenes generated, use full text
        if not scenes:
            scenes = [text]
            
        return scenes

    async def generate_narration_audio(self, text, filepath, voice="en-US-AriaNeural", speed=1.0):
        """Generates voice narration using Edge TTS (high quality) or gTTS (fallback)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        rate_str = f"{'+' if speed >= 1.0 else ''}{int((speed - 1.0) * 100)}%"
        
        if edge_tts:
            try:
                # Map select simple voices to Edge TTS voices
                voice_map = {
                    "male1": "en-US-GuyNeural",
                    "male2": "en-US-ChristopherNeural",
                    "female1": "en-US-AriaNeural",
                    "female2": "en-US-JennyNeural",
                    "uk_male": "en-GB-RyanNeural",
                    "uk_female": "en-GB-SoniaNeural"
                }
                actual_voice = voice_map.get(voice, voice)
                
                # Check if voice is one of the keys or already full name
                if actual_voice not in voice_map.values() and actual_voice in voice_map:
                    actual_voice = voice_map[actual_voice]
                elif actual_voice not in voice_map.values():
                    actual_voice = "en-US-AriaNeural" # Default fallback
                
                communicate = edge_tts.Communicate(text, actual_voice, rate=rate_str)
                await communicate.save(str(filepath))
                return True
            except Exception as e:
                print(f"Edge TTS failed, trying gTTS fallback: {e}")
                
        if gTTS:
            try:
                # gTTS is synchronous, run in threadpool
                lang = "en"
                if "gb" in voice.lower() or "uk" in voice.lower():
                    lang = "en" # gTTS supports tld for regional, we'll keep it simple
                
                tts = gTTS(text=text, lang=lang)
                await asyncio.to_thread(tts.save, str(filepath))
                return True
            except Exception as e:
                print(f"gTTS failed: {e}")
                
        # Hard fallback using local system TTS if available (espeak / festival)
        try:
            subprocess.run(["espeak", "-w", str(filepath), text], check=True)
            return True
        except Exception as e:
            print(f"System espeak failed: {e}")
            
        raise Exception("Failed to generate narration audio. No TTS engine worked.")

    def search_stock_image(self, query):
        """Searches LoremFlickr with robust keyword filtering and fallback to avoid default cat images."""
        import random
        try:
            # Remove common stopwords/verbs
            stopwords = {
                "the", "and", "a", "an", "for", "with", "this", "that", "these", "those", "been", "were", "was", 
                "are", "is", "have", "has", "had", "will", "would", "should", "could", "they", "them", "their", 
                "you", "your", "mine", "ours", "about", "along", "around", "at", "before", "after", "by", "for", 
                "from", "in", "into", "of", "on", "to", "up", "with", "scenic", "beautiful", "amazing", "wonderful", 
                "cool", "great", "awesome", "down", "over", "under", "through", "very", "some", "many", "most", "more", 
                "then", "there", "here", "two", "one", "three", "four", "five", "first", "second", "third", "some",
                "make", "made", "take", "took", "give", "given", "want", "like", "love", "good", "best", "some",
                "real", "realistic", "stock", "photo", "image", "video", "footage", "clip"
            }
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            filtered = [w for w in words if w not in stopwords]
            
            if not filtered:
                keyword_query = "abstract"
            else:
                # Use hyphen join for tag search which searches for the tag string
                keyword_query = "-".join(filtered[:3])
            
            lock_val = random.randint(1, 1000000)
            unsplash_url = f"https://loremflickr.com/1280/720/{keyword_query}?lock={lock_val}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            
            print(f"Searching stock image for query: '{query}' -> tag: '{keyword_query}' with lock={lock_val}")
            response = requests.get(unsplash_url, timeout=10, headers=headers)
            if response.status_code == 200:
                # Check if it returned the default cat placeholder
                if "defaultImage" in response.url or "defaultimage" in response.url:
                    print(f"LoremFlickr returned default cat placeholder for '{keyword_query}'. Trying fallback.")
                    # Fallback to the first keyword alone
                    if filtered:
                        fallback_tag = filtered[0]
                        fallback_lock = random.randint(1, 1000000)
                        fallback_url = f"https://loremflickr.com/1280/720/{fallback_tag}?lock={fallback_lock}"
                        print(f"Trying single keyword fallback: '{fallback_tag}' with lock={fallback_lock}")
                        response = requests.get(fallback_url, timeout=10, headers=headers)
                        if response.status_code == 200 and "defaultImage" not in response.url and "defaultimage" not in response.url:
                            return response.content
                    
                    # If that still fails, fall back to a generic safe tag
                    generic_lock = random.randint(1, 1000000)
                    fallback_url = f"https://loremflickr.com/1280/720/abstract?lock={generic_lock}"
                    print(f"Trying generic fallback: 'abstract' with lock={generic_lock}")
                    response = requests.get(fallback_url, timeout=10, headers=headers)
                    if response.status_code == 200:
                        return response.content
                else:
                    return response.content
        except Exception as e:
            print(f"Stock image search failed: {e}")
        return None

    def download_stock_video(self, query, filepath):
        """Searches Mixkit for free stock video matching query and downloads it."""
        try:
            # Normalize text and extract candidates
            clean_text = re.sub(r'[^a-zA-Z\s]', '', query.lower())
            words = [w for w in clean_text.split() if len(w) >= 3]
            stopwords = {
                "the", "and", "a", "an", "for", "with", "this", "that", "these", "those",
                "scenic", "beautiful", "amazing", "wonderful", "cool", "great", "awesome",
                "down", "from", "into", "over", "under", "through", "about", "along", "around",
                "very", "some", "many", "most", "more", "then", "their", "them", "there", "here"
            }
            filtered = [w for w in words if w not in stopwords]
            
            candidates = []
            # Try adjacent pairs
            for i in range(len(filtered) - 1):
                candidates.append(f"{filtered[i]}-{filtered[i+1]}")
            # Try single words
            for w in filtered:
                if w not in candidates:
                    candidates.append(w)
                    
            if not candidates:
                candidates = ["abstract"]
                
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9"
            }
            
            video_url = None
            # Try top 5 candidates
            for candidate in candidates[:5]:
                url = f"https://mixkit.co/free-stock-video/{requests.utils.quote(candidate)}/"
                print(f"Searching Mixkit for candidate: '{candidate}' at {url}")
                try:
                    res = requests.get(url, headers=headers, timeout=5)
                    if res.status_code == 200:
                        mp4_links = re.findall(r'https://assets\.mixkit\.co/videos/[^\s"\'\\<>]+?\.mp4', res.text)
                        if mp4_links:
                            # Filter resolution preference
                            for res_pref in ["720", "360", "1080"]:
                                matched = [link for link in mp4_links if res_pref in link]
                                if matched:
                                    video_url = matched[0]
                                    break
                            if not video_url:
                                video_url = mp4_links[0]
                            print(f"Found match for candidate '{candidate}': {video_url}")
                            break
                except Exception as e:
                    print(f"Error checking candidate '{candidate}': {e}")
            
            if not video_url:
                # Abstract fallback if all candidates failed
                print("All candidates failed. Trying abstract fallback.")
                url = "https://mixkit.co/free-stock-video/abstract/"
                try:
                    res = requests.get(url, headers=headers, timeout=5)
                    if res.status_code == 200:
                        mp4_links = re.findall(r'https://assets\.mixkit\.co/videos/[^\s"\'\\<>]+?\.mp4', res.text)
                        if mp4_links:
                            video_url = mp4_links[0]
                except Exception as e:
                    print(f"Abstract fallback failed: {e}")
            
            if not video_url:
                print("No stock video could be retrieved.")
                return False
                
            print(f"Downloading stock video from: {video_url}")
            video_res = requests.get(video_url, headers=headers, timeout=20)
            if video_res.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(video_res.content)
                print(f"Successfully downloaded stock video to {filepath}")
                return True
            else:
                print(f"Failed to download video file, status: {video_res.status_code}")
                return False
        except Exception as e:
            print(f"Error in download_stock_video: {e}")
            return False

    def generate_ai_video_clip(self, prompt, filepath):
        """Generates a video clip using the Bytez Text-to-Video API with fallback support."""
        print(f"Generating AI video clip via Bytez for prompt: {prompt}")
        try:
            from bytez import Bytez
            sdk = Bytez("a60b4df1afb2ad1715fdb9d8175544ef")
            model = sdk.model("ali-vilab/text-to-video-ms-1.7b")
            
            result = model.run(prompt)
            if result.error:
                print(f"Bytez API Error: {result.error}")
                return False
                
            output = result.output
            if not output:
                print("Bytez API returned empty output")
                return False
                
            video_url = None
            if isinstance(output, str):
                video_url = output
            elif isinstance(output, list) and len(output) > 0:
                video_url = output[0]
            elif isinstance(output, dict):
                video_url = output.get("url") or output.get("output") or output.get("video")
                
            if not video_url:
                print(f"Could not extract video URL from Bytez output: {output}")
                return False
                
            print(f"Downloading generated video from Bytez URL: {video_url}")
            import requests
            r = requests.get(video_url, stream=True, timeout=60)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Successfully downloaded AI video clip to {filepath}")
            return True
            
        except Exception as e:
            print(f"Exception generating AI video clip via Bytez: {e}")
            return False

    def generate_procedural_image(self, text, filepath, aspect_ratio="16:9"):
        """Generates a beautiful modern gradient background with scene text overlay as a bulletproof fallback."""
        width, height = (1280, 720) if aspect_ratio == "16:9" else (720, 1280) if aspect_ratio == "9:16" else (800, 800)
        
        # Create gradient
        base = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(base)
        
        # Curated harmonious color palettes
        palettes = [
            ((15, 12, 41), (48, 43, 99), (36, 36, 62)),      # Cosmic Cherry
            ((31, 28, 44), (146, 141, 171), (40, 30, 50)),   # Vintage
            ((9, 17, 34), (20, 35, 70), (45, 20, 60)),       # Deep Neon Blue/Purple
            ((4, 21, 45), (16, 52, 98), (9, 21, 45)),        # Dark Ocean
            ((20, 20, 30), (50, 25, 60), (30, 20, 50))       # Velvet Shadow
        ]
        
        c1, c2, c3 = random.choice(palettes)
        
        # Vertical gradient
        for y in range(height):
            # Interpolate colors
            t = y / height
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        # Draw some soft blurred light blobs for modern glassmorphism aesthetic
        blob_layer = Image.new('RGBA', (width, height), (0,0,0,0))
        blob_draw = ImageDraw.Draw(blob_layer)
        for _ in range(3):
            bx = random.randint(0, width)
            by = random.randint(0, height)
            br = random.randint(100, 300)
            blob_draw.ellipse([bx-br, by-br, bx+br, by+br], fill=(
                random.randint(100, 250), 
                random.randint(50, 150), 
                random.randint(200, 255), 
                40
            ))
        blob_layer = blob_layer.filter(ImageFilter.GaussianBlur(50))
        base.paste(blob_layer, (0,0), blob_layer)
        
        # Draw text overlay container (glassmorphism card in middle)
        card_layer = Image.new('RGBA', (width, height), (0,0,0,0))
        card_draw = ImageDraw.Draw(card_layer)
        
        card_w, card_h = int(width * 0.85), int(height * 0.4)
        cx, cy = (width - card_w) // 2, (height - card_h) // 2
        card_draw.rounded_rectangle(
            [cx, cy, cx + card_w, cy + card_h],
            radius=20,
            fill=(0, 0, 0, 100),
            outline=(255, 255, 255, 30),
            width=2
        )
        base.paste(card_layer, (0,0), card_layer)
        
        # Overlay the scene text inside the card
        draw_txt = ImageDraw.Draw(base)
        font_size = int(width * 0.035)
        font = get_pil_font(font_size)
        
        # Word wrap text
        words = text.split(" ")
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            test_line = " ".join(current_line)
            # Check length
            w = font.getbbox(test_line)[2]
            if w > card_w - 60:
                current_line.pop()
                lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
            
        # Draw lines centered
        line_height = font.getbbox("A")[3] + 15
        total_text_h = len(lines) * line_height
        start_y = cy + (card_h - total_text_h) // 2
        
        for line in lines:
            line_w = font.getbbox(line)[2]
            lx = cx + (card_w - line_w) // 2
            # Drop shadow
            draw_txt.text((lx+2, start_y+2), line, font=font, fill=(0,0,0,180))
            # Text
            draw_txt.text((lx, start_y), line, font=font, fill=(255, 255, 255, 240))
            start_y += line_height
            
        base.save(filepath)

    def generate_image(self, prompt, filepath, aspect_ratio="16:9", visual_source="procedural"):
        """Generates visual for a scene using chosen source (SD, stock API, or procedural fallback)."""
        width, height = (1024, 576) if aspect_ratio == "16:9" else (576, 1024) if aspect_ratio == "9:16" else (800, 800)
        
        # If AI generation is selected
        if visual_source == "ai":
            if self.sd_pipeline:
                try:
                    print(f"Generating AI image via local Stable Diffusion for prompt: {prompt}")
                    image = self.sd_pipeline(
                        prompt, 
                        num_inference_steps=4, # Turbo model is fast with 4 steps
                        guidance_scale=0.0,
                        width=width,
                        height=height
                    ).images[0]
                    image.save(filepath)
                    return True
                except Exception as e:
                    print(f"Stable Diffusion generation failed: {e}. Trying online AI fallback.")
            
            # Online free AI generation fallback (Pollinations.ai) to ensure a generated image is returned
            try:
                import urllib.parse
                safe_prompt = urllib.parse.quote(prompt)
                pollinations_url = f"https://image.pollinations.ai/p/{safe_prompt}?width={width}&height={height}&nologo=true"
                print(f"Generating AI image via Pollinations.ai for prompt: {prompt}")
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                response = requests.get(pollinations_url, timeout=20, headers=headers)
                if response.status_code == 200:
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                    return True
                else:
                    print(f"Pollinations.ai returned status code {response.status_code}")
            except Exception as e:
                print(f"Pollinations.ai generation failed: {e}")
                
        # If Stock Image source selected or AI failed
        if visual_source in ["stock", "ai"]:
            img_data = self.search_stock_image(prompt)
            if img_data:
                try:
                    with open(filepath, "wb") as f:
                        f.write(img_data)
                    # Resize to match aspect ratio
                    img = Image.open(filepath)
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    img.save(filepath)
                    return True
                except Exception as e:
                    print(f"Failed to process stock image: {e}")
                    
        # Bulletproof fallback: procedural gradient card
        self.generate_procedural_image(prompt, filepath, aspect_ratio)
        return True

    def get_audio_duration(self, filepath):
        """Uses FFprobe via subprocess to find exact audio duration in seconds."""
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", 
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", 
                str(filepath)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Error getting audio duration for {filepath}: {e}")
            # Dynamic estimate based on word count as fallback
            # Normal speech is around 130-150 words per minute (2.2 - 2.5 words per second)
            return 3.0 # safe fallback default

    def generate_srt_subtitles(self, scenes, scene_audios, filepath):
        """Generates synchronized subtitles in SRT format based on scene durations."""
        srt_content = ""
        current_time = 0.0
        
        for i, (scene_text, audio_path) in enumerate(zip(scenes, scene_audios)):
            duration = self.get_audio_duration(audio_path)
            
            # Split scene text into smaller caption blocks (approx 4 words each)
            words = scene_text.split(" ")
            words_per_block = 4
            blocks = []
            for j in range(0, len(words), words_per_block):
                block = " ".join(words[j:j+words_per_block])
                if block:
                    blocks.append(block)
            
            if not blocks:
                blocks = [scene_text]
                
            # Distribute scene duration proportionally to each block based on character length
            total_chars = sum(len(b) for b in blocks)
            block_durations = []
            for b in blocks:
                ratio = len(b) / total_chars if total_chars > 0 else 1.0/len(blocks)
                block_durations.append(duration * ratio)
                
            block_start = current_time
            for k, (block, b_dur) in enumerate(zip(blocks, block_durations)):
                block_end = block_start + b_dur
                
                # Format timestamps: HH:MM:SS,mmm
                start_str = self.format_srt_timestamp(block_start)
                end_str = self.format_srt_timestamp(block_end)
                
                index = len(block_durations) * i + k + 1
                srt_content += f"{index}\n{start_str} --> {end_str}\n{block}\n\n"
                
                block_start = block_end
                
            current_time += duration
            
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(srt_content)
            
        return filepath

    def format_srt_timestamp(self, seconds):
        """Formats float seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def build_video_ffmpeg(self, task_id, scenes_data, bg_music_path, use_subtitles, aspect_ratio="16:9", resolution="1080p", fps=30):
        """Stitches the video clips, mixes music, and burns subtitles using MoviePy + FFmpeg."""
        from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
        
        task_dir = self.output_dir / task_id
        temp_video_path = task_dir / "temp_stitched.mp4"
        final_video_path = task_dir / "final_output.mp4"
        
        # Setup resolution dimensions
        res_map = {
            "1080p": (1920, 1080) if aspect_ratio == "16:9" else (1080, 1920),
            "720p": (1280, 720) if aspect_ratio == "16:9" else (720, 1280),
            "480p": (854, 480) if aspect_ratio == "16:9" else (480, 854)
        }
        target_res = res_map.get(resolution, res_map["720p"]) # default to 720p for local speed
        width, height = target_res
        
        # Load and stitch visual + audio clips
        clips = []
        for i, scene in enumerate(scenes_data):
            img_path = scene.get("image_path")
            video_path = scene.get("video_path")
            audio_path = scene["audio_path"]
            
            # Find audio duration
            audio_dur = self.get_audio_duration(audio_path)
            
            if video_path and os.path.exists(video_path):
                from moviepy.editor import VideoFileClip
                from moviepy.video.fx.all import loop, crop
                
                print(f"Processing scene {i} with video clip: {video_path}")
                try:
                    video_clip = VideoFileClip(str(video_path))
                    
                    # Adjust duration of the video to match audio duration
                    if video_clip.duration > audio_dur:
                        # Cut to match audio duration
                        video_scene_clip = video_clip.subclip(0, audio_dur)
                    else:
                        # Loop video clip if it's shorter than audio duration
                        video_scene_clip = loop(video_clip, duration=audio_dur)
                        
                    # Calculate scale factor to cover the target box without stretching
                    scale = max(width / video_scene_clip.w, height / video_scene_clip.h)
                    new_w = int(video_scene_clip.w * scale)
                    new_h = int(video_scene_clip.h * scale)
                    
                    # Resize
                    video_scene_clip = video_scene_clip.resize((new_w, new_h))
                    
                    # Center crop to target resolution
                    x_center = new_w / 2
                    y_center = new_h / 2
                    video_scene_clip = crop(video_scene_clip, x_center=x_center, y_center=y_center, width=width, height=height)
                    
                    audio_clip = AudioFileClip(str(audio_path))
                    video_scene_clip = video_scene_clip.set_audio(audio_clip)
                except Exception as e:
                    print(f"Failed to load video file clip: {e}. Falling back to procedural background.")
                    fallback_img_path = task_dir / f"scene_{i}_fallback.png"
                    self.generate_procedural_image(scene.get("text", "Scene"), fallback_img_path, aspect_ratio)
                    img_clip = ImageClip(str(fallback_img_path)).set_duration(audio_dur)
                    audio_clip = AudioFileClip(str(audio_path))
                    video_scene_clip = img_clip.set_audio(audio_clip)
            else:
                # Create video clip from image
                img_clip = ImageClip(str(img_path)).set_duration(audio_dur)
                audio_clip = AudioFileClip(str(audio_path))
                
                # Attach audio
                video_scene_clip = img_clip.set_audio(audio_clip)
                
                # Ken Burns Zoom-in (scale from 1.0 to 1.08 over duration)
                try:
                    video_scene_clip = video_scene_clip.resize(lambda t: 1.0 + 0.08 * (t / audio_dur))
                except Exception as e:
                    print("Could not apply Ken Burns zoom, using static clip:", e)
                
            clips.append(video_scene_clip)
            
        self.update_progress(task_id, 70, "Generating video timeline...")
        
        # Concatenate scenes
        final_clip = concatenate_videoclips(clips, method="compose")
        
        # Write temporary stitched file (without background music or subtitles burned yet)
        final_clip.write_videofile(
            str(temp_video_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(task_dir / "temp_audio.m4a"),
            remove_temp=True,
            threads=4,
            logger=None
        )
        
        # Close clips to release files
        for c in clips:
            c.close()
        final_clip.close()
        
        self.update_progress(task_id, 85, "Mixing audio and subtitles...")
        
        # Step 2: Combine with background music and burn subtitles using FFmpeg command line
        # This is extremely fast and avoids ImageMagick dependencies
        srt_path = task_dir / "subtitles.srt"
        
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(temp_video_path)]
        filter_complex = []
        audio_inputs_count = 1
        
        # Background music logic
        if bg_music_path and os.path.exists(bg_music_path):
            ffmpeg_cmd.extend(["-stream_loop", "-1", "-i", str(bg_music_path)])
            audio_inputs_count += 1
            # Mix narration (input 0) with background music (input 1, volume set to 12%)
            filter_complex.append(f"[0:a]volume=1.0[a0];[1:a]volume=0.12[a1];[a0][a1]amix=inputs=2:duration=first[aout]")
        else:
            filter_complex.append("[0:a]volume=1.0[aout]")
            
        # Subtitle burning logic (using FFmpeg subtitles filter)
        video_filter = ""
        if use_subtitles and srt_path.exists():
            # Escape path for FFmpeg filter. SRT filter can have issues with backslashes on windows,
            # but on Linux we just escape colons and special characters.
            escaped_srt = str(srt_path).replace(":", "\\:").replace("'", "\\'")
            
            # Setup custom modern subtitle styling
            font_size = 22 if aspect_ratio == "16:9" else 15
            margin_v = 40 if aspect_ratio == "16:9" else 100
            
            # Style: Alignment=2 (Bottom Center), PrimaryColour=&H00FFFF& (Cyan/Yellow), Outline=2
            video_filter = f"subtitles='{escaped_srt}':force_style='Alignment=2,FontSize={font_size},MarginV={margin_v},PrimaryColour=&H00FFFF&,OutlineColour=&H000000&,Outline=2'"
            
        if video_filter:
            ffmpeg_cmd.extend(["-vf", video_filter])
            
        if filter_complex:
            ffmpeg_cmd.extend(["-filter_complex", ";".join(filter_complex), "-map", "0:v", "-map", "[aout]"])
        else:
            # Map default video/audio if no custom mix
            ffmpeg_cmd.extend(["-map", "0:v", "-map", "0:a"])
            
        ffmpeg_cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",  # Cut audio at length of video
            str(final_video_path)
        ])
        
        # Run FFmpeg command
        print("Running FFmpeg post-processing:", " ".join(ffmpeg_cmd))
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Clean up temp stitched video
        if temp_video_path.exists():
            temp_video_path.unlink()
            
        return final_video_path

    async def generate_video_task(self, task_id, script, aspect_ratio="16:9", voice="female1", 
                                  use_subtitles=True, use_music=True, music_genre="ambient", 
                                  visual_source="procedural", voice_speed=1.0, resolution="720p", fps=30,
                                  use_narration=True):
        """Orchestrates the entire video generation workflow."""
        try:
            self.update_progress(task_id, 5, "Splitting script into scenes...")
            scenes = self.split_script_into_scenes(script)
            
            task_dir = self.output_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Generate Voiceover narration & Visuals for each scene
            scenes_data = []
            self.update_progress(task_id, 15, f"Generating {len(scenes)} scenes of narration & visuals...")
            
            for i, scene_text in enumerate(scenes):
                audio_path = task_dir / f"scene_{i}_audio.mp3"
                image_path = task_dir / f"scene_{i}_image.png"
                video_path = task_dir / f"scene_{i}_video.mp4"
                
                # Generate voice narration or silence
                if use_narration:
                    await self.generate_narration_audio(scene_text, audio_path, voice=voice, speed=voice_speed)
                else:
                    # Generate 5 seconds of silence for each scene using ffmpeg
                    duration = 5.0
                    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", str(duration), str(audio_path)]
                    await asyncio.to_thread(subprocess.run, cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                
                # Generate visual
                has_video = False
                if visual_source == "video":
                    has_video = await asyncio.to_thread(self.download_stock_video, scene_text, video_path)
                elif visual_source == "ai-video":
                    has_video = await asyncio.to_thread(self.generate_ai_video_clip, scene_text, video_path)
                    if not has_video:
                        print("Bytez AI Video generation failed or returned error. Falling back to stock video...")
                        has_video = await asyncio.to_thread(self.download_stock_video, scene_text, video_path)
                
                if not has_video:
                    # Generate a short visual prompt or use scene text directly
                    await asyncio.to_thread(self.generate_image, scene_text, image_path, aspect_ratio=aspect_ratio, visual_source=visual_source)
                
                scenes_data.append({
                    "text": scene_text,
                    "audio_path": audio_path,
                    "image_path": image_path if not has_video else None,
                    "video_path": video_path if has_video else None
                })
                
                # Update progress incrementally
                scene_progress = 15 + int((i + 1) / len(scenes) * 45)  # takes up to 60% of timeline
                self.update_progress(task_id, scene_progress, f"Processed scene {i+1} of {len(scenes)}")
            
            # Step 2: Generate subtitle file (.srt)
            self.update_progress(task_id, 65, "Synchronizing subtitles...")
            srt_path = task_dir / "subtitles.srt"
            scene_audio_paths = [s["audio_path"] for s in scenes_data]
            await asyncio.to_thread(self.generate_srt_subtitles, scenes, scene_audio_paths, srt_path)
            
            # Step 3: Pick background music track
            bg_music_path = None
            if use_music:
                # Map select genre to files if we have stock music in uploads/music/
                music_dir = self.uploads_dir / "music"
                music_dir.mkdir(parents=True, exist_ok=True)
                
                # Look for mp3 files in the music folder
                music_files = list(music_dir.glob("*.mp3"))
                if music_files:
                    bg_music_path = random.choice(music_files)
                else:
                    # Download a royalty-free track from a public source to help user get started, or write a dummy silent file
                    fallback_music = music_dir / "ambient_fallback.mp3"
                    if not fallback_music.exists():
                        # Standard royalty free sound URL or generate silence
                        url = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" # Short test track
                        try:
                            print("Downloading sample background music...")
                            def download_music():
                                res = requests.get(url, timeout=10)
                                if res.status_code == 200:
                                    with open(fallback_music, "wb") as f:
                                        f.write(res.content)
                                    return fallback_music
                                return None
                            bg_music_path = await asyncio.to_thread(download_music)
                        except Exception as e:
                            print(f"Could not download sample music: {e}")
                    else:
                        bg_music_path = fallback_music
            
            # Step 4: Generate video
            self.update_progress(task_id, 75, "Generating and assembling clips...")
            final_video = await asyncio.to_thread(
                self.build_video_ffmpeg,
                task_id=task_id,
                scenes_data=scenes_data,
                bg_music_path=bg_music_path,
                use_subtitles=use_subtitles,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                fps=fps
            )
            
            # Step 5: Clean up individual scene files to save space
            for s in scenes_data:
                try:
                    if s.get("audio_path") and s["audio_path"].exists():
                        s["audio_path"].unlink()
                    if s.get("image_path") and s["image_path"].exists():
                        s["image_path"].unlink()
                    if s.get("video_path") and s["video_path"].exists():
                        s["video_path"].unlink()
                except Exception:
                    pass
            try:
                if srt_path.exists():
                    srt_path.unlink()
            except Exception:
                pass
                
            self.update_progress(task_id, 100, "Generation complete!", video_url=f"/renders/{task_id}/final_output.mp4")
            
        except Exception as e:
            import traceback
            error_msg = f"Generation failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.update_progress(task_id, -1, f"Error: {str(e)}")

# Test function
if __name__ == "__main__":
    vg = VideoGenerator()
    asyncio.run(vg.generate_video_task(
        "test_task", 
        "Welcome to the AI video generator. This is scene one! And this is scene two, generating locally without cloud dependencies.",
        aspect_ratio="16:9",
        voice="female1",
        use_subtitles=True,
        use_music=True,
        visual_source="procedural"
    ))

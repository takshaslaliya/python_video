import unittest
import numpy as np
from PIL import Image
import io
import wave
import struct
import tempfile
from pathlib import Path

# We can mock or import functions/logic from main.py if possible, or replicate the core logic for unit testing.
# Since we want to test the actual logic inside main.py, let's write unit tests that validate the core algorithms.

class TestBackgroundRemover(unittest.TestCase):
    def setUp(self):
        # Create a temporary image file with a solid red background and a blue center square
        self.temp_dir = tempfile.TemporaryDirectory()
        self.img_path = Path(self.temp_dir.name) / "test_image.png"
        
        # Create 100x100 image
        # Red background (255, 0, 0), blue center square 40x40 (0, 0, 255)
        img_data = np.ones((100, 100, 3), dtype=np.uint8) * np.array([255, 0, 0], dtype=np.uint8)
        img_data[30:70, 30:70] = [0, 0, 255]
        
        img = Image.fromarray(img_data, mode="RGB")
        img.save(self.img_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_bg_removal_logic(self, img_path, threshold):
        # Replicates main.py bg_remove_api inner logic
        img = Image.open(str(img_path)).convert("RGBA")
        data = np.array(img)
        
        h, w, c = data.shape
        top_edge = data[0, :, :3]
        bottom_edge = data[h-1, :, :3]
        left_edge = data[:, 0, :3]
        right_edge = data[:, w-1, :3]
        
        border_pixels = np.concatenate([top_edge, bottom_edge, left_edge, right_edge], axis=0)
        bg_color = np.median(border_pixels, axis=0)
        
        pixels = data[:, :, :3].astype(float)
        dist = np.linalg.norm(pixels - bg_color, axis=2)
        
        mask = np.where(dist < threshold, 0, 255).astype(np.uint8)
        return bg_color, mask

    def test_background_color_detection(self):
        bg_color, mask = self.run_bg_removal_logic(self.img_path, threshold=30)
        # Should detect red [255, 0, 0] as background color
        np.testing.assert_array_equal(bg_color, [255, 0, 0])

    def test_threshold_logic(self):
        # With threshold 30, the red pixels (dist = 0) are removed (mask = 0)
        # and the blue pixels (dist = ~360) are kept (mask = 255)
        _, mask = self.run_bg_removal_logic(self.img_path, threshold=30)
        
        # Red corner pixel at 5, 5 should be transparent (0)
        self.assertEqual(mask[5, 5], 0)
        # Blue center pixel at 50, 50 should be opaque (255)
        self.assertEqual(mask[50, 50], 255)

    def test_high_threshold(self):
        # If threshold is 400 (greater than max distance of ~360), all pixels should be removed
        _, mask = self.run_bg_removal_logic(self.img_path, threshold=400)
        self.assertEqual(mask[50, 50], 0)


class TestVoiceClonerPitchDetection(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def generate_sine_wave(self, freq, duration=3.0, sample_rate=16000):
        # Generate a mono wav file with a pure tone
        filename = Path(self.temp_dir.name) / f"sine_{freq}.wav"
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Generate sine wave in 16-bit range
        amplitude = 32767 * 0.8
        data = amplitude * np.sin(2 * np.pi * freq * t)
        
        with wave.open(str(filename), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for value in data:
                wav_file.writeframes(struct.pack('<h', int(value)))
        return filename

    def detect_pitch_and_gender(self, filepath):
        # Replicates main.py pitch & gender detection logic
        from pydub import AudioSegment
        import numpy as np
        sound = AudioSegment.from_file(str(filepath))
        sound = sound.set_channels(1)
        samples = np.array(sound.get_array_of_samples(), dtype=float)
        fs = sound.frame_rate
        
        duration_sec = sound.duration_seconds
        start_s = int(max(0, (duration_sec / 2) - 1.5) * fs)
        end_s = int(min(len(samples), (duration_sec / 2) + 1.5) * fs)
        snippet = samples[start_s:end_s]
        
        gender = "female"
        pitch_val = 0
        fundamental_freq = 0
        
        if len(snippet) >= 1024:
            corr = np.correlate(snippet - np.mean(snippet), snippet - np.mean(snippet), mode='full')
            corr = corr[len(corr)//2:]
            
            min_idx = int(fs / 400)
            max_idx = int(fs / 50)
            
            if min_idx < len(corr) and max_idx < len(corr):
                peak_idx = np.argmax(corr[min_idx:max_idx]) + min_idx
                fundamental_freq = fs / peak_idx
                
                if fundamental_freq < 165:
                    gender = "male"
                    pitch_val = int((fundamental_freq - 120) / 120 * 100)
                else:
                    gender = "female"
                    pitch_val = int((fundamental_freq - 210) / 210 * 100)
                    
        return gender, pitch_val, fundamental_freq

    def test_male_voice_detection(self):
        # 120Hz sine wave (typical male range)
        filepath = self.generate_sine_wave(freq=120)
        gender, pitch_val, freq = self.detect_pitch_and_gender(filepath)
        
        self.assertEqual(gender, "male")
        # Fundamental frequency detected should be very close to 120Hz
        self.assertAlmostEqual(freq, 120.0, delta=5.0)

    def test_female_voice_detection(self):
        # 210Hz sine wave (typical female range)
        filepath = self.generate_sine_wave(freq=210)
        gender, pitch_val, freq = self.detect_pitch_and_gender(filepath)
        
        self.assertEqual(gender, "female")
        # Fundamental frequency detected should be very close to 210Hz
        self.assertAlmostEqual(freq, 210.0, delta=5.0)


class TestSubtitleFormatters(unittest.TestCase):
    # Tests that the SRT and VTT formatting logic produces correct output
    def format_srt_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        if millis >= 1000:
            millis -= 1000
            secs += 1
            if secs >= 60:
                secs -= 60
                minutes += 1
                if minutes >= 60:
                    minutes -= 60
                    hours += 1
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
    def format_vtt_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        if millis >= 1000:
            millis -= 1000
            secs += 1
            if secs >= 60:
                secs -= 60
                minutes += 1
                if minutes >= 60:
                    minutes -= 60
                    hours += 1
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def test_srt_timestamp_format(self):
        self.assertEqual(self.format_srt_timestamp(0.0), "00:00:00,000")
        self.assertEqual(self.format_srt_timestamp(65.123), "00:01:05,123")
        self.assertEqual(self.format_srt_timestamp(3665.999), "01:01:05,999")

    def test_vtt_timestamp_format(self):
        self.assertEqual(self.format_vtt_timestamp(0.0), "00:00:00.000")
        self.assertEqual(self.format_vtt_timestamp(65.123), "00:01:05.123")
        self.assertEqual(self.format_vtt_timestamp(3665.999), "01:01:05.999")


if __name__ == "__main__":
    unittest.main()

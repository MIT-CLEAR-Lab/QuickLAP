"""
Audio recording utility for speech input in experiments.

This module provides functionality to record audio from the microphone
either for a specified duration or until the user stops speaking.
"""

import os
import time
import tempfile
from typing import Optional

try:
    import sounddevice as sd
    import numpy as np
    from scipy.io.wavfile import write

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice and/or scipy not available. Audio recording disabled.")
    print("Install with: pip install sounddevice scipy")


class AudioRecorder:
    """Records audio from microphone and saves to file."""

    def __init__(self, sample_rate: int = 44100, channels: int = 1):
        """
        Initialize audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        if not AUDIO_AVAILABLE:
            raise ImportError("sounddevice and scipy are required for audio recording")

        self.sample_rate = sample_rate
        self.channels = channels
        self.stop_requested = False
    
    def request_stop(self):
        """Request the recording to stop."""
        self.stop_requested = True
    
    def record_until_stopped(
        self,
        output_path: Optional[str] = None,
        max_duration: float = 60.0,
        device: Optional[str] = None,
        warmup: float = 0.4,
        debug: bool = False,
    ) -> str:
        """
        Record audio until stop_requested is set or max_duration is reached.
        
        Args:
            output_path: Path to save audio file. If None, creates temp file.
            max_duration: Maximum recording duration in seconds
            device: Audio input device
            warmup: Seconds to discard at start
            debug: Print debug info
            
        Returns:
            Path to the saved audio file
        """
        self.stop_requested = False
        
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="audio_")
            os.close(fd)

        dtype = "float32"
        chunk_duration = 0.1  # 100 ms
        frames_per_chunk = int(self.sample_rate * chunk_duration)

        if debug:
            print(f"Recording until stopped (max {max_duration}s)...")
        print("Speak now! (recording will stop after physical input ends + pause)")

        blocks = []
        t0 = time.time()

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=dtype,
            device=device,
        ) as stream:
            # Warm up: discard initial frames
            warm_frames = int(self.sample_rate * warmup)
            if warm_frames > 0:
                _discard, _ = stream.read(warm_frames)

            while True:
                data, _ = stream.read(frames_per_chunk)
                blocks.append(data)
                
                elapsed = time.time() - t0
                
                # Check stop conditions
                if self.stop_requested:
                    if debug:
                        print(f"Stop requested after {elapsed:.1f}s")
                    break
                    
                if elapsed >= max_duration:
                    if debug:
                        print(f"Reached max duration {max_duration}s")
                    break

        if blocks:
            audio = np.concatenate(blocks, axis=0)
            # Save as int16 WAV
            audio_i16 = np.clip(audio, -1, 1)
            audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
            write(output_path, self.sample_rate, audio_i16)
            print(f"Recording saved to: {output_path}")
        else:
            print("No audio captured")

        return output_path

    def record_for_duration(
        self, duration: float, output_path: Optional[str] = None
    ) -> str:
        """
        Record audio for a specified duration.

        Args:
            duration: Recording duration in seconds
            output_path: Path to save audio file. If None, creates temp file.

        Returns:
            Path to the saved audio file
        """
        if output_path is None:
            # Create temporary file
            fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="audio_")
            os.close(fd)

        print(f"Recording for {duration} seconds...")
        print("Speak now!")

        # Record audio
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
        )
        sd.wait()  # Wait until recording is finished

        # Save to file
        write(output_path, self.sample_rate, recording)
        print(f"Recording saved to: {output_path}")

        return output_path

    def record_until_stop(self, output_path: Optional[str] = None) -> str:
        """
        Record audio until user presses Enter.

        Args:
            output_path: Path to save audio file. If None, creates temp file.

        Returns:
            Path to the saved audio file
        """
        if output_path is None:
            # Create temporary file
            fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="audio_")
            os.close(fd)

        print("Recording... Press Enter to stop.")
        print("Speak now!")

        recording_data = []

        def audio_callback(indata, frames, time_info, status):
            """Callback function to capture audio data."""
            if status:
                print(f"Audio status: {status}")
            recording_data.append(indata.copy())

        # Start recording
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=audio_callback,
        ):
            input()  # Wait for user to press Enter

        # Combine all recorded chunks
        if recording_data:
            recording = np.concatenate(recording_data, axis=0)
            # Save to file
            write(output_path, self.sample_rate, recording)
            print(f"Recording saved to: {output_path}")
        else:
            print("No audio data recorded")

        return output_path

    def record_with_silence_detection(
        self,
        output_path: Optional[str] = None,
        silence_threshold: float = 0.008,  # for float32 in [-1, 1]
        silence_duration: float = 2.0,
        max_duration: float = 30.0,
        device: Optional[str] = None,
        warmup: float = 0.4,  # discard first ~400 ms
        min_record_time: float = 1.0,  # don't stop before this
        debug: bool = True,
    ) -> str:

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="audio_")
            os.close(fd)

        dtype = "float32"
        chunk_duration = 0.1  # 100 ms
        frames_per_chunk = int(self.sample_rate * chunk_duration)
        chunks_per_sec = int(round(1.0 / chunk_duration))
        silence_chunks_needed = max(1, int(round(silence_duration * chunks_per_sec)))

        if debug:
            din = None
            try:
                din, _ = sd.default.device
            except Exception:
                pass
            print(
                "Default input device:",
                din,
                "| Using:",
                device if device is not None else din,
            )

        # Validate the input configuration up front
        sd.check_input_settings(
            device=device, samplerate=self.sample_rate, channels=self.channels
        )

        print(
            f"Recording with silence detection… (≤{max_duration}s, stop after {silence_duration}s silence)"
        )
        print("Speak now!")

        blocks = []
        silent_chunks = 0
        t0 = time.time()
        last_debug = -1

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=dtype,
            device=device,
        ) as stream:

            # Warm up: read and discard from THIS stream (CoreAudio may feed zeros initially)
            warm_frames = int(self.sample_rate * warmup)
            if warm_frames > 0:
                _discard, _ = stream.read(warm_frames)

            while True:
                data, _ = stream.read(
                    frames_per_chunk
                )  # shape: (frames, channels), float32 in [-1,1]
                # Compute RMS robustly (float64 accumulation)
                rms = (
                    float(np.sqrt(np.mean(np.square(data), dtype=np.float64)))
                    if data.size
                    else 0.0
                )

                elapsed = time.time() - t0
                if debug and int(elapsed) != last_debug:
                    last_debug = int(elapsed)
                    print(f"[{elapsed:5.2f}s] RMS={rms:.5f}")

                blocks.append(data)

                # Silence detection (only after min_record_time)
                if elapsed >= min_record_time:
                    if rms < silence_threshold:
                        silent_chunks += 1
                        if silent_chunks >= silence_chunks_needed:
                            print(
                                f"Detected {silence_duration:.1f}s of silence. Stopping."
                            )
                            break
                    else:
                        silent_chunks = 0

                if elapsed >= max_duration:
                    print(f"Reached max duration {max_duration}s. Stopping.")
                    break

        if blocks:
            audio = np.concatenate(blocks, axis=0)  # float32
            # Save as int16 WAV (common)
            audio_i16 = np.clip(audio, -1, 1)
            audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
            write(output_path, self.sample_rate, audio_i16)
            print(f"Recording saved to: {output_path}")
        else:
            print("No audio captured (check device/sample rate).")

        return output_path


def quick_record(duration: Optional[float] = None) -> str:
    """
    Quick function to record audio.

    Args:
        duration: If provided, records for this duration. Otherwise records until Enter.

    Returns:
        Path to the saved audio file
    """
    if not AUDIO_AVAILABLE:
        raise ImportError(
            "Audio recording not available. Install: pip install sounddevice scipy"
        )

    recorder = AudioRecorder()

    if duration is not None:
        return recorder.record_for_duration(duration)
    else:
        return recorder.record_until_stop()


if __name__ == "__main__":
    # Example usage
    if AUDIO_AVAILABLE:
        recorder = AudioRecorder()

        print("Choose recording mode:")
        print("1. Record for 5 seconds")
        print("2. Record until you press Enter")
        print("3. Record with silence detection")

        choice = input("Enter choice (1-3): ")

        if choice == "1":
            audio_path = recorder.record_for_duration(5.0)
        elif choice == "2":
            audio_path = recorder.record_until_stop()
        elif choice == "3":
            audio_path = recorder.record_with_silence_detection()
        else:
            print("Invalid choice")
            exit(1)

        print(f"Audio saved to: {audio_path}")
    else:
        print("Audio recording not available. Please install required dependencies.")

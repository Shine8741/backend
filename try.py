import sounddevice as sd
import numpy as np
from scipy.io import wavfile
from scipy import signal

# Audio configuration
SAMPLE_RATE = 44100
BIT_DEPTH = np.int16
CHANNELS = 1
BUFFER_SIZE = 44100
NOISE_DURATION = 2  # Seconds for noise sampling

def record_and_process():
    # Record ambient noise first
    print("Recording ambient noise for 2 seconds...")
    noise = sd.rec(int(NOISE_DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=CHANNELS,
                   dtype=BIT_DEPTH,
                   blocking=True)
    
    # Record main audio
    print("\nRecording speech... (Press Enter to stop)")
    input("Ready? Press Enter to start recording...")
    audio = sd.rec(int(SAMPLE_RATE * 300),  # Max 5 minutes
                   samplerate=SAMPLE_RATE,
                   channels=CHANNELS,
                   dtype=BIT_DEPTH,
                   blocking=False)
    
    input("\nPress Enter to stop recording...")
    sd.stop()
    audio = audio[:int(sd.wait() * SAMPLE_RATE)]

    # Convert to float32 for processing
    noise = noise.astype(np.float32) / 32767.0
    audio = audio.astype(np.float32) / 32767.0

    # 1. Noise reduction using spectral gating
    def reduce_noise(audio_clip, noise_clip):
        # Calculate STFT
        freqs, _, stft = signal.stft(audio_clip, fs=SAMPLE_RATE, nperseg=1024)
        _, _, noise_stft = signal.stft(noise_clip, fs=SAMPLE_RATE, nperseg=1024)

        # Calculate noise profile (mean magnitude)
        noise_profile = np.mean(np.abs(noise_stft), axis=1)

        # Perform spectral subtraction
        stft_denoised = np.zeros_like(stft)
        for i in range(stft.shape[1]):
            magnitude = np.abs(stft[:, i])
            phase = np.angle(stft[:, i])
            magnitude = np.maximum(magnitude - noise_profile, 0)
            stft_denoised[:, i] = magnitude * np.exp(1j * phase)

        # Inverse STFT
        _, denoised = signal.istft(stft_denoised, fs=SAMPLE_RATE)
        return denoised

    processed_audio = reduce_noise(audio[:, 0], noise[:, 0])

    # 2. Amplitude normalization
    processed_audio *= 0.9 / np.max(np.abs(processed_audio))

    # 3. Convert back to 16-bit PCM
    processed_audio = (processed_audio * 32767).astype(np.int16)

    # 4. Save to WAV file
    wavfile.write("processed_output.wav", SAMPLE_RATE, processed_audio)
    print("\nFile saved as processed_output.wav")

if __name__ == "__main__":
    record_and_process()
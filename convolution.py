
import sys
import time
import numpy as np
import sounddevice
import matplotlib.pyplot as plt
import keyboard

# === Settings ===
frequency = 300
amplitude = 0.4
chunkcount = 0
chunksize = 4096
device = None
samplerate = 44100
playing: bool = False

# === Storage for recorded signal ===
recorded_signal = []  # Store all played samples


# === Manual convolution function ===
def convolution(signal1: np.ndarray, signal2: np.ndarray, chunksize: int) -> np.ndarray:                 

    if len(signal1) != chunksize or len(signal2) != chunksize:
        raise ValueError("Both signals must be of length equal to chunksize")

    output_length = 2 * chunksize - 1
    result = np.zeros(output_length)

    for n in range(output_length):
        for k in range(chunksize):
            if 0 <= n - k < chunksize:
                result[n] += signal1[k] * signal2[n - k]

    start = (output_length - chunksize) // 2
    end = start + chunksize
    result = result[start:end]

    return result


# === Keyboard events ===
def play(i):
    global playing
    playing = True

def stop(i):
    global playing
    playing = False

keyboard.on_press_key("p", play)
keyboard.on_release_key("p", stop)

try:
    samplerate = sounddevice.query_devices(device, "output")["default_samplerate"]
    print(f"Using samplerate: {samplerate}")

    def callback(outdata, _, __, status):
        if status:
            print(status, file=sys.stderr)
        global chunkcount
        if playing:
            time = (chunkcount * chunksize + np.arange(chunksize)) / samplerate
            time = time.reshape(-1, 1)  # For multiple Channels

            # Generate signal (example: sine)
            signal = amplitude * np.sin(2 * np.pi * frequency * time)

            outdata[:] = signal

            # Record the generated signal (flatten to 1D)
            recorded_signal.append(signal.flatten())

            chunkcount += 1
        else:
            outdata[:] = 0

    with sounddevice.OutputStream(
        device=device,
        channels=1,
        callback=callback,
        samplerate=samplerate,
        blocksize=chunksize,
    ):
        print("#" * 80)
        print("Press 'p' to play/pause. Press Return to quit.")
        print("#" * 80)
        input()

except KeyboardInterrupt:
    print("Interrupted")
except Exception as e:
    print(e)

# === After playback: Process the recorded signal ===
if recorded_signal:
    recorded_signal = np.concatenate(recorded_signal)  # Merge all chunks
    print(f"Recorded signal length: {len(recorded_signal)}")

    if len(recorded_signal) < chunksize:
        raise ValueError("Recorded signal too short!")

    x = recorded_signal[:chunksize]
    
    # فلتر نفس طول x
    h = np.random.uniform(-0.5, 0.5, size=chunksize)
    
    # Manual convolution
    manual_result = convolution(x, h, chunksize)
    print("Manual convolution done.")

    # === Plot ===
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(x, color='blue')
    axs[0].set_title("Original Signal x")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid()

    axs[1].plot(h, color='green')
    axs[1].set_title("Filter h")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid()

    axs[2].plot(manual_result, color='red')
    axs[2].set_title("Convolved Signal (x * h)")
    axs[2].set_xlabel("Sample")
    axs[2].set_ylabel("Amplitude")
    axs[2].grid()

    plt.tight_layout()
    plt.show()

else:
    print("No signal was recorded.")



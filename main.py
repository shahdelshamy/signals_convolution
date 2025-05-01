import sys
import time
import numpy
import sounddevice
import matplotlib.pyplot as plt
import keyboard

frequency = 300
amplitude = 0.4
chunkcount = 0
chunksize = 4096
device = None
samplerate = 44100
playing: bool = False


def saw(freq, time) -> numpy.ndarray:
    period = 1 / freq
    currtime = numpy.array(((period - (time % period)) - period / 2) / (period / 2))
    return currtime


def square(freq, time) -> numpy.ndarray:
    period = 1 / freq

    halfperiod = period / 2
    currtime = numpy.where((time % period) < halfperiod, 1, -1)
    return currtime


def triangle(freq, time) -> numpy.ndarray:
    period = 1 / freq
    halfperiod = period / 2

    first = ((time % halfperiod) / (halfperiod / 2)) - 1
    second = ((halfperiod - (time % halfperiod)) / (halfperiod / 2)) - 1
    currtime = numpy.where((time % period) < halfperiod, first, second)
    return currtime


def FFT(x) -> numpy.ndarray:
    """
    A recursive implementation of
    the 1D Cooley-Tukey FFT, the
    input should have a length of
    power of 2.
    """
    N = len(x)

    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = numpy.exp(-2j * numpy.pi * numpy.arange(N) / N)

        X = numpy.concatenate(
            [
                X_even + factor[: int(N / 2)] * X_odd,
                X_even + factor[int(N / 2) :] * X_odd,
            ]
        )
        return X


def IFFT(x) -> numpy.ndarray:
    """
    A recursive implementation of
    the 1D Cooley-Tukey IFFT, the
    input should have a length of
    power of 2.
    """
    N = len(x)

    if N == 1:
        return x
    else:
        X_even = IFFT(x[::2])
        X_odd = IFFT(x[1::2])
        factor = numpy.exp(+2j * numpy.pi * numpy.arange(N) / N)

        X = numpy.concatenate(
            [
                X_even + factor[: int(N / 2)] * X_odd,
                X_even + factor[int(N / 2) :] * X_odd,
            ]
        )
        return X


TIME = (chunkcount * chunksize + numpy.arange(chunksize)) / samplerate

SINE = numpy.sin(2 * numpy.pi * frequency * TIME)
SAW = saw(frequency, TIME)
SQRE = square(frequency, TIME)
TRIG = triangle(frequency, TIME)


# ______________
Y = (2 / chunksize) * numpy.abs(FFT(SAW))[1 : int(chunksize / 2)]
X = numpy.arange(len(Y)) * samplerate / chunksize
fig, ax = plt.subplots()
ax.plot(X, Y)
plt.show()


# ______________
# fig, ax = plt.subplots()
# ax.plot(TIME, (1/chunksize) * IFFT(FFT(SQRE)) )
# plt.show()


# ______________
# fig, ax = plt.subplots()
# ax.plot(TIME, TRIG)
# plt.show()


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
    print(samplerate)

    def callback(outdata, _, __, status):
        if status:
            print(status, file=sys.stderr)
        global chunkcount
        if playing:

            time = (chunkcount * chunksize + numpy.arange(chunksize)) / samplerate
            time = time.reshape(-1, 1)  # For multiple Channels

            outdata[:] = amplitude * numpy.sin(2 * numpy.pi * frequency * time)
            # outdata[:] = amplitude * triangle(frequency, time)

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
        print("press Return to quit")
        print("#" * 80)
        input()
except KeyboardInterrupt:
    print("Wdw")
except Exception as e:
    print(e)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter
from pathlib import Path
from typing import List, Union

MAX_NUMBER = 65535
BYTE_ORDER = 'little'

def generate_test_signal(T, N, dt, frequencies, amplitudes):
    t = np.linspace(0.0, T, N, endpoint=False)
    x = np.zeros(N)
    for f, A in zip(frequencies, amplitudes):
        x += A * np.sin(2 * np.pi * f * t)
    return t, x

def filter_signal(signal, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y

def parse_sequence(filepath: Union[Path, str]) -> List[int]:
    chunk_size = 2
    y = []
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            y_i = int.from_bytes(chunk, BYTE_ORDER)
            if y_i > 5000:
                y_i = y_i - MAX_NUMBER
            y.append(y_i)
    return y


signal_type = input("Виберіть тип сигналу (тестовий/робочий/перетворення): ")

if signal_type.lower() == "тестовий":
    frequencies = [50, 60, 400]
    amplitudes = [220, 110, 36]
    T = 1.0
    N = 1000
    dt = 1.0 / N

    t, x = generate_test_signal(T, N, dt, frequencies, amplitudes)

    lowcut = 10
    highcut = 100

    np.savetxt('C:/Users/Khome/Desktop/ЛабаФайл/filename.tst', np.column_stack((t, x)), fmt='%0.6f', delimiter='\t')

    filtered_signal = filter_signal(x, lowcut, highcut, 1.0 / dt)
    np.savetxt('C:/Users/Khome/Desktop/ЛабаФайл/filename.dL1', np.column_stack((t, filtered_signal)), fmt='%0.6f', delimiter='\t')

    filtered_signal_l2 = filter_signal(x, lowcut, 20, 1.0 / dt)
    np.savetxt('C:/Users/Khome/Desktop/ЛабаФайл/filename.dL2', np.column_stack((t, filtered_signal_l2)), fmt='%0.6f', delimiter='\t')

    filtered_signal_h3 = filter_signal(x, 20, highcut, 1.0 / dt)
    np.savetxt('C:/Users/Khome/Desktop/ЛабаФайл/filename.dh3', np.column_stack((t, filtered_signal_h3)), fmt='%0.6f', delimiter='\t')

    filtered_signal_h4 = filter_signal(x, 100, 400, 1.0 / dt)
    np.savetxt('C:/Users/Khome/Desktop/ЛабаФайл/filename.dh4', np.column_stack((t, filtered_signal_h4)), fmt='%0.6f', delimiter='\t')

    t_min, t_max = -0.2, 1.2
    custom_xticks = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                     0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    custom_xlabels = ['-0.2', '-0.15', '-0.1', '-0.05', '0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4',
                      '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1.0', '1.05', '1.1', '1.15', '1.2']

    fig = plt.figure()
    fig.patch.set_facecolor('#393065')
    ax = fig.add_subplot(111)
    ax.plot(t, x, color='orange', linewidth=2)
    ax.set_xlabel('Time S', color='#919193')
    ax.set_ylabel('Amplitude V', color='#919193')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_title('Amplitude - Time', color='white')
    ax.set_facecolor('#434376')
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xlabels, color='#919193')
    ax.tick_params(axis='y', colors='#919193')
    plt.xlim(t_min, t_max)
    plt.show()

    X = np.fft.fft(x)
    freq = np.fft.fftfreq(N, dt)

    fig, ax = plt.subplots()
    ax.plot(freq[:N // 2], np.abs(X)[:N // 2])
    fig.patch.set_facecolor('#393065')
    ax.set_xlabel('Frequency (Hz)', color='#919193')
    ax.set_title('Amplitude - Time', color='#919193')
    ax.set_title('Frequency Spectrum', color='white')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#434376')
    ax.tick_params(axis='y', colors='#919193')
    plt.show()

elif signal_type.lower() == "робочий":

    data_file_path = 'C:/Users/Khome/Desktop/ЛабаФайл/001.dat'

    y = parse_sequence(data_file_path)

    t = np.linspace(-0.2, 1.2, len(y))

    t_min, t_max = -2.5, 3
    custom_xticks = [-2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    custom_xlabels = custom_xticks

    fig = plt.figure()
    fig.patch.set_facecolor('#393065')

    ax = fig.add_subplot(111)

    ax.plot(t, y, color='#2B82D5', linewidth=2)
    ax.set_xlabel('Time S', color='#919193')
    ax.set_ylabel('Amplitude V', color='#919193')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_title('Amplitude - Time', color='white')
    ax.set_facecolor('#434376')
    ax.set_xticks(custom_xticks)
    ax.set_xticklabels(custom_xlabels, color='#919193')
    ax.tick_params(axis='y', colors='#919193')
    plt.xlim(t_min, t_max)
    ax.axhline(y=0, color='#2B82D5', linestyle='-', linewidth=2)
    plt.show()     # Відображення графіку

    Y = np.fft.fft(y)
    freq = np.fft.fftfreq(len(y), abs(t[1] - t[0]))

    fig, ax = plt.subplots()
    ax.plot(freq[:len(y) // 2], np.abs(Y)[:len(y) // 2])
    fig.patch.set_facecolor('#393065')
    ax.set_xlabel('Frequency (Hz)', color='#919193')
    ax.set_title('Amplitude - Time', color='#919193')
    ax.set_title('Frequency Spectrum', color='white')
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#434376')
    ax.tick_params(axis='y', colors='#919193')
    plt.show()

elif signal_type.lower() == "перетворення":

    def compute_fourier_transform(x, dt):
        N = len(x)
        freq = np.fft.fftfreq(N, dt)
        X = np.fft.fft(x)

        Xr = dt * np.sum(x * np.cos(2 * np.pi * np.outer(freq, np.arange(N))), axis=1)
        Xm = dt * np.sum(x * np.sin(2 * np.pi * np.outer(freq, np.arange(N))), axis=1)
        A = np.sqrt(Xr**2 + Xm**2)

        return freq, A

    frequencies_test = [50, 60, 400]
    amplitudes_test = [220, 110, 36]
    T_test = 1.0
    N_test = 1000
    dt_test = 1.0 / N_test

    t_test, x_test = generate_test_signal(T_test, N_test, dt_test, frequencies_test, amplitudes_test)

    freq_test, A_test = compute_fourier_transform(x_test, dt_test)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    custom_xticks = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                     0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    custom_xlabels = ['-0.2', '-0.15', '-0.1', '-0.05', '0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4',
                      '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1.0', '1.05', '1.1', '1.15', '1.2']

    fig.patch.set_facecolor('#393065')

    ax[0].plot(t_test, x_test, color='orange', linewidth=2)
    ax[0].set_xlabel('Time S', color='#919193')
    ax[0].set_ylabel('Amplitude V', color='#919193')
    ax[0].grid(color='gray', linestyle='--', linewidth=0.5)
    ax[0].set_title('Amplitude - Time', color='white')
    ax[0].set_facecolor('#434376')
    ax[0].set_xticks(custom_xticks)
    ax[0].set_xticklabels(custom_xlabels, color='#919193')
    ax[0].tick_params(axis='y', colors='#919193')

    ax[1].plot(freq_test[:N_test // 2], A_test[:N_test // 2], color='blue', linewidth=2)
    ax[1].set_xlabel('Frequency (Hz)', color='#919193')
    ax[1].set_ylabel('Amplitude V', color='#919193')
    ax[1].grid(color='gray', linestyle='--', linewidth=0.5)
    ax[1].set_title('Fourier Transform', color='white')
    ax[1].tick_params(axis='y', colors='#919193')
    ax[1].set_facecolor('#434376')

    plt.tight_layout()
    plt.show()

    data_file_path = 'C:/Users/Khome/Desktop/ЛабаФайл/001.dat'

    y = parse_sequence(data_file_path)

    t = np.linspace(-0.2, 1.2, len(y))

    dt_work = t[1] - t[0]

    freq_work, A_work = compute_fourier_transform(y, dt_work)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    custom_xticks = [-2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    custom_xlabels = custom_xticks

    fig.patch.set_facecolor('#393065')

    ax[0].plot(t, y, color='#2B82D5', linewidth=2)
    ax[0].set_xlabel('Time (s)', color='#919193')
    ax[0].set_ylabel('Amplitude', color='#919193')
    ax[0].set_facecolor('#434376')
    ax[0].set_xticks(custom_xticks)
    ax[0].set_xticklabels(custom_xlabels, color='#919193')
    ax[0].set_title('Original Signal', color='white')
    ax[0].tick_params(axis='y', colors='#919193')
    ax[0].axhline(y=0, color='#2B82D5', linestyle='-', linewidth=2)

    ax[1].plot(freq_work[:len(y) // 2], A_work[:len(y) // 2], color='#2B82D5', linewidth=2,)
    ax[1].set_xlabel('Frequency (Hz)', color='#919193')
    ax[1].grid(color='gray', linestyle='--', linewidth=0.5)
    ax[1].set_facecolor('#434376')
    ax[1].tick_params(axis='y', colors='#919193')
    ax[1].set_ylabel('Amplitude', color='#919193')
    ax[1].set_title('Fourier Transform', color='white')

    plt.tight_layout()
    plt.show()

else:
    print("Ви ввели невірний варіант. Спробуйте ще раз.")
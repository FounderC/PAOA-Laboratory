import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter
from pathlib import Path
from typing import List, Union

MAX_NUMBER = 65535
BYTE_ORDER = 'little'

# Створення вхідного тестового сигналу
def generate_test_signal(T, N, dt, frequencies, amplitudes):
    t = np.linspace(0.0, T, N, endpoint=False)
    x = np.zeros(N)
    for f, A in zip(frequencies, amplitudes):
        x += A * np.sin(2 * np.pi * f * t)
    return t, x

# Фільтрація сигналу
def filter_signal(signal, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y

# Зчитування даних з 001.dat файлу
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

signal_type = input("Виберіть тип сигналу (тестовий/робочий): ")

if signal_type.lower() == "тестовий":
    frequencies = [50, 60, 400]
    amplitudes = [220, 110, 36]
    T = 1.0
    N = 1000
    dt = 1.0 / N

    t, x = generate_test_signal(T, N, dt, frequencies, amplitudes)

    lowcut = 10
    highcut = 100

    # Зберігання результатів у файлах
    np.savetxt('C:/Users/Khome/Desktop/ЛабаФайл/filename.tst', np.column_stack((t, x)), fmt='%0.6f', delimiter='\t')

    filtered_signal = filter_signal(x, lowcut, highcut, 1.0 / dt)
    np.savetxt('C:/Users/Khome/Desktop/ЛабаФайл/filename.dL1', np.column_stack((t, filtered_signal)), fmt='%0.6f', delimiter='\t')

    filtered_signal_l2 = filter_signal(x, lowcut, 20, 1.0 / dt)
    np.savetxt('C:/Users/Khome/Desktop/ЛабаФайл/filename.dL2', np.column_stack((t, filtered_signal_l2)), fmt='%0.6f', delimiter='\t')

    filtered_signal_h3 = filter_signal(x, 20, highcut, 1.0 / dt)
    np.savetxt('C:/Users/Khome/Desktop/ЛабаФайл/filename.dh3', np.column_stack((t, filtered_signal_h3)), fmt='%0.6f', delimiter='\t')

    filtered_signal_h4 = filter_signal(x, 100, 400, 1.0 / dt)
    np.savetxt('C:/Users/Khome/Desktop/ЛабаФайл/filename.dh4', np.column_stack((t, filtered_signal_h4)), fmt='%0.6f', delimiter='\t')

    # Відображення графіків
    t_min, t_max = -0.2, 1.2
    custom_xticks = [-0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                     0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    custom_xlabels = ['-0.2', '-0.15', '-0.1', '-0.05', '0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4',
                      '0.45', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1.0', '1.05', '1.1', '1.15', '1.2']

    # Найстройка графіка
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

elif signal_type.lower() == "робочий":

    # Зчитування файла
    data_file_path = 'C:/Users/Khome/Desktop/ЛабаФайл/001.dat'

    # Зчитування сигналу і відображення його в графік
    y = parse_sequence(data_file_path)

    # Відображення графіка  сигналу
    t = np.linspace(-0.2, 1.2, len(y))

    t_min, t_max = -2.5, 3
    custom_xticks = [-2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    custom_xlabels = custom_xticks

    fig = plt.figure()
    # Встановлення кольору фону
    fig.patch.set_facecolor('#393065')

    # Підграфік
    ax = fig.add_subplot(111)

    # Найстройка графіка
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
    ax.axhline(y=0, color='#2B82D5', linestyle='-', linewidth=2, label='Центр')
    plt.show()     # Відображення графіку
else:
    print("Ви ввели невірний варіант. Спробуйте ще раз.")


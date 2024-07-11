from CSIKit.util.csitools import get_CSI
from CSIKit.util import filters
from CSIKit.reader import IWLBeamformReader
import numpy as np
from scipy.fft import ifft, fft

ALL_CHANNELS = [*range(24), *range(26, 50), *range(62, 85), *range(88, 112), *range(121, 145), *range(147, 170),
                *range(183, 207), *range(209, 233)]
NULL_SUBCARRIERS = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
PILOT_SUBCARRIERS = [0, 1]
USELESS_SUBCARRIERS = NULL_SUBCARRIERS + PILOT_SUBCARRIERS

DEFAULT_FS = 100


def remove_distant_multipath(csi):
    # Convert CSI to CIR
    cir = ifft(csi, axis=1)
    # Remove multipath components with delay > 0.5 microseconds
    max_delay = 0.5e-6
    subcarrier_spacing = 312.5e3  # 312.5 kHz subcarrier spacing for 20 MHz bandwidth in 802.11
    max_index = int(max_delay * subcarrier_spacing * csi.shape[1])
    cir[:, max_index:] = 0
    # Convert CIR back to CSI
    csi = fft(cir, axis=1)
    return csi


def load_csi_data(csi_data, subcarrier_range=ALL_CHANNELS, target_sample_rate=10, lowpass=True):
    frames = csi_data.frames

    # Identify the need for source resampling
    no_frames = len(frames)
    first_timestamp = float(frames[0].real_timestamp)
    last_timestamp = float(frames[-1].real_timestamp)

    final_timestamp = last_timestamp - first_timestamp
    average_sample_rate = no_frames / final_timestamp

    # Check the average sample rate is close enough to that we'd expect
    if abs(average_sample_rate - DEFAULT_FS) > 10:
        if average_sample_rate > DEFAULT_FS:
            downsample_factor = int(average_sample_rate / DEFAULT_FS)
            frames = frames[::downsample_factor]

    # Retrieve CSI for the data we've got now
    csi, _, _ = get_CSI(csi_data)
    timestamps = csi_data.timestamps

    csi = np.squeeze(csi)
    csi = np.transpose(csi)

    # Filter out unwanted subcarriers
    csi = csi[[x for x in range(64) if x not in USELESS_SUBCARRIERS]]

    # Remove distant multipath components
    csi = remove_distant_multipath(csi)

    # Handle Bandpass filter for high frequency noise removal
    if lowpass:
        lowcut = 0.3  # Hz
        highcut = 2.0  # Hz
        sampling_rate = 100  # Hz
        order = 5

        for x in range(csi.shape[0]):
            csi[x] = filters.bandpass(csi[x], lowcut, highcut, sampling_rate, order)
        csi = np.nan_to_num(csi)

    csi_trans = np.transpose(csi)

    # Downsample to 10Hz
    csi_trans = csi_trans[::10]
    timestamps = timestamps[::10]

    return csi_trans, timestamps, timestamps


# 读取文件并返回csi_data对象
def read_csi_data(file_path):
    """读取单个.dat文件的CSI数据"""
    reader = CSVBeamformReader()
    csi_data = reader.read_file(file_path)
    return csi_data


# 文件路径
file_path = "C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/scoliosis/cyd-s01.dat"


if __name__ == "__main__":
    # 示例：从文件读取预处理后的CSI数据
    # 读取CSI数据
    csi_data = read_csi_data(file_path)
# 调用load_csi_data函数进行预处理
    preprocessed_csi, timestamps, _ = load_csi_data(csi_data)
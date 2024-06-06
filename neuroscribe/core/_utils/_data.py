import gzip
import os
import shutil
from struct import calcsize, unpack

import numpy as np
import requests
from tqdm import tqdm


def _request_file(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we notice bad responses
    return response


def _save_file(response, file_path):
    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(file_path)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))


def _decompress_file(file_path):
    if file_path.endswith('.gz'):
        decompressed_file_path = file_path[:-3]
        if not os.path.exists(decompressed_file_path):
            with gzip.open(file_path, 'rb') as f_in:
                with open(decompressed_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(file_path)  # Remove the .gz file


def _read_delimited_data(file_path, chunk_size, delimiter):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        chunk = []
        for row in reader:
            chunk.append([int(x) for x in row])
            if len(chunk) >= chunk_size:
                data.append(np.array(chunk, dtype=np.uint8))
                chunk = []
        if chunk:
            data.append(np.array(chunk, dtype=np.uint8))
    return data


def _read_ubyte_data(file_path, chunk_size):
    data = []
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = _read_binary_header(f, '>4I')
        for _ in range(num):
            img = np.fromfile(f, dtype=np.uint8, count=rows*cols)
            img = img.reshape((rows, cols))
            data.append(img)
    return data


def _read_ubyte_label_data(file_path, chunk_size):
    data = []
    with open(file_path, 'rb') as f:
        magic, num = _read_binary_header(f, '>2I')
        for _ in range(num):
            label = np.fromfile(f, dtype=np.uint8, count=1)
            data.append(label)
    return data


def _read_binary_header(file, format_string):
    header_size = calcsize(format_string)
    header_bytes = file.read(header_size)
    return unpack(format_string, header_bytes)

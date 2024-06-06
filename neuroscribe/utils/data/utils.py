import os

from neuroscribe.utils._utils._data import (
    _read_delimited_data,
    _read_ubyte_data,
    _read_ubyte_label_data,
)


def read_data(file_path, chunk_size=1000):
    file_extension = os.path.splitext(file_path)[1:]
    if file_extension in ['.csv', '.tsv']:
        return _read_delimited_data(file_path, chunk_size, delimiter=',' if file_extension == '.csv' else '\t')
    else:
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic == b'\x00\x00\x08\x03':
                    return _read_ubyte_data(file_path, chunk_size)
                elif magic == b'\x00\x00\x08\x01':
                    return _read_ubyte_label_data(file_path, chunk_size)
                else:
                    raise ValueError("Unsupported file format")
        except FileNotFoundError:
            raise ValueError("File not found")
        except Exception as e:
            raise ValueError("Error reading file: {}".format(e))

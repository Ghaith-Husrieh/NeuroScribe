import os
import platform
from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent


def cuda_available():
    return 'CUDA_PATH' in os.environ or 'CUDA_HOME' in os.environ


def on_mac_silicon():
    return platform.system() == "Darwin" and "arm" in platform.machine().lower()


install_requires = ['numpy==1.26.4']
if cuda_available():
    install_requires.extend(['cupy-cuda12x==13.0.0', 'fastrlock==0.8.2',])
if on_mac_silicon():
    install_requires.extend(['jax==0.4.11', 'jaxlib==0.4.11', 'jax-metal==0.0.4'])
long_description = (BASE_DIR / "README.md").read_text()


setup(
    name='neuro-scribe',
    version='0.1.0.post1',
    author='Ghaith Husrieh',
    description='NeuroScribe - a lightweight deep learning framework.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    project_urls={
        'Source': 'https://github.com/Ghaith-Husrieh/NeuroScribe'
    },
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
    install_requires=install_requires
)

# setup.py
"""
项目安装脚本
"""
from setuptools import setup, find_packages
import os

# 读取README
def read_long_description():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# 读取requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="pest-detection-multimodal",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="多模态病虫害识别系统",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pest-detection-multimodal",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "lora": [
            "accelerate",
            "bitsandbytes",
        ],
    },
    entry_points={
        "console_scripts": [
            "pest-train=scripts.train:main",
            "pest-eval=scripts.evaluate:main",
            "pest-infer=scripts.inference:main",
            "pest-prepare=data.prepare_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
)
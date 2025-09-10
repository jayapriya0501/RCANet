from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="rcanet",
    version="0.1.0",
    author="RCANet Team",
    author_email="rcanet@example.com",
    description="Row-Column Attention Networks: A Dual-Axis Transformer Framework for Enhanced Representation Learning on Tabular Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/RCANet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "PyYAML>=6.0",
        "tqdm>=4.62.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "full": [
            "plotly>=5.0.0",
            "pyarrow>=5.0.0",
            "h5py>=3.1.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "xgboost>=1.5.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0",
            "optuna>=3.0.0",
            "shap>=0.41.0",
            "lime>=0.2.0",
            "rich>=12.0.0",
            "wandb>=0.12.0",
            "memory-profiler>=0.60.0",
            "psutil>=5.8.0",
            "cerberus>=1.3.0",
            "numba>=0.56.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rcanet-train=rcanet.examples.basic_usage:main",
            "rcanet-benchmark=rcanet.examples.benchmark:main",
        ],
    },
    keywords="transformer, attention, tabular data, machine learning, deep learning, pytorch",
    project_urls={
        "Bug Reports": "https://github.com/your-username/RCANet/issues",
        "Source": "https://github.com/your-username/RCANet",
        "Documentation": "https://github.com/your-username/RCANet/wiki",
    },
    include_package_data=True,
    zip_safe=False,
)
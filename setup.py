from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm_master",
    version="0.1.0",
    author="Sina Azizi",
    author_email="sina@uflo.io",
    description="A unified interface for multiple LLM providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uflo-ai/llm_master",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.95.0",
        "anthropic>=0.57.0",
        "requests>=2.31.0",
    ],
)
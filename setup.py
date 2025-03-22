from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="oflo-ai-agent-protocol",
    version="0.1.0",
    author="Ankit Buti",
    author_email="ankit@oflo.ai",
    description="A protocol for building business AI agents with MCP integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ankitbuti/oflo-ai-agent-protocol",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "mcp-python-sdk",
        "slack-sdk",
        "asyncio",
    ],
) 
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
    url="https://github.com/ankitbuti/oflo-agent-protocol",
    packages=find_packages(exclude=["tests*", "examples*"]),
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
        "aiohttp>=3.8.0",
        "anthropic>=0.3.0",
        "openai>=1.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "slack-sdk>=3.0.0",
        "typing-extensions>=4.0.0",
        "cloudflare>=2.8.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=1.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "oflo_agent_protocol": ["py.typed"],
    },
) 
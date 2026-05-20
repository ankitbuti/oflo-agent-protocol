from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="oflo-ai-agent-protocol",
    version="2.0.0",
    author="Ankit Buti",
    author_email="ankit@oflo.ai",
    description=(
        "Modular, auditable, multi-LLM agent protocol with MCP, A2A, "
        "smart routing, voice, sandboxed execution, and 300+ app connectors."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ankitbuti/oflo-agent-protocol",
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "aiohttp>=3.9.0",
        "pydantic>=2.6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        # Install everything: pip install oflo-ai-agent-protocol[all]
        "all": [
            "anthropic>=0.37.0",
            "openai>=1.30.0",
            "composio>=0.7.0",
            "elevenlabs>=1.0.0",
            "daytona>=0.1.0",
            "langchain>=0.2.0",
            "langchain-core>=0.2.0",
            "langchain-anthropic>=0.2.0",
            "langchain-openai>=0.1.0",
            "langgraph>=0.1.0",
            "weaviate-client>=4.5.0",
        ],
        "anthropic": ["anthropic>=0.37.0"],
        "openai": ["openai>=1.30.0"],
        "composio": ["composio>=0.7.0"],
        "voice": ["elevenlabs>=1.0.0", "pyaudio>=0.2.14"],
        "daytona": ["daytona>=0.1.0"],
        "langchain": [
            "langchain>=0.2.0",
            "langchain-core>=0.2.0",
            "langchain-anthropic>=0.2.0",
            "langchain-openai>=0.1.0",
            "langgraph>=0.1.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.0.0",
            "black>=24.0.0",
            "isort>=5.13.0",
            "mypy>=1.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "oflo_agent_protocol": ["py.typed"],
    },
)

"""The agent's backend on Modal (app.py) and a CLI that drives it (cli.py). What both read sits here,
apart from the Modal app."""
from llm_master.types import Config, Model

APP = "llm-master-server"  # the deployed app's name, unless AGENT_APP names another

# each at its own thinking level
MODELS = {model.model: model for model in [
    Model(provider="anthropic", model="claude-sonnet-5-5", config=Config(thinking="medium")),
    Model(provider="openai", model="gpt-6.1-sol", config=Config(thinking="high")),
    Model(provider="gemini", model="gemini-3.8-flash", config=Config(thinking="high")),
    Model(provider="gemini", model="gemini-4-argon", config=Config(thinking="medium")),
]}

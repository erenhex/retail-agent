import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def load_repo_env(repo_root: Path | None = None) -> None:
    """Load ``retail-bench/.env`` and map OpenRouter key to OpenAI client env vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    root = repo_root or _REPO_ROOT
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)

    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]

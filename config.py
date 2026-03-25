import os
from functools import lru_cache


class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # OpenRouter (OpenAI-compatible API)
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

    # Calendar integration requires OAuth credentials; default to dry-run.
    GOOGLE_CALENDAR_ENABLED: bool = os.getenv("GOOGLE_CALENDAR_ENABLED", "false").lower() == "true"
    GOOGLE_OAUTH_CREDENTIALS_JSON: str = os.getenv(
        "GOOGLE_OAUTH_CREDENTIALS_JSON",
        "./google_oauth_credentials.json",
    )
    GOOGLE_OAUTH_TOKEN_JSON: str = os.getenv("GOOGLE_OAUTH_TOKEN_JSON", "./google_oauth_token.json")

    # Email sending configuration
    SMTP_HOST: str = os.getenv("SMTP_HOST", "")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASS: str = os.getenv("SMTP_PASS", "")
    SMTP_FROM_EMAIL: str = os.getenv("SMTP_FROM_EMAIL", "")

    SENDGRID_API_KEY: str = os.getenv("SENDGRID_API_KEY", "")
    SENDGRID_FROM_EMAIL: str = os.getenv("SENDGRID_FROM_EMAIL", "")

    # Where to build candidate test links.
    # Example: http://localhost:8000/tests/take?token=
    TEST_LINK_BASE_URL: str = os.getenv("TEST_LINK_BASE_URL", "http://127.0.0.1:8000/tests/take?token=")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


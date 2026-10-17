from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_key: str = "dev-only-change-me"
    upload_dir: str = "storage/uploads"
    database_path: str = "storage/chiaki.db"
    max_upload_size_mb: int = 200
    job_lease_seconds: int = 360
    max_job_attempts: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
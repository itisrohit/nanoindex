from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "NanoIndex"
    API_V1_STR: str = "/api/v1"

    # Storage settings
    DATA_DIR: str = "data"
    INDEX_FILENAME: str = "nano.index"

    # Search settings
    DEFAULT_TOP_K: int = 10

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env")


settings = Settings()

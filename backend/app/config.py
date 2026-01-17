from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///instructor.db"
    assignment_dir: str = ""
    llm_mode: str = "local"
    vllm_host: str = "localhost:8001"

    class Config:
        env_prefix = ""
        case_sensitive = False


settings = Settings()

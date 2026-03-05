"""
存放配置相关的
"""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "mysql+pymysql://root:root@localhost:3306/graduatedesign"
    SECRET_KEY: str = "don'ttalkyou"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    # 通义千问配置
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_BASE_URL: str = ""

    # LLM 配置
    LLM_MODEL: str = "qwen-plus"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
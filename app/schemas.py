"""
Pydantic 数据验证模型 (请求/响应)
"""

from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True  # 允许从 ORM 对象读取数据


class Token(BaseModel):
    access_token: str
    token_type: str
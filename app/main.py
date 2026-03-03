"""
应用入口，负责注册路由和中间件
"""

from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from app.database import engine, Base, get_db
from app.routers import users
from app import models, auth
import uvicorn

# # 自动建表
# Base.metadata.create_all(bind=engine)

app = FastAPI(title="My FastAPI Project")

# 注册路由
app.include_router(users.router)

# 依赖项：获取当前用户 (供其他受保护接口使用)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


async def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    # 这里复用之前的逻辑，实际项目中可放入 utils 或 auth 模块
    credentials_exception = Exception("Could not validate credentials")
    try:
        payload = auth.jwt.decode(token, auth.settings.SECRET_KEY, algorithms=[auth.settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except auth.JWTError:
        raise credentials_exception

    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None: raise credentials_exception
    return user


@app.get("/users/me")
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return {"id": current_user.id, "username": current_user.username, "email": current_user.email}


if __name__ == "__main__":
    import os, sys
    
    # 确保项目根目录在 Python 路径中
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"CWD: {os.getcwd()}")
    print(f"Sys Path: {sys.path}")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

# 使用轻量级 Python 镜像
FROM python:3.11-slim

# 设置环境变量
# 1. 防止 Python 生成 .pyc 文件
# 2. 确保日志输出不被缓冲，实时打印到控制台
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
# 分层构建：先复制依赖文件，利用 Docker 缓存机制加速构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目所有文件
COPY . .

# 暴露应用端口 (main.py 默认使用 8000)
EXPOSE 8000

# 启动应用
# main.py 中已经包含了 uvicorn 的启动逻辑和参数解析
CMD ["python", "main.py"]

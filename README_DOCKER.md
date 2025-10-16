# Docker 部署说明

## 快速开始

### 使用 Docker Compose（推荐）

```bash
# 构建并启动容器
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止容器
docker-compose down

# 重新构建镜像
docker-compose up -d --build
```

启动后访问: http://localhost:8501

### 使用 Docker 命令

#### 1. 构建镜像

```bash
docker build -t htfa-economic-analysis:latest .
```

#### 2. 运行容器

```bash
docker run -d \
  --name htfa-app \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config:/app/config \
  htfa-economic-analysis:latest
```

Windows PowerShell:
```powershell
docker run -d `
  --name htfa-app `
  -p 8501:8501 `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/logs:/app/logs `
  -v ${PWD}/config:/app/config `
  htfa-economic-analysis:latest
```

#### 3. 管理容器

```bash
# 查看运行状态
docker ps

# 查看日志
docker logs -f htfa-app

# 停止容器
docker stop htfa-app

# 启动容器
docker start htfa-app

# 删除容器
docker rm htfa-app

# 进入容器
docker exec -it htfa-app bash
```

## 镜像说明

### 基础信息

- 基础镜像: `python:3.11-slim`
- 暴露端口: `8501`
- 工作目录: `/app`

### 环境变量

- `PYTHONPATH=/app` - Python模块搜索路径
- `PYTHONUNBUFFERED=1` - 禁用Python输出缓冲
- `STREAMLIT_SERVER_PORT=8501` - Streamlit服务端口
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0` - 绑定所有网络接口
- `STREAMLIT_SERVER_HEADLESS=true` - 无头模式运行
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` - 禁用使用统计

### 数据卷挂载

建议挂载以下目录以实现数据持久化：

- `./data:/app/data` - 数据文件目录
- `./logs:/app/logs` - 日志文件目录
- `./config:/app/config` - 配置文件目录

## 高级配置

### 自定义端口

修改 `docker-compose.yml` 中的端口映射：

```yaml
ports:
  - "8080:8501"  # 将宿主机8080端口映射到容器8501端口
```

或使用 Docker 命令：

```bash
docker run -d -p 8080:8501 htfa-economic-analysis:latest
```

### 生产环境配置

#### 1. 使用环境文件

创建 `.env` 文件：

```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

在 `docker-compose.yml` 中引用：

```yaml
services:
  htfa-app:
    env_file:
      - .env
```

#### 2. 配置资源限制

在 `docker-compose.yml` 中添加：

```yaml
services:
  htfa-app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

#### 3. 配置日志

```yaml
services:
  htfa-app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 健康检查

容器包含健康检查配置，每30秒检查一次服务状态：

```bash
# 查看健康状态
docker inspect --format='{{.State.Health.Status}}' htfa-app
```

## 故障排查

### 容器无法启动

```bash
# 查看详细日志
docker logs htfa-app

# 检查容器状态
docker inspect htfa-app
```

### 端口已被占用

```bash
# 查看端口占用情况
# Linux/Mac
netstat -tuln | grep 8501

# Windows
netstat -ano | findstr 8501

# 修改docker-compose.yml中的端口映射
```

### 数据文件权限问题

```bash
# 修改目录权限
chmod -R 755 ./data ./logs ./config

# 或在Dockerfile中设置用户
```

### 镜像构建失败

```bash
# 清理Docker缓存
docker system prune -a

# 重新构建（不使用缓存）
docker-compose build --no-cache
```

## 镜像优化

### 减小镜像大小

1. 使用 `.dockerignore` 排除不需要的文件
2. 使用多阶段构建（如需要编译步骤）
3. 清理APT缓存（已在Dockerfile中实现）

### 构建速度优化

1. 将不常变动的步骤放在前面（如安装依赖）
2. 使用Docker层缓存
3. 使用国内镜像源（可选）

```dockerfile
# 在Dockerfile中添加（安装系统依赖前）
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 部署到生产环境

### 使用反向代理（Nginx）

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 使用 HTTPS

配置 SSL 证书并更新 Nginx 配置：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # 其他配置...
}
```

## 备份和恢复

### 备份数据

```bash
# 备份data目录
docker run --rm \
  -v htfa_data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/htfa-data-backup.tar.gz /data
```

### 恢复数据

```bash
# 恢复data目录
docker run --rm \
  -v htfa_data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/htfa-data-backup.tar.gz -C /
```

## 更新应用

```bash
# 1. 停止当前容器
docker-compose down

# 2. 拉取最新代码
git pull

# 3. 重新构建并启动
docker-compose up -d --build

# 4. 查看日志确认启动成功
docker-compose logs -f
```

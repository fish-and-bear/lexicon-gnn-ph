# Local Development Setup (Windows)

## Prerequisites

1. **Docker Desktop for Windows**
   - Download and install from: https://www.docker.com/products/docker-desktop
   - Ensure WSL 2 is enabled
   - Minimum requirements: 8GB RAM, Windows 10/11

2. **Python 3.11+**
   - Download and install from: https://www.python.org/downloads/
   - Add Python to PATH during installation
   - Verify with: `python --version`

3. **Node.js 18+**
   - Download and install from: https://nodejs.org/
   - Verify with: `node --version`

4. **PowerShell 5.0+**
   - Windows 10/11 includes this by default
   - Run `$PSVersionTable.PSVersion` to verify

## Initial Setup

1. **Clone the Repository**
   ```powershell
   git clone https://github.com/yourusername/fil-relex.git
   cd fil-relex
   ```

2. **Configure Environment**
   ```powershell
   # Copy example environment file
   Copy-Item .env.example .env.local
   
   # Edit .env.local with your preferred text editor
   notepad .env.local
   ```

3. **Start Development Environment**
   ```powershell
   # Run the development script
   .\dev.ps1 start
   ```

## Available Commands

Run all commands from PowerShell in the project root directory:

- **Start Environment**:
  ```powershell
  .\dev.ps1 start
  ```

- **Stop Environment**:
  ```powershell
  .\dev.ps1 stop
  ```

- **Clean Environment** (removes all data):
  ```powershell
  .\dev.ps1 clean
  ```

- **View Logs**:
  ```powershell
  .\dev.ps1 logs
  ```

- **Show Help**:
  ```powershell
  .\dev.ps1 help
  ```

## Accessing Services

Once the environment is running, you can access:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:10000
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger UI**: http://localhost:16686

## Development Workflow

1. **Frontend Development**
   - Source code is in `src/` directory
   - Changes are hot-reloaded automatically
   - Access the app at http://localhost:3000

2. **Backend Development**
   - Source code is in `backend/` directory
   - Changes are hot-reloaded automatically
   - API documentation at http://localhost:10000/api/v2/docs

3. **Database Changes**
   - Migrations are in `migrations/` directory
   - Create new migration:
     ```powershell
     docker-compose -f docker-compose.local.yml exec backend alembic revision --autogenerate -m "description"
     ```
   - Apply migrations:
     ```powershell
     docker-compose -f docker-compose.local.yml exec backend alembic upgrade head
     ```

4. **Monitoring**
   - View metrics in Grafana: http://localhost:3001
   - Check API traces in Jaeger: http://localhost:16686
   - Monitor system in Prometheus: http://localhost:9090

## Troubleshooting

1. **Docker Issues**
   - Ensure Docker Desktop is running
   - Check Docker resources (CPU/Memory) in Docker Desktop settings
   - Try restarting Docker Desktop

2. **Database Issues**
   - Clean the environment: `.\dev.ps1 clean`
   - Check database logs: `docker-compose -f docker-compose.local.yml logs db`
   - Verify database connection:
     ```powershell
     docker-compose -f docker-compose.local.yml exec db psql -U postgres -d dictionary_dev
     ```

3. **Frontend Issues**
   - Clear node_modules: 
     ```powershell
     Remove-Item -Recurse -Force node_modules
     npm install
     ```
   - Check frontend logs: `docker-compose -f docker-compose.local.yml logs frontend`

4. **Backend Issues**
   - Check backend logs: `docker-compose -f docker-compose.local.yml logs backend`
   - Verify Python dependencies:
     ```powershell
     docker-compose -f docker-compose.local.yml exec backend pip list
     ```

5. **Permission Issues**
   - Run PowerShell as Administrator
   - Check file permissions in WSL
   - Ensure Docker has access to the project directory

## Common Tasks

1. **Adding Dependencies**
   - Frontend (Node.js):
     ```powershell
     docker-compose -f docker-compose.local.yml exec frontend npm install package-name
     ```
   - Backend (Python):
     ```powershell
     docker-compose -f docker-compose.local.yml exec backend pip install package-name
     ```

2. **Running Tests**
   - Frontend:
     ```powershell
     docker-compose -f docker-compose.local.yml exec frontend npm test
     ```
   - Backend:
     ```powershell
     docker-compose -f docker-compose.local.yml exec backend python -m pytest
     ```

3. **Database Operations**
   - Access PostgreSQL shell:
     ```powershell
     docker-compose -f docker-compose.local.yml exec db psql -U postgres -d dictionary_dev
     ```
   - Backup database:
     ```powershell
     docker-compose -f docker-compose.local.yml exec db pg_dump -U postgres dictionary_dev > backup.sql
     ```
   - Restore database:
     ```powershell
     docker-compose -f docker-compose.local.yml exec -T db psql -U postgres dictionary_dev < backup.sql
     ```

4. **Redis Operations**
   - Access Redis CLI:
     ```powershell
     docker-compose -f docker-compose.local.yml exec redis redis-cli -a redis
     ```
   - Monitor Redis:
     ```powershell
     docker-compose -f docker-compose.local.yml exec redis redis-cli -a redis MONITOR
     ```

## Best Practices

1. **Version Control**
   - Create feature branches from `main`
   - Keep commits small and focused
   - Write meaningful commit messages

2. **Code Quality**
   - Run linters before committing:
     ```powershell
     # Frontend
     docker-compose -f docker-compose.local.yml exec frontend npm run lint
     
     # Backend
     docker-compose -f docker-compose.local.yml exec backend flake8
     ```
   - Run type checks:
     ```powershell
     # Frontend
     docker-compose -f docker-compose.local.yml exec frontend npm run type-check
     
     # Backend
     docker-compose -f docker-compose.local.yml exec backend mypy .
     ```

3. **Performance**
   - Use the monitoring tools to identify bottlenecks
   - Profile code when needed:
     ```powershell
     docker-compose -f docker-compose.local.yml exec backend python -m cProfile script.py
     ```

4. **Security**
   - Never commit sensitive data
   - Keep dependencies updated
   - Follow security best practices

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the logs using `.\dev.ps1 logs`
3. Open an issue in the repository
4. Contact the development team 
# Deployment Guide

## Environment Setup

### Requirements

- Python 3.9+ or Docker
- 4GB+ RAM (8GB+ recommended)
- 10GB+ storage (for model and audit data)

### Configuration

1. **Secrets Management**:
   - Set `AUDIT_SECRET` environment variable
   - Use secret manager integration for production

2. **Database**:
   - SQLite for development (included)
   - PostgreSQL/Redis for production scale

3. **Networking**:
   - Port 8000 for metrics (Prometheus)
   - Secure client-orchestrator communication channel

## Docker Deployment

### Build Image

```bash
docker build -t fedgnn-advanced .

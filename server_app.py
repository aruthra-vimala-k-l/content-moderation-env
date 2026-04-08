"""
FastAPI application for the Content Moderation OpenEnv Environment.
Entry point: server.app:main
"""
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from openenv.core.env_server import create_fastapi_app

from environment import ContentModerationEnvironment
from models import ModerationAction, ModerationObservation

app = create_fastapi_app(
    env=ContentModerationEnvironment,
    action_cls=ModerationAction,
    observation_cls=ModerationObservation,
    max_concurrent_envs=10,
)

def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()

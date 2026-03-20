import logging
logging.basicConfig(level=logging.DEBUG)
from app.api import app
import uvicorn
uvicorn.run(app, port=8000, log_level="debug", loop="asyncio")
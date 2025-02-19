import redis
import os
from dotenv import load_dotenv

load_dotenv()

redis_url = os.getenv('REDIS_URL')
if not redis_url:
    raise ValueError("REDIS_URL is not set in the environment variables")

client = redis.Redis.from_url(redis_url)

# Clear the cache
client.flushdb()  # Flush the current database
# client.flushall()  # Use this to flush all databases

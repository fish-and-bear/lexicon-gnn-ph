import redis

# Replace with your Redis URL and credentials
redis_url = "rediss://red-cr22c6ogph6c73belo30:c4Z6nAsWRAHkWNHk7bCTUyvVPCuYoLuM@oregon-redis.render.com:6379"
client = redis.Redis.from_url(redis_url)

# Clear the cache
client.flushdb()  # Flush the current database
# client.flushall()  # Use this to flush all databases

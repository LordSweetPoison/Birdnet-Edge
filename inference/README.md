## Directions 

### Run the following commands in order in seperate terminals:

Start the redis server (you may need to change the permissions of the run_redis.sh)

'./run_redis.sh'

Start the celery background process

'celery -A server.celery worker --loglevel=info' 

Start the web server

'python server.py'
## Directions 

### Run the following commands in order in seperate terminals:

Start the redis server (you may need to change the permissions of the run_redis.sh)

~~~
nohup ./run_redis.sh > redis.out &
# to view the output:
tail -F redis.out
~~~

Start the celery background process

~~~
nohup celery -A server.celery worker --loglevel=info > celery.out &
# to view the output:
tail -F celery.out

celery -A upload worker --loglevel=info 
~~~

Start the web server

~~~
nohup python3 server.py > server.out &
# to view the output:
tail -F server.out

nohup python3 servertpu.py > server.out &
~~~



docker build -f Dockerfile -t liamdetermined/development:deepspeed --build-arg ALWAYS_RUN=$(date +%Y-%m-%d:%H:%M:%S) .
docker push liamdetermined/development:deepspeed

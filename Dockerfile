FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /app
COPY . .
RUN pixi install --locked && rm -rf ~/.cache/rattler
RUN pixi shell-hook -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
RUN echo 'exec "$@"' >> /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
CMD [ "pixi", "run" ]
ENTRYPOINT [ "/app/entrypoint.sh" ] 

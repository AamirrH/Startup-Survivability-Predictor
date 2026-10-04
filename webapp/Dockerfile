FROM python:3.12-slim
WORKDIR /app
COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt
COPY webapp webapp
RUN useradd --create-home appuser
USER appuser
ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000
CMD ["python", "-m", "webapp.app"]

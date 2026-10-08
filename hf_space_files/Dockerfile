FROM python:3.10

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user evaluate_agent.py .
COPY --chown=user start.sh .

RUN chmod +x start.sh
EXPOSE 7860
CMD ["./start.sh"]

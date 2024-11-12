# Use the official Python image from the Docker Hub
FROM langchain/langchain

RUN apt-get update && apt-get install -y libffi-dev libnacl-dev
# RUN apt-get install -y python3-dev

# Set the working directory in the container
WORKDIR /app

RUN python3 -m pip install -U discord.py
RUN python3 -m pip install playwright
RUN playwright install
RUN playwright install-deps  
RUN python3 -m pip install beautifulsoup4
RUN python3 -m pip install wikipedia
RUN python3 -m pip install langgraph langsmith
RUN python3 -m pip install langchain-groq
RUN python3 -m pip install langchain-openai
RUN python3 -m pip install Pillow
RUN python3 -m pip install aiohttp
RUN python3 -m pip install langchain_google_community
RUN python3 -m pip install langchain
RUN python3 -m pip install langchain-community
RUN python3 -m pip install -U duckduckgo-search

# Install the dependencies
# RUN pip install --no-cache-dir -r requirements.txt
# タイムゾーンを設定
RUN ln -snf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && echo "Asia/Tokyo" > /etc/timezone


# Command to run the bot
CMD ["python", "bot.py"]

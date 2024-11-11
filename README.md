discord-bot

## docker compose 起動方法
`.env`を作成し、以下のように環境変数を記述する。
```.env
DISCORD_API_KEY=value
OLLAMA_API_KEY=value
OLLAMA_URL=https://target/ollama/api
```

起動する。

```bash
docker compose up -d
```

## メモ
langchainで使える形式にしなきゃいけない。
chat と generate で送るべきリクエストのjson が変わる。
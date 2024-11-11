import discord
import LangTools
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
import base64
import io
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import re
from typing import List
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferMemory
import os
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

class LangchainBot(discord.Client):
    def __init__(self, llm:BaseChatModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.llm = llm
        
        self.system_prompt_getter = None
        if 'system_prompt_getter' in kwargs:
            self.system_prompt_getter = kwargs['system_prompt_getter']
            print(self.system_prompt_getter())
        
        self.system_prompt = None
        if 'system_prompt' in kwargs:
            self.system_prompt = {
                "role": "system",
                "content": kwargs['system_prompt'],
            }
        # 検索機能の初期化
        self.search = DuckDuckGoSearchAPIWrapper()
        # 分析用プロンプトの設定
        self.query_prompt = PromptTemplate(
            template="""
            あなたは与えられた質問に対して、以下の3つの判断を行うアシスタントです：
            1. 最新の情報が必要かどうか
            2. URLが含まれているかどうか
            3. 通常の会話で対応可能かどうか
            4. 画像生成が必要かどうか

            質問: {question}

            以下の形式で応答してください：
            NEEDS_SEARCH: [true/false] - 最新の情報が必要な場合はtrue
            HAS_URL: [true/false] - URLが含まれている場合はtrue
            SEARCH_QUERY: [検索クエリ] - NEEDS_SEARCHがtrueの場合のみ必要な検索クエリを書いてください
            NEEDS_GENERATEIMG: [true/false] - 画像生成が必要な場合はtrue
            IMAGE_QUERY: [画像生成クエリ] - NEEDS_GENERATEIMGがtrueの場合のみ必要な画像生成プロンプトを書いてください。日本語の場合は英語にして書いてください。
            IMAGE_QUERY_mini: [画像生成クエリ] - NEEDS_GENERATEIMGがtrueの場合のみ画像の特定のビジュアル要素に焦点を当てた短縮させた画像生成クエリを書いてください。日本語の場合は英語にして書いてください。
            """,
            input_variables=["question"]
        )
        self.query_chain = self.query_prompt | self.llm
        
        # スケジューラの設定
        self.scheduler = AsyncIOScheduler()
        self.scheduler.start()
        
    def extract_urls(self, text: str) -> List[str]:
        """URLを検出する関数"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)

    async def get_webpage_content(self, url: str) -> str:
        """Webページの内容を取得する関数"""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text[:5000]
        except Exception as e:
            return f"Error fetching webpage: {str(e)}"

    async def on_ready(self):
        print(f'Logged on as {self.user}!')
        # 例: 毎日9:00にメッセージを送信
        #self.scheduler.add_job(self.send_scheduled_message, 'cron', hour=9, minute=0)
    
    async def generate_chat_prompt(self, message, history_limit:int=10) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        messages_generator = message.channel.history(limit=history_limit)
        # メッセージを取得 (最新のメッセージから取得)
        # messageを取得するたびに、HumanMessageかAIMessageに変換してmessagesに追加
        async for msg in messages_generator:
            content = LangTools.sanitize_mention(msg)
            if msg.author.bot:
                messages.append(AIMessage(content=content))
            else:
                name = LangTools.get_name(msg.author)
                name = name+': '
                messages.append(HumanMessage(content=f'{name}{content}'))
        
        # システムプロンプトを追加
        if self.system_prompt_getter is not None:
            self.system_prompt = self.system_prompt_getter()
        if self.system_prompt is not None:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.reverse()
        
        return messages
    
    async def generate_reply(self, message, history_limit:int=10) -> str:
        async with message.channel.typing():
            messages: list[BaseMessage] = await self.generate_chat_prompt(
                message, history_limit)
            response: AIMessage = await self.llm.ainvoke(messages)
            print(str(response))
            response = response.content
            response = LangTools.sanitize_breakrow(response)

        return response
    
    async def schedule_message(self, channel_id: int, time: str, message_content: str):
        # 入力された時間をパースして日時オブジェクトに変換
        try:
            scheduled_time = datetime.datetime.strptime(time, "%H:%M")
            now = datetime.datetime.now()
            # 今日の予定時刻にするために日付を修正
            scheduled_time = scheduled_time.replace(year=now.year, month=now.month, day=now.day)
            if scheduled_time < now:
                # すでに過ぎた時間の場合、翌日に設定
                scheduled_time += datetime.timedelta(days=1)
        except ValueError:
            print("時間の形式が正しくありません。'HH:MM'形式で指定してください。")
            return

        # 指定された時間にメッセージを送信
        self.scheduler.add_job(
            self.send_scheduled_message,
            'date',
            run_date=scheduled_time,
            args=[channel_id, message_content]
        )
        print(f"メッセージがスケジュールされました: {time} にチャンネル {channel_id} で送信予定")
    
    async def send_scheduled_message(self, channel_id: int, message_content: str):
        # 指定チャンネルにメッセージを送信
        channel = self.get_channel(channel_id)
        if channel:
            await channel.send(message_content)
        else:
            print(f"チャンネルID {channel_id} が見つかりませんでした")
    
    async def generate_web(self, message, prompt, history_limit=10) -> str:
        messages = await self.generate_chat_prompt(message, history_limit)
        messages.append(HumanMessage(content=prompt))
        messages.append(
                SystemMessage(content=f'今話題のものや動画にできそうな事をもとに動画の台本とタイトルを生成してください'))
        # Chat modelに会話の続きを生成させる
        response = await self.llm.ainvoke(messages)
        response = LangTools.sanitize_breakrow(response.content)
        return response

    async def on_message(self, message):
        if message.author.bot or message.author == self.user:
            return
        # メンションされているユーザーのリストを取得
        mentioned_users = message.mentions
        # 特定のユーザーがメンションされているか確認
        if self.user not in mentioned_users:
            return
        # 質問の分析
        analysis = await self.query_chain.ainvoke(prompt)
        # AIMessageからcontentを取得
        content = analysis.content if hasattr(analysis, 'content') else str(analysis)
        needs_search = "NEEDS_SEARCH: true" in content
        # urlを含むか確認
        has_url = "HAS_URL: true" in content
        needs_generateimg = "NEEDS_GENERATEIMG: true" in content
        if has_url:
            urls = self.extract_urls(prompt)
            if urls:
                    webpage_content = await self.get_webpage_content(urls[0])
                    prompt_with_content = f"以下のWebページの内容に基づいて適切な返答を考えてください。広告や関連記事などに気を取られないでください。\n\nWebページ内容: {webpage_content}\n\n質問: {prompt}"
                    #promptreply = await self.llm.ainvoke(prompt_with_content)
                    #reply1 = await self.generate_web(message,promptreply)
                    reply1 = await self.generate_web(message,prompt_with_content)
                    reply = f"**URLを要約中...**\n\n{reply1}"
        elif needs_search:
            search_query = re.search(r'SEARCH_QUERY: (.*)', content)
            #print(search_query)
            if search_query:
                search_results = self.search.run(search_query.group(1))
                prompt_with_search = f"""以下の検索結果の内容に基づいて適切な返答を考えてください。広告や関連記事などに気を取られないでください。
                できるだけ最新の情報を含めて回答してください。

                検索結果: {search_results}

                質問: {prompt}
                """
                # promptreply = await self.llm.ainvoke(prompt_with_search)
                # reply1 = await self.generate_web(message,promptreply)
                #print(prompt_with_search)
                reply1 = await self.generate_web(message,prompt_with_search)
                #print("11111111")
                #print(reply1)
                reply = f"**Webを検索中...**\n\n{reply1}"
        else:
            command_content = message.content.replace(f'<@{self.user.id}>', '').strip()
			if command_content.startswith("!schedule"):
				new_content = command_content[len('!schedule '):].strip()
				match = re.match(r"<#(\d+)> (\d{2}:\d{2}) (.+)", new_content)
				if match:
					channel_id = int(match.group(1))
					time = match.group(2)
					message_content = match.group(3)
					await self.schedule_message(channel_id, time, message_content)
					await message.channel.send(f"{time} にチャンネル {channel_id} でメッセージをスケジュールしました")
				else:
					await message.channel.send("形式が正しくありません。`!schedule #チャンネル 時間 メッセージ` の形式で入力してください。")
			else:
				sentence = await self.generate_reply(message, history_limit=10)
				reply = f"{sentence}"
        await message.reply(reply)
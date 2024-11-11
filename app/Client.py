import discord
import LangTools
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.language_models import BaseChatModel

class HoboJuki(discord.Client):
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

    async def on_ready(self):
        print(f'Logged on as {self.user}!')
    
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
    
    async def generate_reply_with_webpage_content(self, message, url:str, history_limit:int=10) -> str:
        async with message.channel.typing():
            messages: list[BaseMessage] = await self.generate_chat_prompt(
                message, history_limit)
            summary_with_info: tuple[str, list[str]] = LangTools.summarize(
                url, self.llm, 
                read_max_chars=20000, 
                summarize_chunk_size=2000,
                summarize_max_chars=2000)
            summary = summary_with_info[0]
            info = summary_with_info[1]
            if len(info) > 0:
                info = '\n'.join(info)
                messages.append(
                    SystemMessage(content=f'１つ目のURL要約中: {info}'))
            messages.append(
                SystemMessage(content=f'今話題のものや動画にできそうな事をもとに動画の台本とタイトルを生成してください'))
            messages.append(
                SystemMessage(content=f'本日のネタ: {summary}'))
            response: AIMessage = await self.llm.ainvoke(messages)
            response = response.content
            response = LangTools.sanitize_breakrow(response)
            
        return response

    async def on_message(self, message):
        if message.author.bot or message.author == self.user:
            return
        # メンションされているユーザーのリストを取得
        mentioned_users = message.mentions
        # 特定のユーザーがメンションされているか確認
        if self.user not in mentioned_users:
            return
        
         # urlを含むか確認
        urls = LangTools.has_url(message.content)
        if urls:
            # とりあえずひとつだけ読む
            url = urls[0]
            reply = await self.generate_reply_with_webpage_content(
                message, url, history_limit=10)
        else:
            sentence = await self.generate_reply(message, history_limit=10)
            reply = f"{sentence}"
        await message.reply(reply)

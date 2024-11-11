import re
import pathlib
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from LangModel import LangModel as LM
import OllamaLangModel
import urllib.parse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import subprocess
import warnings

def get_name(author)->str:
    if author.display_name is not None:
        return author.display_name
    return author.name

def sanitize_mention(message)->str:
    '''
    <@1291400028593721437>のような文字列を@ユーザー名に変換する
    '''

    content = message.content
    for mention in message.mentions:
        m = mention.mention
        content = content.replace(f'{m}', f'@{mention.name}')
        m = m[:2] + '&' + m[2:]
        content = content.replace(f'{m}', f'@{mention.name}')
    return content

def sanitize_breakrow(message:str)->str:
    '''
    改行コードを削除する
    '''
    pattern = r'\n+'
    content = re.sub(pattern, r'\n', message)
    return content

def ban_system_prompt(message:str)->str:
    '''
    システムプロンプトを出力していたらそれを削除する
    '''
    # TODO path がハードコードされているので修正する
    path = '/prompts/system_prompt_keywords.txt'
    path = pathlib.Path(path)
    if not path.exists():
        return message
    with open(path, 'r') as f:
        keywords = f.readlines()
    keyword_count = 0
    for keyword in keywords:
        if keyword in message:
            keyword_count += 1
    if keyword_count >= len(keywords) * 0.2:
        return '検閲により削除済み'
    return message


def has_url(message:str) -> bool|list[str]:
    """
    retrns urls in the message if they exist else False
    """
    url_pattern = r'https?://\S+'
    urls = re.findall(url_pattern, message)
    return urls if urls else False

def remove_url(message:str) -> str:
    """
    Removes urls from the message
    """
    url_pattern = r'https?://\S+'
    return re.sub(url_pattern, '', message)

def decode_url(encoded_str):
    # URLエンコードされた部分を全てデコード
    return re.sub(r'%[0-9A-Fa-f]{2}', lambda match: urllib.parse.unquote(match.group(0)), encoded_str)

def remove_encoded_url(message:str) -> str:
    """
    Removes urls from the message
    """
    url_pattern = r'%[0-9A-Fa-f]{2}'
    return re.sub(url_pattern, '', message)

def summarize(
        url:str, 
        lang_model:BaseChatModel, 
        debug:bool=False,
        read_max_chars:int=20000, # ページの最大文字数　以降は読まない
        summarize_chunk_size:int=2000, # 要約のchunk size
        summarize_max_chars:int=2000, # 要約の最大文字数
    ) -> tuple[str, list[str]]:
    """
    Summarizes the given URL.
    Args:
        url (str): The URL to summarize.
    Returns:
        str: The summarized content.
    """
    
    info = []
    
    if debug:
        print('summarize url:', url)
    
    # Load HTML
    page_content = subprocess.run(
        ['python', 'page_loader.py', '--url', url],
        text=True,
        capture_output=True
    ).stdout
    page_content = remove_url(page_content)
    page_content = remove_encoded_url(page_content)
    
    prompt_template = PromptTemplate(
        input_variables=['page_content'],
        template='要約タスク: 以下の文章を要約してください。どんな言語でも要約を日本語で行ってください。\n\n{page_content}'
    )
    if len(page_content) > read_max_chars:
        # ページの最大文字数を超えた場合は、最大文字数までに切り捨てる
        # その旨の情報を追加
        page_content = page_content[:read_max_chars]
        info.append(f'info: The page content is too long. Only the first {read_max_chars} characters were read.')
        # 警告も表示
        warnings.warn(f'The page content is too long. Only the first {read_max_chars} characters were read.')
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=summarize_chunk_size, chunk_overlap=summarize_chunk_size//10)
    summarized_page_content = page_content
    if debug:
        loop_count = 0
    current_summarized_chars = len(summarized_page_content)
    while len(summarized_page_content) > summarize_max_chars:
        if debug:
            loop_count += 1
            print('loop_count:', loop_count)
            print('current summarized chars:', len(summarized_page_content))
        split_texts = text_splitter.split_text(summarized_page_content)
        next_summarized_page_content = []
        for text_chunk in split_texts:
            messages = [
                SystemMessage(content=prompt_template.format(page_content=text_chunk)),
            ]
            response = lang_model.invoke(messages)
            next_summarized_page_content.append(response.content)
            if debug:
                print('summarize response:', response)
        summarized_page_content = '\n\n'.join(next_summarized_page_content)
        next_summarized_chars = len(summarized_page_content)
        # 増えたら終了
        if current_summarized_chars <= next_summarized_chars:
            break
        current_summarized_chars = next_summarized_chars

    if debug:
        print('page content:', page_content)
        import pathlib
        dir = pathlib.Path('dump')
        dir.mkdir(exist_ok=True, parents=True)
        with open(dir / 'page_content.txt', 'w') as f:
            f.write(page_content)
    if debug:
        print('summarize response:', summarized_page_content)
        with open(dir / 'summarized_page_content.txt', 'w') as f:
            f.write(summarized_page_content)
    return (summarized_page_content, info)


# 返答すべきか考える関数
def should_reply(model:LM, messages:list[dict[str, str]], debug:bool=False) -> bool:
    """
    Determines whether the AI assistant should reply based on the given conversation messages.
    Args:
        model (LM): The language model used to analyze the conversation.
        messages (list[dict[str, str]]): A list of dictionaries representing the conversation messages. 
                                            Each dictionary contains 'role' and 'content' keys.
    Returns:
        bool: True if the AI assistant should reply, False otherwise.
    """
    
    prompt = [{
        'role':'system',
        'content':'''output: True or False
以下の会話の流れを読んで、AIアシスタントとして返答すべきか考えてください。
返答は必ず、TrueかFalseでお答えください。他の文字列を含めないでください。
あなたの出力をpythonのbool型に変換して返します。そのため、TrueかFalseでお答えください。
'''}]
    messages = prompt + messages
    prompt = [{
        'role':'system',
        'content':'> should_reply? True or False'
    }]
    messages = messages + prompt
    ans = True
    result = model.chat(messages)['content']
    if debug:
        print('should_reply result', result)
    try:
        result = bool(result)
    except Exception as e:
        if debug:
            print('Error:', e)
        result = result.lower()
        if 'true' in result:
            ans = True
        elif 'false' in result:
            ans = False
        else:
            ans = True
    if debug:
        print('should_reply ans:', ans)
    return ans

if __name__ == '__main__':
    def value_from_env():
        import os
        api_key = os.environ['OLLAMA_API_KEY']
        api_url = os.environ['OLLAMA_URL']

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return api_key, api_url, headers
    api_key, api_url, headers = value_from_env()
    lm = LM(api_key, api_url, "gemma2:9b")
    lang_model = OllamaLangModel.OllamaAPIChatModel(
        lang_model=lm
    )
    url = 'https://ja.wikipedia.org/wiki/%E5%B2%A1%E5%B4%8E%E5%B8%82'
    summary_with_info = summarize(url, lang_model=lang_model, debug=True)
    print(summary_with_info[0])
    print(summary_with_info[1])
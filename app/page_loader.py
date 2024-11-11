from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, required=True)
    return parser.parse_args()

def main():
    url = parse_args().url
    loader = AsyncChromiumLoader([url])
    html = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    page_content = bs_transformer.transform_documents(
        html, 
        unwanted_tags=['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'input', 'button', 'select', 'textarea', 'iframe', 'img', 'video', 'audio', 'canvas', 'svg', 'map', 'object', 'embed', 'applet', 'frame', 'frameset', 'noframes', 'base', 'link', 'meta'],
        tags_to_extract=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div', 'a', 'span'],
        # remove_lines=True,
    )
    page_content = page_content[0].page_content
    print(page_content)
    
if __name__ == '__main__':
    # python page_loader.py --url https://www.google.com
    main()
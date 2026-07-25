import bs4 
import requests
from langchain_core.documents import Document


def load_web_page(url: str, bs_kwrg : dict | None = None) -> list[Document]:
    response = requests.get(url,timeout=30)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.txt,"html.parser",**(bs_kwrg or {}))
    
    return [Document(page_content=soup.get_text(),metadata ={"source":url})]


urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [load_web_page(url) for url in urls]
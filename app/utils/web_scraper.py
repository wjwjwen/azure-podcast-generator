"""Module for web scraping utils."""

import os

import requests
from bs4 import BeautifulSoup
from openai import AzureOpenAI
from utils.document import DocumentResponse
from utils.identity import get_token_provider


def scrape_webpage(url: str) -> DocumentResponse:
    """Scrape webpage and extract relevant content using GPT-4."""

    # Fetch webpage content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text content
    text = soup.get_text()

    # Clean up text (remove extra whitespace)
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = " ".join(chunk for chunk in chunks if chunk)

    # Use GPT-4 to extract relevant content
    if os.getenv("AZURE_OPENAI_KEY"):
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    else:
        client = AzureOpenAI(
            api_version="2024-08-01-preview",
            azure_ad_token_provider=get_token_provider(),
        )

    system_prompt = """
    You are an expert content analyzer. Given a webpage's content:
    1. Extract the main article content
    2. Identify key themes and main points
    3. Remove irrelevant content like navigation menus, footers, etc.
    4. Format the output in clean markdown
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT", "gpt-4o"),
        temperature=0.3,
        max_tokens=8000,
    )

    processed_content = chat_completion.choices[0].message.content

    return DocumentResponse(markdown=processed_content, pages=1)

"""Module for LLM utils."""

import json
import os
from dataclasses import dataclass

import streamlit as st
import tiktoken
from openai import AzureOpenAI
from openai.types import CompletionUsage
from utils.identity import get_token_provider

AZURE_OPENAI_API_VERSION = "2024-08-01-preview"

PROMPT = """
Create a highly engaging podcast script between two people based on the input text. Use informal language to enhance the human-like quality of the conversation, including expressions like \"wow,\" laughter, and pauses such as \"uhm.\"

# Steps

1. **Review the Document(s) and Podcast Title**: Understand the main themes, key points, interesting facts and tone.
2. **Adjust your plan to the requested podcast duration**: The conversation should be engaging and take about 5 minutes to read out loud.
3. **Character Development**: Define two distinct personalities for the hosts.
4. **Script Structure**: Outline the introduction, main discussion, and conclusion.
5. **Incorporate Informal Language**: Use expressions and fillers to create a natural dialogue flow.
6. **Engage with Humor and Emotion**: Include laughter and emotional responses to make the conversation lively. Think about how the hosts would react to the content and make it an engaging conversation.


# Output Format

- A conversational podcast script in structured JSON.
- Include informal expressions and pauses.
- Clearly mark speaker turns.
- Name the hosts {voice_1} and {voice_2}.

# Examples

**Input:**
- Document: [Brief overview of the content, main themes, or key points]
- Podcast Title: \"Exploring the Wonders of Space\"

**Output:**
- Speaker 1: \"Hey everyone, welcome to 'Exploring the Wonders of Space!' I'm [Name], and with me is [Name].\"
- Speaker 2: \"Hey! Uhm, I'm super excited about today's topic. Did you see the latest on the new satellite launch?\"
- Speaker 1: \"Wow, yes! It's incredible. I mean, imagine the data we'll get! [laughter]\"
- (Continue with discussion, incorporating humor and informal language)

# Notes

- Maintain a balance between informal language and clear communication.
- Ensure the conversation is coherent and follows a logical progression.
- Adapt the style and tone based on the document's content and podcast title.
- Think step by step, grasp the key points of the document / paper, and explain them in a conversational tone.
""".strip()

JSON_SCHEMA = {
    "name": "podcast",
    "strict": True,
    "description": "An AI generated podcast script.",
    "schema": {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Language code + locale (BCP-47), e.g. en-US or es-PA",
                    }
                },
                "required": ["language"],
                "additionalProperties": False,
            },
            "script": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the host. Use the provided names, don't change the casing or name.",
                        },
                        "message": {"type": "string"},
                    },
                    "required": ["name", "message"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["config", "script"],
        "additionalProperties": False,
    },
}


@dataclass
class PodcastScriptResponse:
    podcast: dict
    usage: CompletionUsage


@dataclass
class PodcastResponse:
    podcast: dict
    tokens: int


def document_to_podcast_script(
    document: str,
    title: str = "AI in Action",
    voice_1: str = "Andrew",
    voice_2: str = "Emma",
    max_tokens: int = 8000,
) -> PodcastScriptResponse:
    """Get LLM response."""

    # Authenticate via API key (not advised for production)
    if os.getenv("AZURE_OPENAI_KEY"):
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    # Authenticate via DefaultAzureCredential (e.g. managed identity or Azure CLI)
    else:
        client = AzureOpenAI(
            api_version=AZURE_OPENAI_API_VERSION,
            azure_ad_token_provider=get_token_provider(),
        )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": PROMPT.format(voice_1=voice_1, voice_2=voice_2),
            },
            # Wrap the document in <documents> tag for Prompt Shield Indirect attacks
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter?tabs=warning%2Cindirect%2Cpython-new#embedding-documents-in-your-prompt
            {
                "role": "user",
                "content": f"<title>{title}</title><documents><document>{document}</document></documents>",
            },
        ],
        model=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT", "gpt-4o"),
        temperature=0.7,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        max_tokens=max_tokens,
    )

    message = chat_completion.choices[0].message.content
    json_message = json.loads(message)

    return PodcastScriptResponse(podcast=json_message, usage=chat_completion.usage)


@st.cache_resource
def get_encoding() -> tiktoken.Encoding:
    """Get TikToken."""
    encoding = tiktoken.encoding_for_model("gpt-4o")

    return encoding


def document_to_english_learning_podcast(
    document: str,
    voice_1: str,  # English host
    voice_2: str,  # Chinese host
    max_tokens: int = 4000,
) -> PodcastResponse:
    """Convert document to English learning podcast format with English-Chinese bilingual hosts."""

    system_prompt = """You are creating an engaging bilingual podcast for advanced English learners who speak Chinese.
    The podcast has three distinct segments with specific roles for each host:

    Part 1: Content Summary (English Host Led)
    - Host 1 (English) provides a comprehensive summary of the content
    - Host 2 (Chinese) occasionally adds (CN) brief cultural context or key points
    Example:
    Host 1: "Today we're exploring an fascinating article about AI development. The key points discuss..."
    Host 2: "(CN) 在开始深入讨论之前，我想指出这个话题在中国和西方的视角有一些有趣的差异..."
    Host 1: "That's an interesting perspective. Looking at the first major point..."

    Part 2: Deep Dive Q&A
    - Host 2 (Chinese) asks insightful questions about the content
    - Host 1 (English) provides detailed explanations
    - Questions should follow a logical progression and build understanding
    Example:
    Host 2: "(CN) 这篇文章提到了AI的伦理问题，能具体解释一下作者的观点吗？"
    Host 1: "The author's perspective on AI ethics is quite nuanced. They argue that..."
    Host 2: "(CN) 这让我想到另一个相关的问题：在实际应用中，这种伦理框架如何落地？"
    Host 1: "That's a crucial question. In practical applications..."

    Part 3: Key Takeaways and Reflection
    - Both hosts discuss main insights and implications
    - Focus on practical applications and broader context
    Example:
    Host 2: "(CN) 让我们总结一下今天讨论的几个重要观点..."
    Host 1: "Yes, I think there are three key takeaways. First..."
    Host 2: "(CN) 特别是第二点，它对我们的日常生活有很大启发..."

    IMPORTANT FORMAT RULES:
    1. Always start lines with "Host 1:" or "Host 2:"
    2. Use (CN) prefix for Chinese content
    3. Maintain natural conversation flow - avoid direct translations
    4. Questions should build upon previous points
    5. Each part should be clearly marked with "Part 1:", "Part 2:", "Part 3:"

    Focus on creating a dynamic, educational dialogue that helps listeners deeply understand the content."""

    user_prompt = f"""Create an English learning podcast based on this content:

{document}

Remember:
- Follow the three-part structure
- Create logical question progression in Part 2
- Focus on deep understanding rather than translation
- Keep the conversation natural and engaging
"""

    try:
        if os.getenv("AZURE_OPENAI_KEY"):
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
        # Authenticate via DefaultAzureCredential (e.g. managed identity or Azure CLI)
        else:
            client = AzureOpenAI(
                api_version=AZURE_OPENAI_API_VERSION,
                azure_ad_token_provider=get_token_provider(),
            )

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT", "gpt-4"),
            temperature=0.7,
            max_tokens=max_tokens,
        )

        # 处理返回的内容，转换成播客脚本格式
        podcast_script = []
        current_section = ""

        for line in chat_completion.choices[0].message.content.split("\n"):
            if line.strip():
                if "Part 1:" in line or "Summary Discussion" in line:
                    current_section = "summary"
                    continue
                elif "Part 2:" in line or "Language Focus" in line:
                    current_section = "language_focus"
                    continue
                elif "Part 3:" in line or "Discussion Questions" in line:
                    current_section = "questions"
                    continue

                if line.startswith("Host 1:"):
                    podcast_script.append(
                        {
                            "name": "Host 1",
                            "message": line.replace("Host 1:", "").strip(),
                            "voice": voice_1,
                            "section": current_section,
                        }
                    )
                elif line.startswith("Host 2:"):
                    podcast_script.append(
                        {
                            "name": "Host 2",
                            "message": line.replace("Host 2:", "").strip(),
                            "voice": voice_2,
                            "section": current_section,
                        }
                    )

        # 确保至少有一个脚本项
        if not podcast_script:
            raise Exception("No valid script content generated")

        return PodcastResponse(
            podcast={
                "script": podcast_script,
                "config": {"language": "en-US"},
            },  # 添加config
            tokens=len(chat_completion.choices[0].message.content),
        )

    except Exception as e:
        raise Exception(f"Error generating English learning podcast: {str(e)}") from e

"""Module for Bilibili video scraping."""

import json
import os
import re

import requests
from openai import AzureOpenAI
from utils.document import DocumentResponse


class BilibiliScraper:
    """Handler for Bilibili videos."""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def get_video_info(self, bvid: str) -> dict:
        """Get Bilibili video information."""
        api_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
        response = requests.get(api_url, headers=self.headers)
        data = response.json()

        if data["code"] == 0:
            return data["data"]
        raise Exception(f"Failed to get video info: {data['message']}")

    def get_video_url(self, bvid: str) -> str:
        """Get video download URL."""
        api_url = f'https://api.bilibili.com/x/player/playurl?bvid={bvid}&cid={self.get_video_info(bvid)["cid"]}&qn=16'
        response = requests.get(api_url, headers=self.headers)
        data = response.json()

        if data["code"] == 0:
            return data["data"]["durl"][0]["url"]
        raise Exception(f"Failed to get video URL: {data['message']}")

    def download_audio(self, url: str, output_path: str):
        """Download video audio."""
        response = requests.get(url, headers=self.headers, stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def process_video(self, bvid: str) -> DocumentResponse:
        """Process Bilibili video."""
        try:
            # 基础设置
            base_url = "https://www.bilibili.com/video"
            url = f"{base_url}/{bvid}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
                "Referer": "https://www.bilibili.com",
            }

            # 1. 获取页面内容
            response = requests.get(url=url, headers=headers)
            html = response.text

            # 2. 提取视频标题
            title = re.findall('title="(.*?)"', html)[0]

            # 3. 提取视频信息
            info = re.findall("window.__playinfo__=(.*?)</script>", html)[0]
            json_data = json.loads(info)

            # 4. 提取音频链接
            audio_url = json_data["data"]["dash"]["audio"][0]["baseUrl"]

            # 5. 获取音频内容
            audio_content = requests.get(url=audio_url, headers=headers).content

            # 6. 保存到临时文件
            temp_dir = "temp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            audio_path = os.path.join(temp_dir, f"{bvid}.mp3")
            with open(audio_path, "wb") as f:
                f.write(audio_content)

            # 7. 使用Azure Whisper转写
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version="2024-06-01",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )

            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file, model=("whisper")
                )

            # 8. 清理临时文件
            os.remove(audio_path)

            # 9. 组织内容
            content = [f"# {title}", f"\nTranscript:\n{transcript.text}"]

            return DocumentResponse(markdown="\n\n".join(content), pages=1)

        except Exception as e:
            raise Exception(f"处理B站视频失败: {str(e)}") from e


def scrape_video(bvid: str) -> DocumentResponse:
    """处理B站视频内容，获取音频并转写"""
    scraper = BilibiliScraper()
    return scraper.process_video(bvid)

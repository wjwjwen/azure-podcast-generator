"""Module for audio utils."""

import os

import azure.cognitiveservices.speech as speechsdk
from const import AZURE_HD_VOICES, LOGGER
from utils.identity import get_speech_token


# TODO leverage streaming to speed up generation
# https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-lower-speech-synthesis-latency?pivots=programming-language-csharp#streaming
def text_to_speech(ssml) -> bytes:
    """Use Azure Speech Service and convert SSML to audio bytes."""

    if os.getenv("AZURE_SPEECH_KEY"):
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ["AZURE_SPEECH_KEY"],
            region=os.environ["AZURE_SPEECH_REGION"],
        )
    else:
        speech_config = speechsdk.SpeechConfig(
            auth_token=get_speech_token(os.environ["AZURE_SPEECH_RESOURCE_ID"]),
            region=os.environ["AZURE_SPEECH_REGION"],
        )

    audio_config = None  # enable in-memory audio stream

    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm
    )

    # Creates a speech synthesizer using the Azure Speech Service.
    speech_synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # Synthesizes the received text to speech.
    result = speech_synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        LOGGER.warning(f"Speech synthesis canceled: {cancellation_details.reason}")

        if (
            cancellation_details.reason == speechsdk.CancellationReason.Error
            and cancellation_details.error_details
        ):
            LOGGER.error(f"Error details: {cancellation_details.error_details}")

        raise Exception(f"Error details: {cancellation_details.error_details}")

    raise Exception(f"Unknown exit reason: {result.reason}")


def podcast_script_to_ssml(podcast_data: dict) -> str:
    """Convert podcast script to SSML format."""

    def process_text(text: str) -> str:
        """处理特殊标记和中文内容"""
        # 处理笑声标记
        text = text.replace(
            "[laughter]",
            '<break time="300ms"/> <say-as interpret-as="interjection">ha ha</say-as> <break time="200ms"/>',
        )
        text = text.replace(
            "[laughs]",
            '<break time="300ms"/> <say-as interpret-as="interjection">ha ha</say-as> <break time="200ms"/>',
        )
        text = text.replace(
            "[chuckles]",
            '<break time="300ms"/> <say-as interpret-as="interjection">heh</say-as> <break time="200ms"/>',
        )

        # 处理中文内容
        if "(CN)" in text:
            text = text.replace("(CN)", "")
            text = f'<lang xml:lang="zh-CN">{text}</lang>'

        return text

    # 获取完整的voice ID
    voice_1 = AZURE_HD_VOICES[podcast_data["voice_1"]]
    voice_2 = AZURE_HD_VOICES[podcast_data["voice_2"]]

    # 动态创建说话者到voice的映射
    speakers = list(set(item["name"] for item in podcast_data["script"]))
    voice_mapping = {
        speaker: voice_1 if i % 2 == 0 else voice_2
        for i, speaker in enumerate(speakers)
    }

    ssml = """<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">\n"""

    for item in podcast_data["script"]:
        voice = voice_mapping[item["name"]]  # 使用说话者名字获取对应的voice
        text = process_text(item["message"])

        if "(CN)" in item["message"]:
            # 中文部分
            ssml += f"""<voice name="{voice}"><mstts:express-as style="chat"><prosody rate="0.95">{text}<break time="500ms"/></prosody></mstts:express-as></voice>\n"""
        else:
            # 英语部分
            ssml += f"""<voice name="{voice}"><mstts:express-as style="chat">{text}<break time="500ms"/></mstts:express-as></voice>\n"""

    ssml += "</speak>"
    return ssml

import subprocess
from pytube import YouTube
import os
from pathlib import Path
from transformers import pipeline
import torch


def read_file_content(filename):
    with open(filename, 'r') as file:
        return file.read()

def save_video(url, video_filename):
    youtubeObject = YouTube(url)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")
    
    return video_filename

def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first() #this will return only audio data from the youtube video
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    try:
        os.rename(out_file, file_name)
    except WindowsError:
        os.remove(file_name)
        os.rename(out_file, file_name)
    audio_filename = Path(file_name).stem+'.mp3'
    video_filename = save_video(url, Path(file_name).stem+'.mp4')
    print(yt.title + " Has been successfully downloaded")
    return yt.title, audio_filename, video_filename

def load_model():

    #transcriber = FlaxWhisperPipline("openai/whisper-large-v2",dtype=jnp.float32)
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
 
    return transcriber



def transcription(audio_file):

    model = load_model()
    output = model(audio_file, chunk_length_s=30,
        batch_size=24,
        return_timestamps=True
    ) 
    return output

def transcribe2(audio_file):
    if audio_file:
        head, tail = os.path.split(audio_file)
        path = head
        
        if tail[-3:] != 'wav':
            subprocess.call(['ffmpeg', '-i', audio_file, "audio.wav", '-y'])
            tail = "audio.wav"
  
        subprocess.call(['ffmpeg', '-i', audio_file, "audio.wav", '-y'])
        tail = "audio.wav"
        print("before diarize") 
        #os.system(f"insanely-fast-whisper --file-name {tail} --hf_token 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 --flash True")
        subprocess.run([f"insanely-fast-whisper --file-name {tail} --diarization_model --hf_token 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 --flash True"], shell=True, capture_output=True, text=True)
        print("after diarize")
        subprocess.run(["python cleanup.py"], shell=True, capture_output=True, text=True)

        text = read_file_content('audio.txt')
        torch.cuda.empty_cache()
        return text


from django.http import HttpResponse
from django.shortcuts import render
from AudioTranscription.utility.helper import *
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from transformers import AutoModelForCausalLM, AutoTokenizer

# pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", token= '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', torch_dtype=torch.float16, device="cuda")

# pipe = pipeline("text-generation", model="TheBloke/zephyr-7B-beta-GPTQ")

#pipe = pipeline("text-generation", model="microsoft/phi-2")

model_name = "microsoft/phi-2"

pipe = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto",
    trust_remote_code=True,
)

def home(request):
    final_response = {}
    
    try:
        if request.method == 'POST':
            query = request.POST.get('userquery')
            
            
            prompt = "You are an Q&A expert. Answer the user question: "
            prompt = prompt + query

            outputs = pipe(
                prompt,
                max_new_tokens=500,
                do_sample=True, 
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            final_res = outputs[0]["generated_text"]
            # final_res = final_res.split("<|assistant|>",1)[1]
            
            print(final_res)
            
            final_response = {
                'title' : 'Home Page',
                'response' : final_res, 

            }
   
    except:
        pass

    return render(request,"index.html",final_response)
    # return HttpResponse("Work in progess!")

def youtube_transcription(request):
    result = {}
    try:
        if request.method == 'POST':
            url = request.POST.get('audioInput')
            video_title, audio_filename, video_filename = save_audio(url)
            transcription_result = transcription(audio_filename)
            transcription_result = transcription_result['chunks']
            
            result = {
                'transcription' : transcription_result,
                'video' : video_filename,
                'video_title' : video_title
            }
            print(result['transcription'])
    except:
        pass
    
    return render(request,"youtube.html", result)


def audio_file_transcription(request):
    result = {}
    try:
        if request.method == 'POST':

            if 'audioFile' in request.FILES:
                audio_file = request.FILES['audioFile']
                aud_result = transcription(audio_file)
                result = {
                    "aud_result" : aud_result,
                }

                print(aud_result)
            
    except:
        pass

    return render(request,"audio.html",result)

def real_time_transcription(request):
    return render(request,"realtime.html")
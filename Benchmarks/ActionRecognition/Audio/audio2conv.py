import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, HarmBlockThreshold
import json
import re
# from moviepy.editor import VideoFileClip
from tqdm import tqdm
import os
from transformers import pipeline
import torch
from google.cloud import speech, storage
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# import whisper_timestamped as whisper
# import whisperx
import gc 
from pydub import AudioSegment
from openai import OpenAI

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

OPENAI_API_KEY=''
client = OpenAI(api_key=OPENAI_API_KEY)

def gemini_generate(prompt, audio, max_token=8192, temperature=1, top_p=0.95):

    generation_config = {
        "max_output_tokens": max_token,
        "temperature": temperature,
        "top_p": top_p,
    }

    output = []
    vertexai.init(project="", location="us-central1")
    # vertexai.init(project="", location="us-central1")
    model = GenerativeModel(
        # "gemini-1.5-pro-001",
        "gemini-1.5-flash-002",
    )

    if audio != None:
        responses = model.generate_content(
            [prompt, audio],
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )
    else:
        responses = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )

    # print(responses.text)
    return responses.text

def extract_json(response, pattern=None):

    if type(response) == json:
        return None, response

    # Regular expression pattern to match all JSON objects
    # pattern = r'\[.*\]'
    if not pattern:
        pattern = r'\[\s*{\s*"Role":\s*".+?",\s*"Utterance":\s*".+?"\s*}(?:,\s*{\s*"Role":\s*".+?",\s*"Utterance":\s*".+?"\s*})*\s*\]'


    # Find all matches in the text
    matches = re.findall(pattern, response, re.DOTALL)
    print('+++++'*100)
    print(matches)
    print(response)
    print('+++++'*100)

    if not matches:
        print('====='*100)
        print("No JSON object found in the text.")
        print(response)
        print('====='*100)
        return "No JSON object found in the text.", None
    
    # Select the JSON object based on the number of matches
    json_data = matches[0] if len(matches) == 1 else matches[-1]
    
    try:
        # Load the JSON data
        data = json.loads(json_data)
        return None, data
    except json.JSONDecodeError as e:
        print('*****'*100)
        print(f"Error decoding JSON: {e}")
        print(response)
        print(json_data)
        print('*****'*100)
        return e, json_data

def handleError(prompt, audio, next_response, model="gpt4o"):

    if model == "gpt4o":
        generate = gpt4o_generate
    elif model == "gemini":
        generate = gemini_generate
    else:
        generate = None

    error, next_response_dict = extract_json(next_response)
    ################################ no json file, regenerate ################################
    cnt = 1
    while error == "No JSON object found in the text." and next_response_dict == None and next_response:
        print(f"No json, repeat generating for the {cnt} time")
        next_response = generate(prompt, audio)
        error, next_response_dict = extract_json(next_response)
        cnt += 1

    ################################ json file incorrect ################################
    cnt = 1
    while error and cnt < 10:
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Directly return the json without explanation.
        """
  
        new_response = generate(prompt, audio=None, temperature=0.3)
        error, next_response_dict = extract_json(new_response)
        cnt += 1
    
    if error:
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Make sure it can be loaded using python (json.loads()). Directly return the json without explanation.
        """

        new_response = generate(prompt, audio=None, temperature=0.3)
        next_response_dict = json.loads(new_response)
    return next_response_dict

def whisper_model(audio_file):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # torch_dtype = torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, 
                                                      torch_dtype=torch_dtype, 
                                                      low_cpu_mem_usage=True)
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    asr_pipeline = pipeline("automatic-speech-recognition", 
                            model=model_id, 
                            tokenizer=processor.tokenizer,
                            feature_extractor=processor.feature_extractor,
                            batch_size=1,
                            chunk_length_s=30, ###### reduce the chuck size ######
                            torch_dtype=torch_dtype,
                            device=device,
                            )
    
    res = asr_pipeline(audio_file, 
                       generate_kwargs={"max_new_tokens": 445,
                                        "language": "english"}, 
                       return_timestamps="word",
                       )
    return res

def process_gspeech(bucket_name, folder_prefix):
    # Initialize a Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)
    
    # Define save path
    # save_path = '/standard/UVA-DSA/NIST EMS Project Data/CognitiveEMS_Datasets/North_Garden/Final'
    save_path = '/scratch/zar8jw/Audio/manual_check_transcripts'

    # List all blobs in the specified folder
    blobs = bucket.list_blobs(prefix=folder_prefix)
    # Iterate through the blobs and print their names
    for blob in blobs:
        mp3_file_path = "gs://" + bucket_name + "/" + blob.name
        if "_encoded_trimmed.mp3" not in mp3_file_path:
            continue

        path_list = blob.name.split('/')
        path_parts, file_name = path_list[1:-1], path_list[-1]
        base_name = '/'.join(path_parts)

        save_file_path = os.path.join(save_path, base_name)

        if os.path.exists(os.path.join(save_file_path, 'google_speech_' + file_name.replace('.mp3', '.json'))):
            print(f"Find {os.path.join(save_file_path, 'google_speech_' + file_name.replace('.mp3', '.json'))}, move to next")
            continue

        res = google_speech(mp3_file_path)


        with open(os.path.join(save_file_path, folder_prefix, 'google_speech_' + file_name.replace('.mp3', '.json')), 'w') as f:
            json.dump(res, f, indent=4)
        
        print(os.path.join(save_file_path, folder_prefix, 'google_speech_' + file_name.replace('.mp3', '.json')))

def google_speech(audio_file: str):
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=audio_file)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    result = operation.result(timeout=450)
    
    res_seg = []
    full_transcript = ""
    for result in result.results:
        cur = {}
        alternative = result.alternatives[0]
        
        cur["Transcript"] = alternative.transcript
        cur["Confidence"] = alternative.confidence
        full_transcript += alternative.transcript + " "

        word_segment = []
        for word_info in alternative.words:
            word_dict = {}
            word = word_info.word
            start_time = word_info.start_time
            end_time = word_info.end_time
            word_dict["word"] = word
            word_dict["start_time"] = start_time.total_seconds()
            word_dict["end_time"] = end_time.total_seconds()
            word_segment.append(word_dict)
        cur["Word Segment"] = word_segment
        res_seg.append(cur)
    res = {
        "Full Transcript": full_transcript.strip(),
        "Segment": res_seg
    }
    return res

def whisper_timestamp(audio_file):
    model_id = "openai/whisper-large-v3"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    audio = whisper.load_audio(audio_file)
    model = whisper.load_model(model_id, device=device)
    result = whisper.transcribe(model, 
                                audio, 
                                language="en",
                                beam_size=5, 
                                best_of=5, 
                                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                # vad=True, 
                                )
    # thresh = 0.1
    

    return result

def whisper_x(audio_file):
    batch_size = 1 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
    model_id = "large-v3"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model_id, device, compute_type=compute_type)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    audio = whisperx.load_audio(audio_file)
    whisper_result = model.transcribe(audio, batch_size=batch_size)
    # print(whisper_result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=whisper_result["language"], device=device)
    whisperx_result = whisperx.align(whisper_result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # print(whisperx_result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # # 3. Assign speaker labels
    # diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

    # # add min/max number of speakers if known
    # diarize_segments = diarize_model(audio)
    # # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    # result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    # print(result["segments"]) # segments are now assigned speaker IDs
    return whisper_result, whisperx_result

def gpt4o_generate(mp3_file_path, word_timestamp=None):
    prompt = f"""There is an audio about Enmergency Medical Service. Transcribe the audio to text. In the transcipts, you must ignore background sounds and are not allowed to include texts like (Sounds of CPR being performed). And assign a word-level timestamp for every word in your transcript based on the provided dictionary. If you find provided timestamp dictionary has errors, you can decide the timestamp based on the audio. Note that every word in your transcripts must has only one word-level timestamp. Return the transcripts in the json format defined as follow. Every word must have a timestamp in the transcript,
    {{
        "transcript": ""
        "words": 
        [
            [word1, start, end],
            ...
        ]
    }}

    Here is provided dictionary:
    {word_timestamp}
    """

    with open(mp3_file_path, "rb") as mp3_file:
        mp3_data = mp3_file.read()
    
    encoded_string = base64.b64encode(mp3_data).decode('utf-8')

    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "mp3"},
        messages=[
            {
                "role": "user",
                "content": [
                    { 
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string,
                            "format": "mp3"
                        }
                    }
                ]
            },
        ]
    )

    return completion.choices[0].message

def whisper1(mp3_file_path):
    audio_file = open(mp3_file_path, "rb")
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )

    words = transcript.words
    texts = transcript.text
    return texts, words

if __name__ == "__main__":

    ################################# google-speech #################################
    # bucket_name = 'ems_ego_exo'
    # folder_prefix = 'ng8/'
    # process_gspeech(bucket_name, folder_prefix)



    # res = google_speech(mp3_file_path)
    # print(res)


    prompt = """Given a recording, you are going to transcribe it to a conversation. The audio records an emergency medical service conversation between first responders (there could be multiple first responders), patient, bystanders(if available). The first responder is taking interventions to save the patient\'s lives. There is also AED (Automated External Defibrillator) machine pads (if available) in the conversation. You must remove irrelevant utterances and make sure the conversation is ONLY about the first responder taking interventions to save a patient. 

    Let’s think step by step,

    Step1: Do a speech recognition to transcribe an audio to transcripts and convert the transcripts to a conversation between first responders (there could be multiple first responders), patient, bystanders(if available). In the conversation, you must also include the utterance made by AED (Automated External Defibrillator) machine pads (if exists). You must identify the utterances of different roles in the conversation (by provided audio). Note that there might be some noises or irrelevant dialogues at the beginning. You must remove irrelevant utterances and make sure the conversation is ONLY about the first responder taking interventions to save a patient.

    Step2: Double check the conversation. Correct any errors you found (e.g.: wrong roles for the utterance, EMS-irrelevant dialogues at the beginning).

    Step3: Organize and return the conversation in the json format defined as follows,
    [
        {
            “Role”:
            “Utterance”:
        },
    ]

    Let’s think Step by Step,

    Audio:
    """

    path = '/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final'

    manual_list = [
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng5/cardiac_arrest/2/audio/GX010318_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng5/cardiac_arrest/3/audio/GX010319_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng3/cardiac_arrest/0/audio/GX010332_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng3/cardiac_arrest/4/audio/GX010336_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng3/cardiac_arrest/5/audio/GX010364_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng8/cardiac_arrest/0/audio/GX010321_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng8/cardiac_arrest/1/audio/GX010322_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng8/cardiac_arrest/2/audio/GX010323_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng8/cardiac_arrest/3/audio/GX010324_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng8/cardiac_arrest/4/audio/GX010325_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/0/audio/GX010387_encoded.mp3"
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/1/audio/GX010388_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/2/audio/GX010389_encoded_trimmed.mp3",
        "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/3/audio/GX010390_encoded_trimmed.mp3",
    ]

    for root, dirs, files in tqdm(os.walk(path)):
        if 'audio' in dirs:
            for file in os.listdir(os.path.join(root, 'audio')):

                if "_encoded_trimmed.mp3" in file:
                    
                    mp3_file_path = os.path.join(root, 'audio', file)

                    # if mp3_file_path not in manual_list:
                    #     continue
                    
                    print(mp3_file_path)
                    
                    ################################ whisper-x #################################
                    # wsp_res, wsp_x_res = whisper_x(mp3_file_path)
                    # json_name = os.path.join(root, 'audio', f"whisper_x_{file.split('.mp3')[0]}.json")
                    # with open(json_name, 'w') as f:
                    #     json.dump(wsp_x_res, f, indent=4)

                    ################################ whisper-timestamp #################################
                    # res = whisper_timestamp(mp3_file_path)
                    # json_name = os.path.join(root, 'audio', f"whisper_timestamp_{file.split('.mp3')[0]}.json")
                    # with open(json_name, 'w') as f:
                    #     json.dump(res, f, indent=4)

                    ################################ whisper-v3-large #################################
                    # json_name = os.path.join(root, 'audio', f"transcripts_{file.split('.mp3')[0]}.json")
                    # if os.path.exists(json_name):
                    #     print(f"find {json_name}, move to next")
                    #     continue
                    # res = whisper_model(mp3_file_path)
                    # with open(json_name, 'w') as f:
                    #     json.dump(res, f, indent=4)


                    ################################ gpt-4o ##################################
                    # json_name = os.path.join(root, 'audio', f"gpt4o_{file.split('.mp3')[0]}.json")
                    # with open(os.path.join(root, "audio", f"whisper_timestamp_{file.split('.mp3')[0]}.json"), 'r') as f:
                    #     whisper_timestamp_res = json.load(f)
                    
                    # word_timestamp = []
                    # segments = whisper_timestamp_res["segments"]
                    # for i in range(len(segments)):
                    #     word_timestamp.extend(segments[i]["words"])

                    # output = gpt4o_generate(mp3_file_path, word_timestamp)
                    # print(output)
                    # print(output.keys())

                    # with open("./test.txt", 'w') as f:
                    #     f.write(output["transcript"])
                    # error, res = extract_json(output["transcript"], pattern=r'\{.*?\}')

                    # with open(json_name, 'w') as f:
                    #     json.dump(res, f, indent=4)

                    
                    ############################### whisper1 ##################################
                    # cur_res = {}
                    # cur_words = []
                    # json_name = os.path.join(root, 'audio', f"whisper1_{file.split('.mp3')[0]}.json")
                    # if json_name in os.listdir(os.path.join(root, 'audio')):
                    #     continue
                    # text, words = whisper1(mp3_file_path)
                    # # print(text)
                    # for i in range(len(words)):
                    #     word = words[i].word
                    #     start = words[i].start
                    #     end = words[i].end
                    #     cur_words.append([word, start, end])
                    # cur_res = {
                    #     "text": text,
                    #     "words": cur_words
                    # }
                    # with open(json_name, 'w') as f:
                    #     json.dump(cur_res, f, indent=4)

                    ################################ gemini #################################
                    json_name = os.path.join(root, 'audio', f"gemini_{file.split('.mp3')[0]}.json")
                    
                    if json_name in os.listdir(os.path.join(root, 'audio')):
                        continue

                    with open(mp3_file_path, "rb") as mp3_file:
                        mp3_data = mp3_file.read()

                    mp3_base64_string = base64.b64encode(mp3_data)
                    audio = Part.from_data(
                        mime_type="audio/mpeg",
                        data=base64.b64decode(mp3_base64_string),
                    )
                    output = gemini_generate(prompt, audio)

                    error, json_data = extract_json(output)
                    if error:
                        json_data = handleError(prompt, audio, output, model="gemini")
                    
                    print(json_name)
                    with open(json_name, 'w') as f:
                        json.dump(json_data, f, indent=4)

                
                    

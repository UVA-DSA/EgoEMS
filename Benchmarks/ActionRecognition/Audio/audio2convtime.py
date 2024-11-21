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
import time

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

def gemini_generate(prompt, audio, max_token=8192, temperature=1, top_p=0.95):

    # print(audio)
    generation_config = {
        "max_output_tokens": max_token,
        "temperature": temperature,
        "top_p": top_p,
    }

    vertexai.init(project="", location="us-central1")
    model = GenerativeModel(
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

    if "Please provide the audio recording." in response or "no audio" in response:
        return None, []

    # Regular expression pattern to match all JSON objects
    # pattern = r'\[.*\]'
    if not pattern:
        pattern = r'\[\s*{\s*"role":\s*".+?",\s*"utterance":\s*".+?"\s*}(?:,\s*{\s*"role":\s*".+?",\s*"utterance":\s*".+?"\s*})*\s*\]'


    # Find all matches in the text
    matches = re.findall(pattern, response, re.DOTALL)
    # print('+++++'*100)
    # print(matches)
    # print(response)
    # print('+++++'*100)

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
    # ################################ no json file, regenerate ################################
    # cnt = 1
    # while error == "No JSON object found in the text." and next_response_dict == None and next_response:
    #     print(f"No json, repeat generating for the {cnt} time")
    #     next_response = generate(prompt, audio)
    #     error, next_response_dict = extract_json(next_response)
    #     cnt += 1

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


def segment_process():
    """segment the audio sentence by sentence, transcribe each segmented audio"""

    # prompt = """Given a recording, you are going to transcribe it to a conversation. The audio records an emergency medical service conversation between first responders (there could be multiple first responders), patient, bystanders(if available). The first responder is taking interventions to save the patient\'s lives. There is also AED (Automated External Defibrillator) machine pads (if available) in the conversation. You must remove irrelevant utterances and make sure the conversation is ONLY about the first responder taking interventions to save a patient. If there is no audio, return 
    # [
    #     {
    #         "role": "",
    #         “utterance”: ""
    #     }
    # ]

    # Let’s think step by step,

    # Step1: Do a speech recognition to transcribe an audio to transcripts and convert the transcripts to a conversation between first responders (there could be multiple first responders), patient, bystanders(if available). In the conversation, you must also include the utterance made by AED (Automated External Defibrillator) machine pads (if exists). You must identify the utterances of different roles in the conversation (by provided audio). Note that there might be some noises or irrelevant dialogues at the beginning. You must remove irrelevant utterances and make sure the conversation is ONLY about the first responder taking interventions to save a patient. If there is no audio, return the utterance as "".

    # Step2: Double check the conversation. Correct any errors you found (e.g.: wrong roles for the utterance, EMS-irrelevant dialogues at the beginning).

    # Step3: Organize and return the conversation in the json format defined as follows. If there is no audio, set the value of utterance and role as empty string "".
    # [
    #     {
    #         "role": "",
    #         “utterance”: ""
    #     }
    # ]

    # Let’s think Step by Step,

    # Audio:
    # """

    prompt = """
    Given a recording, you are going to transcribe it to a text. The audio has an emergency medical service recording between first responders (there could be multiple first responders), patient, bystanders(if available). The first responder is taking interventions to save the patient\'s lives. There is also AED (Automated External Defibrillator) machine pads (if available) in the recording. You must remove irrelevant utterances and make sure the transcript is ONLY about the first responder taking interventions to save a patient. If there is no audio, return 
    [
        "None"
    ]

    Let's think step by step,
    
    Step1: Do a speech recognition to transcribe an audio to transcripts. Note that there might be some noises or irrelevant dialogues. You must remove irrelevant utterances and make sure the transcripts is ONLY about the first responder taking interventions to save a patient. If there is no audio, return the transcript as "".

    Step2: Double check the transcript. Correct any errors you found (e.g.: EMS-irrelevant dialogues at the beginning), you are not allowed to include text that never occurs in the audio.

    Step3: Organize and return the conversation in the json format defined as follows. If there is no audio, return string "None" in the list.
    [
        "transcript"
    ]
    """


    # audio_path = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng5/cardiac_arrest/2/audio/GX010318_encoded_trimmed.mp3"
    # timestamp_path = "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/ng5/cardiac_arrest/2/audio/whisper_timestamp_GX010318_encoded_trimmed.json"


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
        # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/0/audio/GX010387_encoded.mp3"
        # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/1/audio/GX010388_encoded_trimmed.mp3",
        # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/2/audio/GX010389_encoded_trimmed.mp3",
        # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/3/audio/GX010390_encoded_trimmed.mp3",
    ]

    for audio_path in tqdm(manual_list):
        file_name = os.path.basename(audio_path).split(".mp3")[0].strip()
        timestamp_path = os.path.join(os.path.dirname(audio_path), 'whisper_timestamp_'+ file_name + '.json')


        with open(timestamp_path, 'r') as f:
            whisper_timestamp = json.load(f)
        
        path = os.path.join("/scratch/zar8jw/Audio/segments_v1", file_name)
        if not os.path.exists(path):
            os.makedirs(path)
        
        if os.path.exists(os.path.join(path, "transcript.json")):
            continue

        audio = AudioSegment.from_file(audio_path)

        segments = whisper_timestamp["segments"]

        timestamped_segment = []
        
        for i in range(len(segments)):

            id = segments[i]["id"]
            text = segments[i]["text"]
            start_t = segments[i]["start"] * 1000
            end_t = segments[i]["end"] * 1000
            print(file_name, id)
            if os.path.exists(os.path.join(path, f"gemini_output{id}.json")):
                cur_transcipt = ""

                with open(os.path.join(path, f"gemini_output{id}.json"), 'r') as f:
                    prev_output = json.load(f)

                error, jsondata = extract_json(prev_output, pattern=r'\[.*\]')

                # for j in range(len(jsondata)):
                #     if jsondata[j]["utterance"] != None:
                #         cur_transcipt += jsondata[j]["utterance"] + " "

                if jsondata:
                    cur_transcipt = jsondata[0]

                if cur_transcipt.strip() not in ["None", "none", ""]:
                    cur_dct = {}
                    cur_dct["transcript"] = cur_transcipt.strip()
                    cur_dct["start time"] = segments[i]["start"]
                    cur_dct["end time"] = segments[i]["end"]
                    timestamped_segment.append(cur_dct)
            else:
                audio_segment = audio[start_t:end_t]
                audio_segment.export(os.path.join(path, f"audio_segment_{id}.mp3"), format="mp3")

                # with open(os.path.join(path, f"audio_segment_{id}.json"), "w") as f:
                #     json.dump(wordlevel_timestamp, f, indent=4)

                with open(os.path.join(path, f"audio_segment_{id}.mp3"), "rb") as mp3_file:
                    cur_mp3_data = mp3_file.read()

                mp3_base64_string = base64.b64encode(cur_mp3_data)
                cur_audio = Part.from_data(
                    mime_type="audio/mpeg",
                    data=base64.b64decode(mp3_base64_string),
                )
                try:
                    output = gemini_generate(prompt, cur_audio)
                except:
                    print("sleep for 120s and them resume calling gemini...")
                    time.sleep(120)
                    output = gemini_generate(prompt, cur_audio)

                with open(os.path.join(path, f"gemini_output{id}.json"), 'w') as f:
                    json.dump(output, f, indent=4)

                error, jsondata = extract_json(output, pattern=r'\[.*\]') #pattern=r'\[.*\]'
                if error:
                    jsondata = handleError(prompt, cur_audio, output, model="gemini")
                
                # print(jsondata)
                # cur_transcipt = ""
                # for j in range(len(jsondata)):
                #     if jsondata[j]["utterance"] != None:
                #         cur_transcipt += jsondata[j]["utterance"] + " "
                
                if jsondata:
                    cur_transcipt = jsondata[0]

                if cur_transcipt not in ["None", "none", ""]:
                    cur_dct = {}
                    cur_dct["transcript"] = cur_transcipt.strip()
                    cur_dct["start time"] = segments[i]["start"]
                    cur_dct["end time"] = segments[i]["end"]
                    timestamped_segment.append(cur_dct)
        
        with open(os.path.join(path, "transcript.json"), 'w') as f:
            json.dump(timestamped_segment, f, indent=4)

def align_whisper_gemini_output():
    # """whisper-1: get the time-level stamp
    # gemini: get the transcripts"""

    
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

    # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/0/audio/GX010387_encoded.mp3"
    # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/1/audio/GX010388_encoded_trimmed.mp3",
    # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/2/audio/GX010389_encoded_trimmed.mp3",
    # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/3/audio/GX010390_encoded_trimmed.mp3",
    ]

    for audio_path in tqdm(manual_list):
        file_name = os.path.basename(audio_path).split(".mp3")[0].strip()
        print(file_name)
        # timestamp_path = os.path.join(os.path.dirname(audio_path), 'whisper_timestamp_'+ file_name + '.json')
        timestamp_path = os.path.join(os.path.dirname(audio_path), 'whisper1_'+ file_name + '.json')
        gemini_text_path = os.path.join(os.path.dirname(audio_path), 'gemini_'+ file_name + '.json')

        with open(timestamp_path, 'r') as f:
            whisper_timestamp = json.load(f)
        
        word2time = whisper_timestamp["words"]

        # word2time = []
        # for j in range(len(whisper_timestamp["segments"])):
        #     for k in range(len(whisper_timestamp["segments"][j]["words"])):

        #         word = whisper_timestamp["segments"][j]["words"][k]["text"]
        #         start_t = whisper_timestamp["segments"][j]["words"][k]["start"]
        #         end_t = whisper_timestamp["segments"][j]["words"][k]["end"]
        #         word2time.append([word, start_t, end_t])
        
        with open(gemini_text_path, 'r') as f:
            gemini_text = json.load(f)

        transcript = ""
        for i in range(len(gemini_text)):
            transcript += gemini_text[i]["Utterance"] + " "
        
        prompt = f"""Assign every word in the transcript a timestamp based on the provided word-to-time mapping. Remember every word in the transcript must appear in the your result, and do not add additional words to the transcript. The returned transcript must be the exact the same with provided transcript. If you do not find the timestamp for the word from the word-to-time mapping. Do an approximated time for the word. Return the result in the json format defined as follows,
        {{
            "transcript": "",
            "word-level timestamp":
                ["word1", start time, end time]
        }}

        Transcript:
        {transcript}

        Word-to-time Mapping:
        {word2time}
        """
        
        path = os.path.join("/scratch/zar8jw/Audio/word_time", file_name)
        if not os.path.exists(path):
            os.makedirs(path)
        
        if "pred.json" in os.listdir(path):
            print(f"prediction already in {file_name}")
            continue

        try:
            output = gemini_generate(prompt, audio=None)
        except:
            print("sleep for 120s and them resume calling gemini...")
            time.sleep(120)
            output = gemini_generate(prompt, audio=None)
        
        with open(os.path.join(path, "output.json"), 'w') as f:
            json.dump(output, f, indent=4)

        error, jsondata = extract_json(output, pattern=r'\{.*?\}')
        if error:
            jsondata = handleError(prompt, audio=None, next_response=output, model="gemini")
        
        with open(os.path.join(path, "pred.json"), 'w') as f:
            json.dump(jsondata, f, indent=4)




if __name__ == "__main__":
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

        # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/0/audio/GX010387_encoded.mp3"
        # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/1/audio/GX010388_encoded_trimmed.mp3",
        # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/2/audio/GX010389_encoded_trimmed.mp3",
        # "/standard/UVA-DSA/NIST EMS Project Data/EgoExoEMS_CVPR2025/Dataset/Final/wa1/cardiac_scenario/3/audio/GX010390_encoded_trimmed.mp3",
    ]

    for audio_path in tqdm(manual_list):
        file_name = os.path.basename(audio_path).split(".mp3")[0].strip()
        print(file_name)

        path = os.path.join("/scratch/zar8jw/Audio/accumulate_segments", file_name)
        if not os.path.exists(path):
            os.makedirs(path)

        if os.path.exists(os.path.join(path, "transcript.json")):
            continue

        timestamp_path = os.path.join(os.path.dirname(audio_path), 'whisper_x_'+ file_name + '.json')
        gemini_text_path = os.path.join(os.path.dirname(audio_path), 'gemini_'+ file_name + '.json')

        with open(timestamp_path, 'r') as f:
            whisper_timestamp = json.load(f)
        
        segments = whisper_timestamp["segments"]
        timestamped_segment = []
        audio = AudioSegment.from_file(audio_path)
        
        history = ""
        for i in range(len(segments)):
            # text = segments[i]["text"]

            if os.path.exists(os.path.join(path, f"gemini_output{i}.json")):
                print(f"found {file_name} gemini_output{i}...")
                cur_transcipt = ""

                with open(os.path.join(path, f"gemini_output{i}.json"), 'r') as f:
                    prev_output = json.load(f)

                error, jsondata = extract_json(prev_output, pattern=r'\[.*\]')

                # for j in range(len(jsondata)):
                #     if jsondata[j]["utterance"] != None:
                #         cur_transcipt += jsondata[j]["utterance"] + " "

                if jsondata:
                    cur_transcipt = jsondata[0]

                if cur_transcipt.strip() not in ["None", "none", ""]:
                    cur_dct = {}
                    cur_dct["transcript"] = cur_transcipt.strip()
                    cur_dct["start time"] = segments[i]["start"]
                    cur_dct["end time"] = segments[i]["end"]
                    history += cur_transcipt + " "
                    timestamped_segment.append(cur_dct)
            else:
                start_t = segments[i]["start"] * 1000
                end_t = segments[i]["end"] * 1000

                if start_t > end_t:
                    start_t = segments[i]["end"] * 1000
                    end_t = segments[i]["start"] * 1000

                prompt = f"""
                Given a recording, you are going to transcribe it to a text. The audio has an emergency medical service recording between first responders (there could be multiple first responders), patient, bystanders(if available). The first responder is taking interventions to save the patient\'s lives. There is also AED (Automated External Defibrillator) machine pads (if available) in the recording. You must remove irrelevant utterances and make sure the transcript is ONLY about the first responder taking interventions to save a patient. If there is no audio, return 
                [
                    "None"
                ]

                Let's think step by step,
                
                Step1: Do a speech recognition to transcribe an audio to transcripts. Note that there might be some noises or irrelevant dialogues. You must remove irrelevant utterances and make sure the transcripts is ONLY about the first responder taking interventions to save a patient. If there is no audio, return the transcript as "".

                Step2: Based on the provided history transcripts, further refine the current transcript. Correct any errors you find for the current transcript. However, you are not allowed to add provided history transcripts to the current transcript.

                Step3: Organize and return ONLY the current transcript in the json format defined as follows. If there is no audio, return string "None" in the list. Do not include the previous history in your return result,
                [
                    "transcript"
                ]

                History transcript: 
                {history}

                Current Audio: 
                """
                
                # print(i, start_t, end_t)
                audio_segment = audio[start_t:end_t]
                audio_segment.export(os.path.join(path, f"audio_segment_{i}.mp3"), format="mp3")


                with open(os.path.join(path, f"audio_segment_{i}.mp3"), "rb") as mp3_file:
                    cur_mp3_data = mp3_file.read()

                mp3_base64_string = base64.b64encode(cur_mp3_data)
                cur_audio = Part.from_data(
                    mime_type="audio/mpeg",
                    data=base64.b64decode(mp3_base64_string),
                )

                try:
                    time.sleep(10)
                    output = gemini_generate(prompt, cur_audio)
                except:
                    print("sleep for 60s and them resume calling gemini...")
                    time.sleep(120)
                    output = gemini_generate(prompt, cur_audio)
                
                with open(os.path.join(path, f"gemini_output{i}.json"), 'w') as f:
                    json.dump(output, f, indent=4)

                error, jsondata = extract_json(output, pattern=r'\[.*\]') #pattern=r'\[.*\]'
                if error:
                    jsondata = handleError(prompt, cur_audio, output, model="gemini")
                
                if jsondata:
                    cur_transcipt = jsondata[0]

                if cur_transcipt not in ["None", "none", ""]:
                    cur_dct = {}
                    cur_dct["transcript"] = cur_transcipt.strip()
                    cur_dct["start time"] = segments[i]["start"]
                    cur_dct["end time"] = segments[i]["end"]
                    history += cur_transcipt.strip() + " "
                    timestamped_segment.append(cur_dct)
        
        with open(os.path.join(path, "transcript.json"), 'w') as f:
            json.dump(timestamped_segment, f, indent=4)


    




    


    



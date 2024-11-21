import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, HarmBlockThreshold
import json
import re
from collections import defaultdict
# from moviepy.editor import VideoFileClip
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import itertools
import string
from data_loader import EgoExoEMSDataset, collate_fn, transform
import argparse
from openai import OpenAI
import torchaudio
import whisper

# safety_settings = [
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
#         threshold=HarmBlockThreshold.BLOCK_NONE
#     ),
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
#         threshold=HarmBlockThreshold.BLOCK_NONE
#     ),
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
#         threshold=HarmBlockThreshold.BLOCK_NONE
#     ),
#     SafetySetting(
#         category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
#         threshold=HarmBlockThreshold.BLOCK_NONE
#     ),
# ]
# vertexai.init(project="", location="us-central1")
# model = GenerativeModel("gemini-1.5-pro-001",)


OPENAI_API_KEY=''
client = OpenAI(api_key=OPENAI_API_KEY)

whisper_model = whisper.load_model("large-v3")

keysteps = {
    "approach_patient": "Approach the patient",
    "check_responsiveness": "Check for responsiveness",
    "check_pulse": "Check patient's pulse",
    "check_breathing": "Check if patient is breathing",
    "chest_compressions": "Perform chest compressions",
    "request_aed": "Request an AED",
    "request_assistance": "Request additional assistance",
    "turn_on_aed": "Turn on the AED",
    "attach_defib_pads": "Attach defibrillator pads",
    "clear_for_analysis": "Clear for analysis",
    "clear_for_shock": "Clear for shock",
    "administer_shock_aed": "Administer shock using AED",
    "open_airway": "Open patient's airway",
    "place_bvm": "Place bag valve mask (BVM)",
    "ventilate_patient": "Ventilate patient",
    "no_action": "No action",
    "assess_patient": "Assess the patient",
    "explain_procedure": "Explain the ECG procedure to the patient",
    "shave_patient": "Shave/Cleanse the patient for ECG",
    "place_left_arm_lead": "Place the lead on left arm for ECG",
    "place_right_arm_lead": "Place the lead on right arm for ECG",
    "place_left_leg_lead": "Place the lead on left leg for ECG",
    "place_right_leg_lead": "Place the lead on right leg for ECG",
    "place_v1_lead": "Place the V1 lead on the patient",
    "place_v2_lead": "Place the V2 lead on the patient",
    "place_v3_lead": "Place the V3 lead on the patient",
    "place_v4_lead": "Place the V4 lead on the patient",
    "place_v5_lead": "Place the V5 lead on the patient",
    "place_v6_lead": "Place the V6 lead on the patient",
    "ask_patient_age_sex": "Ask the age and or sex of the patient",
    "request_patient_to_not_move": "Request the patient to not move",
    "turn_on_ecg": "Turn on the ECG machine",
    "connect_leads_to_ecg": "Verify all ECG leads are properly connected",
    "obtain_ecg_recording": "Obtain the ECG recording",
    "interpret_and_report": "Interpret the ECG and report findings"
}


def none_or_int(value):
    if value == 'None':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid int value: {value}")

def gemini_generate(prompt, audio=None, max_token=8192, temperature=1, top_p=0.95):
    generation_config = {
        "max_output_tokens": max_token,
        "temperature": temperature,
        "top_p": top_p,
    }

    if audio != None:
        input = [prompt, audio]
    else:
        input = prompt
    
    time.sleep(10)
    try:
        responses = model.generate_content(
            input,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )
    except:
        time.sleep(65)
        responses = model.generate_content(
            input,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )
    
    if not responses.text:
        responses = model.generate_content(
            input,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )


    return responses.text


def gpt4_generate(prompt, audio=None, max_token=16384, temperature=1, top_p=0.95):
    completion = client.chat.completions.create(model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a professional EMS first responder, skilled in EMS concepts and medical knowledge"},
            {"role": "user", "content": f"{prompt}"},
        ],
        max_tokens=max_token,
        temperature=temperature,
        top_p=top_p
    )

    resp = str(completion.choices[0].message.content)
    return resp

def extract_json(response, pattern=r'\[.*\]'):

    if type(response) == json:
        return None, response

    # Regular expression pattern to match all JSON objects
    # pattern = r'\[.*\]'
    # pattern = r'\{[^{}]*\}'

    # Find all matches in the text
    matches = re.findall(pattern, response, re.DOTALL)
    # print('+++++'*100)
    # print(matches)
    # print(response)
    # print('+++++'*100)

    if not matches:
        # print('====='*100)
        # print("No JSON object found in the text.")
        # print(response)
        # print('====='*100)
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

def handleError(prompt, next_response, model):

    if model == 'gemini':
        genetate = gemini_generate
    elif model == 'gpt4':
        generate = gpt4_generate
    else:
        generate = None
        raise Exception("check model")

    error, next_response_dict = extract_json(next_response)
    ################################ no json file, regenerate ################################
    cnt = 1
    while error == "No JSON object found in the text." and next_response_dict == None and next_response:
        print(f"No json, repeat generating for the {cnt} time")
        next_response = generate(prompt)
        error, next_response_dict = extract_json(next_response, pattern='\{.*?\}')
        cnt += 1

    ################################ json file incorrect ################################
    cnt = 1
    while error and cnt < 10:
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Directly return the json without explanation.
        """
  
        new_response = generate(prompt, temperature=0.3)
        error, next_response_dict = extract_json(new_response, pattern='\{.*?\}')
        cnt += 1
    
    if error:
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Make sure it can be loaded using python (json.loads()). Directly return the json without explanation.
        """

        new_response = generate(prompt, temperature=0.3)
        next_response_dict = json.loads(new_response)
    return next_response_dict


def classify_keystep(text, model):
    prompt = f"""Here is a text about EMS. Identify the keystep (action first responder is taking) in the text. There is only one keystep in the text. Note that the clear before "shock advised" is clear_for_analysis, and the clear after "shock advised" is clear_for_shock.
    
    There are all the keysteps, in the dictionary, the key is the label, the value is the description.
    {keysteps}


    Let's think step by step,
    Step 1: Make a classification on the text.
        Here is serveral example: 
        Example 1: 
        TEXT: Assessing patient.
        LABEL: approach_patient

        Example 2: 
        TEXT: Hello can you hear me? 
        LABEL: check_responsiveness

        Example 3:
        TEXT: No pulse.
        LABEL: check_pulse

        Example 4:
        TEXT: no breaths.
        LABEL: check_breathing
        
        Example 3:
        TEXT: Doing CPR
        LABEL: chest_compressions
        
        Example 4:
        TEXT: Call 911
        LABEL: request_assistance
        
        Example 5:
        TEXT: Applying defib, Adult patient If the patient is a child, press the child button. Press pads firmly on skin. Press the pads as shown in the picture. 
        LABEL: request_aed
        
        Example 6:
        TEXT: Do not touch the patient, everyone clear, analyzing heart rhythm, shock advised
        LABEL: clear_for_analysis
        
        Example 9:
        TEXT: Do not touch the patient, everyone clear. 
        LABEL: clear_for_shock
        
        Example 10:
        TEXT: Press the flashing shocking button, shock delivered.
        LABEL: administer_shock_aed
        
        Example 11:
        TEXT: Begin CPR 10 20 30. 
        LABEL: chest_compressions
        
        Example 12:
        TEXT: give two breaths. 
        LABEL: place_bvm, ventilate_patient
        
        Example 13:
        TEXT: Resume CPR, ten, twenty, thirty
        LABEL: chest_compressions

    Step 2: Return your result in the defined json format,
    {{
        "text": "",
        "label": ""
    }}


    Now is the real text,
    Text:
    {text}

    Let's think step by step,
    """

    if model == 'gemini':
        genetate = gemini_generate
    elif model == 'gpt4':
        generate = gpt4_generate
    else:
        generate = None

    response = generate(prompt)
    return prompt, response


def audio2text(audio):
    prompt = f"""
    Here is an audio about EMS. Transcribe the audio to text. If there is nothing in the audio. please return text as "". Return the text in json format defined as follows,
    {{
        "text": ""
    }}


    Audio: 
    {audio}
    """
    response = generate(prompt, audio)
    return prompt, response


def dataloader(args):
    alldata_dataset = EgoExoEMSDataset(annotation_file=args.train_annotation_path,
                                     data_base_path='',
                                    fps=args.fps, 
                                    frames_per_clip=args.observation_window, 
                                    transform=transform, 
                                    data_types=args.modality)
    alldata_loader = DataLoader(alldata_dataset, batch_size=args.batch_size, shuffle=False)
    return alldata_loader


def llm_process(data_loader, datatype="train", task="classification", model="gpt-4"):
    res = {}
    if not os.path.exists(f"./logs/classification/{datatype}"):
        os.makedirs(f"./logs/classification/{datatype}")

    pbar = tqdm(colour="blue", desc=f"{datatype} dataLoader", total=len(data_loader), dynamic_ncols=True)
    # conv = ""
    # subject_list = []
    # trial_list = []
    res = {}
    for i, batch in enumerate(data_loader):
        if f"output_{i}.mp3" in os.listdir(f"./logs/{datatype}/"):
            continue
        label = batch['keystep_label']
        subject_id = batch['subject_id'][0]
        trial_id = batch['trial_id'][0]

        if subject_id not in res:
            res[subject_id] = {}
        if trial_id not in res[subject_id]:
            res[subject_id][trial_id] = defaultdict(list)


        audio_tensor = batch['audio'].squeeze(0).T
        torchaudio.save(f"./logs/classification/{datatype}/output_{i}.mp3", audio_tensor, 48000)

        # Open the audio file in binary mode and read its content
        whisper_output = whisper_model.transcribe(f"./logs/classification/{datatype}/output_{i}.mp3", language='en')
        text = whisper_output["text"]
        # try:
        #     with open(f"./logs/{datatype}/output_{i}.mp3", "rb") as audio_file:
        #         audio_data = audio_file.read()
        #     base64_encoded_audio = base64.b64encode(audio_data)
        #     base64_string = base64_encoded_audio.decode('utf-8')
        #     audio = Part.from_data(
        #         mime_type="audio/mpeg",
        #         data=base64.b64decode(base64_string)
        #     )
        #     prompt, response = audio2text(audio)
        #     error, res_json = extract_json(response, pattern='\{.*?\}')
        #     if error:
        #         res_json = handleError(prompt, response, audio)
        #     text = res_json["text"]
        # except:
        #     whisper_output = whisper_model.transcribe(f"./logs/{datatype}/output_{i}.mp3", language='en')
        #     text = whisper_output["text"]

        print(subject_id, trial_id, text)

        prompt, response = classify_keystep(text, model)
        error, res_json = extract_json(response, pattern='\{.*?\}')
        if error:
            res_json = handleError(prompt, response, model)
        
        res[subject_id][trial_id]["index"].append(i)        
        res[subject_id][trial_id]["pred"].append(res_json["label"])        
        res[subject_id][trial_id]["label"].append(label[0])

        if i%10 == 0:
            with open(f"./results/{datatype}_{task}.json", 'w') as f:
                json.dump(res, f, indent=4)

    return res


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training script for recognition")
    parser.add_argument('--data_split', type=str, default="test")
    parser.add_argument('--train_annotation_path', type=str, default="/scratch/zar8jw/EgoExoEMS/Annotations/splits/trials/test_split_classification.json")
    parser.add_argument('--task', type=str, default="classification")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--observation_window', type=none_or_int, default=None)
    parser.add_argument('--modality', type=list, default=["audio"])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model', type=str, default="gpt4")

    args = parser.parse_args()

    print(args)
    all_loader = dataloader(args)
    result = llm_process(all_loader, datatype=args.data_split, task=args.task, model=args.model)

    with open(f"./results/{args.data_split}_{args.task}.json", 'w') as f:
        json.dump(result, f, indent=4)
    
    print("finished")




                        




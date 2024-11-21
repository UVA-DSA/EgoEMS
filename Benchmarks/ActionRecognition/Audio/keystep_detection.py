import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
import json
import re
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import os
import time
import itertools
import string

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
]

vertexai.init(project="erudite-vent-435920-t9", location="us-central1")
model = GenerativeModel("gemini-1.5-pro-001",)


def generate(prompt, max_token=8192, temperature=1, top_p=0.95):
    generation_config = {
        "max_output_tokens": max_token,
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        responses = model.generate_content(
            prompt,
            generation_config=generation_config,
            # safety_settings=safety_settings,
            stream=False,
        )
    except:
        time.sleep(65)
        responses = model.generate_content(
            prompt,
            generation_config=generation_config,
            # safety_settings=safety_settings,
            stream=False,
        )


    return responses.text


def checkInterventions(utterance):
    prompt = f"""
    You are a professional First Responder in Emergency Medical Service. Classify whether each utterance in the conversation includes the following interventions: "CPR", "Defib", "BVM", "None", and explain your classification. If The utterance includes multiple interventions, return all interventions in a list. You must do the intervention classification for each utterance based on the conversation context rather than one single utterance. "Give two breath" is the intervention "BVM".
        - "CPR" stands for cardiopulmonary resuscitation, which is a life-saving procedure performed when someone's heart or breathing has stopped. 
        - "Defib" stands for defibrillator, which is a device that delivers an electric shock to the heart to restore a normal heartbeat. 
        - "BVM" stands for bag-valve-mask, a handheld device used to provide oxygen and ventilation to patients who are having trouble breathing.
        - "None" stands for the this text does not contain any interventions above.
    Return the results in the defined json format as follows,
    [
        {{
            "Utterance":
            "Intervention":
            "Explanation":
        }}
    ]

    Conversation: 
    {utterance}
    """

    response = generate(prompt)
    return prompt, response

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

def handleError(prompt, next_response):
    error, next_response_dict = extract_json(next_response)
    ################################ no json file, regenerate ################################
    cnt = 1
    while error == "No JSON object found in the text." and next_response_dict == None and next_response:
        print(f"No json, repeat generating for the {cnt} time")
        next_response = generate(prompt)
        error, next_response_dict = extract_json(next_response)
        cnt += 1

    ################################ json file incorrect ################################
    cnt = 1
    while error and cnt < 10:
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Directly return the json without explanation.
        """
  
        new_response = generate(prompt, temperature=0.3)
        error, next_response_dict = extract_json(new_response)
        cnt += 1
    
    if error:
        prompt = f"""There is an Error decoding JSON: {error} in the following json data
        {next_response_dict}, Can you fix the error and return the correct json format. Make sure it can be loaded using python (json.loads()). Directly return the json without explanation.
        """

        new_response = generate(prompt, temperature=0.3)
        next_response_dict = json.loads(new_response)
    return next_response_dict

def align_gemini_whisper(conv_gemini, conv_whisper_x):
    prompt = f"""There is a conversation about EMS. Find the right timestamp for each keystep. The timestamp must copied from the provided time-stamped conversation. If lack timestamps in the provided data, return start time, end time as null.

    Return results as follows,
    {{
        "Utterance": "",
        "KeyStep": "",
        "Explanation": "",
        "start time": "",
        "end time": ""
    }}

    Conversation includes keystep: {conv_gemini}
    Conversation include word-level timestamp: {conv_whisper_x}
    """

def word_level_timestamp(conv):
    prompt = f"""Here is a conversation about EMS. The conversation is segmented as word-level with start/end timestamp,
    Identify the keysteps (what First Responder is doing) from the word-level conversation and the timestamp (start_time, end_time) for each keystep. Include all words of keystep based on timestamp.
    You must return all utterances in the conversation. You must exactly rely on the provided word-level conversation to generate the result and can not generate anything beyond the provided content. If there are multiple keysteps in one utterance, return all keysteps in a list.
    
    Return the result in the defined json format as follows,
    {{
	    "Utterance": "",
	    "Keystep": ["Keystep1", "Keystep2", ...],
	    "Keystep corresponding words": [[word1, word2, ...], [word3, word4, ...], ...]
	    "Keystep timestamp": [[start, end], [start, end], ...],
    }}
    
    Keystep: ["approach_patient", "check_responsiveness", "check_pulse", "check_breathing", "chest_compressions", "request_aed", "request_assistance", "turn_on_aed", "attach_defib_pads", "clear_for_analysis", "clear_for_shock", "administer_shock_aed", "open_airway", "place_bvm", "ventilate_patient", "none"]
    
    Conversation:
    {conv}"""

def detectKeystep(conv):
    prompt = f"""Here is a conversation about EMS. Identify the keystep (action first responder is taking) and its corresponding words. Note that the clear before "shock advised" is clear_for_analysis, and the clear after "shock advised" is clear_for_shock.
    
    Here is one example: 
    Conversation: 
    Assessing patient. Hello can you hear me? No pulse, no breaths, Doing CPR, Call 911, Applying defib, Adult patient If the patient is a child, press the child button. Press pads firmly on skin. Press the pads as shown in the picture. Do not touch the patient, everyone clear, analyzing heart rhythm, shock advised. Do not touch the patient, everyone clear. Press the flashing shocking button, shock delivered. Begin CPR 10 20 30. give two breaths. Resume CPR, ten, twenty, thirty.

    Let's think step by step,
    Step 1: Token classification: classify every token in the conversation in following classes,  ["approach_patient", "check_responsiveness", "check_pulse", "check_breathing", "chest_compressions", "request_aed", "request_assistance", "turn_on_aed", "attach_defib_pads", "clear_for_analysis", "clear_for_shock", "administer_shock_aed", "open_airway", "place_bvm", "ventilate_patient", "none"]. If you think the token contains multiple keysteps, you should assign all keysteps to that token. You are not allowed to use any json format in Step 1. If the text is too long, you don't need to do an output for Step 1.
    Assessing: approach_patient
    patient: approach_patient
    Hello: check_responsiveness
    can: check_responsiveness
    you: check_responsiveness
    hear: check_responsiveness
    me: check_responsiveness
    No: check_pulse
    pulse: check_pulse
    no: check_breathing
    breaths: check_breathing
    Doing: chest_compressions
    CPR: chest_compressions
    Call: request_assistance
    911: request_assistance
    Applying: turn_on_aed, request_aed
    defib: turn_on_aed, request_aed
    Adult: none
    patient: none
    If: none
    the: none
    patient: none
    is: none
    a: none
    child: none
    press: none
    the: none
    child: none
    button: none
    Press: attach_defib_pads
    pads: attach_defib_pads
    firmly: attach_defib_pads
    on: attach_defib_pads
    skin: attach_defib_pads
    Press: attach_defib_pads
    the: attach_defib_pads
    pads: attach_defib_pads
    as: attach_defib_pads
    shown: attach_defib_pads
    in: attach_defib_pads
    the: attach_defib_pads
    picture: attach_defib_pads
    Do: clear_for_analysis
    not: clear_for_analysis
    touch: clear_for_analysis
    the: clear_for_analysis
    patient: clear_for_analysis
    everyone: clear_for_analysis
    clear: clear_for_analysis
    analyzing: clear_for_analysis
    heart: clear_for_analysis
    rhythm: clear_for_analysis
    shock: none
    advised: none
    Do: clear_for_shock
    not: clear_for_shock
    touch: clear_for_shock
    the: clear_for_shock
    patient: clear_for_shock
    everyone: clear_for_shock
    clear: clear_for_shock
    Press: administer_shock_aed
    the: administer_shock_aed
    flashing: administer_shock_aed
    shocking: administer_shock_aed
    button: administer_shock_aed
    shock: none
    delivered: none
    Begin: chest_compressions
    CPR: chest_compressions
    10: chest_compressions
    20: chest_compressions
    30: chest_compressions
    give: place_bvm, ventilate_patient
    two: place_bvm, ventilate_patient
    breaths: place_bvm, ventilate_patient
    Resume: chest_compressions
    CPR: chest_compressions
    ten: chest_compressions
    twenty: chest_compressions
    thirty: chest_compressions



    Step 2: Span Detection: concatenate tokens with the same labels into utterances, and identify all keysteps for the utterance. Every Keystep must has its corresponding words. Return the results in the defined json format as follows, in the json file EVERY Keystep must have a list with corresponding words. The words for keysteps can fully overlapped (one sentence can have multiple keysteps). However, the words for keystep are not allowed partially overlapped. For example, "Keystep1": ["word1", "word2"] and "Keystep2": ["word1", "word2"] are valid, but "Keystep1": ["word1", "word2"] and "Keystep2": ["word2", "word3", "word4"] are not.
    [
        {{
            "Utterance": "",
            "Keystep": {{
                "Keystep1": [word1, word2, ...],
                "Keystep2": [word1, word2, ...],
                "Keystep3": [word3, word4, ...]
            }}
        }}
    ]

[
  {{
      "Utterance": "Assessing patient",
      "Keystep": {{
        "approach_patient": ["Assessing", "patient"]
        }}
  }},
  {{
      "Utterance": "Hello can you hear me",
      "Keystep": {{
        "check_responsiveness": ["Hello", "can", "you", "hear", "me"]
        }}
  }},
  {{
      "Utterance": "No Pulse",
      "Keystep": {{
        "check_pulse": ["No", "Pulse"]
        }}
  }},
  {{
      "Utterance": "no breaths",
      "Keystep": {{
        "check_breathing": ["no", "breaths"]
        }}
  }},
  {{
      "Utterance": "Doing CPR",
      "Keystep": {{
        "chest_compressions": ["Doing", "CPR"]
        }}
  }},
    {{
      "Utterance": "Call 911",
      "Keystep": {{
        "request_assistance": ["Call", "911"], 
        }},
  }},
  {{
      "Utterance": "Applying defib",
      "Keystep": {{
        "turn_on_aed": ["Applying", "defib"], 
        "attach_defib_pads": ["Applying", "defib"]
        }},
  }},
  {{
      "Utterance": "Adult patient If the patient is a child, press the child button",
      "Keystep": {{
        "none": ["Adult", "patient", "If", "the", "patient", "is", "a", "child", "press", "the", "child", "button"]
      }}
  }},
  {{
      "Utterance": "Press pads firmly on skin. Press the pads as shown in the picture",
      "Keystep": {{
        "attach_defib_pads": ["Press", "pads", "firmly", "on", "skin", "Press", "the", "pads", "as", "shown", "in", "the", "picture"]
      }}
  }},
  {{
      "Utterance": "Do not touch the patient, everyone clear, analyzing heart rhythm.",
      "Keystep": {{
        "clear_for_analysis": ["Do", "not", "touch", "the", "patient", "everyone", "clear", "analyzing", "heart", "rhythm"]
        }}
  }},
  {{
      "Utterance": "shock advised",
      "Keystep": {{
        "none": ["shock", "advised"]
        }},
  }},
  {{
      "Utterance": "Do not touch the patient, everyone clear",
      "Keystep": {{
        "clear_for_shock": ["Do", "not", "touch", "the", "patient", "everyone", "clear"]
        }},
  }},
  {{
      "Utterance": "Press the flashing shocking button, shock delivered",
      "Keystep": {{
        "administer_shock_aed": ["Press", "the", "flashing", "shocking", "button", "shock", "delivered"]
        }},
  }},
  {{
      "Utterance": "Begin CPR 10 20 30",
      "Keystep": {{
        "chest_compressions": ["Begin", "CPR", "10", "20", "30"]
      }}
  }},
  {{
      "Utterance": "give two breaths",
      "Keystep": {{
        "place_bvm": ["give", "two", "breaths"], 
        "ventilate_patient": ["give", "two", "breaths"]
        }},
  }},
  {{
      "Utterance": "Resume CPR ten twenty thirty",
      "Keystep": {{
        "chest_compressions": ["Resume", "CPR", "ten", "twenty", "thirty"], 
        }},
  }},
]


    Here is the real conversation, you are not allowed to skip any results in returning json file.
    Conversation:
    {conv}

    Let's think step by step,
    """
    response = generate(prompt)
    return prompt, response

def getWordTimestamp(whisper_timestamp):
    segments = whisper_timestamp["segments"]
    words_timestamp = []
    for i in range(len(segments)):
        if "words" not in segments[i]:
            continue
        for w in segments[i]["words"]:
            words_timestamp.append(w)
    return words_timestamp

# Function to get the timestamps for each utterance in the conversation
def get_conversation_timestamps(conversation, word_timestamps):
    conversation_timestamps = []
    timestamp_idx = 0  # Pointer to track the word-level timestamp

    # Iterate through each utterance in the conversation
    for utterance in conversation:
        utterance_timestamps = []

        # For each word in the utterance, find the next matching word in the word-level timestamps
        for word in utterance:
            while timestamp_idx < len(word_timestamps):
                ts = word_timestamps[timestamp_idx]
                # Check for a match and move the pointer forward if matched
                if re.sub(r'[^\w\s]', '', word.lower()) == re.sub(r'[^\w\s]', '', ts['text'].lower()):
                    utterance_timestamps.append((word, ts['start'], ts['end']))
                    timestamp_idx += 1
                    break
                timestamp_idx += 1  # Move pointer forward to check next word
            else:
                print('---'*100)
                print(utterance_timestamps)
                print(utterance)
                print(word, ts['text'])
                print('---'*100)
                print("something wrong, transfer to gemini alignment...")
                return []
        
        conversation_timestamps.append(utterance_timestamps)

    return conversation_timestamps


def getUniqueUtterance(all_info):
    unique_list = []
    for i in range(len(all_info)):
        string_set = []
        for k, v in all_info[i]["Keystep"].items():
            cur_string = " ".join(v)
            if cur_string not in string_set:
                string_set.append(cur_string)
                unique_list.append(v)
    return unique_list


def insertKeywordTimestamp(all_info, keystep_words_timestamp):
    cnt = 0
    d = []
    for utterance in all_info:
        cur_dict = {}
        keystep_words = utterance["Keystep"]
        timestamp = []
        for k, v in keystep_words.items():
            cur_string = " ".join(v)
            cur_table_string = " ".join([j[0] for j in keystep_words_timestamp[cnt]])

            while cur_string != cur_table_string:
                cnt += 1
                cur_table_string = " ".join([j[0] for j in keystep_words_timestamp[cnt]])

            timestamp.append(keystep_words_timestamp[cnt])


        cur_dict["Utterance"] = utterance["Utterance"]
        cur_dict["Keystep"] = utterance["Keystep"]
        cur_dict["Keystep words timestamp"] = timestamp
        d.append(cur_dict)
    return d


def gemini_insert_keyword_timestamp(all_info, words_timestamp):
    prompt = f"""Based on provided Timestamp, insert the word start and end time for the keystep information list. In the list, there are individual dictionary representing each utterance and the keystep label. You need to align the timestamp for each word for each keystep within each utterance. Return the results in defined json format. Directly generate and return the result. In your returned result, you must make sure the returned list has the same length with Keystep information list {len(all_info)}, double check the length.
    [
	    {{
		    "Utterance": "",
		    "Keystep words timestamp":  
		    {{
			    "Keystep1": [[word1, start1, end1], [word2, start2, end2], ...]
			    "Keystep2": [[word3, start3, end3), (word4, start4, end4], ...]
            }}
	    }}
    ]

    Keystep information list:
    {all_info}

    Timestamp:
    {words_timestamp}
    """
    response = generate(prompt)
    error, json_data = extract_json(response)
    if error:
        json_data = handleError(prompt, response)
    
    print(json_data)
    print(len(all_info), len(json_data))
    assert len(all_info) == len(json_data)

    n = len(all_info)
    keystep_word_time_l = []
    for i in range(n):
        cur = {}
        raw_utter = all_info[i]["Utterance"]
        time_utter = json_data[i]["Utterance"]
        
        if raw_utter == time_utter:
            cur["Utterance"] = raw_utter
            cur["Keystep"] = all_info[i]["Keystep"]
            timestamp = []
            for k, v in all_info[i]["Keystep"].items():
                timestamp.append(json_data[i]["Keystep words timestamp"][k])
            cur["Keystep words timestamp"] = timestamp
        keystep_word_time_l.append(cur)
    return keystep_word_time_l


#### this section of code is to recheck the output of gemini json ####
def check_sequence(data):
    """
    Check if words in keystep have the same sequence as in utterance.

    Returns:
    - bool: True if keystep phrases are sequential and non-duplicate, False otherwise.
    """

    utterance = data["Utterance"]
    keystep = data["Keystep"]

    utterance_words = utterance.split()
    keystep_string = ""
    prev_string = ""
    
    for phrase_name, phrase_words in keystep.items():
        phrase_string = " ".join(phrase_words)
        
        # Skip duplicate phrases
        if phrase_string in keystep_string:
            if phrase_string == prev_string:
                continue
            else:
                return False
        
        keystep_string += phrase_string + " "
        prev_string = phrase_string

    # Remove trailing space
    keystep_string = keystep_string.strip()
    keystep_nopunct = re.sub(r'[^\w\s]', '', keystep_string)
    utterance_nopunct = re.sub(r'[^\w\s]', '', utterance)
    
    if len(keystep_nopunct) == len(utterance_nopunct):
        return keystep_nopunct == utterance_nopunct
    else:
        sub_string = ''.join(e for e in keystep_nopunct if e.isalnum() or e.isspace())
        return ' '.join(sub_string.split()) in utterance_nopunct

def reorder_keysteps(data):
    """
    Reorder keystep phrases based on their sequence in the utterance.

    Returns:
    - dict: Reordered keystep dictionary.
    """
    utterance = data["Utterance"]
    keystep = data["Keystep"]

    utterance_words = utterance.split()
    keystep_words = [word for phrase in keystep.values() for word in phrase]

    # Find indices of keystep words in utterance
    indices = [(word, utterance_words.index(word)) for word in keystep_words if word in utterance_words]

    # Sort keystep phrases based on indices
    sorted_phrases = sorted(keystep.items(), key=lambda x: [utterance_words.index(word) for word in x[1] if word in utterance_words])

    return {"Utterance": utterance, "Keystep": dict(sorted_phrases)}

def attach_punctuation(data):
    utterance = data["Utterance"]
    keystep = data["Keystep"]
    for key, sequence in keystep.items():
        # Initialize a new sequence to store the modified words
        modified_sequence = []
        
        for i, word in enumerate(sequence):
            # If the word is a punctuation and there's a previous word, attach it
            if word in string.punctuation and modified_sequence:
                modified_sequence[-1] += word
            else:
                modified_sequence.append(word)
        
        # Go through the list to attach any punctuation marks that are part of the next word
        for i in range(len(modified_sequence) - 1):
            # If the next word starts with punctuation, attach it to the current word
            if modified_sequence[i+1] in string.punctuation:
                modified_sequence[i] += modified_sequence[i+1]
                modified_sequence[i+1] = ''  # Replace punctuation entry with empty string
        
        # Remove any empty strings that may have been created
        keystep[key] = [word for word in modified_sequence if word]

    return {"Utterance": utterance, "Keystep": keystep}

def has_punctuation_in_keystep(data):
    keystep = data['Keystep']
    
    # Function to check if a list has punctuation
    def has_punctuation(lst):
        for item in lst:
            if any(char in string.punctuation for char in item):
                return True
        return False
    
    # Go through each keystep and check for punctuation
    for sequence in keystep.values():
        if has_punctuation(sequence):
            return True  # Return True if any list contains punctuation
    
    return False  # Return False if no punctuation is found
#### this section of code is to recheck the output of gemini json ####

def combKeystepIntervention(keystep, intervention, whisper_timestamp):
    ##### following 635-638 line of code is temporary solution #####
    if len(keystep) != len(intervention):
        print(f"Lengh misaligned, keystep: {len(keystep)}, intervention: {len(intervention)}")
        last_one = intervention[-1]
        intervention.append(last_one)

    # Check if the sequence of keywords in keystep is correct
    # example: 
    # {
    #   'Utterance': 'You call 9-1-1, get an AED. 666 we!',
    #   'Keystep': {
    #       'request_aed': ["get", "an", "AED"], 
    #       "none": ["666", "we!"], 
    #       'request_assistance': ["You", "call", "9-1-1"]
    #    }
    # }
    #
    # Check if it contains indicidual punctuations
    # example:
    # {
    #   "Utterance": "10, 20, 30, give two breaths",
    #   "Keystep": {
    #     "chest_compressions": ["10", ",", "20", ",", "30"], 
    #     "open_airway": ["give", "two", "breaths"], 
    #     "place_bvm": ["give", "two", "breaths"], 
    #     "ventilate_patient": ["give", "two", "breaths"]
    #     }
    # }
    for i, each in enumerate(keystep):
        if not check_sequence(each):
            reorder_dict = reorder_keysteps(each)
            keystep[i] = reorder_dict
        updated_each = keystep[i]

        if has_punctuation_in_keystep(updated_each):
            rm_punct_dict = attach_punctuation(updated_each)
            keystep[i] = rm_punct_dict


    word_timestamps = getWordTimestamp(whisper_timestamp)
    # print(word_timestamps)
    conversation = getUniqueUtterance(keystep)
    # print(conversation)
    keystep_words_timestamp = get_conversation_timestamps(conversation, word_timestamps)
    # print(keystep_words_timestamp)
    if keystep_words_timestamp:
        keystep_timestamp = insertKeywordTimestamp(keystep, keystep_words_timestamp)
    else:
        keystep_timestamp = gemini_insert_keyword_timestamp(keystep, word_timestamps)

    n = len(keystep)
    comb = []
    for i in range(n):
        cur = {}
        cur["Utterance"] = keystep_timestamp[i]["Utterance"]
        cur["Intervention"] = intervention[i]["Intervention"]
        cur["Keystep"] = keystep_timestamp[i]["Keystep"]
        cur["Keystep words timestamp"] = keystep_timestamp[i]["Keystep words timestamp"]
        comb.append(cur)
    return comb


def get_keystep_sequence(keystep_info):
    """
    get ketstep sequence from full info dict

    Args:
        keystep_info: list of dict

    Returns:
        list: list of sequence with timestamp [(keystep, start, end), (), [(), ()], ...]
    """

    n = len(keystep_info)
    keystep_sequence = []
    for i in range(n):
        idx = 0
        cur_sequence = []
        for keystep, words in keystep_info[i]["Keystep"].items():
            word_time = keystep_info[i]["Keystep words timestamp"][idx]
            start = word_time[0][1]
            end = word_time[-1][2]
            idx += 1

            flag = False
            if cur_sequence:
                
                if (cur_sequence[-1][1] != start and cur_sequence[-1][2] != end):
                    
                    if len(cur_sequence) == 1:
                        keystep_sequence.append(cur_sequence[0])
                    else:
                        keystep_sequence.append(cur_sequence)
                    keystep_sequence.append((keystep, start, end))
                    cur_sequence = []
                    flag = True
            if not flag:
                cur_sequence.append((keystep, start, end))
        
        if len(cur_sequence) == 1:
            keystep_sequence.append(cur_sequence[0])
        elif len(cur_sequence) > 1:
            keystep_sequence.append(cur_sequence)
        else:
            continue
    return keystep_sequence

def gemini_process_keystep_timestamp(sequence):
    print("use gemini combines the same keystep")
    prompt = f"""Here is a sequence [(keystep, start, end), ...]. Merge the consecutive same keystep into one and combine the start and end time.
    
    For example, 
    example1: 
    [('administer_shock_aed', 68.5, 69.64), ('clear_for_shock', 70.54, 72.46),  ('clear_for_shock', 73.5, 74.7)]. 
    We can combine clear_for_shock as ('clear_for_shock', 70.54,  74.7), the returned result is [['administer_shock_aed', 68.5, 69.64], ['clear_for_shock', 70.54,  74.7]]

    example2: 
    [[('check_pulse', 6.86, 9.54), ('check_breathing', 6.86, 9.54)], ('check_breathing', 11.46, 11.72), ('check_pulse', 11.72, 12.38), ('chest_compressions', 12.38, 13.46)]. 
    We can combine check_pulse and check_breathing as ('check_pulse', 6.86, 12.38), ('check_breathing', 6.86, 11.72), the returned result is [['check_pulse', 6.86, 9.54], ['check_breathing', 6.86, 9.54], ['chest_compressions', 12.38, 13.46]]

    example3: 
    [["none", 51.5, 57.02],  ["none", 58.5, 60.0]]. 
    We can combine "none" as ("none", 51.5, 60.0), the returned result is [["none", 51.5, 60.0]]

    example4:
    [('place_bvm', 145.62, 147.96), ('ventilate_patient', 145.62, 147.96)], [('place_bvm', 147.96, 151.48), ('ventilate_patient', 147.96, 151.48)], [('place_bvm', 151.48, 152.12), ('ventilate_patient', 151.48, 152.12)].
    We can combine place_bvm and ventilate_patient as (place_bvm, 145.62, 152.12), (ventilate_patient, 145.62, 152.12), the returned result is [['place_bvm', 145.62, 152.12], ['ventilate_patient', 145.62, 152.12]]
    
    Return results in json format. Directly generate the result and return.
    [
	    ["keystep", start, end]
    ]

    Here is the real sequence: 
    {sequence}
    """
    response = generate(prompt)
    error, json_data = extract_json(response)
    if error:
        json_data = handleError(prompt, response)

    # double check 
    # combine consecutive the same keystep
    n = len(json_data)
    i = 0
    seq_clean = []
    while i < n-1:
        j = i+1
        cur = json_data[i]
        end_time = cur[2]
        while j < n:
            nxt = json_data[j]
            
            if cur[0] == nxt[0]:
                end_time = nxt[2]
                j += 1
            else:
                break
        i = j
        seq_clean.append([cur[0], cur[1], end_time])
    seq_clean.append(json_data[-1])

    # reogranize by combine the same start time into a group
    # [["chest_compressions", 82.5, 102.64],
    #  ["open_airway", 102.64, 104.08],
    #  ["place_bvm", 102.64, 104.08],
    #  ["ventilate_patient", 102.64, 104.08]]
    # ===>
    # [
    #   ("chest_compressions", 82.5, 102.64),
    #   [("open_airway", 102.64, 104.08),
    #    ("place_bvm", 102.64, 104.08),
    #    ("ventilate_patient", 102.64, 104.08)]
    # ]
    n = len(seq_clean)
    res = []
    i = 0
    while i < n-1:
        j = i+1
        cur = seq_clean[i]
        group = [(cur[0], cur[1], cur[2])]
        while j < n:
            nxt = seq_clean[j]
            if cur[1] == nxt[1]:
                group.append((nxt[0], nxt[1], nxt[2]))
                j += 1
            else:
                break
        i = j
        
        if len(group) == 1:
            res.append(group[0])
        else:
            res.append(group)
    if (seq_clean[-1][0], seq_clean[-1][1], seq_clean[-1][2]) not in res[-1]:
        res.append((seq_clean[-1][0], seq_clean[-1][1], seq_clean[-1][2]))
    return res
    
def recal_time_keystep(keystep_sequence_combined):
    keystep = []
    for i, cur_seq in enumerate(keystep_sequence_combined[:-1]):
        nxt_seq = keystep_sequence_combined[i+1] if not isinstance(keystep_sequence_combined[i+1], list) else keystep_sequence_combined[i+1][0]
        if isinstance(cur_seq, list):
            cur_seq_list = []
            for step in cur_seq:
                name = step[0]
                start = step[1]
                end = nxt_seq[1]
                cur_seq_list.append((name, start, end))
            keystep.append(cur_seq_list)
        else:
            name = cur_seq[0]
            start = cur_seq[1]
            end = nxt_seq[1]
            keystep.append((name, start, end))
    keystep.append(keystep_sequence_combined[-1])
    return keystep

if __name__ == "__main__":
    # path = '/standard/storage/CognitiveEMS_Datasets/North_Garden/Sep_2024/Raw/05-09-2024/bryan/cardiac_arrest/0'

    path = '/standard/storage/CognitiveEMS_Datasets/North_Garden/Final'
    for root, dirs, files in tqdm(os.walk(path)):
        if 'audio' in dirs:
            for file in os.listdir(os.path.join(root, 'audio')):
                if "_encoded_trimmed.mp3" in file:

                    existing_file_name = os.path.join(root, 'audio', f"keystep_timestamp_{file.split('.mp3')[0]}.json")
                    if os.path.exists(existing_file_name):
                        print(f"find the json file {existing_file_name}, skip to the next")
                        continue

                    print(f"Now working on {os.path.join(root, 'audio', file)}")
                    json_name = os.path.join(root, 'audio', f"whisper_timestamp_{file.split('.mp3')[0]}.json")

                    with open(json_name, 'r') as f:
                        whisper_output = json.load(f)

                    dialogue = whisper_output["text"]
                    segments = whisper_output["segments"]

                    #classify keysteps
                    print(f"Detect keysteps for {file}")
                    prompt, response = detectKeystep(dialogue)
                    error, keystep_json = extract_json(response, pattern=r'\[\s*\{.*?"Utterance".*?\}\s*\]')
                    if error:
                        keystep_json = handleError(prompt, response)
                    
                    #classify interventions
                    conv = []
                    for each in keystep_json:
                        cur = {
                            "Utterance": each["Utterance"]
                        }
                        conv.append(cur)
                    print(f"Detect intervention for {file}")
                    prompt, response = checkInterventions(conv)
                    error, intervention_json = extract_json(response)
                    if error:
                        intervention_json = handleError(prompt, response)

                    # add timestamp for keystep, add intervention
                    print(f"Add timestamp of keystep and intervention for {file}")
                    res = combKeystepIntervention(keystep_json, intervention_json, whisper_output)
                    with open(os.path.join(root, 'audio', f"keystep_intervention_{file.split('.mp3')[0]}.json"), 'w') as f:
                        json.dump(res, f, indent=4)
                    

                    # post-processing to get timestamp for keystep
                    print(f"Post processing to calculate keystep timestamp")
                    keystep_intervention_path = os.path.join(root, 'audio', f"keystep_intervention_{file.split('.mp3')[0]}.json")

                    with open(keystep_intervention_path, 'r') as f:
                        keystep_intervention = json.load(f)

                    keystep_seq = get_keystep_sequence(keystep_intervention)
                    keystep_seq_sync = gemini_process_keystep_timestamp(keystep_seq)
                    keystep_seq_recal = recal_time_keystep(keystep_seq_sync)

                    with open(os.path.join(root, 'audio', f"keystep_timestamp_{file.split('.mp3')[0]}.json"), 'w') as f:
                        json.dump(keystep_seq_recal, f, indent=4)
                        




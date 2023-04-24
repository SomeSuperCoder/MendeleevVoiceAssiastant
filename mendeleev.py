#!/usr/bin/python3

from pandas import *
import torch
import sounddevice as sd
import time
import vosk
import sys
import queue
from fuzzywuzzy import fuzz
import json

lang = "ru"
model_id = "ru_v3"
sample_rate = 48000
speaker = "aidar"
put_accent = True
put_yo = True
device = torch.device("cpu")

vosk_model = vosk.Model("model")
vosk_samplerate = 16000
vosk_device = 7
q = queue.Queue()

xls = ExcelFile('data.xlsx')
df = xls.parse(xls.sheet_names[0])
dataset = df.to_dict()
print(dataset)
print(type(dataset["A"]))

def read_all():
    for i in range(len(dataset["A"])):
        name = dataset["A"][i]
        info = dataset["B"][i]
        current_text = f"{name}.{info}."
        model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models",
                                  model='silero_tts',
                                  language=lang,
                                  speaker=model_id)

        model.to(device)
        audio = model.apply_tts(text=current_text,
                                speaker=speaker,
                                sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo)
        sd.play(audio, sample_rate)
        time.sleep(len(audio) / sample_rate)
        sd.stop()



def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

with sd.RawInputStream(samplerate=vosk_samplerate, blocksize=8000, device=vosk_device, dtype="int16",
                       channels=1, callback=callback):
    rec = vosk.KaldiRecognizer(vosk_model, vosk_samplerate)
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = rec.Result()
            print(result)
            matches = []
            for i in range(len(dataset["A"])):
                name = dataset["A"][i]

                matches.append(fuzz.partial_ratio(json.loads(result)["text"].replace("менделеев", ""), name))

                print(result)
                print(matches)

            print(f"The index is: {matches.index(max(matches))}")

            if max(matches) < 50:
                continue

            if fuzz.partial_ratio("менделеев", result) < 75:
                continue



            info = dataset["B"][matches.index(max(matches))]
            name = dataset["A"][matches.index(max(matches))]
            current_text = f"{name}. {info}."
            model, _ = torch.hub.load(repo_or_dir="snakers4/silero-models",
                                      model='silero_tts',
                                      language=lang,
                                      speaker=model_id)

            model.to(device)
            audio = model.apply_tts(text=current_text,
                                    speaker=speaker,
                                    sample_rate=sample_rate,
                                    put_accent=put_accent,
                                    put_yo=put_yo)
            sd.play(audio, sample_rate)
            time.sleep(len(audio) / sample_rate)
            sd.stop()

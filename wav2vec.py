import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)

# tokenize
input_values = processor(ds[1]["audio"]["array"], return_tensors="pt").input_values

# retrieve logits
with torch.no_grad():
    logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids, clean_up_tokenization_spaces=False)
print(transcription)
# => should give ['m ɪ s t ɚ k w ɪ l t ɚ ɹ ɪ z ð ɪ ɐ p ɑː s əl ʌ v ð ə m ɪ d əl k l æ s ᵻ z æ n d w iː ɑːɹ ɡ l æ d t ə w ɛ l k ə m h ɪ z ɡ ɑː s p əl']

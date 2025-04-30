import spacy
import json
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

def merge_strings(s1: str, s2: str, keep_overlap=True) -> str:
    max_overlap = 0
    min_len = min(len(s1), len(s2))

    # Check all possible overlaps from end of s1 and start of s2
    for i in range(1, min_len + 1):
        if s1[-i:] == s2[:i]:
            max_overlap = i

    if keep_overlap:
        return s1 + s2[max_overlap:]
    else:
        return s2[max_overlap:]

def recursive_merge(texts):
    if len(texts) == 0:
        return ""
    if len(texts) == 1:
        return texts[0]

    # Merge adjacent pairs
    merged = []
    i = 0
    while i < len(texts):
        if i + 1 < len(texts):
            merged_pair = merge_strings(texts[i], texts[i + 1])
            merged.append(merged_pair)
        else:
            merged.append(texts[i])  # Odd one out
        i += 2

    # Recurse on merged list
    return recursive_merge(merged)

def combine_subtitles(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    texts = [item["text"] for item in data]
    return recursive_merge(texts).replace("\n", " ")

# combined_subtitles = combine_subtitles('test.json')
combined_subtitles = "in this video I would like to start the discussion about convolutional new networks which is another architecture of neural networks that we are going to see specifically kind of engineered to address problems that we are facing in computer vision I want to"
print(combined_subtitles)

tokenizer = T5Tokenizer.from_pretrained('SJ-Ray/Re-Punctuate')
model = TFT5ForConditionalGeneration.from_pretrained('SJ-Ray/Re-Punctuate')

inputs = tokenizer.encode("punctuate: " + combined_subtitles, return_tensors="tf") 
result = model.generate(inputs)

decoded_output = tokenizer.decode(result[0], skip_special_tokens=True)
print(decoded_output)

# print()
# nlp = spacy.load("en_core_web_sm")
# doc = nlp(combined_subtitles)
# assert doc.has_annotation("SENT_START")
# for sent in doc.sents:
#     print("sent")
#     print(sent.text)
# print()

# def chunk_subtitles(filepath, max_len = 999999999999):
#     chunks = []
#     # Open and read the JSON file
#     with open(filepath, 'r') as file:
#         data = json.load(file)
    
#     prev = None
#     for item in data:
#         subtitle = item['text'].replace("\n", " ")
#         if prev == None:
#             cur = subtitle
#             chunks.append({
#                 "start": item["start"],
#                 "end": item["end"],
#                 "text": cur
#             })
#         else:
#             cur = merge_strings(prev, subtitle)
#             if len(cur) <= max_len:
#                 chunks[-1]["end"] = item["end"]
#                 chunks[-1]["text"] = cur
#             else:
#                 cur = merge_strings(prev, subtitle, keep_overlap=False)
#                 chunks.append({
#                     "start": item["start"],
#                     "end": item["end"],
#                     "text": cur
#                 })
#         prev = cur
#     return chunks

# print(chunk_subtitles('test.json'))
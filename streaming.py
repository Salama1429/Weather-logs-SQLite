from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import datetime
import sys
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

model = Wav2Vec2ForCTC.from_pretrained("C:\\Users\\moham\\Desktop\\RDI\\ASR\\Wav2Vec\\1 wav2vec live test SA 43hrs 84% vast ai\\finalmodel")
processor = Wav2Vec2Processor.from_pretrained("C:\\Users\\moham\\Desktop\\RDI\\ASR\\Wav2Vec\\1 wav2vec live test SA 43hrs 84% vast ai\\finalmodel")
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("C:\\Users\\moham\\Desktop\\RDI\\ASR\\Wav2Vec\\1 wav2vec live test SA 43hrs 84% vast ai\\finalmodel", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

pipe  = pipeline("automatic-speech-recognition",model=model,processor=processor,tokenizer=tokenizer, feature_extractor=feature_extractor )
sampling_rate = pipe.feature_extractor.sampling_rate

start = datetime.datetime.now()

chunk_length_s = 5
stream_chunk_s = 0.1
mic = ffmpeg_microphone_live(
    sampling_rate=sampling_rate,
    chunk_length_s=chunk_length_s,
    stream_chunk_s=stream_chunk_s,
)
print("Start talking...")
for item in pipe(mic):
    sys.stdout.write("\033[K")
    print(item["text"], end="\r")
    if not item["partial"][0]:
        print("")
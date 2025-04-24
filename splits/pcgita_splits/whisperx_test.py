import whisperx
import gc 

device = "cuda" 
batch_size = 1 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v3-turbo", device, compute_type=compute_type)


# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio_path = '/data/storage1t/PC-GITA/PC-GITA//sentences2/2_juan/non-normalized/pd/AVPEPUDEA0016_juan.wav'
audio = whisperx.load_audio(audio_path)
result = model.transcribe(audio, batch_size=batch_size, language="es")
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

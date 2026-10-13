"""ACE annotation models: local Qwen2.5-Omni audio-to-text inference."""
import argparse
import json
from pathlib import Path
from shared import enforce_offline, local_directory, local_file

PROMPTS = {
    'captioner': '*Task* Describe this audio in detail',
    'transcriber': '*Task* Transcribe this audio in detail',
}
SYSTEM = ('You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, '
          'capable of perceiving auditory and visual inputs, as well as generating text and speech.')

class AudioAnnotator:
    def __init__(self, options):
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        self.path = local_directory(options.get('model'), 'ACE.model')
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.path, torch_dtype='auto', device_map=options.get('device_map', 'auto'),
            local_files_only=True, attn_implementation='sdpa')
        self.model.disable_talker()
        self.model.eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.path, local_files_only=True)
        self.max_new_tokens = int(options.get('max_new_tokens', 2048))

    def predict(self, audio, task):
        import torch
        from qwen_omni_utils import process_mm_info
        audio = local_file(audio, 'ACE.audio')
        conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': PROMPTS[task]}, {'type': 'audio', 'audio': audio}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors='pt', padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, return_audio=False, use_audio_in_video=False,
                                      do_sample=False, max_new_tokens=self.max_new_tokens)
        # Decoder-only generation includes the prompt. Only retain newly generated text.
        generated = ids[:, inputs['input_ids'].shape[1]:]
        if generated.shape[1] >= self.max_new_tokens:
            raise ValueError('Output may be truncated; increase max_new_tokens')
        return self.processor.batch_decode(generated, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0].strip()

def cli(task):
    parser = argparse.ArgumentParser(description='Local ACE-Step ' + task)
    parser.add_argument('--model', required=True, help='Complete local model directory')
    parser.add_argument('--audio', required=True)
    parser.add_argument('--output', required=True, help='Save raw generated text as JSON')
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    args = parser.parse_args()
    enforce_offline()
    runner = AudioAnnotator(vars(args))
    text = runner.predict(args.audio, task)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'task': task, 'model': runner.path,
                                  'audio': str(Path(args.audio).resolve()), 'text': text},
                                 ensure_ascii=False, indent=2), encoding='utf-8')
    print(text)

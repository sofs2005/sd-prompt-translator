import modules.scripts as scripts
import gradio as gr
from modules.shared import opts
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
import string
import csv
import time
from typing import List, Tuple, Optional
from functools import lru_cache

# 常量定义
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'translations.csv')

class Translator:
    def __init__(self, model_name: str, cache_dir: str = CACHE_DIR):
        self.model_name = model_name
        self.model = self._load_model(cache_dir)
        self.tokenizer = self._load_tokenizer(cache_dir)

    def _load_model(self, cache_dir: str):
        raise NotImplementedError

    def _load_tokenizer(self, cache_dir: str):
        raise NotImplementedError

    def translate(self, text: str, input_language: str, output_language: str) -> str:
        raise NotImplementedError

class LlamaTranslator(Translator):
    def __init__(self, model_name: str = "impactframes/llama3_if_ai_sdpromptmkr_q4km", cache_dir: str = CACHE_DIR):
        super().__init__(model_name, cache_dir)

    def _load_model(self, cache_dir: str):
        return AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=cache_dir, torch_dtype=torch.float16, device_map="auto")

    def _load_tokenizer(self, cache_dir: str):
        return AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)

    def translate(self, text: str, input_language: str, output_language: str) -> str:
        prompt = f"Translate the following Stable Diffusion prompt from {input_language} to {output_language}: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)
        
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取翻译结果（假设翻译结果在冒号后面）
        translated_text = translated_text.split(":")[-1].strip()
        
        return translated_text

class LanguageOption:
    def __init__(self, label: str, language_code: str):
        self.label = label
        self.language_code = language_code

# 语言选项列表
language_options = [
    LanguageOption("中文", "Chinese"),
    LanguageOption("English", "English"),
    LanguageOption("Español", "Spanish"),
    LanguageOption("日本語", "Japanese"),
    LanguageOption("Deutsch", "German"),
    LanguageOption("Français", "French"),
    # 添加更多语言选项...
]

@lru_cache(maxsize=1000)
def custom_translate(text: str, cache: dict) -> Optional[str]:
    return cache.get(text)

class TranslatorScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.ln_code = "Chinese"
        self.is_active = True
        self.is_negative_translate_active = False
        self.translator = None
        self.cache = self.load_csv(CSV_FILE_PATH)
        self.cache_translate = {}

    def title(self):
        return "自动翻译提示词"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Row():
            with gr.Column():
                with gr.Accordion("提示词翻译器", open=False):
                    with gr.Accordion("帮助", open=False):
                        gr.Markdown(self.get_help_markdown())
                    with gr.Column():
                        self.disable_translation = gr.Checkbox(label="禁用翻译", value=False)
                        with gr.Column() as options:
                            self.options = options
                            self.translate_negative_prompt = gr.Checkbox(label="翻译负面提示词")
                            self.language = gr.Dropdown(
                                label="源语言",
                                choices=[x.label for x in language_options],
                                value="中文",
                                type="index",
                                elem_id=self.elem_id("x_type")
                            )
                        self.output = gr.Label("首次运行加载模型耗时较长，请耐心等待", visible=False)
                        
                        self.disable_translation.change(self.set_active, [self.disable_translation], [self.output, self.options], show_progress=True)
                        self.translate_negative_prompt.change(self.set_negative_translate_active, [self.translate_negative_prompt])
                        self.language.change(self.set_ln_code, [self.language])

        self.options.visible = True
        return [self.language]

    @staticmethod
    def get_help_markdown():
        return """
        # 描述
        这个扩展可以让您使用母语直接编写提示词，无需翻译。
        # 如何使用
        默认开启翻译正面提示词，如果需要翻译负面提示词，请在下方勾选"翻译负面提示词"，如果需要关闭翻译，请在下方勾选"禁用翻译"。
        # 注意事项
        第一次启用脚本时可能需要很长时间下载翻译模型和加载模型，但一旦加载完成，它将更快。自定义提示词翻译在extensions/sd-prompt-translator/scripts/translation.csv中，您可以自行添加。
        若有问题前往https://github.com/studyzy/sd-prompt-translator留言或者Email作者:studyzy@gmail.com
        """

    def set_active(self, disable):
        self.is_active = not disable
        if not disable and self.translator is None:
            self.translator = LlamaTranslator()
        return "准备好了", self.output.update(visible=True)

    def set_negative_translate_active(self, negative_translate_active):
        self.is_negative_translate_active = negative_translate_active

    def set_ln_code(self, language):
        language_option = language_options[language]
        self.ln_code = language_option.language_code

    @staticmethod
    def load_csv(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as f:
            return dict(csv.reader(f))

    def process(self, p, language, **kwargs):
        if not isinstance(language, int) or language < 0:
            language = 0
        language_option = language_options[language]
        self.ln_code = language_option.language_code

        if self.translator is None and self.is_active:
            self.translator = LlamaTranslator()

        if self.translator and self.is_active:
            original_prompts, original_negative_prompts = self.get_prompts(p)
            translated_prompts = self.translate_prompts(original_prompts, language_option)
            
            p.prompt = translated_prompts[0]
            p.prompt_for_display = translated_prompts[0]
            p.all_prompts = translated_prompts

            if p.negative_prompt and self.is_negative_translate_active:
                translated_negative_prompts = self.translate_prompts(original_negative_prompts, language_option)
                p.negative_prompt = translated_negative_prompts[0]
                p.all_negative_prompts = translated_negative_prompts

    def get_prompts(self, p):
        return (
            p.all_prompts if p.all_prompts else [p.prompt],
            p.all_negative_prompts if p.all_negative_prompts else [p.negative_prompt]
        )

    def translate_prompts(self, prompts, language_option):
        translated_prompts = []
        previous_prompt = ""
        previous_translated_prompt = ""

        for prompt in prompts:
            if prompt != previous_prompt:
                print(f"Translating prompt to English from {language_option.label}")
                print(f"Initial prompt: {prompt}")

                start_time = time.time()
                translated_prompt = self.process_text(prompt)
                translated_prompt = self.post_process_prompt(prompt, translated_prompt)
                end_time = time.time()

                print(f"Translated prompt: {translated_prompt}, time taken: {end_time - start_time:.2f} seconds")
                translated_prompts.append(translated_prompt)

                previous_prompt = prompt
                previous_translated_prompt = translated_prompt
            else:
                translated_prompts.append(previous_translated_prompt)

        return translated_prompts

    def process_text(self, text):
        text = text.translate(str.maketrans('，。！？；：''""（）【】', ',.!?;:\'\'\"\"()[]'))
        parts = re.split(r'(<[^>]*>)', text)
        translated_parts = []
        
        for part in parts:
            if part.startswith('<') and part.endswith('>'):
                translated_parts.append(part)
            else:
                translated_segments = [
                    self.transfer_processing(segment) if not self.is_english(segment) else segment
                    for segment in part.split(',')
                ]
                translated_parts.append(','.join(translated_segments))
        
        return ''.join(translated_parts)

    @staticmethod
    def is_english(text):
        return all(c.isascii() or c.isspace() for c in text)

    def transfer(self, text):
        if text in self.cache_translate:
            return self.cache_translate[text]
        
        result = custom_translate(text, self.cache)
        if result is not None:
            self.cache_translate[text] = result
            return result
        
        en_prompt = self.translator.translate(text, self.ln_code, "English")
        self.cache_translate[text] = en_prompt
        return en_prompt

    def transfer_processing(self, text):
        pattern = re.compile(r'[\(\[\{]*([^:\]\)\}]*)[:)\]\}]*')
        matches = pattern.match(text)
        if matches and matches.group(1):
            return text.replace(matches.group(1), self.transfer(matches.group(1)))
        return self.transfer(text)

    @staticmethod
    def post_process_prompt(original, translated):
        translated = re.sub(r'\)\s*\+\+|\)\+\+\s*', ')++', translated)
        return TranslatorScript.match_pluses(original, translated)

    @staticmethod
    def match_pluses(original_text, translated_text):
        def extract_plus_positions(text):
            pattern = re.compile(r'\++')
            matches = pattern.finditer(text)
            positions = []
            last_match_end = None
            for match in matches:
                if last_match_end is not None and match.start() != last_match_end:
                    j = last_match_end - 1
                    while text[j] == "+":
                        j -= 1
                    j += 1
                    positions.append([j, last_match_end, last_match_end - j])
                last_match_end = match.end()
            if last_match_end is not None and last_match_end == len(text):
                j = last_match_end - 1
                while text[j] == "+":
                    j -= 1
                j += 1
                positions.append([j, last_match_end, last_match_end - j])
            return positions

        in_positions = extract_plus_positions(original_text)
        out_positions = extract_plus_positions(translated_text)    
        
        out_vals = []
        out_current_pos = 0
        
        if len(in_positions) == len(out_positions):
            for in_, out_ in zip(in_positions, out_positions):
                out_vals.append(translated_text[out_current_pos:out_[0]])
                out_vals.append(original_text[in_[0]:in_[1]])
                out_current_pos = out_[1]
                
                if in_[2] != out_[2]:
                    print("detected different + count")

        out_vals.append(translated_text[out_current_pos:])
        
        output = "".join(out_vals)
        return output

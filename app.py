import tkinter as tk
from tkinter import ttk, scrolledtext, font, messagebox
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from duckduckgo_search import DDGS
import os, time, re, logging, requests, io, psutil, threading, datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
from cachetools import TTLCache, cached
from concurrent.futures import ThreadPoolExecutor
import emoji
import speech_recognition as sr
import pyttsx3
from PIL import Image, ExifTags

# Configurações iniciais do NLTK
nltk.download('punkt')
nltk.download('vader_lexicon')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    'Português': {'code': 'pt-BR', 'instruction': 'Responda em português brasileiro'},
    'English': {'code': 'en-US', 'instruction': 'Respond in English'},
    'Español': {'code': 'es-ES', 'instruction': 'Responde en español'},
    'Français': {'code': 'fr-FR', 'instruction': 'Réponds en français'},
    'Deutsch': {'code': 'de-DE', 'instruction': 'Antworte auf Deutsch'}
}

DEFAULT_MODEL_NAME = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
DEFAULT_MODEL_FILE = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
DEFAULT_LOCAL_MODEL_DIR = "models"


class CyberChatAI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cyber Assistant v4.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0a0a0a')
        
        # Configurações principais
        self.model = None
        self.cache = TTLCache(maxsize=500, ttl=3600)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Configurações de voz
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.listening_active = False
        self.stop_listening_flag = False
        self.speaking = False

        # Foco para investigações
        self.investigation_focus = None
        
        self.loading_window = None  # Janela de carregamento
        
        self.setup_ui()
        self.check_hardware()
        self.load_model()
        self.configure_voice()
        self.toggle_image_metadata()  # Ajusta os controles de metadados de imagem

    def setup_ui(self):
        # Fonte e estilos
        self.main_font = font.Font(family='Courier New', size=12)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#0a0a0a')
        style.configure('TLabel', background='#0a0a0a', foreground='#00ff00', font=self.main_font)
        style.configure('TButton', background='#1a1a1a', foreground='#00ff00', font=self.main_font)
        style.configure('TEntry', fieldbackground='#1a1a1a', foreground='#00ff00', font=self.main_font)
        style.configure('TCheckbutton', background='#0a0a0a', foreground='#00ff00', font=self.main_font)

        # Container principal
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Painel esquerdo (controles)
        left_frame = ttk.Frame(container, width=250)
        left_frame.grid(row=0, column=0, sticky="NS", padx=(0, 10))
        
        # Seção: Configurações gerais
        ttk.Label(left_frame, text="Idioma:").grid(row=0, column=0, sticky="w", pady=(0,2))
        self.lang_selector = ttk.Combobox(left_frame, values=list(LANGUAGE_MAP.keys()), state='readonly', width=20)
        self.lang_selector.set('Português')
        self.lang_selector.grid(row=1, column=0, sticky="ew", pady=(0,5))
        
        ttk.Label(left_frame, text="Estilo:").grid(row=2, column=0, sticky="w", pady=(0,2))
        self.style_selector = ttk.Combobox(left_frame, values=["Técnico", "Livre"], state='readonly', width=20)
        self.style_selector.set("Técnico")
        self.style_selector.grid(row=3, column=0, sticky="ew", pady=(0,5))
        
        self.listen_btn = ttk.Button(left_frame, text="🎤 Ativar Voz", command=self.toggle_listening)
        self.listen_btn.grid(row=4, column=0, sticky="ew", pady=5)
        
        # Seção: Investigações
        ttk.Separator(left_frame, orient="horizontal").grid(row=5, column=0, sticky="ew", pady=10)
        ttk.Label(left_frame, text="Investigações").grid(row=6, column=0, sticky="w")
        
        ttk.Label(left_frame, text="Meta de sites:").grid(row=7, column=0, sticky="w", pady=(5,2))
        self.meta_sites_entry = ttk.Entry(left_frame, width=10)
        self.meta_sites_entry.insert(0, "5")
        self.meta_sites_entry.grid(row=8, column=0, sticky="ew", pady=(0,5))
        self.use_meta_sites = tk.BooleanVar(value=True)
        self.meta_sites_checkbox = ttk.Checkbutton(left_frame, text="Usar meta", variable=self.use_meta_sites, command=self.toggle_meta_sites)
        self.meta_sites_checkbox.grid(row=9, column=0, sticky="w", pady=(0,5))
        
        ttk.Label(left_frame, text="Foco:").grid(row=10, column=0, sticky="w", pady=(5,2))
        self.investigation_focus_entry = ttk.Entry(left_frame, width=20)
        self.investigation_focus_entry.grid(row=11, column=0, sticky="ew", pady=(0,5))
        self.investigation_focus_btn = ttk.Button(left_frame, text="Definir Foco", command=self.set_investigation_focus)
        self.investigation_focus_btn.grid(row=12, column=0, sticky="ew", pady=5)
        
        self.search_news_var = tk.BooleanVar(value=False)
        self.search_news_checkbox = ttk.Checkbutton(left_frame, text="Procurar Notícias", variable=self.search_news_var)
        self.search_news_checkbox.grid(row=13, column=0, sticky="w", pady=2)
        self.search_leaked_data_var = tk.BooleanVar(value=False)
        self.search_leaked_data_checkbox = ttk.Checkbutton(left_frame, text="Procurar Dados Vazados", variable=self.search_leaked_data_var)
        self.search_leaked_data_checkbox.grid(row=14, column=0, sticky="w", pady=2)
        
        # Seção: Análise de metadados de imagem
        ttk.Separator(left_frame, orient="horizontal").grid(row=15, column=0, sticky="ew", pady=10)
        ttk.Label(left_frame, text="Metadados de Imagem").grid(row=16, column=0, sticky="w")
        self.image_metadata_enabled = tk.BooleanVar(value=False)
        self.image_metadata_checkbox = ttk.Checkbutton(left_frame, text="Ativar análise", variable=self.image_metadata_enabled, command=self.toggle_image_metadata)
        self.image_metadata_checkbox.grid(row=17, column=0, sticky="w", pady=(0,5))
        ttk.Label(left_frame, text="Link da Imagem:").grid(row=18, column=0, sticky="w", pady=(5,2))
        self.image_url_entry = ttk.Entry(left_frame, width=20)
        self.image_url_entry.grid(row=19, column=0, sticky="ew", pady=(0,5))
        self.image_metadata_btn = ttk.Button(left_frame, text="Analisar", command=self.process_image_metadata)
        self.image_metadata_btn.grid(row=20, column=0, sticky="ew", pady=5)
        
        # Painel direito: Notebook com abas Chat e Investigação
        right_frame = ttk.Frame(container)
        right_frame.grid(row=0, column=1, sticky="NSEW")
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)
        
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.grid(row=0, column=0, sticky="NSEW")
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        
        # Aba Chat
        self.chat_frame = ttk.Frame(self.notebook)
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, font=self.main_font,
                                                       bg='#1a1a1a', fg='#00ff00', insertbackground='#00ff00', padx=10, pady=10)
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.notebook.add(self.chat_frame, text="Chat")
        
        # Aba Investigação
        self.investigation_frame = ttk.Frame(self.notebook)
        self.investigation_display = scrolledtext.ScrolledText(self.investigation_frame, wrap=tk.WORD, font=self.main_font,
                                                                bg='#1a1a1a', fg='#ffa500', insertbackground='#ffa500', padx=10, pady=10)
        self.investigation_display.pack(fill=tk.BOTH, expand=True)
        # Tabela de links e botão para limpar
        table_frame = ttk.Frame(self.investigation_frame)
        table_frame.pack(fill=tk.X, padx=10, pady=5)
        self.investigation_table = ttk.Treeview(table_frame, columns=("title", "link"), show="headings", height=5)
        self.investigation_table.heading("title", text="Título")
        self.investigation_table.heading("link", text="Link")
        self.investigation_table.column("title", width=150)
        self.investigation_table.column("link", width=300)
        self.investigation_table.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.investigation_table.bind("<Double-1>", self.copy_link)
        self.clear_investigation_btn = ttk.Button(table_frame, text="🗑️ Limpar", command=self.clear_investigation_results)
        self.clear_investigation_btn.pack(side=tk.RIGHT, padx=5)
        self.notebook.add(self.investigation_frame, text="Investigação")
        
        # Área de entrada e botão de envio (abaixo do Notebook)
        input_frame = ttk.Frame(right_frame)
        input_frame.grid(row=1, column=0, sticky="ew", pady=10)
        self.user_input = ttk.Entry(input_frame, font=self.main_font)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.user_input.bind('<Return>', self.process_input)
        self.send_btn = ttk.Button(input_frame, text="🚀 Enviar", command=self.process_input)
        self.send_btn.pack(side=tk.LEFT)
        
        # Barra de status
        self.status_bar = ttk.Label(self.root, text="🟢 Sistema Inicializado", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configura tag para cabeçalhos
        for display in (self.chat_display, self.investigation_display):
            display.tag_configure('header', font=(self.main_font.actual('family'), 12, 'bold'))

    def set_investigation_focus(self):
        foco = self.investigation_focus_entry.get().strip()
        if foco:
            self.investigation_focus = foco
            self.investigation_focus_entry.config(state='disabled')
            self.investigation_focus_btn.config(state='disabled')
            self.update_status(f"Foco definido: {foco}")
            self.display_message("ℹ️ Investigação", f"Foco definido: {foco}")
        else:
            self.show_warning("Por favor, insira um valor para o foco.")

    def toggle_meta_sites(self):
        if self.use_meta_sites.get():
            self.meta_sites_entry.config(state='normal')
        else:
            self.meta_sites_entry.config(state='disabled')

    def get_active_display(self):
        # Retorna a área de exibição conforme a aba ativa
        if self.notebook.index("current") == 0:
            return self.chat_display
        else:
            return self.investigation_display

    def display_message(self, sender, message):
        widget = self.get_active_display()
        widget.config(state='normal')
        widget.insert(tk.END, f"{sender}:\n", ('header',))
        widget.insert(tk.END, f"{message}\n\n")
        widget.config(state='disabled')
        widget.see(tk.END)

    def process_input(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        self.user_input.delete(0, tk.END)
        self.display_message("👤 Usuário", user_text)
        # Se estiver na aba Chat, gera resposta; se não, processa investigação
        if self.notebook.index("current") == 0:
            self.executor.submit(self.generate_response, user_text, self.lang_selector.get(), self.style_selector.get())
        else:
            try:
                sites_meta = int(self.meta_sites_entry.get()) if self.use_meta_sites.get() else 5
            except ValueError:
                sites_meta = 5
            self.executor.submit(self.process_investigation, user_text, sites_meta)

    def generate_response(self, query, lang, style):
        start_time = time.time()
        try:
            lang_config = LANGUAGE_MAP[lang]
            if style == "Técnico":
                system_instruction = f"{lang_config['instruction']}. Seja detalhado e técnico."
                temperature = 0.7
            else:
                system_instruction = f"{lang_config['instruction']}. Responda de forma livre e criativa."
                temperature = 0.9
            
            system_msg = {"role": "system", "content": system_instruction}
            user_msg = {"role": "user", "content": query}
            response = self.model.create_chat_completion(
                messages=[system_msg, user_msg],
                temperature=temperature,
                max_tokens=800,
                stop=["</s>"]
            )
            raw_response = response['choices'][0]['message']['content']
            final_response = self.validate_language(raw_response, lang_config)
            self.root.after(0, lambda: self.display_message("🤖 CyberAI", final_response))
            threading.Thread(target=self.speak, args=(final_response,), daemon=True).start()
            self.root.after(0, lambda: self.update_status(f"✅ Resposta gerada em {time.time()-start_time:.2f}s"))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(str(e)))

    def validate_language(self, text, lang_config):
        try:
            if detect(text) != lang_config['code'].split('-')[0]:
                return self.correct_language(text, lang_config)
            return text
        except Exception:
            return text

    def correct_language(self, text, lang_config):
        correction_prompt = f"Traduza para {lang_config['instruction']}:\n{text}"
        corrected = self.model.create_chat_completion(
            messages=[{"role": "user", "content": correction_prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return f"[Traduzido]\n{corrected['choices'][0]['message']['content']}"

    def process_investigation(self, target, sites_meta):
        try:
            self.root.after(0, self.show_loading_screen, "Investigando...")
            with DDGS() as ddgs:
                results = ddgs.text(target, max_results=sites_meta)
                news_results = ddgs.text("notícias " + target, max_results=3) if self.search_news_var.get() else []
                leaked_results = ddgs.text("dados vazados " + target, max_results=3) if self.search_leaked_data_var.get() else []
            
            if len(results) < sites_meta:
                info_msg = f"⚠️ Apenas {len(results)} sites encontrados para '{target}'."
                self.root.after(0, lambda: self.display_message("🕵️ Investigação", info_msg))
            
            formatted_results = "\n".join(
                f"• {res.get('title', 'Sem título')}\n  {res.get('href', 'Sem link')}\n  {res.get('body', '')}"
                for res in results
            )
            investigation_prompt = f"Analise os dados obtidos sobre '{target}'"
            if self.investigation_focus:
                investigation_prompt += f", focando em '{self.investigation_focus}'"
            investigation_prompt += "\n\nResultados de sites:\n" + formatted_results
            if news_results:
                formatted_news = "\n".join(
                    f"• {res.get('title', 'Sem título')}\n  {res.get('href', 'Sem link')}\n  {res.get('body', '')}"
                    for res in news_results
                )
                investigation_prompt += "\n\nResultados de notícias:\n" + formatted_news
            if leaked_results:
                formatted_leaked = "\n".join(
                    f"• {res.get('title', 'Sem título')}\n  {res.get('href', 'Sem link')}\n  {res.get('body', '')}"
                    for res in leaked_results
                )
                investigation_prompt += "\n\nResultados de dados vazados:\n" + formatted_leaked
            investigation_prompt += "\n\nElabore um relatório detalhado com ligações, riscos e informações relevantes."
            
            investigation_response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Você é um perito em investigação online. Seja minucioso e detalhado."},
                    {"role": "user", "content": investigation_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                stop=["</s>"]
            )
            report = investigation_response['choices'][0]['message']['content']
            self.root.after(0, lambda: self.display_message("🕵️ Investigação", report))
            self.root.after(0, lambda: self.update_investigation_table(results))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"❌ Erro na investigação: {str(e)}"))
        finally:
            self.root.after(0, self.hide_loading_screen)

    def update_investigation_table(self, results):
        for item in self.investigation_table.get_children():
            self.investigation_table.delete(item)
        for res in results:
            title = res.get("title", "Sem título")
            link = res.get("href", "Sem link")
            self.investigation_table.insert("", tk.END, values=(title, link))

    def copy_link(self, event):
        selected = self.investigation_table.focus()
        if selected:
            values = self.investigation_table.item(selected, "values")
            if values and len(values) >= 2:
                self.root.clipboard_clear()
                self.root.clipboard_append(values[1])
                self.update_status("Link copiado para a área de transferência!")

    def toggle_listening(self):
        self.listening_active = not self.listening_active
        if self.listening_active:
            self.stop_listening_flag = False
            threading.Thread(target=self.listen_loop, daemon=True).start()
            self.listen_btn.config(text="🔴 Desativar Voz")
            self.update_status("🔈 Ouvindo...")
        else:
            self.stop_listening_flag = True
            self.listen_btn.config(text="🎤 Ativar Voz")
            self.update_status("🟢 Sistema em modo texto")

    def listen_loop(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while self.listening_active and not self.stop_listening_flag:
                try:
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=7)
                    text = self.recognizer.recognize_google(audio, language=LANGUAGE_MAP[self.lang_selector.get()]['code'])
                    self.root.after(0, self.process_voice_command, text)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    self.root.after(0, self.update_status, "🔇 Áudio não reconhecido")
                except sr.RequestError as e:
                    self.root.after(0, self.show_error, f"❌ Erro no serviço de voz: {e}")
                except Exception as e:
                    self.root.after(0, self.show_error, str(e))

    def process_voice_command(self, text):
        self.user_input.delete(0, tk.END)
        self.user_input.insert(0, text)
        self.process_input()

    def speak(self, text):
        try:
            self.speaking = True
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            self.show_error(f"❌ Erro na síntese de voz: {str(e)}")
        finally:
            self.speaking = False

    def check_hardware(self):
        try:
            mem = psutil.virtual_memory()
            self.update_status(f"💾 RAM: {mem.available//(1024**2)}MB | 🖥️ CPUs: {psutil.cpu_count()}")
        except Exception as e:
            self.show_error(f"Erro na verificação de hardware: {str(e)}")

    def load_model(self):
        model_path = os.path.join(DEFAULT_LOCAL_MODEL_DIR, DEFAULT_MODEL_FILE)
        if not os.path.exists(model_path):
            self.download_model()
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=psutil.cpu_count(),
                n_gpu_layers=33 if psutil.virtual_memory().available > 4*1024**3 else 15
            )
            self.update_status("🤖 Modelo Neural Carregado")
        except Exception as e:
            self.show_error(f"❌ Erro na Inicialização: {str(e)}")

    def download_model(self):
        try:
            self.update_status("⏬ Baixando Modelo...")
            hf_hub_download(
                repo_id=DEFAULT_MODEL_NAME,
                filename=DEFAULT_MODEL_FILE,
                local_dir=DEFAULT_LOCAL_MODEL_DIR,
                resume_download=True
            )
        except Exception as e:
            self.show_error(f"❌ Falha no Download: {str(e)}")

    @cached(cache=TTLCache(maxsize=100, ttl=300))
    def analyze_text(self, text):
        return {
            'sentimento': self.sentiment_analyzer.polarity_scores(text),
            'emojis': emoji.emoji_count(text),
            'links': re.findall(r'http[s]?://\S+', text),
            'linguagem': detect(text)
        }

    def update_status(self, message):
        self.status_bar.config(text=message)

    def show_error(self, message):
        self.root.after(0, lambda: messagebox.showerror("❌ Erro", message))
        self.root.after(0, lambda: self.update_status(f"❌ {message}"))

    def show_warning(self, message):
        self.root.after(0, lambda: messagebox.showwarning("⚠️ Aviso", message))

    def show_loading_screen(self, message="Carregando..."):
        if not self.loading_window:
            self.loading_window = tk.Toplevel(self.root)
            self.loading_window.title("Aguarde")
            self.loading_window.geometry("300x100")
            self.loading_window.configure(bg='#1a1a1a')
            lbl = tk.Label(self.loading_window, text=message, font=self.main_font, bg='#1a1a1a', fg='#00ff00')
            lbl.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
            self.loading_window.protocol("WM_DELETE_WINDOW", lambda: None)
            self.loading_window.transient(self.root)
            self.loading_window.grab_set()

    def hide_loading_screen(self):
        if self.loading_window:
            self.loading_window.destroy()
            self.loading_window = None

    def configure_voice(self):
        try:
            voices = self.engine.getProperty('voices')
            selected_voice = next((v for v in voices if 'português' in v.name.lower() or 'brazil' in v.name.lower()), None)
            if selected_voice:
                self.engine.setProperty('voice', selected_voice.id)
                self.engine.setProperty('rate', 160)
            else:
                self.show_warning("Voz em português não encontrada, usando a padrão.")
        except Exception as e:
            self.show_error(f"Erro na configuração de voz: {str(e)}")

    # ============================
    # Funções para Metadados de Imagem
    # ============================
    def toggle_image_metadata(self):
        if self.image_metadata_enabled.get():
            self.image_url_entry.config(state='normal')
            self.image_metadata_btn.config(state='normal')
        else:
            self.image_url_entry.config(state='disabled')
            self.image_metadata_btn.config(state='disabled')

    def process_image_metadata(self):
        url = self.image_url_entry.get().strip()
        if not url:
            self.show_warning("Por favor, insira um link de imagem.")
            return
        self.update_status("🔍 Analisando metadados da imagem...")
        self.executor.submit(self.run_image_metadata_analysis, url)

    def run_image_metadata_analysis(self, url):
        metadata = self.analyze_image_metadata(url)
        formatted = "\n".join(f"{k}: {v}" for k, v in metadata.items()) if metadata else "Nenhum metadado encontrado ou erro."
        self.root.after(0, lambda: self.display_message("🖼️ Metadados da Imagem", formatted))

    def analyze_image_metadata(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            image_data = response.content
            image = Image.open(io.BytesIO(image_data))
            exif = image._getexif()
            meta = {}
            if exif:
                for tag_id, value in exif.items():
                    tag = Image.ExifTags.TAGS.get(tag_id, tag_id)
                    meta[tag] = value
            else:
                meta["info"] = "Nenhum metadado EXIF encontrado."
            return meta
        except Exception as e:
            return {"error": str(e)}
    # ============================
    
    def clear_investigation_results(self):
        self.investigation_display.config(state='normal')
        self.investigation_display.delete('1.0', tk.END)
        self.investigation_display.config(state='disabled')
        for item in self.investigation_table.get_children():
            self.investigation_table.delete(item)
        self.update_status("🔄 Resultados de investigação limpos.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CyberChatAI(root)
    root.mainloop()

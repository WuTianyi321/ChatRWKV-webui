# from modules.ui import create_ui
import gradio as gr
import os,json,types
from typing import Optional, List, Tuple
import copy
os.environ["RWKV_JIT_ON"] = '1' # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
from rwkv.model import RWKV
from rwkv.utils import PIPELINE,PIPELINE_ARGS
css = "style.css"
current_path = os.path.dirname(os.path.abspath(__file__))
CHUNK_LEN = 256
AVOID_REPEAT = '，：？！'
args = types.SimpleNamespace()
args.strategy = 'cuda fp16'
# args.MODEL_NAME = '/home/wty/ChatModel/RWKVModel/RWKV-4-Pile-169M-20220807-8023'
# args.MODEL_NAME = '/home/wty/ChatModel/RWKVModel/RWKV-4-Raven-1B5-v8-Eng-20230408-ctx4096'
args.strategy = 'cuda fp16i8'
# args.MODEL_NAME = '/home/wty/ChatModel/RWKVModel/RWKV-4-Raven-7B-v8-Eng49%-Chn50%-Other1%-20230412-ctx4096'
# args.MODEL_NAME = '/home/wty/ChatModel/RWKVModel/RWKV-4-Raven-7B-v8-Eng49%-Chn50%-Other1%-20230412-AltVersion-ctx4096'
args.MODEL_NAME = '/home/wty/ChatModel/RWKVModel/RWKV-4-Raven-7B-v9-Eng99%-Other1%-20230412-ctx8192'
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
END_OF_TEXT = 0
END_OF_LINE = 187
END_OF_LINE_2 = 535
CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
# English Chinese
PROMPT_FILE = f'{current_path}/prompt/English-2.py'
# PROMPT_FILE = f'{current_path}/prompt/Chinese-2.py'
alpha_presence = 0.5
alpha_frequency = 0.5
with open(PROMPT_FILE, 'rb') as file:
    user = None
    bot = None
    interface = None
    init_prompt = None
    exec(compile(file.read(), PROMPT_FILE, 'exec'))
init_prompt = init_prompt.strip().split('\n')
for c in range(len(init_prompt)):
    init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = pipeline.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd
if not os.path.isfile('config.json'):
    save_config()

CHAR_STOP='\n\n'

class Context:
    def __init__(self,model_state=None,model_state_init=None,model_state_gen_0=None, history: Optional[List[Tuple[str, str]]] = None):
        if history != None:
            self.history = history
        else:
            self.history = []
        if model_state_init != None:
            self.model_state_init=copy.deepcopy(model_state_init)
        else:
            self.model_state_init=None
        if model_state != None:
            self.model_state=copy.deepcopy(model_state)
        else:
            self.model_state=copy.deepcopy(model_state_init)
        self.model_state_gen_0=copy.deepcopy(model_state_init)

    def append(self, query, output):
        self.history.append((query,output))

    def clear_history(self):
        self.history = []
        self.model_state = copy.deepcopy(self.init_model_state)

    def clear_all_history(self):
        self.history = []
        self.model_state=None
        self.init_model_state=None

    def load_last_state(self):
        self.model_state=copy.deepcopy(self.model_state_gen_0)

def run_rnn(model_state,tokens,last_token, newline_adj = 0):
    tokens = [int(x) for x in tokens]
    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]
    out[END_OF_LINE] += newline_adj # adjust \n probability
    if last_token in AVOID_REPEAT_TOKENS:
        out[last_token] = -999999999
    return out,model_state

def read_tokens(tokens,model_state,chunk_len=256):
    while len(tokens)>0:
        out, model_state = model.forward(tokens[:chunk_len], model_state)
        tokens = tokens[chunk_len:]
    return out, model_state
    
def out_prob_adj_chat(
    out,
    occurrence,
    alpha_presence,
    alpha_frequency):
    for n in occurrence:
        out[n] -= (alpha_presence + occurrence[n] * alpha_frequency)
    return out

def out_prob_adj_gen(
    out,
    occurrence,
    alpha_presence,
    alpha_frequency):
    for n in occurrence:
        out[n] -= (alpha_presence + occurrence[n] * alpha_frequency)
    return out

def stream_generate_from_out(out, model_state, out_prob_adj=out_prob_adj_gen, token_count=100,args=PIPELINE_ARGS()):
    out_tokens = []
    out_last = 0
    occurrence = {}
    i=0
    out_str=""
    while (token_count<0)|(i<token_count):
        out=out_prob_adj(out,occurrence,alpha_presence,alpha_frequency)
        token = pipeline.sample_logits(
            out, 
            temperature=args.temperature, 
            top_p=args.top_p, 
            top_k=args.top_k)
        if token == END_OF_TEXT:
            break
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        out_tokens+=[token]
        out, model_state = model.forward([token], model_state)
        tmp = pipeline.decode(out_tokens[out_last:])
        if '\ufffd' not in tmp: # avoid utf-8 display issues
            if CHAR_STOP in tmp:
                break
            out_str += tmp
            out_last = i + 1
            yield out_str,model_state
        i=i+1
    yield out_str,model_state

def chat(query,history,model_state):
    # history = history + [(query,None)]
    # ctx.history[-1][1]=""
    # query=ctx.history[-1][0]
    history = history + [(query,"")]
    query_text=f"\n\n{user}{interface}{query}\n\n{bot}{interface}"
    tokens = pipeline.encode(query_text)
    out,model_state=read_tokens(tokens,model_state)
    last_model_state=copy.deepcopy(model_state)
    last_out=copy.deepcopy(out)
    for response,model_state in stream_generate_from_out(out, model_state, out_prob_adj=out_prob_adj_chat, token_count=-1,args=PIPELINE_ARGS()):
        history[-1]=(query,response)
        if "\n\n" in response:
            break
        yield "",history, model_state,last_out,last_model_state

def regen_last(history,last_out,last_model_state):
    if len(history)==0:
        yield "",history,model_state
    for response,model_state in stream_generate_from_out(last_out, last_model_state, out_prob_adj=out_prob_adj_chat, token_count=-1,args=PIPELINE_ARGS()):
        history[-1][1]=response
        yield history, model_state
    

#     query_text = f'''
# Below is an instruction that describes a task. Write a response that appropriately completes the request.

# # Instruction:
# {query.strip()}

# # Response:
# '''
    
    # yield "",history,ctx

# def chat(query,history,ctx,regen_flag):
#     # history = history + [(query,None)]
#     # ctx.history[-1][1]=""
#     # query=ctx.history[-1][0]
#     if regen_flag==1:
#         ctx.load_last_state()
#     else:
#         query_text=f"\n\n{user}{interface}{query}\n\n{bot}{interface}"
#         out,model_state=run_rnn(model_state,tokens,last_token, newline_adj = 0)
# #     query_text = f'''
# # Below is an instruction that describes a task. Write a response that appropriately completes the request.

# # # Instruction:
# # {query.strip()}

# # # Response:
# # '''
#     tokens=pipeline.encode(query_text)
#     # ctx.model_state=None
#     occurrence = {}
#     last_token=0
#     out_last=0
#     unused_tokens = []
#     history = history + [(query,"")]
#     # ctx.history = ctx.history + [(query,"")]
#     response=""
#     i=0
#     while True:
#         # token,ctx.model_state=generate_1token(ctx.model_state,tokens,occurrence,last_token,i)
#         out,model_state=run_rnn(model_state,tokens,last_token)
#         token=pipeline.sample_logits(
#             out,
#             temperature=1,
#             top_p=0.6)
#         if token == END_OF_TEXT:
#             break
#         if token not in occurrence:
#             occurrence[token] = 1
#         else:
#             occurrence[token] += 1
#         last_token=token
#         unused_tokens+=[token]
#         tokens=[token]
#         tmp,unused_tokens=unused_tokens_decode(unused_tokens)
#         if tmp!="":
#             response+=tmp
#             history[-1] = (query,response)
#             yield "",history,ctx
#         i+=1
#         if '\n\n' in response:
#             break
#     # response = response.replace('\r\n', '\n').replace('\\n', '\n').replace('\n\n','')
#     if response[-2:]=="\n\n": 
#         response=response[:-2]
#     history[-1] = (query,response)
#     yield "",history,ctx

# def chat_wrapper(query,history,ctx):
#     ctx.last_model_state=ctx.model_state
#     query,history,ctx=chat(query,history,ctx)
#     yield query,history,ctx


# def regen_last(history,ctx):



def parse_text(text):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if '```' in line:
            item = line.split('`')[-1]
            if item:
                lines[i] = f'<pre><code class="{item}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                line = line.replace('<', '&lt;').replace('>', '&gt;')
                lines[i] = f'<br/>{line}'
    return ''.join(lines)

# def clear_history(ctx):
#     ctx.clear_history()
#     return gr.update(value=[]),ctx

def clear_history(model_state,init_model_state):
    model_state = copy.deepcopy(init_model_state)
    return gr.update(value=[]),model_state

with open(f'{current_path}/config.json', 'r', encoding='utf-8') as f:
    configs = json.loads(f.read())

# init_prompt=""
# init_state=None
# init_tokens=pipeline.encode(init_prompt)
# tokens=init_tokens
# while len(tokens)>0:
#     out,init_state=model.forward(tokens[:CHUNK_LEN],init_state)
#     tokens = tokens[CHUNK_LEN:]

# # init_state=None
# init_ctx=Context(init_state,init_state,[])

def init_chat_interface(chat_init_prompt):
    init_state=None
    init_tokens=pipeline.encode(chat_init_prompt)
    tokens=init_tokens
    while len(tokens)>0:
        out,init_state=model.forward(tokens[:CHUNK_LEN],init_state)
        tokens = tokens[CHUNK_LEN:]
    init_ctx=Context(init_state,init_state,[])
    init_model_state=init_state
    with gr.Blocks(css=css,analytics_enabled=False) as chat_interface:
        # chat_ctx=gr.State(init_ctx)
        model_state=gr.State(init_state)
        init_model_state=gr.State(init_state)
        last_model_state=gr.State(model_state)
        last_out=gr.State(out)
        chatbot = gr.Chatbot(elem_id='chatbot', show_label=False).style(height=450)
        message = gr.Textbox(
            show_label=False,
            placeholder="输入内容后按回车发送",
        ).style(container=False)
        # input_list = [message,chatbot,chat_ctx]
        # output_list = [message,chatbot,chat_ctx]
        message.submit(chat,[message,chatbot,model_state],[message,chatbot,model_state,last_out,last_model_state])
        clear_history_btn = gr.Button('清空对话')
        regen_last_btn = gr.Button('重新生成上一条回答')
        clear_history_btn.click(clear_history,inputs=[model_state,init_model_state],outputs=[chatbot,model_state])
        regen_last_btn.click(regen_last,inputs=[chatbot,last_out,last_model_state],outputs=[chatbot,model_state])
    return chat_interface

def create_ui():
    chat_interface=init_chat_interface(init_prompt)
    tab_gen = types.SimpleNamespace()
    with gr.Blocks(css=css,analytics_enabled=False) as generate_interface:
        with gr.Row():
            with gr.Column():
                tab_gen.prompt = gr.Textbox(label="提示")
                tab_gen.generate_btn=gr.Button("生成")
            tab_gen.output = gr.Textbox(label="输出")
    

    interfaces = [
        (chat_interface, "Chat", "chat"),
        (generate_interface, "Generate", "generate")
    ]
    with gr.Blocks(css=css, analytics_enabled=False, title="ChatRWKV WebUI") as demo:
        gr.Markdown("""<h2><center>ChatRWKV WebUI</center></h2>""")
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id="tab_" + ifid):
                    interface.render()
        # chat_interface.render()

    return demo

create_ui().queue().launch(server_name='0.0.0.0')


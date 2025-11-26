import streamlit as st
import asyncio
import nest_asyncio
import json
import os
import platform

if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# nest_asyncio 应用：允许在已运行的事件循环中嵌套调用
nest_asyncio.apply()

# 全局事件循环的创建与复用（创建一次后持续使用）
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from utils import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# 加载环境变量（从 .env 文件中获取 API 密钥等设置）
load_dotenv(override=True)

# 设置 config.json 文件路径
CONFIG_FILE_PATH = "config.json"

# 加载 JSON 配置文件的函数
def load_config_from_json():
    """
    从 config.json 文件加载设置。
    如果文件不存在，则使用默认设置创建文件。

    返回值：
        dict: 加载的设置
    """
    default_config = {
        "get_current_time": {
            "command": "python",
            "args": ["./mcp_server_time.py"],
            "transport": "stdio"
        }
    }
    
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            # 如果文件不存在，则用默认设置创建文件
            save_config_to_json(default_config)
            return default_config
    except Exception as e:
        st.error(f"配置文件加载出错: {str(e)}")
        return default_config

# 保存 JSON 配置文件的函数
def save_config_to_json(config):
    """
    将设置保存到 config.json 文件。

    参数：
        config (dict): 要保存的设置
    
    返回值：
        bool: 保存是否成功
    """
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"配置文件保存出错: {str(e)}")
        return False

# 登录会话变量初始化
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 检查是否需要登录
use_login = os.environ.get("USE_LOGIN", "false").lower() == "true"

# 根据登录状态更改页面设置
if use_login and not st.session_state.authenticated:
    # 登录页面使用默认（窄）布局
    st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠")
else:
    # 主应用使用宽布局
    st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠", layout="wide")

# 如果启用了登录功能且尚未认证，则显示登录界面
if use_login and not st.session_state.authenticated:
    st.title("🔐 登录")
    st.markdown("系统使用需要登录。")

    # 登录表单居中显示
    with st.form("login_form"):
        username = st.text_input("账号")
        password = st.text_input("密码", type="password")
        submit_button = st.form_submit_button("登录")

        if submit_button:
            expected_username = os.environ.get("USER_ID")
            expected_password = os.environ.get("USER_PASSWORD")

            if username == expected_username and password == expected_password:
                st.session_state.authenticated = True
                st.success("✅ 登录成功！请稍候……")
                st.rerun()
            else:
                st.error("❌ 账号或密码不正确。")

    # 登录界面不显示主应用
    st.stop()

# 在侧边栏顶部添加作者信息（优先于其他侧边栏元素）
st.sidebar.markdown("### ✍️ 作者：[Teddynote](https://youtube.com/c/teddynote) 🚀")
st.sidebar.markdown(
    "### 💻 [Project Page](https://github.com/teddynote-lab/langgraph-mcp-agents)"
)

st.sidebar.divider()  # 添加分割线

# 页面标题与描述
st.title("💬 MCP 工具智能体")
st.markdown("✨ 向基于 MCP 工具的 ReAct 智能体提问吧。")

SYSTEM_PROMPT = """<ROLE>
You are a smart agent with an ability to use tools. 
You will be given a question and you will use the tools to answer the question.
Pick the most relevant tool to answer the question. 
If you are failed to answer the question, try different tools to get context.
Your answer should be very polite and professional.
</ROLE>

----

<INSTRUCTIONS>
Step 1: Analyze the question
- Analyze user's question and final goal.
- If the user's question is consist of multiple sub-questions, split them into smaller sub-questions.

Step 2: Pick the most relevant tool
- Pick the most relevant tool to answer the question.
- If you are failed to answer the question, try different tools to get context.

Step 3: Answer the question
- Answer the question in the same language as the question.
- Your answer should be very polite and professional.

Step 4: Provide the source of the answer(if applicable)
- If you've used the tool, provide the source of the answer.
- Valid sources are either a website(URL) or a document(PDF, etc).

Guidelines:
- If you've used the tool, your answer should be based on the tool's output(tool's output is more important than your own knowledge).
- If you've used the tool, and the source is valid URL, provide the source(URL) of the answer.
- Skip providing the source if the source is not URL.
- Answer in the same language as the question.
- Answer should be concise and to the point.
- Avoid response your output with any other information than the answer and the source.  
</INSTRUCTIONS>

----

<OUTPUT_FORMAT>
(concise answer to the question)

**Source**(if applicable)
- (source1: valid URL)
- (source2: valid URL)
- ...
</OUTPUT_FORMAT>
"""

OUTPUT_TOKEN_INFO = {
    "claude-3-5-sonnet-latest": {"max_tokens": 8192},
    "claude-3-5-haiku-latest": {"max_tokens": 8192},
    "claude-3-7-sonnet-latest": {"max_tokens": 64000},
    "gpt-4o": {"max_tokens": 16000},
    "gpt-4o-mini": {"max_tokens": 16000},
}

# 会话状态初始化
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False  # 会话初始化状态标志
    st.session_state.agent = None  # 存储 ReAct 代理对象
    st.session_state.history = []  # 存储对话记录的列表
    st.session_state.mcp_client = None  # 存储 MCP 客户端对象
    st.session_state.timeout_seconds = 120  # 响应生成超时时间（秒），默认120秒
    st.session_state.selected_model = "claude-3-7-sonnet-latest"  # 默认模型选择
    st.session_state.recursion_limit = 100  # 递归调用限制，默认100

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()


# --- 函数定义部分 ---


async def cleanup_mcp_client():
    """
    安全关闭现有的 MCP 客户端。

    如果存在旧客户端，则正常释放资源。
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:

            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback

            # st.warning(f"MCP 客户端关闭出错: {str(e)}")
            # st.warning(traceback.format_exc())


def print_message():
    """
    在界面上输出聊天记录。

    区分用户和助手的消息，并在助手消息容器中显示工具调用信息。
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]

        if message["role"] == "user":
            st.chat_message("user", avatar="🧑‍💻").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # 创建助手消息容器
            with st.chat_message("assistant", avatar="🤖"):
                # 显示助手消息内容
                st.markdown(message["content"])

                # 检查下一个消息是否为工具调用信息
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    # 在同一容器内以 expander 形式显示工具调用信息
                    with st.expander("🔧 工具调用信息", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2  # 两条消息一起处理，所以加2
                else:
                    i += 1  # 只处理普通消息时加1
        else:
            # assistant_tool 消息已在上面处理，跳过
            i += 1


def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    创建流式回调函数。

    该函数用于将 LLM 生成的响应实时显示在界面上。
    分别在不同区域显示文本响应和工具调用信息。

    参数：
        text_placeholder: 用于显示文本响应的 Streamlit 组件
        tool_placeholder: 用于显示工具调用信息的 Streamlit 组件

    返回值：
        callback_func: 流式回调函数
        accumulated_text: 存储累计文本响应的列表
        accumulated_tool: 存储累计工具调用信息的列表
    """
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)

        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            # 内容为列表时的处理（主要出现在 Claude 模型等）
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                # 处理文本类型的情况
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                # 处理工具使用类型的情况
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append(
                            "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                        )
                    with tool_placeholder.expander("🔧 工具调用信息", expanded=True):
                        st.markdown("".join(accumulated_tool))
            # 处理 tool_calls 属性的情况（主要出现在 OpenAI 模型等）
            elif (
                hasattr(message_content, "tool_calls")
                and message_content.tool_calls
                and len(message_content.tool_calls[0]["name"]) > 0
            ):
                tool_call_info = message_content.tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander("🔧 工具调用信息", expanded=True):
                    st.markdown("".join(accumulated_tool))
            # 处理纯字符串的情况
            elif isinstance(content, str):
                accumulated_text.append(content)
                text_placeholder.markdown("".join(accumulated_text))
            # 处理无效的工具调用信息
            elif (
                hasattr(message_content, "invalid_tool_calls")
                and message_content.invalid_tool_calls
            ):
                tool_call_info = message_content.invalid_tool_calls[0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander(
                    "🔧 工具调用信息（无效）", expanded=True
                ):
                    st.markdown("".join(accumulated_tool))
            # 处理 tool_call_chunks 属性的情况
            elif (
                hasattr(message_content, "tool_call_chunks")
                and message_content.tool_call_chunks
            ):
                tool_call_chunk = message_content.tool_call_chunks[0]
                accumulated_tool.append(
                    "\n```json\n" + str(tool_call_chunk) + "\n```\n"
                )
                with tool_placeholder.expander("🔧 工具调用信息", expanded=True):
                    st.markdown("".join(accumulated_tool))
            # 处理 additional_kwargs 中包含 tool_calls 的情况（支持多种模型兼容性）
            elif (
                hasattr(message_content, "additional_kwargs")
                and "tool_calls" in message_content.additional_kwargs
            ):
                tool_call_info = message_content.additional_kwargs["tool_calls"][0]
                accumulated_tool.append("\n```json\n" + str(tool_call_info) + "\n```\n")
                with tool_placeholder.expander("🔧 工具调用信息", expanded=True):
                    st.markdown("".join(accumulated_tool))
        # 处理工具消息（工具响应）
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append(
                "\n```json\n" + str(message_content.content) + "\n```\n"
            )
            with tool_placeholder.expander("🔧 工具调用信息", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None

    return callback_func, accumulated_text, accumulated_tool


async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    处理用户问题并生成响应。

    该函数将用户问题传递给代理，并以流式方式实时显示响应。
    如果在指定时间内未完成响应，则返回超时错误。

    参数：
        query: 用户输入的问题文本
        text_placeholder: 用于显示文本响应的 Streamlit 组件
        tool_placeholder: 用于显示工具调用信息的 Streamlit 组件
        timeout_seconds: 响应生成超时时间（秒）

    返回值：
        response: 代理的响应对象
        final_text: 最终文本响应
        final_tool: 最终工具调用信息
    """
    try:
        if st.session_state.agent:
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = (
                get_streaming_callback(text_placeholder, tool_placeholder)
            )
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=RunnableConfig(
                            recursion_limit=st.session_state.recursion_limit,
                            thread_id=st.session_state.thread_id,
                        ),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ 请求时间超过 {timeout_seconds} 秒，请稍后重试。"
                return {"error": error_msg}, error_msg, ""

            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            return response, final_text, final_tool
        else:
            return (
                {"error": "🚫 智能体尚未初始化。"},
                "🚫 智能体尚未初始化。",
                "",
            )
    except Exception as e:
        import traceback

        error_msg = f"❌ 查询处理出错: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""


async def initialize_session(mcp_config=None):
    """
    初始化 MCP 会话和代理。

    参数：
        mcp_config: MCP 工具配置信息（JSON）。为 None 时使用默认设置

    返回值：
        bool: 初始化是否成功
    """
    with st.spinner("🔄 正在连接 MCP 服务器……"):
        # 先安全清理旧客户端
        await cleanup_mcp_client()

        if mcp_config is None:
            # 从 config.json 文件加载设置
            mcp_config = load_config_from_json()
        client = MultiServerMCPClient(mcp_config)
        await client.__aenter__()
        tools = client.get_tools()
        st.session_state.tool_count = len(tools)
        st.session_state.mcp_client = client

        # 根据所选模型进行初始化
        selected_model = st.session_state.selected_model

        if selected_model in [
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
        ]:
            model = ChatAnthropic(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
            )
        else:  # 使用 OpenAI 模型
            model = ChatOpenAI(
                model=selected_model,
                temperature=0.1,
                max_tokens=OUTPUT_TOKEN_INFO[selected_model]["max_tokens"],
            )
        agent = create_react_agent(
            model,
            tools,
            checkpointer=MemorySaver(),
            prompt=SYSTEM_PROMPT,
        )
        st.session_state.agent = agent
        st.session_state.session_initialized = True
        return True


# --- 侧边栏：系统设置部分 ---
with st.sidebar:
    st.subheader("⚙️ 系统设置")

    # 模型选择功能
    # 生成可用模型列表
    available_models = []

    # 检查 Anthropic API 密钥
    has_anthropic_key = os.environ.get("ANTHROPIC_API_KEY") is not None
    if has_anthropic_key:
        available_models.extend(
            [
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
            ]
        )

    # 检查 OpenAI API 密钥
    has_openai_key = os.environ.get("OPENAI_API_KEY") is not None
    if has_openai_key:
        available_models.extend(["gpt-4o", "gpt-4o-mini"])

    # 如果没有可用模型则显示提示信息
    if not available_models:
        st.warning(
            "⚠️ 未设置 API 密钥。请在 .env 文件中添加 ANTHROPIC_API_KEY 或 OPENAI_API_KEY。"
        )
        # 默认添加 Claude 模型（即使没有密钥也显示 UI）
        available_models = ["claude-3-7-sonnet-latest"]

    # 模型选择下拉框
    previous_model = st.session_state.selected_model
    st.session_state.selected_model = st.selectbox(
        "🤖 选择要使用的模型",
        options=available_models,
        index=(
            available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models
            else 0
        ),
        help="Anthropic 模型需设置 ANTHROPIC_API_KEY，OpenAI 模型需设置 OPENAI_API_KEY 环境变量。",
    )

    # 当模型更改时提示需要重新初始化会话
    if (
        previous_model != st.session_state.selected_model
        and st.session_state.session_initialized
    ):
        st.warning(
            "⚠️ 模型已更改，请点击‘应用设置’按钮以应用更改。"
        )

    # 添加超时设置滑块
    st.session_state.timeout_seconds = st.slider(
        "⏱️ 响应生成超时时间（秒）",
        min_value=60,
        max_value=300,
        value=st.session_state.timeout_seconds,
        step=10,
        help="设置智能体生成响应的最大时间。复杂任务可能需要更长时间。",
    )

    st.session_state.recursion_limit = st.slider(
        "⏱️ 递归调用限制（次数）",
        min_value=10,
        max_value=200,
        value=st.session_state.recursion_limit,
        step=10,
        help="设置递归调用次数限制。设置过高可能导致内存不足。",
    )

    st.divider()  # 添加分割线

    # 添加工具设置部分
    st.subheader("🔧 工具设置")

    # 用 session_state 管理 expander 状态
    if "mcp_tools_expander" not in st.session_state:
        st.session_state.mcp_tools_expander = False

    # MCP 工具添加界面
    with st.expander("🧰 添加 MCP 工具", expanded=st.session_state.mcp_tools_expander):
        # 从 config.json 文件加载设置并显示
        loaded_config = load_config_from_json()
        default_config_text = json.dumps(loaded_config, indent=2, ensure_ascii=False)
        
        # 如果没有 pending config，则基于现有 mcp_config_text 创建
        if "pending_mcp_config" not in st.session_state:
            try:
                st.session_state.pending_mcp_config = loaded_config
            except Exception as e:
                st.error(f"初始化 pending config 失败: {e}")

        # 用于单独添加工具的 UI
        st.subheader("添加工具")
        st.markdown(
            """
            [如何设置？](https://teddylee777.notion.site/MCP-1d324f35d12980c8b018e12afdf545a1?pvs=4)

            ⚠️ **重要**: JSON 必须用大括号（`{}`）包裹。"""
        )

        # 提供更清晰的示例
        example_json = {
            "github": {
                "command": "npx",
                "args": [
                    "-y",
                    "@smithery/cli@latest",
                    "run",
                    "@smithery-ai/github",
                    "--config",
                    '{"githubPersonalAccessToken":"your_token_here"}',
                ],
                "transport": "stdio",
            }
        }

        default_text = json.dumps(example_json, indent=2, ensure_ascii=False)

        new_tool_json = st.text_area(
            "工具 JSON",
            default_text,
            height=250,
        )

        # 添加按钮
        if st.button(
            "添加工具",
            type="primary",
            key="add_tool_button",
            use_container_width=True,
        ):
            try:
                # 校验输入值
                if not new_tool_json.strip().startswith(
                    "{"
                ) or not new_tool_json.strip().endswith("}"):
                    st.error("JSON 必须以大括号（{}）开头和结尾。")
                    st.markdown('正确格式: `{ "工具名": { ... } }`')
                else:
                    # 解析 JSON
                    parsed_tool = json.loads(new_tool_json)

                    # 检查是否为 mcpServers 格式并处理
                    if "mcpServers" in parsed_tool:
                        # 将 mcpServers 内的内容提升到最外层
                        parsed_tool = parsed_tool["mcpServers"]
                        st.info(
                            "检测到 'mcpServers' 格式，已自动转换。"
                        )

                    # 检查输入的工具数量
                    if len(parsed_tool) == 0:
                        st.error("请至少输入一个工具。")
                    else:
                        # 处理所有工具
                        success_tools = []
                        for tool_name, tool_config in parsed_tool.items():
                            # 检查 URL 字段并设置 transport
                            if "url" in tool_config:
                                # 如果有 URL，则将 transport 设置为 "sse"
                                tool_config["transport"] = "sse"
                                st.info(
                                    f"检测到 '{tool_name}' 工具有 URL，已将 transport 设置为 'sse'。"
                                )
                            elif "transport" not in tool_config:
                                # 如果没有 URL 且没有 transport，则默认设置为 "stdio"
                                tool_config["transport"] = "stdio"

                            # 检查必填字段
                            if (
                                "command" not in tool_config
                                and "url" not in tool_config
                            ):
                                st.error(
                                    f"'{tool_name}' 工具设置需要 'command' 或 'url' 字段。"
                                )
                            elif "command" in tool_config and "args" not in tool_config:
                                st.error(
                                    f"'{tool_name}' 工具设置需要 'args' 字段。"
                                )
                            elif "command" in tool_config and not isinstance(
                                tool_config["args"], list
                            ):
                                st.error(
                                    f"'{tool_name}' 工具的 'args' 字段必须为数组（[]）格式。"
                                )
                            else:
                                # 向 pending_mcp_config 添加工具
                                st.session_state.pending_mcp_config[tool_name] = (
                                    tool_config
                                )
                                success_tools.append(tool_name)

                        # 成功消息
                        if success_tools:
                            if len(success_tools) == 1:
                                st.success(
                                    f"已添加 {success_tools[0]} 工具。请点击 '应用设置' 按钮以生效。"
                                )
                            else:
                                tool_names = ", ".join(success_tools)
                                st.success(
                                    f"共添加 {len(success_tools)} 个工具（{tool_names}）。请点击 '应用设置' 按钮以生效。"
                                )
                            # 添加后收起 expander
                            st.session_state.mcp_tools_expander = False
                            st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"JSON 解析错误: {e}")
                st.markdown(
                    f"""
                **修正方法**：
                1. 请确保 JSON 格式正确。
                2. 所有键都要用双引号（"）包裹。
                3. 字符串值也要用双引号（"）包裹。
                4. 字符串中如需使用双引号请使用转义（\\"）。
                """
                )
            except Exception as e:
                st.error(f"发生错误: {e}")

    # 显示已注册工具列表并添加删除按钮
    with st.expander("📋 已注册工具列表", expanded=True):
        try:
            pending_config = st.session_state.pending_mcp_config
        except Exception as e:
            st.error("无效的 MCP 工具设置。")
        else:
            # 遍历 pending config 的键（工具名）并显示
            for tool_name in list(pending_config.keys()):
                col1, col2 = st.columns([8, 2])
                col1.markdown(f"- **{tool_name}**")
                if col2.button("删除", key=f"delete_{tool_name}"):
                    # 从 pending config 删除该工具（不会立即生效）
                    del st.session_state.pending_mcp_config[tool_name]
                    st.success(
                        f"已删除 {tool_name} 工具。请点击 '应用设置' 按钮以生效。"
                    )

    st.divider()  # 添加分割线

# --- 侧边栏：系统信息与操作按钮部分 ---
with st.sidebar:
    st.subheader("📊 系统信息")
    st.write(f"🛠️ MCP 工具数量: {st.session_state.get('tool_count', '初始化中...')}")
    selected_model_name = st.session_state.selected_model
    st.write(f"🧠 当前模型: {selected_model_name}")

    # 将“应用设置”按钮移到这里
    if st.button(
        "应用设置",
        key="apply_button",
        type="primary",
        use_container_width=True,
    ):
        # 显示应用中消息
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 正在应用更改，请稍候……")
            progress_bar = st.progress(0)

            # 保存设置
            st.session_state.mcp_config_text = json.dumps(
                st.session_state.pending_mcp_config, indent=2, ensure_ascii=False
            )

            # 保存设置到 config.json 文件
            save_result = save_config_to_json(st.session_state.pending_mcp_config)
            if not save_result:
                st.error("❌ 设置文件保存失败。")
            
            progress_bar.progress(15)

            # 准备会话初始化
            st.session_state.session_initialized = False
            st.session_state.agent = None

            # 更新进度状态
            progress_bar.progress(30)

            # 执行初始化
            success = st.session_state.event_loop.run_until_complete(
                initialize_session(st.session_state.pending_mcp_config)
            )

            # 更新进度状态
            progress_bar.progress(100)

            if success:
                st.success("✅ 新设置已应用。")
                # 收起工具添加 expander
                if "mcp_tools_expander" in st.session_state:
                    st.session_state.mcp_tools_expander = False
            else:
                st.error("❌ 设置应用失败。")

        # 页面刷新
        st.rerun()

    st.divider()  # 添加分割线

    # 操作按钮部分
    st.subheader("🔄 操作")

    # 对话初始化按钮
    if st.button("重置对话", use_container_width=True, type="primary"):
        # 初始化 thread_id
        st.session_state.thread_id = random_uuid()

        # 初始化对话历史
        st.session_state.history = []

        # 提示消息
        st.success("✅ 对话已重置。")

        # 页面刷新
        st.rerun()

    # 仅在启用登录功能时显示登出按钮
    if use_login and st.session_state.authenticated:
        st.divider()  # 添加分割线
        if st.button("登出", use_container_width=True, type="secondary"):
            st.session_state.authenticated = False
            st.success("✅ 已登出。")
            st.rerun()

# --- 默认会话初始化（未初始化时） ---
if not st.session_state.session_initialized:
    st.info(
        "MCP 服务器和代理尚未初始化。请点击左侧边栏的 '应用设置' 按钮进行初始化。"
    )


# --- 输出对话记录 ---
print_message()

# --- 用户输入与处理 ---
user_query = st.chat_input("💬 请输入您的问题")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user", avatar="🧑‍💻").markdown(user_query)
        with st.chat_message("assistant", avatar="🤖"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = (
                st.session_state.event_loop.run_until_complete(
                    process_query(
                        user_query,
                        text_placeholder,
                        tool_placeholder,
                        st.session_state.timeout_seconds,
                    )
                )
            )
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append(
                {"role": "assistant", "content": final_text}
            )
            if final_tool.strip():
                st.session_state.history.append(
                    {"role": "assistant_tool", "content": final_tool}
                )
            st.rerun()
    else:
        st.warning(
            "⚠️ MCP 服务器和智能体尚未初始化。请点击左侧边栏的 '应用设置' 按钮进行初始化。"
        )

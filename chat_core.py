import warnings
# 屏蔽所有弃用类警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv
import os
# 大模型
from langchain_openai import ChatOpenAI
# 核心解析器、提示词（从langchain_core导入）
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
# 向量库独立包，不再用community
from langchain_chroma import Chroma
# 阿里云Embedding暂时还在community，无独立分包，保留不变
from langchain_community.embeddings import DashScopeEmbeddings
# 数据模型
from pydantic import BaseModel, Field

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL")

# 初始化大模型
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=api_key,
    base_url=base_url,
    temperature=0.1
)

# ---------------------- 1. 意图识别数据结构 ----------------------
class UserIntent(BaseModel):
    intent_type: str = Field(description="只能二选一：record_bill 记账指令 / knowledge_ask 理财防骗问答")
parser_intent = PydanticOutputParser(pydantic_object=UserIntent)

# 意图判断提示词
intent_prompt = PromptTemplate(
    template="""
    用户输入：{user_input}
    判断用户意图：
    1. 包含金额、吃饭、购物、充值、消费数字等开销内容 → record_bill
    2. 询问诈骗、存钱、贷款、理财、防坑知识 → knowledge_ask
    {format_instructions}
    """,
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser_intent.get_format_instructions()}
)
intent_chain = intent_prompt | llm | parser_intent

# ---------------------- 2. 记账信息结构化提取 ----------------------
class BillInfo(BaseModel):
    money: float = Field(description="消费金额，纯数字")
    time: str = Field(description="消费时间，无明确时间默认今天")
    category: str = Field(description="分类：餐饮/服饰/交通/学习/娱乐/充值/其他")
parser_bill = PydanticOutputParser(pydantic_object=BillInfo)

bill_prompt = PromptTemplate(
    template="""
    从用户消费语句提取信息：{text}
    必须输出金额、时间、消费分类，无时间填“今日”
    {format_instructions}
    """,
    input_variables=["text"],
    partial_variables={"format_instructions": parser_bill.get_format_instructions()}
)
bill_chain = bill_prompt | llm | parser_bill

# ---------------------- 3. RAG知识库问答检索 ----------------------
def get_knowledge_answer(query):
    # 加载本地向量库
    embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=api_key)
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    # 组装问答提示词
    rag_prompt = f"""
    参考知识库内容回答学生问题，通俗易懂，重点提示金融安全风险：
    知识库：{context}
    用户问题：{query}
    """
    res = llm.invoke(rag_prompt)
    # 修复返回格式：同时返回回答和文档（适配main.py的调用）
    return res.content, docs

# ---------------------- 对外统一对话入口 ----------------------
def chat_handler(user_text):
    # 第一步：意图分发
    intent_res = intent_chain.invoke({"user_input": user_text})
    if intent_res.intent_type == "record_bill":
        # 记账分支：提取账单信息
        bill_data = bill_chain.invoke({"text": user_text})
        return f"【账单记录成功】\n金额：{bill_data.money}元\n时间：{bill_data.time}\n分类：{bill_data.category}"
    else:
        # 问答分支：检索知识库返回建议（只取回答内容）
        answer, _ = get_knowledge_answer(user_text)
        return answer

# 测试入口
if __name__ == "__main__":
    print("=== 大学生记账理财对话系统 ===")
    while True:
        user_input = input("请输入你的消息：")
        if user_input == "exit":
            break
        reply = chat_handler(user_input)
        print(f"系统回复：{reply}\n")

__all__ = ["chat_handler", "get_knowledge_answer"]
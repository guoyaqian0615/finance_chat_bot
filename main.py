from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 导入对话处理函数
from chat_core import chat_handler

app = FastAPI(title="理财反诈RAG服务", version="1.0")

# 跨域配置，允许前端页面访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求体格式
class ChatQuery(BaseModel):
    input: str

# 返回体格式
class ChatResponse(BaseModel):
    status: str
    answer: str
    context: str

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_api(query: ChatQuery):
    content = query.input.strip()
    if not content:
        raise HTTPException(status_code=400, detail="输入内容不能为空")
    try:
        # 账单/记账类指令
        bill_keywords = ["元", "块", "花了", "花费", "支出"]
        if content == "我的账单" or any(k in content for k in bill_keywords):
            answer = chat_handler(content)
            context = ""
        # 反诈问答类，获取回答+知识库文档
        else:
            from chat_core import get_knowledge_answer
            answer, docs = get_knowledge_answer(content)
            context = "\n\n".join([doc.page_content for doc in docs])
        return ChatResponse(
            status="success",
            answer=answer,
            context=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务异常：{str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
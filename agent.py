import json
import re
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, TEMPERATURE, MAX_TOKENS, MEMORY_K
from tools import get_tools, execute_tool

class ReActAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        self.memory = ConversationBufferWindowMemory(
            k=MEMORY_K,
            return_messages=True
        )
        
        self.tools = get_tools()
        self.conversation_history = []
        
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """创建系统提示词"""
        tools_description = "\n".join([
            f"- {tool['name']}: {tool['description']}" 
            for tool in self.tools
        ])
        
        return f"""你是一个智能助手，使用ReAct框架来思考和行动。

可用工具:
{tools_description}

思考过程:
1. 分析用户的问题
2. 决定是否需要使用工具
3. 如果需要工具，选择最合适的工具并调用
4. 基于工具结果给出最终答案

回答格式:
思考: [你的分析过程]
行动: [工具名称] [参数]
观察: [工具返回结果]
回答: [最终答案]

记住:
- 每次只能调用一个工具
- 如果工具结果不完整，可以继续调用其他工具
- 保持对话的连贯性，记住之前的对话内容
- 用中文回答"""
    
    def _extract_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """从LLM响应中提取工具调用"""
        action_pattern = r"行动:\s*(\w+)\s*(.*)"
        match = re.search(action_pattern, response)
        
        if match:
            tool_name = match.group(1).strip()
            params_str = match.group(2).strip()
            
            params = {}
            if params_str:
                try:
                    params = json.loads(params_str)
                except json.JSONDecodeError:
                    params = {"query": params_str}
            
            return {
                "tool_name": tool_name,
                "parameters": params
            }
        return None
    
    def _format_conversation(self, messages: List) -> str:
        """格式化对话历史"""
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"助手: {msg.content}")
        return "\n".join(formatted)
    
    def process_message(self, user_input: str) -> str:
        """处理用户消息"""
        print(f"用户: {user_input}")
        
        messages = self.memory.chat_memory.messages
        conversation_context = self._format_conversation(messages)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", f"对话历史:\n{conversation_context}\n\n当前问题: {user_input}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        
        print(f"思考过程: {response.content}")
        
        tool_call = self._extract_tool_call(response.content)
        
        if tool_call:
            tool_name = tool_call["tool_name"]
            parameters = tool_call["parameters"]
            
            print(f"调用工具: {tool_name} 参数: {parameters}")
            
            tool_result = execute_tool(tool_name, **parameters)
            print(f"工具结果: {tool_result}")
            
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", "基于工具结果给出最终答案，用中文回答。"),
                ("human", f"用户问题: {user_input}\n工具结果: {tool_result}")
            ])
            
            final_chain = final_prompt | self.llm
            final_response = final_chain.invoke({})
            
            answer = final_response.content
        else:
            answer = response.content
        
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(answer)
        
        print(f"助手: {answer}")
        return answer
    
    def get_memory_summary(self) -> str:
        """获取记忆摘要"""
        messages = self.memory.chat_memory.messages
        if not messages:
            return "暂无对话历史"
        
        summary = "对话历史:\n"
        for i, msg in enumerate(messages[-MEMORY_K:], 1):
            if isinstance(msg, HumanMessage):
                summary += f"{i}. 用户: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                summary += f"   助手: {msg.content}\n"
        return summary
    
    def clear_memory(self):
        """清除记忆"""
        self.memory.chat_memory.clear()
        print("记忆已清除")
    
    def list_tools(self) -> str:
        """列出所有可用工具"""
        tools_list = "可用工具:\n"
        for i, tool in enumerate(self.tools, 1):
            tools_list += f"{i}. {tool['name']}: {tool['description']}\n"
        return tools_list 
from agent import ReActAgent

def main():
    print("=== ReAct Agent 智能助手 ===")
    print("输入 'help' 查看帮助")
    print("输入 'quit' 退出程序")
    print("=" * 30)
    
    agent = ReActAgent()
    
    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            elif user_input.lower() == 'help':
                print("\n=== 帮助信息 ===")
                print("1. 直接提问 - 我会使用工具来帮助您")
                print("2. 输入 'tools' - 查看所有可用工具")
                print("3. 输入 'memory' - 查看对话历史")
                print("4. 输入 'clear' - 清除对话历史")
                print("5. 输入 'quit' - 退出程序")
                print("=" * 20)
                continue
            
            elif user_input.lower() == 'tools':
                print(agent.list_tools())
                continue
            
            elif user_input.lower() == 'memory':
                print(agent.get_memory_summary())
                continue
            
            elif user_input.lower() == 'clear':
                agent.clear_memory()
                continue
            
            response = agent.process_message(user_input)
            
        except KeyboardInterrupt:
            print("\n\n程序被中断，再见！")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main() 
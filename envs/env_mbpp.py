# 抛弃了React范式。记得改fewshots
import sys
import re
import string
import textwrap
from .base_env import Env
import signal
import multiprocessing
import traceback

class MBPPEnv(Env):
    def __init__(self, ground_truth, test_list, is_react=True):
        super().__init__(
            ground_truth,
            invalid_hint="Invalid Action. Valid Action inputs are supporting code within [BEGIN] and [END] without any addional explanations.",
        )
        self.test_list = test_list
        self.is_react = is_react
        self._cached_answer = ""
        self._cached_error = ""
        self._is_correct = False
        self._timeout_occurred = False

    # 这里是处理格式的部分
    def parse_action(self, string):
        
        if '[BEGIN]' in string and '[END]' in string:
            return "Finish", string.split("[BEGIN]")[1].split("[END]")[0]
        '''
        start_token = "[BEGIN]"
        end_tokens = ["[END]", "(END)"]

        if start_token in string:
            for end_token in end_tokens:
                if end_token in string:
                    extracted_text = string.split(start_token, 1)[1].split(end_token, 1)[0]
                    cleaned_text = extracted_text.replace("\\_", "_").replace("\_", "_").replace("\\r\\n", "\n").replace("\\n", "\n")

                    # 修复缩进（如果代码没有正常缩进，则重新格式化）
                    cleaned_text = textwrap.dedent(cleaned_text)
                    return "Finish", cleaned_text
        '''
        return None
            
    def get_observation(self, action_type, argument,is_free_text=False):
        result = None
        if action_type == "Finish":
            flag, obs = self.is_correct(argument)
            if flag == True:
                obs += "\nAnswer is CORRECT"
            else:
                obs += "\nAnswer is INCORRECT"
            result = (flag, obs, argument)
        return result

    def _run_code_in_process(self, queue, candidate_code, test_list):
        """子进程执行的代码（独立环境）"""
        ns = {}
        error = ""
        is_correct = False
        try:
            # 执行候选代码
            exec(candidate_code, ns, ns)
            # 执行测试断言
            for test_cmd in test_list:
                try:
                    exec(test_cmd, ns, ns)
                except Exception as e:
                    error = f"Test failed: {str(e)}"
                    break
            else:
                is_correct = True
        except Exception as e:
            error = f"Code execution error: {traceback.format_exc()}"
        # 将结果放入队列
        queue.put((is_correct, error))


    def is_correct(self, candidate_code: str) -> (bool, str):
        # 缓存逻辑保持不变
        if candidate_code == self._cached_answer:
            return self._is_correct, self._cached_error
        
        self._cached_answer = candidate_code
        self._cached_error = ""
        self._is_correct = False

        # 使用多进程    
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self._run_code_in_process,
            args=(queue, candidate_code, self.test_list)
        )
        process.start()
        
        # 等待最多5秒
        process.join(timeout=5)
        
        # 如果进程仍然存活，说明超时
        if process.is_alive():
            process.terminate()  # 强制终止进程
            process.join()
            self._cached_error = "Execution timed out after 5 seconds."
        else:
            # 从队列中获取结果
            if not queue.empty():
                self._is_correct, self._cached_error = queue.get()
            else:
                self._cached_error = "Unknown error occurred during execution."

        return self._is_correct, self._cached_error

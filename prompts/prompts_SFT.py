REASON_PROMPT='''In this task. You are required to solve a question by following a structured approach that involves interleaving Thought, Action, and Observation steps. Thought allows you to reason and analyze the current situation. Finally, you need to {format}. The obervations will be provided to you automatically after you act.
 
You can think step-by-step to reach the answer. Here are some examples:
{examples}
(END OF EXAMPLES)
 
You are solving the following question: {question}

{reflections}

Below is the history of what your solving progress so far:
(BEGIN)
{scratchpad}
(END)
 
Please complete the current step (i.e., the last step before (END) ). '''

REASON_PROMPT_SFT='''In this task. You are required to solve a question.

Here are some examples:
{examples}
(END OF EXAMPLES)

You are solving the following question: {question}

Please give the answer directly without further explanation or other words within {format}.'''

LOGIQA_FORMAT = 'call the `Finish` function and fill in your answer in [] after it'
MATH_FORMAT = 'call the `Finish` function and fill in your answer in <<<>>> after it'
MBPP_FORMAT = 'fill in your answer in [BEGIN] and [END] after `Action`'
BIGBENCH_FORMAT = 'call the `Finish` function and fill in your answer in [] after it'
BIGBENCH_FREE_FORMAT = 'call the `Finish` function and fill in your answer in [] after it'

REFLECTION_HEADER = 'Below is your previous reflection that help to revise the incorrect solutions and correctly answer the question. It localizes the errors, summarizes the potential reasons for your failure and outlines a plan to mitigate the same failure:\n'

REFLECTION_PROMPT_TEST_C2='''You are an advanced reasoning agent that can improve based on self-reflection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high-level plan that aims to mitigate the same failure. Use complete sentences.

Question: {question}
Previous trial and your incorrect solution: {scratchpad}

Using your reflection, generate a new answer to the question, your answer SHOULD not contain any reasoning. provide your answer in the format \"Answer: YOUR_ANSWER\"
'''

REFLECTION_PROMPT_TEST='''You are an advanced reasoning agent that can improve based on self-reflection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high-level plan that aims to mitigate the same failure. Use complete sentences.

Question: {question}
Previous trial and your incorrect solution: {scratchpad}
'''

REFLECTION_PROMPT='''Given the question and relevant context, you were unsuccessful in answering the question. As an advanced reasoning agent, you are required to enhance your incorrect solution and correctly answer the question based on self-reflection.

Question: {question}
Previous trial and your incorrect solution: {scratchpad}

Based on this information, please provide the following:
{demand}

Please follow the instructions without any additional introductory or concluding statements. Do not provide the answer directly. You will be punished to output more than 100 words.'''     # 这里加不超过200token的限制，让refl尽量短


DEMAND_TYPES = {
"1-1-1": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your calculation process to ensure that all the operations are accurate.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"1-1-2": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your calculation process to ensure that all the operations are accurate.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"1-2-1": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your algorithm logic to ensure all steps follow the correct order.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"1-2-2": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your algorithm logic to ensure all steps follow the correct order.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"1-3-1": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your solution to ensure it maintains logical coherence.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"1-3-2": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your solution to ensure it maintains logical coherence.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"1-4-1": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your solution to check statements and conclusions for internal consistency.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"1-4-2": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your solution to check statements and conclusions for internal consistency.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"1-5-1": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review the context and requirements presented in the question to confirm that the response aligns with them.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"1-5-2": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review the context and requirements presented in the question to confirm that the response aligns with them.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"1-6-1": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your solution to ensure that it is relevant to the question and addresses each aspect of the question.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"1-6-2": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your solution to ensure that it is relevant to the question and addresses each aspect of the question.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"1-7-1": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your solution to ensure it conforms to the required format and guidelines in a well-organized structure.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"1-7-2": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your solution to ensure it conforms to the required format and guidelines in a well-organized structure.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"1-8-1": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your solution to ensure all provided facts are accurate and up to date.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"1-8-2": '''Analyze the failed solution by tracing and examining its execution with step-by-step verification.
Review your solution to ensure all provided facts are accurate and up to date.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"2-1-1": '''Quickly go through the failed solution without step-by-step verification.
Review your calculation process to ensure that all the operations are accurate.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"2-1-2": '''Quickly go through the failed solution without step-by-step verification.
Review your calculation process to ensure that all the operations are accurate.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"2-2-1": '''Quickly go through the failed solution without step-by-step verification.
Review your algorithm logic to ensure all steps follow the correct order.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"2-2-2": '''Quickly go through the failed solution without step-by-step verification.
Review your algorithm logic to ensure all steps follow the correct order.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"2-3-1": '''Quickly go through the failed solution without step-by-step verification.
Review your solution to ensure it maintains logical coherence.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"2-3-2": '''Quickly go through the failed solution without step-by-step verification.
Review your solution to ensure it maintains logical coherence.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"2-4-1": '''Quickly go through the failed solution without step-by-step verification.
Review your solution to check statements and conclusions for internal consistency.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"2-4-2": '''Quickly go through the failed solution without step-by-step verification.
Review your solution to check statements and conclusions for internal consistency.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"2-5-1": '''Quickly go through the failed solution without step-by-step verification.
Review the context and requirements presented in the question to confirm that the response aligns with them.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"2-5-2": '''Quickly go through the failed solution without step-by-step verification.
Review the context and requirements presented in the question to confirm that the response aligns with them.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"2-6-1": '''Quickly go through the failed solution without step-by-step verification.
Review your solution to ensure that it is relevant to the question and addresses each aspect of the question.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"2-6-2": '''Quickly go through the failed solution without step-by-step verification.
Review your solution to ensure that it is relevant to the question and addresses each aspect of the question.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"2-7-1": '''Quickly go through the failed solution without step-by-step verification.
Review your solution to ensure it conforms to the required format and guidelines in a well-organized structure.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"2-7-2": '''Quickly go through the failed solution without step-by-step verification.
Review your solution to ensure it conforms to the required format and guidelines in a well-organized structure.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
"2-8-1": '''Quickly go through the failed solution without step-by-step verification.
Review your solution to ensure all provided facts are accurate and up to date.
Outline a high-level plan explaining how these changes will mitigate similar issues.''',
"2-8-2": '''Quickly go through the failed solution without step-by-step verification.
Review your solution to ensure all provided facts are accurate and up to date.
Outline a low-level plan proposing specific strategies or corrections to address these issues.''',
}

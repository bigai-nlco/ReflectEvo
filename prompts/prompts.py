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

Please give the answer directly, {format}.'''

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
"1": 
'''1.Identify the potential reasons for the incorrect solution.
2.Propose specific strategies or corrections to address these issues.
3.Outline a high-level plan explaining how these changes will mitigate similar issues. ''',

"2": 
'''1.Identify the potential reasons for the incorrect solution.
2.Propose specific strategies or corrections to address these issues.
3.Outline a detailed plan explaining how these changes will mitigate similar issues in the future.''',

"3": 
'''1.Analyze the solution by tracing through its execution and identifying specific steps where errors occur.
2.Diagnose potential reasons for the incorrect solution.
3.Propose specific strategies or corrections to address these issues.''',

"4": 
'''1.Verify the failed solution to identify potential reasons for its incorrectness.
2.Propose specific strategies or corrections to address these issues.''',

"1+5":
'''1.Review the requirements to confirm that the response aligns with them.
2.Identify the potential reasons for the incorrect solution.
3.Propose specific strategies or corrections to address these issues.
4.Outline a high-level plan explaining how these changes will mitigate similar issues.
''',

"1+6": 
'''1. Cite factual sources that support your solution.
2. Verify that your solution does not contain incorrect or outdated information.
3.Identify the potential reasons for the incorrect solution.
4.Propose specific strategies or corrections to address these issues.
5.Outline a high-level plan explaining how these changes will mitigate similar issues.''',

"1+7": 
'''1.Review your solution to ensure it conforms to the required format and guidelines.
2.Identify the potential reasons for the incorrect solution.
3.Propose specific strategies or corrections to address these issues.
4.Outline a high-level plan explaining how these changes will mitigate similar issues.''',

"1+8":
'''1.Review your solution to ensure it maintains logical coherence.
2.Identify the potential reasons for the incorrect solution.
3.Propose specific strategies or corrections to address these issues.
4.Outline a high-level plan explaining how these changes will mitigate similar issues.''',

"1+9": 
'''1.Review your solution to ensure that it is relevant to the question and presented with clarity, sufficient detail, and a well-organized structure.
2.Identify the potential reasons for the incorrect solution.
3.Propose specific strategies or corrections to address these issues.
4.Outline a high-level plan explaining how these changes will mitigate similar issues.''',

"1+10": 
'''1.Review your response to ensure it comprehensively addresses each aspect of the question.
2.Identify the potential reasons for the incorrect solution.
3.Propose specific strategies or corrections to address these issues.
4.Outline a high-level plan explaining how these changes will mitigate similar issues.''',

"1+11": 
'''1. Review your calculation process to ensure that each step is accurate.
2.Identify the potential reasons for the incorrect solution.
3.Propose specific strategies or corrections to address these issues.
4.Outline a high-level plan explaining how these changes will mitigate similar issues.''',

"2+5":
'''1. Identify the potential reasons for the incorrect solution.
2. Propose specific strategies or corrections to address these issues.
3. Outline a detailed plan explaining how these changes will mitigate similar issues in the future.
4. Review the requirements to confirm that the response aligns with them.
''',

"2+6":
'''
1. Identify the potential reasons for the incorrect solution.
2. Propose specific strategies or corrections to address these issues.
3. Outline a detailed plan explaining how these changes will mitigate similar issues in the future.
4. Cite factual sources that support your solution.
5. Verify that your solution does not contain incorrect or outdated information.
''',

"2+7":
'''1. Identify the potential reasons for the incorrect solution.
2. Propose specific strategies or corrections to address these issues.
3. Outline a detailed plan explaining how these changes will mitigate similar issues in the future.
4. Review your solution to ensure it conforms to the required format and guidelines.
''',

"2+8":
'''
1. Identify the potential reasons for the incorrect solution.
2. Propose specific strategies or corrections to address these issues.
3. Outline a detailed plan explaining how these changes will mitigate similar issues in the future.
4. Review your solution to ensure it maintains logical coherence.
''',

"2+9":
'''
1. Identify the potential reasons for the incorrect solution.
2. Propose specific strategies or corrections to address these issues.
3. Outline a detailed plan explaining how these changes will mitigate similar issues in the future.
4. Review your solution to ensure that it is relevant to the question and presented with clarity, sufficient detail, and a well-organized structure.
''',

"2+10":
'''
1. Identify the potential reasons for the incorrect solution.
2. Propose specific strategies or corrections to address these issues.
3. Outline a detailed plan explaining how these changes will mitigate similar issues in the future.
4. Review your response to ensure it comprehensively addresses each aspect of the question.
''',

"2+11":
'''
1. Identify the potential reasons for the incorrect solution.
2. Propose specific strategies or corrections to address these issues.
3. Outline a detailed plan explaining how these changes will mitigate similar issues in the future.
4. Review your calculation process to ensure that each step is accurate.
''',

"3+5":
'''1. Analyze the solution by tracing through its execution and identifying specific steps where errors occur.
2. Diagnose potential reasons for the incorrect solution.
3. Propose specific strategies or corrections to address these issues.
4. Review the requirements to confirm that the response aligns with them.
''',

"3+6":
'''1. Analyze the solution by tracing through its execution and identifying specific steps where errors occur.
2. Diagnose potential reasons for the incorrect solution.
3. Propose specific strategies or corrections to address these issues.
4. Cite factual sources that support your analysis and proposed solutions.
5. Verify that your analysis does not contain incorrect or outdated information.
''',

"3+7":
'''1. Analyze the solution by tracing through its execution and identifying specific steps where errors occur.
2. Diagnose potential reasons for the incorrect solution.
3. Propose specific strategies or corrections to address these issues.
4. Review your analysis to ensure it conforms to the required format and guidelines.
''',

"3+8":
'''1. Analyze the solution by tracing through its execution and identifying specific steps where errors occur.
2. Diagnose potential reasons for the incorrect solution.
3. Propose specific strategies or corrections to address these issues.
4. Review your analysis to ensure it maintains logical coherence.
''',

"3+9":
'''1. Analyze the solution by tracing through its execution and identifying specific steps where errors occur.
2. Diagnose potential reasons for the incorrect solution.
3. Propose specific strategies or corrections to address these issues.
4. Review your analysis to ensure that it is relevant to the question and presented with clarity, sufficient detail, and a well-organized structure.
''',

"3+10":
'''1. Analyze the solution by tracing through its execution and identifying specific steps where errors occur.
2. Diagnose potential reasons for the incorrect solution.
3. Propose specific strategies or corrections to address these issues.
4. Review your response to ensure it comprehensively addresses each aspect of the analysis.
''',

"3+11":
'''1. Analyze the solution by tracing through its execution and identifying specific steps where errors occur.
2. Diagnose potential reasons for the incorrect solution.
3. Propose specific strategies or corrections to address these issues.
4. Review your calculation process within the solution to ensure that each step is accurate.
''',

"4+5":
'''1. Verify the failed solution to identify potential reasons for its incorrectness.
2. Propose specific strategies or corrections to address these issues.
3. Review the requirements to confirm that the response aligns with them.
''',

"4+6":
'''1. Verify the failed solution to identify potential reasons for its incorrectness.
2. Propose specific strategies or corrections to address these issues.
3. Cite factual sources that support your verification and proposed solutions.
4. Verify that your verification does not contain incorrect or outdated information.
''',

"4+7":
'''1. Verify the failed solution to identify potential reasons for its incorrectness.
2. Propose specific strategies or corrections to address these issues.
3. Review your verification to ensure it conforms to the required format and guidelines.
''',

"4+8":
'''1. Verify the failed solution to identify potential reasons for its incorrectness.
2. Propose specific strategies or corrections to address these issues.
3. Review your verification to ensure it maintains logical coherence.
''',

"4+9":
'''1. Verify the failed solution to identify potential reasons for its incorrectness.
2. Propose specific strategies or corrections to address these issues.
3. Review your verification to ensure that it is relevant to the question and presented with clarity, sufficient detail, and a well-organized structure.
''',

"4+10":
'''1. Verify the failed solution to identify potential reasons for its incorrectness.
2. Propose specific strategies or corrections to address these issues.
3. Review your response to ensure it comprehensively addresses each aspect of the verification.
''',

"4+11":
'''1. Verify the failed solution to identify potential reasons for its incorrectness.
2. Propose specific strategies or corrections to address these issues.
3. Review your calculation process within the verification to ensure that each step is accurate.
''',
}


INT_TO_DEMAND_TYPES = {
1: "1",#high-level plan
2: "2",#detailed plan
3: "3",#trace through the execution
4: "4",#without step by step
5: "1+5",#1+check requirements
6: "1+6",#1+list reference
7: "1+7",#1_phrasing discrepancy
8: "1+8",#1+logic errors
9: "1+9",#1+irrelevent answers
10: "1+10",#1+coverage
11: "1+11",#1+mathematical calculations
12: "2+5",
13: "2+6",
14: "2+7",
15: "2+8",
16: "2+9",
17: "2+10",
18: "2+11",
19: "3+5",
20: "3+6",
21: "3+7",
22: "3+8",
23: "3+9",
24: "3+10",
25: "3+11",
26: "4+5",
27: "4+6",
28: "4+7",
29: "4+8",
30: "4+9",
31: "4+10",
32: "4+11",
}

DEMAND_TYPES_TO_INT = {
"1": 1,
"2": 2,
"3": 3,
"4": 4,
"1+5": 5,
"1+6": 6,
"1+7": 7,
"1+8": 8,
"1+9": 9,
"1+10": 10,
"1+11": 11,
"2+5": 12,
"2+6": 13,
"2+7": 14,
"2+8": 15,
"2+9": 16,
"2+10": 17,
"2+11": 18,
"3+5": 19,
"3+6": 20,
"3+7": 21,
"3+8": 22,
"3+9": 23,
"3+10": 24,
"3+11": 25,
"4+5": 26,
"4+6": 27,
"4+7": 28,
"4+8": 29,
"4+9": 30,
"4+10": 31,
"4+11": 32,
}

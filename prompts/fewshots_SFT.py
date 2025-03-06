HOTPOTQA_FEWSHOTS = '''Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: Let's think step by step. The eastern sector of Colorado orogeny extends into the High Plains. High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft]

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action: Finish[Richard Nixon]

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought: Let's think step by step. Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action: Finish[The Saimaa Gesture]

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought: Let's think step by step. Professions of Nicholas Ray are director, screenwriter, and actor. Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action: Finish[director, screenwriter, actor]

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: Let's think step by step. Arthur's Magazine was started in 1844. First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action: Finish[Arthur's Magazine]

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: Let's think step by step. Pavel Urysohn is a mathematician. Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.
Action: Finish[Yes]'''

LOGIQA_FEWSHOTS = '''Question: G2B (Business-to-Government) is an e-government model, which refers to the operation mode of the use of computers, the Internet, and other modern information technologies between the government and enterprises to transcend the constraints of time, space, and departmental separation to carry out related business transaction activities. It helps realize the optimization and reorganization of government organizations and work processes, and improve government work efficiency.
According to the above definition, which of the following options belong to the G2B service model?
A. Xiao Zhang watched an HD movie after registering on an audiovisual service website and paying the membership fee
B. Xiao Guo paid the fine of the previous quarter on the "motor vehicle illegal inquiry online platform" in a certain province
C. Xiao Wang purchased the latest smartphone using online banking in a well-known online mall
D. Xiao Li declares and pays his company's taxes last month in the "online tax collection system" of a city
Thought: The G2B service model involves interactions between businesses and the government using digital platforms to perform official transactions. Option D describes Xiao Li using an online tax collection system to declare and pay taxes, which is a direct interaction with a government service. The other options involve interactions that are more consumer-oriented or unrelated to government transactions.
Action: Finish[D]

Question: The independent proof method and the fallacy method are two methods of indirect argumentation. Among them, the independent proof method is to prove that the contradicted proposition is true, thereby determining the refuted proposition to be false. The fallacy method is to assume the proposition is true and then derive a ridiculous conclusion, thus proving that the proposition to be refuted is false.
According to the above definition, which of the following is the independent proof method used in the following arguments?
A: Humans evolved from apes. B: Impossible! Has anyone seen this, and which monkey has become a human?
B: Heaven does not give birth to Zhong Ni, and eternity is like a long night. B: Does anyone before Zhong Ni live in darkness?
C: Human nature is evil. B: If human nature is evil, where does the moral code come from?
D: Sufficient food and clothing is a prerequisite for talking about morality. B: Sufficient food and clothing is not a prerequisite for talking about morality. In the past, societies that have not solved the problem of clothing and food are talking about morality.
Thought: The independent proof method involves proving that the opposite of the proposition is true to refute it. Option A presents an argument where the claim "Humans evolved from apes" is taken as true to refute the opposing view. This aligns with the independent proof method by establishing the truth of the proposition to invalidate the contradiction.
Action: Finish[A]

Question: It is important to cultivate the aesthetic taste of design students, so schools should offer Chinese and Western art history courses for design students.
If the following options are true, what will most weaken the conclusion?
A. There is no significant difference in aesthetic taste between students who have taken Chinese and Western art history courses and those who have not taken them.
B. Whether there is aesthetic interest has little to do with whether students can design excellent works
C. The degree of a student's hard work in the course is proportional to the exquisiteness of the designed work
D. Not all students who have taken Chinese and Western art history courses can become outstanding designers
Thought: The conclusion is that offering art history courses will cultivate aesthetic taste and improve design skills. Option A states that there is no significant difference in aesthetic taste between students who took the courses and those who didn't, directly challenging the effectiveness of these courses in achieving the intended outcome. This directly weakens the conclusion by suggesting the courses have no impact.
Action: Finish[A]'''

MATH_FEWSHOTS='''Question: Let $\\mathbf{a} = \\begin{pmatrix} 1 \\\\ -2 \\\\ -5 \\end{pmatrix},$ $\\mathbf{b} = \\begin{pmatrix} \\sqrt{7} \\\\ 4 \\\\ -1 \\end{pmatrix},$ and $\\mathbf{c} = \\begin{pmatrix} 13 \\\\ -4 \\\\ 17 \\end{pmatrix}.$  Find the angle between the vectors $\\mathbf{a}$ and $(\\mathbf{a} \\cdot \\mathbf{c}) \\mathbf{b} - (\\mathbf{a} \\cdot \\mathbf{b}) \\mathbf{c},$ in degrees.
Thought: Note that the dot product of $\\mathbf{a}$ and $(\\mathbf{a} \\cdot \\mathbf{c}) \\mathbf{b} - (\\mathbf{a} \\cdot \\mathbf{b}) \\mathbf{c}$ is
\\[\\mathbf{a} \\cdot [(\\mathbf{a} \\cdot \\mathbf{c}) \\mathbf{b} - (\\mathbf{a} \\cdot \\mathbf{b}) \\mathbf{c}] = (\\mathbf{a} \\cdot \\mathbf{c}) (\\mathbf{a} \\cdot \\mathbf{b}) - (\\mathbf{a} \\cdot \\mathbf{b}) (\\mathbf{a} \\cdot \\mathbf{c}) = 0.\\]
Therefore, the angle between the vectors is $\\boxed{90^\\circ}$.
Action: Finish<<<90^\\circ>>>

Question: In the diagram below, $\\|\\overrightarrow{OA}\\| = 1,$ $\\|\\overrightarrow{OB}\\| = 1,$ and $\\|\\overrightarrow{OC}\\| = \\sqrt{2}.$  Also, $\\tan \\angle AOC = 7$ and $\\angle BOC = 45^\\circ.$\n\n[asy]\nunitsize(2 cm);\n\npair A, B, C, O;\n\nA = (1,0);\nB = (-0.6,0.8);\nC = (0.2,1.4);\nO = (0,0);\n\ndraw(O--A,Arrow(6));\ndraw(O--B,Arrow(6));\ndraw(O--C,Arrow(6));\n\nlabel("$A$", A, E);\nlabel("$B$", B, NW);\nlabel("$C$", C, N);\nlabel("$O$", O, S);\n[/asy]\n\nThere exist constants $m$ and $n$ so that\n\\[\\overrightarrow{OC} = m \\overrightarrow{OA} + n \\overrightarrow{OB}.\\]Enter the ordered pair $(m,n).$
Thought: By constructing a right triangle with adjacent side 1, opposite side 7, and hypotenuse $\\sqrt{1^2 + 7^2} = 5 \\sqrt{2}$, we see that\n\\[\\cos \\angle AOC = \\frac{1}{5 \\sqrt{2}} \\quad \\text{and} \\quad \\sin \\angle AOC = \\frac{7}{5 \\sqrt{2}}.\\]Then\n\\begin{align*}\n\\cos \\angle AOB &= \\cos (\\angle AOC + \\angle BOC) \\\\\n&= \\cos \\angle AOC \\cos \\angle BOC - \\sin \\angle AOC \\sin \\angle BOC \\\\\n&= \\frac{1}{5 \\sqrt{2}} \\cdot \\frac{1}{\\sqrt{2}} - \\frac{7}{5 \\sqrt{2}} \\cdot \\frac{1}{\\sqrt{2}} \\\\\n&= -\\frac{3}{5}.\n\\end{align*}Taking the dot product of the equation $\\overrightarrow{OC} = m \\overrightarrow{OA} + n \\overrightarrow{OB}$ with $\\overrightarrow{OA},$ we get\n\\[\\overrightarrow{OA} \\cdot \\overrightarrow{OC} = m \\overrightarrow{OA} \\cdot \\overrightarrow{OA} + n \\overrightarrow{OA} \\cdot \\overrightarrow{OB}.\\]Then $\\|\\overrightarrow{OA}\\| \\|\\overrightarrow{OC}\\| \\cos \\angle AOC = m \\|\\overrightarrow{OA}\\|^2 + n \\|\\overrightarrow{OA}\\| \\|\\overrightarrow{OB}\\| \\cos \\angle AOB,$ or\n\\[\\frac{1}{5} = m - \\frac{3}{5} n.\\]Taking the dot product of the equation $\\overrightarrow{OC} = m \\overrightarrow{OA} + n \\overrightarrow{OB}$ with $\\overrightarrow{OB},$ we get\n\\[\\overrightarrow{OB} \\cdot \\overrightarrow{OC} = m \\overrightarrow{OA} \\cdot \\overrightarrow{OB} + n \\overrightarrow{OB} \\cdot \\overrightarrow{OB}.\\]Then $\\|\\overrightarrow{OB}\\| \\|\\overrightarrow{OC}\\| \\cos \\angle BOC = m \\|\\overrightarrow{OA}\\| \\|\\overrightarrow{OB}\\| \\cos \\angle AOB + n \\|\\overrightarrow{OB}\\|^2,$ or\n\\[1 = -\\frac{3}{5} m + n.\\]Solving the system $\\frac{1}{5} = m - \\frac{3}{5} n$ and $1 = -\\frac{3}{5} m + n,$ we find $(m,n) = \\boxed{\\left( \\frac{5}{4}, \\frac{7}{4} \\right)}.$
Action: Finish<<<\\left( \\frac{5}{4}, \\frac{7}{4} \\right)>>>

Question: What is the equation of the line parallel to $4x+2y=8$ and passing through the point $(0,1)$? Write the equation in slope-intercept form.
Thought: First, subtract $4x$ from both sides and divide by 2 to write the given line in slope intercept form.  This gives $y=-2x+4$.  The slope of this line is $-2$.  So, we need to find a line with a slope of $-2$ that passes through the point $(0,1)$.  Writing this requirement in point-slope form gives the line $y-1=-2x$, or equivalently, $\\boxed{y=-2x+1}.$
Action: Finish<<<y=-2x+1>>>'''

MBPP_FEWSHOTS='''Question: Write a python function to check whether the count of inversion of two types are same or not. \nTest list: ['assert solve([1,0,2],3) == True', 'assert solve([1,2,0],3) == False', 'assert solve([1,2,1],3) == True']
Thought: To tackle this problem, we need to understand that an inversion is a pair of elements in an array that are out of their natural order. The function will iterate through the array while keeping track of the maximum value encountered so far. If any element is found to be less than this maximum value, it indicates an inversion, and the function should return False. If no such inversion is found by the end of the iteration, the function returns True.
Action: 
[BEGIN]
import sys \r\n def solve(a,n):   \r\n    mx = -sys.maxsize - 1\r\n    for j in range(1,n):  \r\n        if (mx > a[j]):  \r\n            return False  \r\n        mx = max(mx,a[j - 1])    \r\n    return True
[END]

Question: Write a python function to find the kth element in an array containing odd elements first and then even elements. \nTest list: ['assert get_Number(8,5) == 2', 'assert get_Number(7,2) == 3', 'assert get_Number(5,2) == 3']
Thought: The approach to solve this problem involves creating an array that first contains all the odd numbers up to a certain limit and then all the even numbers. We will use two separate loops to fill the array with odd and even numbers respectively. Once the array is constructed, we can directly access the kth element by using the index k-1, since array indices start at 0.
Action: 
[BEGIN]
def get_Number(n, k): \r\n    arr = [0] * n; \r\n    i = 0; \r\n    odd = 1; \r\n    while (odd <= n):   \r\n        arr[i] = odd; \r\n        i += 1; \r\n        odd += 2;\r\n    even = 2; \r\n    while (even <= n): \r\n        arr[i] = even; \r\n        i += 1;\r\n        even += 2; \r\n    return arr[k - 1]; 
[END]

Question: Write a function to find the n-th rectangular number. \nTest list: ['assert find_rect_num(4) == 20', 'assert find_rect_num(5) == 30', 'assert find_rect_num(6) == 42']
Thought: The n-th rectangular number is the sum of the first n natural numbers multiplied by n. This can be directly calculated using the formula n*(n + 1), which is a simple arithmetic progression sum formula.
Action: 
[BEGIN]
def find_rect_num(n):\r\n  return n*(n + 1)
[END]
'''


BIGBENCH_FEWSHOTS = '''Question: Determine whether the following pairs of sentences embody an entailment relation or not.

Sentences: The meeting starts in less than an hour. So the meeting starts in less than ten minutes.
Options:
A. entailment
B. no-entailment
Thought: The first sentence indicates the meeting starts in less than an hour, which is a broader statement. The second sentence specifies that the meeting starts in less than ten minutes. Since the second sentence is a more specific case of the first, there is no entailment.
Action: Finish[B]

Question: Determine whether the following pairs of sentences embody an entailment relation or not.

Sentences: Lina met two nurses. So, Lina met at least one woman.
Options:
A. entailment
B. no-entailment
Thought: The first sentence states that Lina met two nurses, and since nurses are women, it can be logically concluded that Lina met at least one woman. Therefore, this is an entailment.
Action: Finish[A]

Question: Determine whether the following pairs of sentences embody an entailment relation or not.

Sentences: Sally met two actresses. So Sally met at least one woman.
Options:
A. entailment
B. no-entailment
Thought: The first sentence states that Sally met two actresses. Since all actresses are women, it logically follows that Sally met at least one woman. Hence, this is an entailment.
Action: Finish[A]
'''


BIGBENCH_FREE_FEWSHOTS = '''
Question: People usually hate mice, but one mouse has won the hearts of the people all over the world---the famous Mickey Mouse. Fifty years ago most movies were silent. A man called Walt Disney made a cartoon mouse that could talk his movies. He named his mouse Mickey Mouse. People, both young and old, were very excited at the talking mouse. Children loved to see their friend, because he brought joy and laughter to them. Mickey was a clean mouse right from the beginning. Perhaps this is one reason why people hate real mice and love Mickey Mouse. In his early life Mickey made some mistakes. People were angry. They wrote to Disney and said that they did not want Mickey to do silly things. So Disney made a new animal named Donald Duck. He also made a dog called Pluto. This dog does stupid things and makes mistakes wherever he goes. Now our Mickey Mouse is not only clean but also important. He appears as a beautiful and clever star. He has his friends in almost every country. Who is the most famous mouse?
Thought: The passage mentions Mickey Mouse several times and emphasizes how important and famous Mickey Mouse is. Based on this, we can conclude that the most famous mouse is Mickey Mouse.
Action: Finish[Mickey Mouse.]

Question: People usually hate mice, but one mouse has won the hearts of the people all over the world---the famous Mickey Mouse. Fifty years ago most movies were silent. A man called Walt Disney made a cartoon mouse that could talk his movies. He named his mouse Mickey Mouse. People, both young and old, were very excited at the talking mouse. Children loved to see their friend, because he brought joy and laughter to them. Mickey was a clean mouse right from the beginning. Perhaps this is one reason why people hate real mice and love Mickey Mouse. In his early life Mickey made some mistakes. People were angry. They wrote to Disney and said that they did not want Mickey to do silly things. So Disney made a new animal named Donald Duck. He also made a dog called Pluto. This dog does stupid things and makes mistakes wherever he goes. Now our Mickey Mouse is not only clean but also important. He appears as a beautiful and clever star. He has his friends in almost every country. About when was he created?
Thought: The passage explicitly states that Mickey Mouse was created 50 years ago, during the time when movies were silent. Therefore, Mickey Mouse was created 50 years ago.
Action: Finish[50 years ago, during the time when movies were silent.]

Question: People usually hate mice, but one mouse has won the hearts of the people all over the world---the famous Mickey Mouse. Fifty years ago most movies were silent. A man called Walt Disney made a cartoon mouse that could talk his movies. He named his mouse Mickey Mouse. People, both young and old, were very excited at the talking mouse. Children loved to see their friend, because he brought joy and laughter to them. Mickey was a clean mouse right from the beginning. Perhaps this is one reason why people hate real mice and love Mickey Mouse. In his early life Mickey made some mistakes. People were angry. They wrote to Disney and said that they did not want Mickey to do silly things. So Disney made a new animal named Donald Duck. He also made a dog called Pluto. This dog does stupid things and makes mistakes wherever he goes. Now our Mickey Mouse is not only clean but also important. He appears as a beautiful and clever star. He has his friends in almost every country. Who created him?
Thought: The passage mentions that Walt Disney created Mickey Mouse, so the creator of Mickey Mouse is Walt Disney.
Action: Finish[Walt Disney.]
'''

LOGIQA_FEWSHOTS_SFT = '''Question: G2B (Business-to-Government) is an e-government model, which refers to the operation mode of the use of computers, the Internet, and other modern information technologies between the government and enterprises to transcend the constraints of time, space, and departmental separation to carry out related business transaction activities. It helps realize the optimization and reorganization of government organizations and work processes, and improve government work efficiency.
According to the above definition, which of the following options belong to the G2B service model?
A. Xiao Zhang watched an HD movie after registering on an audiovisual service website and paying the membership fee
B. Xiao Guo paid the fine of the previous quarter on the "motor vehicle illegal inquiry online platform" in a certain province
C. Xiao Wang purchased the latest smartphone using online banking in a well-known online mall
D. Xiao Li declares and pays his company's taxes last month in the "online tax collection system" of a city
D

Question: The independent proof method and the fallacy method are two methods of indirect argumentation. Among them, the independent proof method is to prove that the contradicted proposition is true, thereby determining the refuted proposition to be false. The fallacy method is to assume the proposition is true and then derive a ridiculous conclusion, thus proving that the proposition to be refuted is false.
According to the above definition, which of the following is the independent proof method used in the following arguments?
A: Humans evolved from apes. B: Impossible! Has anyone seen this, and which monkey has become a human?
B: Heaven does not give birth to Zhong Ni, and eternity is like a long night. B: Does anyone before Zhong Ni live in darkness?
C: Human nature is evil. B: If human nature is evil, where does the moral code come from?
D: Sufficient food and clothing is a prerequisite for talking about morality. B: Sufficient food and clothing is not a prerequisite for talking about morality. In the past, societies that have not solved the problem of clothing and food are talking about morality.
A

Question: It is important to cultivate the aesthetic taste of design students, so schools should offer Chinese and Western art history courses for design students.
If the following options are true, what will most weaken the conclusion?
A. There is no significant difference in aesthetic taste between students who have taken Chinese and Western art history courses and those who have not taken them.
B. Whether there is aesthetic interest has little to do with whether students can design excellent works
C. The degree of a student's hard work in the course is proportional to the exquisiteness of the designed work
D. Not all students who have taken Chinese and Western art history courses can become outstanding designers
A'''
MATH_FEWSHOTS_SFT='''Question: Let $\\mathbf{a} = \\begin{pmatrix} 1 \\\\ -2 \\\\ -5 \\end{pmatrix},$ $\\mathbf{b} = \\begin{pmatrix} \\sqrt{7} \\\\ 4 \\\\ -1 \\end{pmatrix},$ and $\\mathbf{c} = \\begin{pmatrix} 13 \\\\ -4 \\\\ 17 \\end{pmatrix}.$  Find the angle between the vectors $\\mathbf{a}$ and $(\\mathbf{a} \\cdot \\mathbf{c}) \\mathbf{b} - (\\mathbf{a} \\cdot \\mathbf{b}) \\mathbf{c},$ in degrees.
90^\\circ

Question: In the diagram below, $\\|\\overrightarrow{OA}\\| = 1,$ $\\|\\overrightarrow{OB}\\| = 1,$ and $\\|\\overrightarrow{OC}\\| = \\sqrt{2}.$  Also, $\\tan \\angle AOC = 7$ and $\\angle BOC = 45^\\circ.$\n\n[asy]\nunitsize(2 cm);\n\npair A, B, C, O;\n\nA = (1,0);\nB = (-0.6,0.8);\nC = (0.2,1.4);\nO = (0,0);\n\ndraw(O--A,Arrow(6));\ndraw(O--B,Arrow(6));\ndraw(O--C,Arrow(6));\n\nlabel("$A$", A, E);\nlabel("$B$", B, NW);\nlabel("$C$", C, N);\nlabel("$O$", O, S);\n[/asy]\n\nThere exist constants $m$ and $n$ so that\n\\[\\overrightarrow{OC} = m \\overrightarrow{OA} + n \\overrightarrow{OB}.\\]Enter the ordered pair $(m,n).$
\\left( \\frac{5}{4}, \\frac{7}{4} \\right)

Question: What is the equation of the line parallel to $4x+2y=8$ and passing through the point $(0,1)$? Write the equation in slope-intercept form.
y=-2x+1'''

MBPP_FEWSHOTS_SFT='''Question: Write a python function to check whether the count of inversion of two types are same or not. \nTest list: ['assert solve([1,0,2],3) == True', 'assert solve([1,2,0],3) == False', 'assert solve([1,2,1],3) == True']
import sys \r\n def solve(a,n):   \r\n    mx = -sys.maxsize - 1\r\n    for j in range(1,n):  \r\n        if (mx > a[j]):  \r\n            return False  \r\n        mx = max(mx,a[j - 1])    \r\n    return True

Question: Write a python function to find the kth element in an array containing odd elements first and then even elements. \nTest list: ['assert get_Number(8,5) == 2', 'assert get_Number(7,2) == 3', 'assert get_Number(5,2) == 3']
def get_Number(n, k): \r\n    arr = [0] * n; \r\n    i = 0; \r\n    odd = 1; \r\n    while (odd <= n):   \r\n        arr[i] = odd; \r\n        i += 1; \r\n        odd += 2;\r\n    even = 2; \r\n    while (even <= n): \r\n        arr[i] = even; \r\n        i += 1;\r\n        even += 2; \r\n    return arr[k - 1]; 

Question: Write a function to find the n-th rectangular number. \nTest list: ['assert find_rect_num(4) == 20', 'assert find_rect_num(5) == 30', 'assert find_rect_num(6) == 42']
def find_rect_num(n):\r\n  return n*(n + 1)
'''
BIGBENCH_FEWSHOTS_SFT = '''Question: Determine whether the following pairs of sentences embody an entailment relation or not.

Sentences: The meeting starts in less than an hour. So the meeting starts in less than ten minutes.
Options:
A. entailment
B. no-entailment
B

Question: Determine whether the following pairs of sentences embody an entailment relation or not.

Sentences: Lina met two nurses. So, Lina met at least one woman.
Options:
A. entailment
B. no-entailment
A

Question: Determine whether the following pairs of sentences embody an entailment relation or not.

Sentences: Sally met two actresses. So Sally met at least one woman.
Options:
A. entailment
B. no-entailment
A
'''

BIGBENCH_FREE_FEWSHOTS_SFT = '''
Question: People usually hate mice, but one mouse has won the hearts of the people all over the world---the famous Mickey Mouse. Fifty years ago most movies were silent. A man called Walt Disney made a cartoon mouse that could talk his movies. He named his mouse Mickey Mouse. People, both young and old, were very excited at the talking mouse. Children loved to see their friend, because he brought joy and laughter to them. Mickey was a clean mouse right from the beginning. Perhaps this is one reason why people hate real mice and love Mickey Mouse. In his early life Mickey made some mistakes. People were angry. They wrote to Disney and said that they did not want Mickey to do silly things. So Disney made a new animal named Donald Duck. He also made a dog called Pluto. This dog does stupid things and makes mistakes wherever he goes. Now our Mickey Mouse is not only clean but also important. He appears as a beautiful and clever star. He has his friends in almost every country. Who is the most famous mouse?
Mickey Mouse.

Question: People usually hate mice, but one mouse has won the hearts of the people all over the world---the famous Mickey Mouse. Fifty years ago most movies were silent. A man called Walt Disney made a cartoon mouse that could talk his movies. He named his mouse Mickey Mouse. People, both young and old, were very excited at the talking mouse. Children loved to see their friend, because he brought joy and laughter to them. Mickey was a clean mouse right from the beginning. Perhaps this is one reason why people hate real mice and love Mickey Mouse. In his early life Mickey made some mistakes. People were angry. They wrote to Disney and said that they did not want Mickey to do silly things. So Disney made a new animal named Donald Duck. He also made a dog called Pluto. This dog does stupid things and makes mistakes wherever he goes. Now our Mickey Mouse is not only clean but also important. He appears as a beautiful and clever star. He has his friends in almost every country. About when was he created?
50 years ago, during the time when movies were silent.

Question: People usually hate mice, but one mouse has won the hearts of the people all over the world---the famous Mickey Mouse. Fifty years ago most movies were silent. A man called Walt Disney made a cartoon mouse that could talk his movies. He named his mouse Mickey Mouse. People, both young and old, were very excited at the talking mouse. Children loved to see their friend, because he brought joy and laughter to them. Mickey was a clean mouse right from the beginning. Perhaps this is one reason why people hate real mice and love Mickey Mouse. In his early life Mickey made some mistakes. People were angry. They wrote to Disney and said that they did not want Mickey to do silly things. So Disney made a new animal named Donald Duck. He also made a dog called Pluto. This dog does stupid things and makes mistakes wherever he goes. Now our Mickey Mouse is not only clean but also important. He appears as a beautiful and clever star. He has his friends in almost every country. Who created him?
Walt Disney.
'''

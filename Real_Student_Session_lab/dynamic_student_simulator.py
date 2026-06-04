"""
Self-contained dynamic English MCQ student simulator.

This file intentionally does not import from student_simulator.py. It includes
its own question metadata, difficulty helpers, baseline student utilities, and
dynamic normal/boost/slump behavior so the dynamic env and trainer can be copied
as a standalone set.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import numpy as np


OPTIONS = ["A", "B", "C", "D"]
MIN_ABILITY = 10
MAX_ABILITY = 30
DIFFICULTY_LOW = 0.0
DIFFICULTY_HIGH = 15.0


# ---------------------------------------------------------------------
# 1. Your 50 questions
# ---------------------------------------------------------------------

QUESTIONS = [{'question_id': 'Q1',
  'question': 'Choose the correct sentence:',
  'option_A': 'Had I known the truth, I would have acted differently.',
  'option_B': 'Had I knew the truth, I would have acted differently.',
  'option_C': 'If I would know the truth, I acted differently.',
  'option_D': 'If I had knew the truth, I would acted differently.',
  'answer': 'A',
  'distractor_strength': {
      'B': 2.5,
      'C': 1.2,
      'D': 2.0,
  },
  'topic': 'Third conditional',
  'subtopic': 'inversion',
  'explanation': '“Had I known” is the inverted form of “If I had known.” It correctly uses past perfect + '
                 '“would have.”'},
 {'question_id': 'Q2',
  'question': 'Fill in the blank: I ___ a student.',
  'option_A': 'is',
  'option_B': 'am',
  'option_C': 'are',
  'option_D': 'be',
  'answer': 'B',
  'distractor_strength': {
      'A': 1.7,
      'C': 2.0,
      'D': 1.2,
  },
  'topic': 'Verb “be”',
  'subtopic': 'subject-verb agreement',
  'explanation': 'With the subject “I,” the correct form of the verb “be” is am.'},
 {'question_id': 'Q3',
  'question': 'Choose the correct sentence:',
  'option_A': 'She is senior than me.',
  'option_B': 'She is senior to me.',
  'option_C': 'She is senior from me.',
  'option_D': 'She is more senior me.',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.6,
      'C': 1.1,
      'D': 1.4,
  },
  'topic': 'Comparative adjective usage',
  'subtopic': 'Comparative adjective usage',
  'explanation': '“Senior” is followed by to, not “than.” Correct form: “senior to me.”'},
 {'question_id': 'Q4',
  'question': 'Choose the sentence that avoids a dangling modifier:',
  'option_A': 'Walking to school, the rain started suddenly.',
  'option_B': 'While walking to school, the rain started suddenly.',
  'option_C': 'Walking to school, I was caught in the rain.',
  'option_D': 'The rain, walking to school, caught me suddenly.',
  'answer': 'C',
  'distractor_strength': {
      'A': 2.0,
      'B': 2.2,
      'D': 0.7,
  },
  'topic': 'Dangling modifier',
  'subtopic': 'Dangling modifier',
  'explanation': 'The phrase “Walking to school” must clearly refer to a person. “I was caught in the rain” '
                 'makes the modifier logical.'},
 {'question_id': 'Q5',
  'question': 'Choose the correct plural form: One child -> Two ___',
  'option_A': 'childs',
  'option_B': 'childes',
  'option_C': 'children',
  'option_D': 'child',
  'answer': 'C',
  'distractor_strength': {
      'A': 2.5,
      'B': 0.8,
      'D': 1.4,
  },
  'topic': 'Irregular plural noun',
  'subtopic': 'Irregular plural noun',
  'explanation': 'The plural of “child” is children, not “childs.”'},
 {'question_id': 'Q6',
  'question': 'Fill in the blank: Rarely ___ such a powerful speech.',
  'option_A': 'I have heard',
  'option_B': 'have I heard',
  'option_C': 'I heard',
  'option_D': 'did I heard',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.4,
      'C': 1.0,
      'D': 1.5,
  },
  'topic': 'Negative adverbial inversion',
  'subtopic': 'Negative adverbial inversion',
  'explanation': 'After negative/limiting adverbs like “rarely,” the auxiliary comes before the subject: '
                 '“Rarely have I heard.”'},
 {'question_id': 'Q7',
  'question': 'Choose the correct passive form: “They built the bridge in 1990.”',
  'option_A': 'The bridge built in 1990.',
  'option_B': 'The bridge was built in 1990.',
  'option_C': 'The bridge has built in 1990.',
  'option_D': 'The bridge is built in 1990.',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.1,
      'C': 1.2,
      'D': 1.8,
  },
  'topic': 'Passive voice',
  'subtopic': 'Passive voice',
  'explanation': '“They built the bridge” becomes “The bridge was built.” The object becomes the subject in '
                 'passive voice.'},
 {'question_id': 'Q8',
  'question': 'Choose the correct word: This is ___ apple.',
  'option_A': 'a',
  'option_B': 'an',
  'option_C': 'the',
  'option_D': 'no article',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.4,
      'C': 1.0,
      'D': 0.6,
  },
  'topic': 'Articles',
  'subtopic': 'Articles',
  'explanation': '“Apple” begins with a vowel sound, so the correct article is an.'},
 {'question_id': 'Q9',
  'question': 'Choose the sentence with correct parallel structure:',
  'option_A': 'She likes reading, to swim, and cycling.',
  'option_B': 'She likes to read, swimming, and cycle.',
  'option_C': 'She likes reading, swimming, and cycling.',
  'option_D': 'She likes read, swim, and cycling.',
  'answer': 'C',
  'distractor_strength': {
      'A': 2.5,
      'B': 1.6,
      'D': 0.8,
  },
  'topic': 'Parallel structure',
  'subtopic': 'Parallel structure',
  'explanation': 'The items in the list must follow the same grammatical pattern: reading, swimming, and '
                 'cycling.'},
 {'question_id': 'Q10',
  'question': 'Identify the grammatical function of the phrase: “that he refused the offer” in the sentence: '
              'I was surprised that he refused the offer.',
  'option_A': 'Adjective clause',
  'option_B': 'Noun clause',
  'option_C': 'Adverb clause',
  'option_D': 'Prepositional phrase',
  'answer': 'B',
  'distractor_strength': {
      'A': 1.7,
      'C': 1.3,
      'D': 0.7,
  },
  'topic': 'Noun clause',
  'subtopic': 'Noun clause',
  'explanation': '“That he refused the offer” acts as the object/complement of the adjective “surprised,” so '
                 'it is a noun clause.'},
 {'question_id': 'Q11',
  'question': 'Choose the correct tense: She ___ her homework before dinner yesterday.',
  'option_A': 'finishes',
  'option_B': 'has finished',
  'option_C': 'finished',
  'option_D': 'finishing',
  'answer': 'C',
  'distractor_strength': {
      'A': 1.2,
      'B': 2.3,
      'D': 0.8,
  },
  'topic': 'Simple past tense',
  'subtopic': 'Simple past tense',
  'explanation': '“Yesterday” indicates past time, so the simple past form finished is correct.'},
 {'question_id': 'Q12',
  'question': 'Choose the most precise word: His explanation was so ___ that no one understood what he '
              'meant.',
  'option_A': 'lucid',
  'option_B': 'obvious',
  'option_C': 'transparent',
  'option_D': 'vague',
  'answer': 'D',
  'distractor_strength': {
      'A': 1.5,
      'B': 1.3,
      'C': 1.2,
  },
  'topic': 'Vocabulary',
  'subtopic': 'word meaning',
  'explanation': '“Vague” means unclear or not specific. If no one understood him, his explanation was '
                 'vague.'},
 {'question_id': 'Q13',
  'question': 'Choose the correct sentence:',
  'option_A': 'Neither of the answers are correct.',
  'option_B': 'Neither of the answers is correct.',
  'option_C': 'Neither answers is correct.',
  'option_D': 'Neither answer are correct.',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.6,
      'C': 0.8,
      'D': 1.4,
  },
  'topic': 'Subject-verb agreement',
  'subtopic': 'Subject-verb agreement',
  'explanation': '“Neither” is singular, so it takes the singular verb is.'},
 {'question_id': 'Q14',
  'question': 'What is the opposite of “hot”?',
  'option_A': 'Cold',
  'option_B': 'Warm',
  'option_C': 'Big',
  'option_D': 'Fast',
  'answer': 'A',
  'distractor_strength': {
      'B': 2.2,
      'C': 0.7,
      'D': 0.6,
  },
  'topic': 'Antonym',
  'subtopic': 'Antonym',
  'explanation': 'The opposite of “hot” is cold.'},
 {'question_id': 'Q15',
  'question': 'Choose the sentence that best avoids redundancy:',
  'option_A': 'The final outcome of the experiment was unexpected.',
  'option_B': 'The outcome of the experiment was unexpected.',
  'option_C': 'The end outcome of the experiment was unexpected.',
  'option_D': 'The final end result of the experiment was unexpected.',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.4,
      'C': 1.4,
      'D': 0.9,
  },
  'topic': 'Redundancy',
  'subtopic': 'Redundancy',
  'explanation': '“Final outcome” is redundant because “outcome” already means result. “The outcome…” is '
                 'clearer.'},
 {'question_id': 'Q16',
  'question': 'Fill in the blank: I have lived here ___ 2020.',
  'option_A': 'for',
  'option_B': 'since',
  'option_C': 'during',
  'option_D': 'from',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.6,
      'C': 0.9,
      'D': 1.8,
  },
  'topic': 'Preposition with time',
  'subtopic': 'Preposition with time',
  'explanation': 'Use since with a starting point in time, such as 2020.'},
 {'question_id': 'Q17',
  'question': 'Choose the sentence with correct punctuation:',
  'option_A': 'The result, however, was unexpected.',
  'option_B': 'The result however, was unexpected.',
  'option_C': 'The result, however was unexpected.',
  'option_D': 'The result however was, unexpected.',
  'answer': 'A',
  'distractor_strength': {
      'B': 2.1,
      'C': 2.0,
      'D': 0.7,
  },
  'topic': 'Punctuation',
  'subtopic': 'comma use',
  'explanation': '“However” is an interrupter, so it should be set off with commas: “The result, however, '
                 'was unexpected.”'},
 {'question_id': 'Q18',
  'question': 'Identify the adjective: “The old man walked slowly across the road.”',
  'option_A': 'man',
  'option_B': 'walked',
  'option_C': 'old',
  'option_D': 'slowly',
  'answer': 'C',
  'distractor_strength': {
      'A': 1.3,
      'B': 0.8,
      'D': 2.4,
  },
  'topic': 'Parts of speech',
  'subtopic': 'adjective',
  'explanation': '“Old” describes the noun “man,” so it is an adjective.'},
 {'question_id': 'Q19',
  'question': 'Choose the sentence with the most formal and grammatically precise style:',
  'option_A': 'Everyone should bring their book to class.',
  'option_B': 'Each student should bring their book to class.',
  'option_C': 'Each student should bring his or her book to class.',
  'option_D': 'Students should bring his or her book to class.',
  'answer': 'C',
  'distractor_strength': {
      'A': 2.3,
      'B': 2.6,
      'D': 1.0,
  },
  'topic': 'Formal pronoun agreement',
  'subtopic': 'Formal pronoun agreement',
  'explanation': 'In formal grammar, singular “Each student” matches “his or her.” This avoids informal '
                 'singular “their.”'},
 {'question_id': 'Q20',
  'question': 'Fill in the blank: If I ___ enough money, I would buy a new laptop.',
  'option_A': 'have',
  'option_B': 'has',
  'option_C': 'had',
  'option_D': 'will have',
  'answer': 'C',
  'distractor_strength': {
      'A': 2.0,
      'B': 0.8,
      'D': 1.5,
  },
  'topic': 'Second conditional',
  'subtopic': 'Second conditional',
  'explanation': 'In unreal present conditionals, use past simple after “if”: “If I had enough money…”'},
 {'question_id': 'Q21',
  'question': 'Choose the correct sentence:',
  'option_A': 'The data shows that the theory is correct.',
  'option_B': 'The datas show that the theory is correct.',
  'option_C': ' The data show that the theory is correct.',
  'option_D': 'The data showing that the theory is correct.',
  'answer': 'C',
  'distractor_strength': {
      'A': 2.4,
      'B': 0.9,
      'D': 1.2,
  },
  'topic': 'Subject-verb agreement',
  'subtopic': 'plural noun',
  'explanation': 'Traditionally, “data” is plural in formal academic English, so the verb should be show.'},
 {'question_id': 'Q22',
  'question': 'Choose the best meaning of “brave”:',
  'option_A': 'Afraid',
  'option_B': 'Lazy',
  'option_C': 'Courageous',
  'option_D': 'Angry',
  'answer': 'C',
  'distractor_strength': {
      'A': 1.5,
      'B': 0.7,
      'D': 1.1,
  },
  'topic': 'Vocabulary',
  'subtopic': 'synonym',
  'explanation': '“Brave” means courageous.'},
 {'question_id': 'Q23',
  'question': 'Choose the best revision: Original: “The committee discussed the proposal, and it was '
              'rejected.”',
  'option_A': 'The proposal was discussed and rejected.',
  'option_B': 'The committee discussed the proposal, rejected.',
  'option_C': 'The committee discussed the proposal and rejected it.',
  'option_D': 'The committee discussed and it rejected the proposal.',
  'answer': 'C',
  'distractor_strength': {
      'A': 2.3,
      'B': 1.0,
      'D': 0.8,
  },
  'topic': 'Clear sentence revision',
  'subtopic': 'Clear sentence revision',
  'explanation': '“The committee discussed the proposal and rejected it” removes ambiguity and clearly shows '
                 'who rejected the proposal.'},
 {'question_id': 'Q24',
  'question': 'Choose the correct sentence:',
  'option_A': 'She go to school every day.',
  'option_B': 'She going to school every day.',
  'option_C': 'She goes to school every day.',
  'option_D': 'She gone to school every day.',
  'answer': 'C',
  'distractor_strength': {
      'A': 2.5,
      'B': 1.4,
      'D': 0.8,
  },
  'topic': 'Simple present tense',
  'subtopic': 'Simple present tense',
  'explanation': 'With third-person singular subject “She,” the verb takes -s: “goes.”'},
 {'question_id': 'Q25',
  'question': 'Fill in the blank: The teacher insisted that he ___ the assignment again.',
  'option_A': 'does',
  'option_B': 'do',
  'option_C': 'did',
  'option_D': 'done',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.4,
      'C': 1.5,
      'D': 0.7,
  },
  'topic': 'Subjunctive mood',
  'subtopic': 'Subjunctive mood',
  'explanation': 'After verbs like “insisted,” the base form of the verb is used: “he do.”'},
 {'question_id': 'Q26',
  'question': 'Choose the correct sentence:',
  'option_A': 'He don’t like tea.',
  'option_B': 'He doesn’t likes tea.',
  'option_C': 'He doesn’t like tea.',
  'option_D': 'He not like tea.',
  'answer': 'C',
  'distractor_strength': {
      'A': 1.8,
      'B': 2.6,
      'D': 0.9,
  },
  'topic': 'Negative sentence',
  'subtopic': 'auxiliary verb',
  'explanation': 'With “doesn’t,” the main verb stays in base form: “doesn’t like,” not “doesn’t likes.”'},
 {'question_id': 'Q27',
  'question': 'Identify the type of sentence: “Although it was raining, we continued the match.”',
  'option_A': 'Simple',
  'option_B': 'Compound',
  'option_C': 'Complex',
  'option_D': 'Fragment',
  'answer': 'C',
  'distractor_strength': {
      'A': 1.2,
      'B': 1.8,
      'D': 0.7,
  },
  'topic': 'Sentence types',
  'subtopic': 'Sentence types',
  'explanation': 'The sentence has a dependent clause “Although it was raining” and an independent clause '
                 '“we continued the match,” so it is complex.'},
 {'question_id': 'Q28',
  'question': 'Choose the sentence with the clearest and most elegant style:',
  'option_A': 'Because he was late, he missed the important announcement.',
  'option_B': 'Due to the fact that he was late, he missed the important announcement.',
  'option_C': 'He missed the announcement due to him being late.',
  'option_D': 'The important announcement was missed by him due to lateness.',
  'answer': 'A',
  'distractor_strength': {
      'B': 2.2,
      'C': 1.7,
      'D': 1.1,
  },
  'topic': 'Concise and clear style',
  'subtopic': 'Concise and clear style',
  'explanation': 'Option A is direct, clear, and avoids wordy phrases like “due to the fact that.”'},
 {'question_id': 'Q29',
  'question': 'Identify the error: “Each of the students have submitted their assignment.”',
  'option_A': 'Each',
  'option_B': 'students',
  'option_C': 'assignment',
  'option_D': 'have',
  'answer': 'D',
  'distractor_strength': {
      'A': 1.7,
      'B': 1.4,
      'C': 0.8,
  },
  'topic': 'Subject-verb agreement',
  'subtopic': 'Subject-verb agreement',
  'explanation': '“Each” is singular, so the verb should be has, not “have.”'},
 {'question_id': 'Q30',
  'question': 'Choose the best meaning of “ambiguous”:',
  'option_A': 'Clearly stated',
  'option_B': 'Having more than one possible meaning',
  'option_C': 'Extremely dangerous',
  'option_D': 'Very beautiful',
  'answer': 'B',
  'distractor_strength': {
      'A': 1.8,
      'C': 0.8,
      'D': 0.7,
  },
  'topic': 'Vocabulary',
  'subtopic': 'word meaning',
  'explanation': '“Ambiguous” means having more than one possible meaning.'},
 {'question_id': 'Q31',
  'question': 'Choose the sentence with correct inversion:',
  'option_A': 'Not only did he apologize, but he also offered compensation.',
  'option_B': 'Not only he apologized, but he also offered compensation.',
  'option_C': 'Not only apologized he, but he also offered compensation.',
  'option_D': 'Not only he did apologize, but he also offered compensation.',
  'answer': 'A',
  'distractor_strength': {
      'B': 2.4,
      'C': 1.0,
      'D': 1.7,
  },
  'topic': 'Correlative conjunction',
  'subtopic': 'inversion',
  'explanation': 'After “Not only” at the beginning, auxiliary inversion is needed: “did he apologize.”'},
 {'question_id': 'Q32',
  'question': 'Choose the best word: Her argument was logically strong but emotionally ___.',
  'option_A': 'compelling',
  'option_B': 'fluent',
  'option_C': 'detached',
  'option_D': 'literal',
  'answer': 'C',
  'distractor_strength': {
      'A': 1.8,
      'B': 0.9,
      'D': 1.3,
  },
  'topic': 'Vocabulary',
  'subtopic': 'academic word choice',
  'explanation': '“Detached” means emotionally distant. The sentence contrasts logical strength with lack of '
                 'emotion.'},
 {'question_id': 'Q33',
  'question': 'Identify the error: “The number of students in the class are increasing every year.”',
  'option_A': 'The number',
  'option_B': 'students',
  'option_C': 'every year',
  'option_D': 'are',
  'answer': 'D',
  'distractor_strength': {
      'A': 1.5,
      'B': 1.1,
      'C': 0.6,
  },
  'topic': 'Subject-verb agreement',
  'subtopic': 'Subject-verb agreement',
  'explanation': '“The number” is singular, so the verb should be is, not “are.”'},
 {'question_id': 'Q34',
  'question': 'Choose the sentence with correct subject-verb agreement:',
  'option_A': 'A list of important documents were missing.',
  'option_B': 'A list of important documents was missing.',
  'option_C': 'A list of important documents have been missing.',
  'option_D': 'A list of important documents are missing.',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.5,
      'C': 1.3,
      'D': 2.0,
  },
  'topic': 'Subject-verb agreement',
  'subtopic': 'Subject-verb agreement',
  'explanation': 'The main subject is “A list,” which is singular. Therefore, the correct verb is was.'},
 {'question_id': 'Q35',
  'question': 'Choose the best transformation: Original: “He was tired, but he continued working.”',
  'option_A': 'Despite he was tired, he continued working.',
  'option_B': 'Although being tired, he continued working.',
  'option_C': 'Despite being tired, he continued working.',
  'option_D': 'In spite of the fact that he was tired, he continued working.',
  'answer': 'C',
  'distractor_strength': {
      'A': 1.2,
      'B': 1.6,
      'D': 2.2,
  },
  'topic': 'Sentence transformation',
  'subtopic': 'despite',
  'explanation': '“Despite” should be followed by a noun or gerund phrase, so “Despite being tired” is '
                 'correct.'},
 {'question_id': 'Q36',
  'question': 'Choose the sentence with the correct use of “whom”:',
  'option_A': 'Whom do you think will win the prize?',
  'option_B': 'Whom did you give the book to?',
  'option_C': 'Whom is coming to dinner?',
  'option_D': 'Whom won the prize?',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.1,
      'C': 1.6,
      'D': 1.4,
  },
  'topic': 'Who vs whom',
  'subtopic': 'Who vs whom',
  'explanation': '“Whom” is used as the object. In “did you give the book to,” the receiver is the object.'},
 {'question_id': 'Q37',
  'question': 'Identify the type of clause in the sentence: “The idea that he proposed yesterday was '
              'rejected.” The clause is “that he proposed yesterday.”',
  'option_A': 'Noun clause',
  'option_B': 'Adverb clause',
  'option_C': 'Independent clause',
  'option_D': 'Adjective clause',
  'answer': 'D',
  'distractor_strength': {
      'A': 1.7,
      'B': 0.8,
      'C': 1.0,
  },
  'topic': 'Adjective clause',
  'subtopic': 'relative clause',
  'explanation': '“That he proposed yesterday” describes the noun “idea,” so it functions as an adjective '
                 'clause.'},
 {'question_id': 'Q38',
  'question': 'Choose the most concise sentence:',
  'option_A': 'She left because she was tired.',
  'option_B': 'The reason why she left was because she was tired.',
  'option_C': 'She left due to the reason that she was tired.',
  'option_D': 'The cause of her leaving was tiredness.',
  'answer': 'A',
  'distractor_strength': {
      'B': 2.2,
      'C': 1.8,
      'D': 1.2,
  },
  'topic': 'Conciseness',
  'subtopic': 'Conciseness',
  'explanation': '“She left because she was tired” is the shortest and clearest version without unnecessary '
                 'words.'},
 {'question_id': 'Q39',
  'question': 'Choose the correct sentence:',
  'option_A': 'I wish I was taller.',
  'option_B': 'I wish I were taller.',
  'option_C': 'I wish I am taller.',
  'option_D': 'I wish I will be taller.',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.7,
      'C': 1.0,
      'D': 0.9,
  },
  'topic': 'Subjunctive mood',
  'subtopic': 'Subjunctive mood',
  'explanation': 'For unreal wishes, formal English uses were: “I wish I were taller.”'},
 {'question_id': 'Q40',
  'question': 'Choose the sentence with correct tense consistency:',
  'option_A': 'He said that he is tired yesterday.',
  'option_B': 'He says that he was tired yesterday.',
  'option_C': 'He said that he tired yesterday.',
  'option_D': 'He said that he was tired yesterday.',
  'answer': 'D',
  'distractor_strength': {
      'A': 1.7,
      'B': 1.9,
      'C': 0.8,
  },
  'topic': 'Tense consistency',
  'subtopic': 'Tense consistency',
  'explanation': '“Said” is past tense, and “yesterday” also indicates past time, so “was tired” is '
                 'correct.'},
 {'question_id': 'Q41',
  'question': 'Choose the correct conditional form:',
  'option_A': 'If he would have studied, he would have passed.',
  'option_B': 'If he had studied, he would have passed.',
  'option_C': 'If he studied, he would have passed.',
  'option_D': 'If he had study, he would pass.',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.6,
      'C': 1.8,
      'D': 0.9,
  },
  'topic': 'Third conditional',
  'subtopic': 'Third conditional',
  'explanation': 'The correct third conditional structure is: If + past perfect, would have + past '
                 'participle.'},
 {'question_id': 'Q42',
  'question': 'Identify the error: “Neither the manager nor the employees was prepared for the crisis.”',
  'option_A': 'was',
  'option_B': 'Neither',
  'option_C': 'manager',
  'option_D': 'employees',
  'answer': 'A',
  'distractor_strength': {
      'B': 1.2,
      'C': 0.9,
      'D': 1.8,
  },
  'topic': 'Subject-verb agreement with neither',
  'subtopic': 'nor',
  'explanation': 'With “neither...nor,” the verb usually agrees with the nearer subject. “Employees” is '
                 'plural, so the verb should be were.'},
 {'question_id': 'Q43',
  'question': 'Choose the sentence with correct use of a relative pronoun:',
  'option_A': 'The student which won the prize is my friend.',
  'option_B': 'The student whom won the prize is my friend.',
  'option_C': 'The student who won the prize is my friend.',
  'option_D': 'The student whose won the prize is my friend.',
  'answer': 'C',
  'distractor_strength': {
      'A': 2.2,
      'B': 1.8,
      'D': 1.2,
  },
  'topic': 'Relative pronoun',
  'subtopic': 'Relative pronoun',
  'explanation': 'For a person as the subject of the relative clause, use who: “The student who won…”'},
 {'question_id': 'Q44',
  'question': 'Choose the sentence with the clearest pronoun reference:',
  'option_A': 'When Ali met Rahim, he was angry.',
  'option_B': 'He was angry when Ali met Rahim.',
  'option_C': 'When he met Rahim, Ali was angry.',
  'option_D': 'Ali was angry when he met Rahim.',
  'answer': 'D',
  'distractor_strength': {
      'A': 2.0,
      'B': 1.0,
      'C': 2.4,
  },
  'topic': 'Clear pronoun reference',
  'subtopic': 'Clear pronoun reference',
  'explanation': 'Option D clearly shows that Ali was angry. The pronoun reference is less confusing.'},
 {'question_id': 'Q45',
  'question': 'Choose the sentence that uses a semicolon correctly:',
  'option_A': 'I studied all night; because the exam was difficult.',
  'option_B': 'I studied all night; and the exam was difficult.',
  'option_C': 'I studied; all night the exam was difficult.',
  'option_D': 'I studied all night; the exam was difficult.',
  'answer': 'D',
  'distractor_strength': {
      'A': 1.8,
      'B': 2.0,
      'C': 0.8,
  },
  'topic': 'Semicolon usage',
  'subtopic': 'Semicolon usage',
  'explanation': 'A semicolon correctly joins two closely related independent clauses.'},
 {'question_id': 'Q46',
  'question': 'Choose the sentence with the most precise academic style:',
  'option_A': 'The results prove that the method always works.',
  'option_B': 'The results indicate that the method may be effective.',
  'option_C': 'The results say the method is good.',
  'option_D': 'The results clearly and definitely prove the method.',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.0,
      'C': 1.0,
      'D': 1.4,
  },
  'topic': 'Academic style',
  'subtopic': 'cautious language',
  'explanation': 'Academic writing avoids absolute claims like “prove” or “always.” “Indicate” and “may be '
                 'effective” are more precise and cautious.'},
 {'question_id': 'Q47',
  'question': 'Choose the sentence with the most natural academic tone:',
  'option_A': 'This result suggests that the theory may be valid.',
  'option_B': 'This thing shows that the theory is kind of true.',
  'option_C': 'This stuff proves the theory is totally right.',
  'option_D': 'This result says the theory is correct.',
  'answer': 'A',
  'distractor_strength': {
      'B': 1.2,
      'C': 1.0,
      'D': 1.8,
  },
  'topic': 'Academic tone',
  'subtopic': 'Academic tone',
  'explanation': '“Suggests that the theory may be valid” sounds formal, natural, and appropriately '
                 'cautious.'},
 {'question_id': 'Q48',
  'question': 'Choose the sentence with correct modifier placement:',
  'option_A': 'He almost drove his car for six hours every day.',
  'option_B': 'He drove almost his car for six hours every day.',
  'option_C': 'He drove his car for almost six hours every day.',
  'option_D': 'Almost he drove his car for six hours every day.',
  'answer': 'C',
  'distractor_strength': {
      'A': 2.3,
      'B': 0.8,
      'D': 1.0,
  },
  'topic': 'Modifier placement',
  'subtopic': 'Modifier placement',
  'explanation': '“Almost” should modify “six hours,” so the correct placement is “for almost six hours.”'},
 {'question_id': 'Q49',
  'question': 'It is imperative that she __________ the office immediately to sign the contract.',
  'option_A': 'visits',
  'option_B': 'visit',
  'option_C': 'visited',
  'option_D': 'will visit',
  'answer': 'B',
  'distractor_strength': {
      'A': 2.4,
      'C': 1.5,
      'D': 1.1,
  },
  'topic': 'Subjunctive mood',
  'subtopic': 'Subjunctive mood',
  'explanation': 'After expressions like “It is imperative that,” the base verb is used: “she visit,” not '
                 '“she visits.”'},
 {'question_id': 'Q50',
  'question': 'Choose the sentence with the most logical comparison:',
  'option_A': 'My writing is better than Rahim.',
  'option_B': "My writing is better as Rahim's writing.",
  'option_C': 'My writing is better than Rahim writes.',
  'option_D': "My writing is better than Rahim's writing.",
  'answer': 'D',
  'distractor_strength': {
      'A': 2.3,
      'B': 1.6,
      'C': 1.2,
  },
  'topic': 'Logical comparison',
  'subtopic': 'Logical comparison',
  'explanation': 'The sentence compares “my writing” with “Rahim’s writing.” The comparison must be between '
                 'the same type of thing.'}]


# ---------------------------------------------------------------------
# 2. Assumed inherent difficulty for each question
#    Scale: 1 = very easy, 10 = very hard
# ---------------------------------------------------------------------

ASSUMED_INHERENT_DIFFICULTY = {
    
    "Q1": 6.5,
    "Q2": 1.5,
    "Q3": 4.8,
    "Q4": 9.3,
    "Q5": 2.8,
    "Q6": 6.4,
    "Q7": 4.0,
    "Q8": 1.0,
    "Q9": 6.7,
    "Q10": 8.7,
    "Q11": 2.5,
    "Q12": 3.6,
    "Q13": 7.5,
    "Q14": 1.0,
    "Q15": 8.7,
    "Q16": 5.0,
    "Q17": 5.7,
    "Q18": 5.8,
    "Q19": 8.6,
    "Q20": 4.1,
    "Q21": 8.1,
    "Q22": 3.3,
    "Q23": 9.4,
    "Q24": 2.0,
    "Q25": 8.6,
    "Q26": 2.9,
    "Q27": 4.7,
    "Q28": 9.6,
    "Q29": 7.6,
    "Q30": 3.8,
    "Q31": 9.7,
    "Q32": 3.7,
    "Q33": 7.0,
    "Q34": 6.9,
    "Q35": 7.9,
    "Q36": 4.3,
    "Q37": 9.6,
    "Q38": 7.8,
    "Q39": 5.4,
    "Q40": 5.5,
    "Q41": 6.4,
    "Q42": 8.2,
    "Q43": 4.5,
    "Q44": 8.8,
    "Q45": 9.5,
    "Q46": 8.6,
    "Q47": 9.2,
    "Q48": 5.8,
    "Q49": 8.3,
    "Q50": 8.6,
}


# ---------------------------------------------------------------------
# 3. Utility functions
# ---------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def ability_to_10_scale(ability: int | float) -> float:
    """
    Convert ability from 10-30 scale into 0-10 scale.

    Why?
    - This experiment is designed for students with ability 10-30.
    - This scale is kept for reporting/backward compatibility.
    - Do not use this for direct comparison with perceived difficulty.
    """
    ability = clip(float(ability), MIN_ABILITY, MAX_ABILITY)
    return ((ability - MIN_ABILITY) / (MAX_ABILITY - MIN_ABILITY)) * 10.0


def ability_to_difficulty_scale(ability: int | float) -> float:
    """
    Convert ability from 10-30 scale into the perceived-difficulty scale.

    Perceived difficulty is sampled from DIFFICULTY_LOW..DIFFICULTY_HIGH,
    currently 0..15. This is the scale used for answer probability and
    response-time simulation.
    """
    ability = clip(float(ability), MIN_ABILITY, MAX_ABILITY)
    ability_norm = (ability - MIN_ABILITY) / (MAX_ABILITY - MIN_ABILITY)
    return DIFFICULTY_LOW + ability_norm * (DIFFICULTY_HIGH - DIFFICULTY_LOW)


def make_default_distractor_strength(correct_answer: str) -> dict[str, float]:
    """
    If you do not manually give distractor strengths,
    all wrong options are treated as equally attractive.
    """
    return {option: 1.0 for option in OPTIONS if option != correct_answer}


# ---------------------------------------------------------------------
# 4. Build difficulty profile for ability 10-30
# ---------------------------------------------------------------------

def build_difficulty_profile(inherent_difficulty: float) -> dict[int, dict[str, float]]:
    """
    Creates:
    {
        10: {"center": ..., "spread": ...},
        11: {"center": ..., "spread": ...},
        ...
        30: {"center": ..., "spread": ...},
    }

    center:
        Most likely perceived difficulty for that ability.

    spread:
        How much the sampled difficulty varies around the center.

    Important:
        In this experiment, ability=10 is the weakest supported student and
        ability=30 is the strongest. The model is not trained for absolute
        beginner behavior below ability 10.
    """

    # Weak supported students feel questions harder, but high-difficulty
    # questions should not collapse into one identical perceived difficulty.
    low_ability_center = clip(1.18 * inherent_difficulty + 1.00, 1.0, 13.5)

    # Strong students still find questions easier, but easy and medium-easy
    # questions should remain distinguishable instead of all becoming 0.3.
    high_ability_center = clip(0.45 * inherent_difficulty + 0.35, 0.5, 5.2)

    profile: dict[int, dict[str, float]] = {}

    for ability in range(MIN_ABILITY, MAX_ABILITY + 1):
        t = (ability - MIN_ABILITY) / (MAX_ABILITY - MIN_ABILITY)

        # Linear interpolation from low ability center to high ability center.
        center = low_ability_center * (1.0 - t) + high_ability_center * t

        # Spread can be slightly larger for harder/uncertain questions.
        # Keep it controlled so difficulty stays near the center.
        spread = 0.25 + 0.035 * inherent_difficulty + 0.015 * abs(center - 5.0)
        spread = clip(spread, 0.25, 0.75)

        profile[ability] = {
            "center": round(center, 2),
            "spread": round(spread, 2),
        }

    return profile


def estimate_base_time(question: dict[str, Any], inherent_difficulty: float) -> float:
    """
    Assumed base response time in seconds.

    Higher difficulty + longer text => more base time.
    """
    text = (
        question["question"]
        + " "
        + question["option_A"]
        + " "
        + question["option_B"]
        + " "
        + question["option_C"]
        + " "
        + question["option_D"]
    )

    length_factor = min(len(text) / 90.0, 7.0)
    base_time = 5.0 + inherent_difficulty * 3.2 + length_factor

    return round(base_time, 2)


def attach_assumed_question_metadata(raw_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Adds:
    - inherent_difficulty
    - base_time
    - difficulty_profile
    - distractor_strength
    """
    questions = deepcopy(raw_questions)

    for question in questions:
        qid = question["question_id"]

        inherent_difficulty = ASSUMED_INHERENT_DIFFICULTY[qid]

        question["inherent_difficulty"] = inherent_difficulty
        question["base_time"] = estimate_base_time(question, inherent_difficulty)
        question["difficulty_profile"] = build_difficulty_profile(inherent_difficulty)

        if "distractor_strength" not in question:
            question["distractor_strength"] = make_default_distractor_strength(question["answer"])

    return questions


# This is the final question list you should use in simulation.
QUESTIONS_WITH_METADATA = attach_assumed_question_metadata(QUESTIONS)


# ---------------------------------------------------------------------
# 5. Difficulty sampling
# ---------------------------------------------------------------------

def sample_difficulty_from_distribution(
    center: float,
    spread: float,
    rng: np.random.Generator,
    low: float = DIFFICULTY_LOW,
    high: float = DIFFICULTY_HIGH,
    step: float = 0.1,
) -> float:
    """
    Discrete bell-shaped distribution.

    Highest probability is near 'center'.
    Probability decreases on both left and right sides.
    """
    values = np.round(np.arange(low, high + step, step), 2)

    weights = np.exp(-0.5 * ((values - center) / spread) ** 2)
    probabilities = weights / weights.sum()

    sampled = rng.choice(values, p=probabilities)
    return float(sampled)


def sample_perceived_difficulty(
    student: dict[str, Any],
    question: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    """
    Picks the difficulty profile row for the student's ability,
    then samples one perceived difficulty value.
    """
    ability = int(student["ability"])

    if ability < MIN_ABILITY or ability > MAX_ABILITY:
        raise ValueError(f"Student ability must be between {MIN_ABILITY} and {MAX_ABILITY}. Got {ability}.")

    params = question["difficulty_profile"][ability]
    center = params["center"]
    spread = params["spread"]

    return sample_difficulty_from_distribution(center=center, spread=spread, rng=rng)


# ---------------------------------------------------------------------
# 6. Option distribution
# ---------------------------------------------------------------------

def get_option_distribution(
    student: dict[str, Any],
    question: dict[str, Any],
    perceived_difficulty: float,
) -> dict[str, float]:
    """
    Returns probability of choosing each option.

    Example:
    {
        "A": 0.70,
        "B": 0.10,
        "C": 0.12,
        "D": 0.08,
    }
    """

    ability_difficulty_scale = ability_to_difficulty_scale(student["ability"])
    correct = question["answer"]

    # If ability on the same scale is greater than perceived difficulty,
    # mastery becomes high.
    mastery = sigmoid(0.85 * (ability_difficulty_scale - perceived_difficulty))

    # Random guessing floor for 4-option MCQ.
    min_correct = 1.0 / len(OPTIONS)

    # Even a strong student can make careless mistakes.
    carelessness = float(student.get("carelessness", 0.05))
    max_correct = clip(0.98 - carelessness, min_correct, 0.98)

    p_correct = min_correct + (max_correct - min_correct) * mastery
    p_correct = clip(p_correct, min_correct, max_correct)

    distribution: dict[str, float] = {}
    distribution[correct] = p_correct

    wrong_probability = 1.0 - p_correct

    distractors = question.get("distractor_strength") or make_default_distractor_strength(correct)

    # Safety: only wrong options should be in distractors.
    distractors = {opt: strength for opt, strength in distractors.items() if opt != correct}

    total_strength = sum(distractors.values())

    if total_strength <= 0:
        distractors = make_default_distractor_strength(correct)
        total_strength = sum(distractors.values())

    for option, strength in distractors.items():
        distribution[option] = wrong_probability * (strength / total_strength)

    # Ensure all options exist.
    for option in OPTIONS:
        distribution.setdefault(option, 0.0)

    # Normalize for numerical safety.
    total = sum(distribution.values())
    distribution = {option: prob / total for option, prob in distribution.items()}

    return distribution


# ---------------------------------------------------------------------
# 7. Response time sampling
# ---------------------------------------------------------------------

def sample_response_time(
    student: dict[str, Any],
    question: dict[str, Any],
    perceived_difficulty: float,
    rng: np.random.Generator,
) -> float:
    """
    Generates different time every time, even for same student + same question.

    Uses same sampled perceived_difficulty that was used for answer probability.
    """

    base_time = float(question["base_time"])
    ability_difficulty_scale = ability_to_difficulty_scale(student["ability"])

    time_multiplier = float(student.get("time_multiplier", 1.0))

    difficulty_gap = perceived_difficulty - ability_difficulty_scale

    # If question feels harder than student ability, time increases.
    struggle_factor = 1.0 + max(0.0, difficulty_gap) * 0.095

    # If question feels easier, time can become slightly lower.
    ease_factor = 1.0 - min(max(0.0, ability_difficulty_scale - perceived_difficulty) * 0.017, 0.25)

    min_time = max(2.0, base_time * time_multiplier * 0.35 * ease_factor)
    max_time = max(min_time + 1.0, base_time * time_multiplier * 2.0 * struggle_factor)

    alpha = float(student.get("time_alpha", 2.5))
    beta = float(student.get("time_beta", 4.0))

    x = rng.beta(alpha, beta)

    time_taken = min_time + x * (max_time - min_time)

    return round(float(time_taken), 2)


# ---------------------------------------------------------------------
# 8. Student creation
# ---------------------------------------------------------------------

def create_student(
    student_id: str,
    ability: int,
    carelessness: float | None = None,
    time_multiplier: float | None = None,
    time_alpha: float | None = None,
    time_beta: float | None = None,
) -> dict[str, Any]:
    """
    Create one simulated student.

    ability:
        10 = weakest supported student in this experiment
        30 = strongest supported student in this experiment
    """

    if ability < MIN_ABILITY or ability > MAX_ABILITY:
        raise ValueError(f"ability must be between {MIN_ABILITY} and {MAX_ABILITY}")

    ability_10 = ability_to_10_scale(ability)
    ability_difficulty_scale = ability_to_difficulty_scale(ability)

    if carelessness is None:
        # Stronger students are usually less careless, but not zero.
        carelessness = clip(0.16 - ability_10 * 0.011, 0.025, 0.16)

    if time_multiplier is None:
        # Stronger students are usually faster.
        time_multiplier = clip(1.35 - ability_10 * 0.055, 0.65, 1.35)

    if time_alpha is None:
        # Lower alpha / higher beta means values tend to be closer to min_time.
        time_alpha = clip(2.2 + ability_10 * 0.06, 2.0, 3.0)

    if time_beta is None:
        time_beta = clip(3.2 + ability_10 * 0.10, 3.0, 4.5)

    return {
        "student_id": student_id,
        "ability": int(ability),
        "ability_10": round(ability_10, 2),
        "ability_difficulty_scale": round(ability_difficulty_scale, 2),
        "carelessness": round(float(carelessness), 3),
        "time_multiplier": round(float(time_multiplier), 3),
        "time_alpha": round(float(time_alpha), 3),
        "time_beta": round(float(time_beta), 3),
    }


def create_student_population(
    variants_per_ability: int = 1,
    seed: int | None = None,
    student_id_prefix: str = "S",
) -> list[dict[str, Any]]:
    """Create varied student profiles across the supported ability range.

    Ability controls the main knowledge level. Variants add small differences in
    carelessness and speed so the model does not see only one personality for a
    given ability.
    """

    if variants_per_ability < 1:
        raise ValueError("variants_per_ability must be at least 1")

    rng = np.random.default_rng(seed)
    population: list[dict[str, Any]] = []

    for ability in range(MIN_ABILITY, MAX_ABILITY + 1):
        base = create_student(f"{student_id_prefix}{ability:02d}", ability=ability)

        if variants_per_ability == 1:
            population.append(base)
            continue

        for variant in range(1, variants_per_ability + 1):
            carelessness = clip(
                float(base["carelessness"]) + float(rng.normal(0.0, 0.025)),
                0.02,
                0.22,
            )
            time_multiplier = clip(
                float(base["time_multiplier"]) * float(rng.lognormal(mean=0.0, sigma=0.10)),
                0.55,
                1.65,
            )
            time_alpha = clip(
                float(base["time_alpha"]) + float(rng.normal(0.0, 0.12)),
                1.7,
                3.4,
            )
            time_beta = clip(
                float(base["time_beta"]) + float(rng.normal(0.0, 0.15)),
                2.5,
                5.0,
            )

            population.append(
                create_student(
                    f"{student_id_prefix}{ability:02d}_v{variant:02d}",
                    ability=ability,
                    carelessness=carelessness,
                    time_multiplier=time_multiplier,
                    time_alpha=time_alpha,
                    time_beta=time_beta,
                )
            )

    return population


# ---------------------------------------------------------------------
# 9. Simulate answer
# ---------------------------------------------------------------------

def simulate_answer(
    student: dict[str, Any],
    question: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """
    Simulates one student answering one question.
    """

    if rng is None:
        rng = np.random.default_rng()

    perceived_difficulty = sample_perceived_difficulty(student, question, rng)

    option_distribution = get_option_distribution(
        student=student,
        question=question,
        perceived_difficulty=perceived_difficulty,
    )

    probs = [option_distribution[opt] for opt in OPTIONS]

    chosen_option = str(rng.choice(OPTIONS, p=probs))
    time_taken = sample_response_time(
        student=student,
        question=question,
        perceived_difficulty=perceived_difficulty,
        rng=rng,
    )

    is_correct = chosen_option == question["answer"]

    return {
        "student_id": student["student_id"],
        "student_ability": student["ability"],
        "student_ability_10": student.get("ability_10", ability_to_10_scale(student["ability"])),
        "student_ability_difficulty_scale": student.get(
            "ability_difficulty_scale",
            ability_to_difficulty_scale(student["ability"]),
        ),

        "question_id": question["question_id"],
        "question": question["question"],
        "topic": question["topic"],
        "subtopic": question["subtopic"],

        "inherent_difficulty": question["inherent_difficulty"],
        "base_time": question["base_time"],
        "sampled_perceived_difficulty": perceived_difficulty,

        "chosen_option": chosen_option,
        "correct_answer": question["answer"],
        "is_correct": is_correct,
        "time_taken": time_taken,

        "option_distribution": option_distribution,
    }


def simulate_dataset(
    students: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    attempts_per_student_question: int = 1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Simulate many student-question interactions.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for student in students:
        for question in questions:
            for _ in range(attempts_per_student_question):
                rows.append(simulate_answer(student, question, rng))

    return rows


# ---------------------------------------------------------------------
# 10. Example run
# ---------------------------------------------------------------------


# Preserve the baseline student factory for the dynamic factory below.
create_base_student = create_student


# ---------------------------------------------------------------------
# Dynamic student behavior
# ---------------------------------------------------------------------

def _all_topics() -> list[str]:
    return sorted({str(question["topic"]) for question in QUESTIONS_WITH_METADATA})


def create_student(
    student_id: str,
    ability: int,
    carelessness: float | None = None,
    time_multiplier: float | None = None,
    time_alpha: float | None = None,
    time_beta: float | None = None,
    confidence: float | None = None,
    fatigue: float | None = None,
    guessing_tendency: float | None = None,
    speed_tendency: float | None = None,
    learning_rate: float | None = None,
    fatigue_rate: float | None = None,
    recovery_rate: float | None = None,
    time_noise_sigma: float | None = None,
    topic_mastery_bias: dict[str, float] | None = None,
) -> dict[str, Any]:
    student = create_base_student(
        student_id=student_id,
        ability=ability,
        carelessness=carelessness,
        time_multiplier=time_multiplier,
        time_alpha=time_alpha,
        time_beta=time_beta,
    )

    ability_10 = float(student["ability_10"])

    if confidence is None:
        confidence = clip(0.25 + ability_10 * 0.045, 0.15, 0.85)
    if fatigue is None:
        fatigue = 0.05
    if guessing_tendency is None:
        guessing_tendency = clip(0.30 - ability_10 * 0.020, 0.04, 0.30)
    if speed_tendency is None:
        speed_tendency = 1.0
    if learning_rate is None:
        learning_rate = clip(0.08 + ability_10 * 0.004, 0.08, 0.14)
    if fatigue_rate is None:
        fatigue_rate = clip(0.16 - ability_10 * 0.006, 0.07, 0.16)
    if recovery_rate is None:
        recovery_rate = clip(0.07 + ability_10 * 0.006, 0.07, 0.14)
    if time_noise_sigma is None:
        time_noise_sigma = 0.16

    student.update(
        {
            "confidence": round(float(confidence), 3),
            "fatigue": round(float(fatigue), 3),
            "guessing_tendency": round(float(guessing_tendency), 3),
            "speed_tendency": round(float(speed_tendency), 3),
            "learning_rate": round(float(learning_rate), 3),
            "fatigue_rate": round(float(fatigue_rate), 3),
            "recovery_rate": round(float(recovery_rate), 3),
            "time_noise_sigma": round(float(time_noise_sigma), 3),
            "topic_mastery_bias": {
                str(topic): round(float(bias), 3)
                for topic, bias in (topic_mastery_bias or {}).items()
                if abs(float(bias)) > 1e-9
            },
        }
    )
    return student


def ensure_dynamic_student(student: dict[str, Any]) -> dict[str, Any]:
    ability = int(round(clip(float(student["ability"]), MIN_ABILITY, MAX_ABILITY)))
    dynamic = create_student(
        student_id=str(student.get("student_id", "dynamic_student")),
        ability=ability,
        carelessness=student.get("carelessness"),
        time_multiplier=student.get("time_multiplier"),
        time_alpha=student.get("time_alpha"),
        time_beta=student.get("time_beta"),
        confidence=student.get("confidence"),
        fatigue=student.get("fatigue"),
        guessing_tendency=student.get("guessing_tendency"),
        speed_tendency=student.get("speed_tendency"),
        learning_rate=student.get("learning_rate"),
        fatigue_rate=student.get("fatigue_rate"),
        recovery_rate=student.get("recovery_rate"),
        time_noise_sigma=student.get("time_noise_sigma"),
        topic_mastery_bias=student.get("topic_mastery_bias"),
    )
    return dynamic


def with_updated_ability(student: dict[str, Any], ability: int | float) -> dict[str, Any]:
    updated = ensure_dynamic_student(student)
    ability_int = int(round(clip(float(ability), MIN_ABILITY, MAX_ABILITY)))
    updated["ability"] = ability_int
    updated["ability_10"] = round(ability_to_10_scale(ability_int), 2)
    updated["ability_difficulty_scale"] = round(ability_to_difficulty_scale(ability_int), 2)
    return updated


def choose_acting_ability(
    real_ability: int,
    rng: np.random.Generator,
) -> tuple[int, str]:
    """
    Pick a hidden acting ability independently for each question.

    The student's real ability is unchanged. acting_ability is used only for the
    current simulated answer/time so same-ability students do not behave almost
    deterministically across all questions.
    """
    real_ability = int(round(clip(real_ability, MIN_ABILITY, MAX_ABILITY)))
    roll = float(rng.random())

    if 10 <= real_ability <= 16:
        if roll < 0.75:
            return real_ability, "normal"
        low = min(MAX_ABILITY, real_ability + 8)
        return int(rng.integers(low, MAX_ABILITY + 1)), "boost"

    if 17 <= real_ability <= 23:
        if roll < 0.75:
            return real_ability, "normal"
        if roll < 0.88:
            low = min(MAX_ABILITY, real_ability + 5)
            return int(rng.integers(low, MAX_ABILITY + 1)), "boost"
        high = max(MIN_ABILITY, real_ability - 5)
        return int(rng.integers(MIN_ABILITY, high + 1)), "slump"

    if roll < 0.75:
        return real_ability, "normal"
    high = max(MIN_ABILITY, real_ability - 9)
    return int(rng.integers(MIN_ABILITY, high + 1)), "slump"


def create_student_population(
    variants_per_ability: int = 1,
    seed: int | None = None,
    student_id_prefix: str = "DS",
) -> list[dict[str, Any]]:
    if variants_per_ability < 1:
        raise ValueError("variants_per_ability must be at least 1")

    rng = np.random.default_rng(seed)
    topics = _all_topics()
    population: list[dict[str, Any]] = []

    for ability in range(MIN_ABILITY, MAX_ABILITY + 1):
        for variant in range(1, variants_per_ability + 1):
            base = create_base_student(f"{student_id_prefix}{ability:02d}", ability=ability)

            topic_bias = {
                topic: float(rng.normal(0.0, 0.75))
                for topic in topics
            }

            carelessness = clip(
                float(base["carelessness"]) + float(rng.normal(0.0, 0.035)),
                0.02,
                0.28,
            )
            time_multiplier = clip(
                float(base["time_multiplier"]) * float(rng.lognormal(mean=0.0, sigma=0.16)),
                0.50,
                1.80,
            )
            confidence = clip(
                0.25 + float(base["ability_10"]) * 0.045 + float(rng.normal(0.0, 0.10)),
                0.10,
                0.90,
            )
            fatigue = clip(float(rng.beta(1.5, 8.0)) * 0.45, 0.0, 0.45)
            guessing_tendency = clip(
                0.30 - float(base["ability_10"]) * 0.020 + float(rng.normal(0.0, 0.045)),
                0.03,
                0.42,
            )
            speed_tendency = clip(float(rng.lognormal(mean=0.0, sigma=0.13)), 0.70, 1.35)
            time_noise_sigma = clip(float(rng.normal(0.17, 0.04)), 0.08, 0.28)

            population.append(
                create_student(
                    student_id=f"{student_id_prefix}{ability:02d}_v{variant:02d}",
                    ability=ability,
                    carelessness=carelessness,
                    time_multiplier=time_multiplier,
                    confidence=confidence,
                    fatigue=fatigue,
                    guessing_tendency=guessing_tendency,
                    speed_tendency=speed_tendency,
                    time_noise_sigma=time_noise_sigma,
                    topic_mastery_bias=topic_bias,
                )
            )

    return population


def get_option_distribution(
    student: dict[str, Any],
    question: dict[str, Any],
    perceived_difficulty: float,
) -> dict[str, float]:
    student = ensure_dynamic_student(student)
    ability_scale = ability_to_difficulty_scale(student["ability"])
    topic = str(question["topic"])
    topic_bias = float(student.get("topic_mastery_bias", {}).get(topic, 0.0))
    confidence = float(student.get("confidence", 0.50))
    fatigue = float(student.get("fatigue", 0.0))
    guessing = float(student.get("guessing_tendency", 0.10))

    effective_scale = (
        ability_scale
        + topic_bias
        + 0.75 * (confidence - 0.50)
        - 1.05 * fatigue
        - 0.45 * guessing
    )

    mastery = sigmoid(0.58 * (effective_scale - perceived_difficulty))
    correct = question["answer"]
    min_correct = 1.0 / len(OPTIONS)

    carelessness = float(student.get("carelessness", 0.05))
    effective_carelessness = clip(
        carelessness
        + 0.05 * guessing
        + 0.06 * fatigue
        - 0.03 * confidence,
        0.02,
        0.30,
    )
    max_correct = clip(0.92 - effective_carelessness, min_correct + 0.05, 0.92)
    p_correct = clip(min_correct + (max_correct - min_correct) * mastery, min_correct, max_correct)

    distribution: dict[str, float] = {correct: p_correct}
    wrong_probability = 1.0 - p_correct
    distractors = question.get("distractor_strength") or make_default_distractor_strength(correct)
    distractors = {opt: strength for opt, strength in distractors.items() if opt != correct}

    total_strength = sum(distractors.values())
    if total_strength <= 0:
        distractors = make_default_distractor_strength(correct)
        total_strength = sum(distractors.values())

    for option, strength in distractors.items():
        distribution[option] = wrong_probability * (strength / total_strength)

    for option in OPTIONS:
        distribution.setdefault(option, 0.0)

    total = sum(distribution.values())
    return {option: prob / total for option, prob in distribution.items()}


def sample_response_time(
    student: dict[str, Any],
    question: dict[str, Any],
    perceived_difficulty: float,
    rng: np.random.Generator,
) -> float:
    student = ensure_dynamic_student(student)
    base_time = float(question["base_time"])
    ability_scale = ability_to_difficulty_scale(student["ability"])
    topic = str(question["topic"])
    topic_bias = float(student.get("topic_mastery_bias", {}).get(topic, 0.0))
    confidence = float(student.get("confidence", 0.50))
    fatigue = float(student.get("fatigue", 0.0))
    guessing = float(student.get("guessing_tendency", 0.10))

    effective_scale = ability_scale + topic_bias + 0.50 * (confidence - 0.50) - 0.85 * fatigue

    time_multiplier = float(student.get("time_multiplier", 1.0))
    time_multiplier *= float(student.get("speed_tendency", 1.0))
    time_multiplier *= 1.0 + 0.25 * fatigue - 0.10 * confidence + 0.12 * guessing
    time_multiplier = clip(time_multiplier, 0.45, 1.90)

    difficulty_gap = perceived_difficulty - effective_scale
    struggle_factor = 1.0 + max(0.0, difficulty_gap) * 0.095
    ease_factor = 1.0 - min(max(0.0, effective_scale - perceived_difficulty) * 0.017, 0.25)

    min_time = max(2.0, base_time * time_multiplier * 0.35 * ease_factor)
    max_time = max(min_time + 1.0, base_time * time_multiplier * 2.0 * struggle_factor)

    alpha = float(student.get("time_alpha", 2.5))
    beta = float(student.get("time_beta", 4.0))
    x = rng.beta(alpha, beta)

    time_taken = min_time + x * (max_time - min_time)
    time_taken *= float(rng.lognormal(mean=0.0, sigma=float(student.get("time_noise_sigma", 0.16))))
    time_taken = clip(time_taken, 2.0, max_time * 1.35)
    return round(float(time_taken), 2)


def simulate_answer(
    student: dict[str, Any],
    question: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng()

    student = ensure_dynamic_student(student)
    real_ability = int(student["ability"])
    acting_ability, acting_mode = choose_acting_ability(real_ability, rng)
    acting_student = with_updated_ability(student, acting_ability)

    perceived_difficulty = sample_perceived_difficulty(acting_student, question, rng)
    option_distribution = get_option_distribution(acting_student, question, perceived_difficulty)
    probs = [option_distribution[opt] for opt in OPTIONS]
    chosen_option = str(rng.choice(OPTIONS, p=probs))
    time_taken = sample_response_time(acting_student, question, perceived_difficulty, rng)
    is_correct = chosen_option == question["answer"]

    return {
        "student_id": student["student_id"],
        "student_ability": student["ability"],
        "student_ability_10": student["ability_10"],
        "student_ability_difficulty_scale": student["ability_difficulty_scale"],
        "acting_mode": acting_mode,
        "acting_ability": acting_ability,
        "acting_ability_10": acting_student["ability_10"],
        "acting_ability_difficulty_scale": acting_student["ability_difficulty_scale"],
        "student_confidence": student["confidence"],
        "student_fatigue": student["fatigue"],
        "student_guessing_tendency": student["guessing_tendency"],
        "student_speed_tendency": student["speed_tendency"],
        "question_id": question["question_id"],
        "question": question["question"],
        "topic": question["topic"],
        "subtopic": question["subtopic"],
        "inherent_difficulty": question["inherent_difficulty"],
        "base_time": question["base_time"],
        "sampled_perceived_difficulty": perceived_difficulty,
        "chosen_option": chosen_option,
        "correct_answer": question["answer"],
        "is_correct": is_correct,
        "time_taken": time_taken,
        "option_distribution": option_distribution,
    }


def update_dynamic_student_state(
    student: dict[str, Any],
    question: dict[str, Any],
    is_correct: bool,
    time_ratio: float,
) -> dict[str, Any]:
    updated = ensure_dynamic_student(student)
    topic = str(question["topic"])
    topic_bias = deepcopy(updated.get("topic_mastery_bias", {}))
    learning_rate = float(updated.get("learning_rate", 0.10))
    fatigue_rate = float(updated.get("fatigue_rate", 0.10))
    recovery_rate = float(updated.get("recovery_rate", 0.10))

    confidence = float(updated.get("confidence", 0.50))
    fatigue = float(updated.get("fatigue", 0.0))
    guessing = float(updated.get("guessing_tendency", 0.10))
    topic_delta = 0.0

    if is_correct:
        if time_ratio <= 0.80:
            confidence += recovery_rate * 0.95
            fatigue -= recovery_rate * 0.75
            guessing -= 0.020
            topic_delta = learning_rate * 0.95
        elif time_ratio <= 1.50:
            confidence += recovery_rate * 0.55
            fatigue -= recovery_rate * 0.35
            guessing -= 0.012
            topic_delta = learning_rate * 0.60
        elif time_ratio <= 2.20:
            confidence += recovery_rate * 0.10
            fatigue += 0.025
            topic_delta = learning_rate * 0.20
        else:
            confidence -= 0.035
            fatigue += 0.060
            topic_delta = learning_rate * 0.05
    else:
        if time_ratio < 0.70:
            confidence -= 0.060
            fatigue += 0.045
            guessing += 0.040
            topic_delta = -learning_rate * 0.40
        elif time_ratio <= 1.50:
            confidence -= 0.090
            fatigue += 0.075
            guessing += 0.020
            topic_delta = -learning_rate * 0.75
        else:
            confidence -= 0.120
            fatigue += fatigue_rate
            guessing -= 0.005
            topic_delta = -learning_rate * 1.00

    fatigue += 0.008
    topic_bias[topic] = clip(float(topic_bias.get(topic, 0.0)) + topic_delta, -3.0, 3.0)

    updated["confidence"] = round(clip(confidence, 0.05, 0.95), 4)
    updated["fatigue"] = round(clip(fatigue, 0.0, 1.0), 4)
    updated["guessing_tendency"] = round(clip(guessing, 0.02, 0.45), 4)
    updated["topic_mastery_bias"] = {
        str(key): round(float(value), 4)
        for key, value in topic_bias.items()
        if abs(float(value)) > 1e-9
    }
    updated["last_dynamic_topic"] = topic
    updated["last_dynamic_topic_delta"] = round(topic_delta, 4)
    return updated

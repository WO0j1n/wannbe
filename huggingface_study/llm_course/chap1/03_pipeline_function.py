# Transformers pipeline function example
# pipeline? -> 라고 하면 pipeline에 대한 각종 다양한 기능을 확인할 수 있음
# 모델 추론(inference)에 필요한 모든 절차를 하나로 묶어서, 사용자가 모델 내부를 몰라도 바로 결과를 얻을 수 있게 해주는 고수준 API


# 텍스트 파이프라인
    # text-generation - 프롬프트에서 텍스트를 생성합니다.
    # text-classification - 텍스트를 미리 정의된 범주로 분류합니다.
    # summarization - 핵심 정보는 유지하면서 텍스트를 더 짧게 줄이세요.
    # translation - 한 언어에서 다른 언어로 텍스트를 번역합니다.
    # zero-shot-classification - 특정 레이블에 대한 사전 학습 없이 다른 분야에서의 텍스트를 분류합니다.
    # feature-extraction - 텍스트의 벡터 표현을 추출합니다.

# 이미지 파이프라인
    # image-to-text - 이미지에 대한 텍스트 설명을 생성합니다.
    # image-classification - 이미지에서 객체를 식별하세요
    # object-detection - 이미지에서 물체를 찾고 식별합니다.

# 오디오 파이프라인
    # automatic-speech-recognition - 음성을 텍스트로 변환
    # audio-classification - 오디오를 카테고리별로 분류합니다.
    # text-to-speech - 텍스트를 음성 오디오로 변환

# 멀티모달 파이프라인
    # image-text-to-text - 제시된 텍스트에 따라 이미지에 응답하세요.

from transformers import pipeline

classifier = pipeline('sentiment-analysis') # 감정 분석 파이프라인 생성
result = classifier("I love using transformers library!")
print(result)


result2 = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)
print(result2)


zero_shot_classifier = pipeline('zero-shot-classification') # 제로샷 분류 파이프라인 생성
result3 = zero_shot_result = zero_shot_classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

print(result3)

generator = pipeline('text-generation') # 텍스트 생성 파이프라인 생성
result4 = generator("In a future world, AI and humans") 
print(result4)

generator = pipeline("text-generation", model="distilgpt2") # distilgpt2 모델을 사용하는 텍스트 생성 파이프라인 생성
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

unmasker = pipeline("fill-mask") # 마스크 채우기 파이프라인 생성
result5 = unmasker("This course will teach you all about <mask> models.", top_k=2)
print(result5)


ner = pipeline("ner", grouped_entities=True) # 개체명 인식 파이프라인 생성
print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

question_answerer = pipeline("question-answering") # 질문 답변 파이프라인 생성
print(question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
))



summarizer = pipeline("summarization")
print(summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
))


translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")# 프랑스어-영어 번역 파이프라인 생성
print(translator("Ce cours est produit par Hugging Face."))
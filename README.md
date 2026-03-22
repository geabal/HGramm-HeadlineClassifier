
# HGramm

HGramm(Headline classifier by Grammar)는 한글 문법 요소를 바탕으로 표제어(headline) 문장을 찾는 분류기입니다.    
JusText에서는 웹문서에서 유의미한 정보를 수집할 때, 불용어가 많은 문장은 본문에 가깝다는 text heuristic을 사용합니다. 이 개념을 한글 문법에 맞춰 응용한 표제어(비본문)-본문을 구분하는 분류기를 만들어보았습니다.
  
**참고사항**

JusText와 Trafilatura는 HTML 본문 전체를 사용하지만, 해당 프로젝트에서는 HTML에서 추출된 text만을 사용합니다. 즉, 앞서 언급한 두 라이브러리와 다르게 태그 정보를 활용하지 않고 텍스트 문법과 의미만으로 본문 영역을 선출합니다.  
제가 수집한 텍스트 데이터에 태그 정보가 없어서 개인적인 용도로 사용하기 위해 제작했습니다. 너른 마음으로 봐주시면 감사하겠습니다.  

## 모델 소개


본문 영역을 추출할 때는 표제어 분류기 HGramm의 결과와 LLM 벡터 임베딩을 함께 사용합니다. LLM은 sentence-transformers로 호출해 사용합니다.  
(2026.03.22 추가) LLM을 사용할 수 없는 환경에서도 HGramm을 본문 영역을 선출할 수 있도록 코드를 업데이트했습니다. 이제 HGramm만 사용해서 본문 영역을 선출할 수 있습니다.
  
1. HGramm  
문서 내 각 문장이 표제어일 확률(is_head) 계산하는 GBM(Gradient Boosting Machine) 모델입니다. is_head가 급격히 변하는 지점을 비본문-본문 전환 후보 영역으로 지정합니다.  
모델 학습 시에는 제가 직접 수집한 문장 데이터(59,250개)와 빅카인즈에서 제공하는 문장 데이터(159,420개)를 가공해 사용했습니다. 총 218,670개의 문장을 사용해 학습했습니다.
  
2. LLM(optional)  
비본문-본문 전환 후보 영역에서 명확한 전환점을 정의할 때 사용합니다. 의미상으로 가장 거리가 먼 지점을 전환점으로 정의합니다. (계산 기준: 코사인 유사도).  
모델은 sentence-transformers로 허깅페이스 모델을 불러와 사용합니다. 저는 HyperCLOVAX-SEED-Text-Instruct-0.5B를 사용했으나, 다른 모델 무엇이든 사용하셔도 됩니다. 개인적으로 한국어 이해도가 좋은 모델을 추천합니다.
  

## HGramm target label: is_head


- 표제어(headline): 1에 가까울수록 표제어에 가깝습니다. 제목이나 광고 문구에 사용되는 유형의 문장입니다.
- 본문(text): 0에 가까울수록 본문 문장에 가깝습니다. 조사나 어미가 생략되지 않은 온전한 문장이 이 유형에 속합니다.

## requirements


- mecab : 전처리 시 POS 태깅에 사용합니다.
- pandas : 데이터 입력 형식을 pandas.DataFrame을 받습니다.
- sentence-transformers(optional) : 문장 벡터 임베딩에 사용합니다. llm을 사용하지 않는다면 설치되어있지 않아도 됩니다.

---

# 사용 예제

아래 사용 예제는 HGramm을 사용해 본문영역(textarea)를 선출하는 코드입니다.  
자세한 사용 설명 및 전체 사용 과정은 tutorial 폴더에서 확인해주세요.

## import


```python
from HGramm import HGramm

#HGramm 객체를 불러옵니다
hg = HGramm()

# llm 경로 설정은 옵션입니다. HGramm만 사용한다면 llm 경로 설정은 필요 없습니다.
# llm 경로는 local path나 hugging face url을 사용합니다.
# llm_path는 SentenceTransformer(llm_path)로 모델을 불러올 때 사용됩니다.
llm_path = "C:/Users/psjoy/huggingface_models/HyperCLOVAX-SEED-Text-Instruct-0.5B"
hg.set_llm(llm_path=llm_path)

# HGramm에서 본문 선출을 하기 전에 반드시 mecab 설정이 필요합니다.
# mecab 사용을 위한 dicpath 경로를 설정합니다
mecab_path = 'C:/mecab/mecab-ko-dic'
hg.set_mecab(mecab_path=mecab_path)
```

```python
import json

# 본문 영역을 선출할 raw 데이터 로드
filename = 'tutorial/test_data.json'
try:
    file_handle = open(filename, 'r', encoding='UTF8')
    data = json.load(file_handle)
except Exception as e:
    print(e)
```

## Test 1: 본문 선출


```python
# data: 여러개의 문서 정보를 List[Dict]형으로 받습니다. 각 Dict 객체는 하나의 문서를 의미합니다.
# text_col: 문서의 텍스트 정보가 담긴 key 값입니다. 텍스트는 여러 개의 문장을 List[str]로 입력받습니다.
text_col = 'text'
# id_col: 각 문서를 구분하는 id 정보가 담긴 key 값입니다.
id_col = 'url'
# is_llm: True라면 HGramm과 LLM으로 본문 영역을 선출합니다. False일 때는 LLM 없이 HGramm만으로 본문 영역을 선출합니다. (기본값: False)
df_textarea = hg.get_textarea(data, text_col=text_col, id_col=id_col, is_llm=True)
```

## Test 2: is_head

```python
# is_head는 표제어에 가까운 문장일수록 1에 가까운 값을 갖습니다.
sent = df_textarea.loc[0,'text'][0]
is_head = df_textarea.loc[0,'is_head'][0]
print('[표제어 문장]\nis_head는 표제어에 가까운 문장일수록 1에 가까운 값을 갖습니다.')
print(f'문장: {sent}')
print(f'is_head socre: {is_head}\n')

# 반대로 본문 문장에 가까울수록 0에 가까운 값을 가집니다.
sent = df_textarea.loc[0,'text'][5]
is_head = df_textarea.loc[0,'is_head'][5]
print('[본문 문장]\n반대로 본문 문장에 가까울수록 is_head는 0에 가까운 값을 가집니다.')
print(f'문장: {sent}')
print(f'is_head socre: {is_head}\n')
```

```python
'''
[output]

[표제어 문장]
is_head는 표제어에 가까운 문장일수록 1에 가까운 값을 갖습니다.
문장: LG유플러스, 서울 지하철 9호선 1·2·3단계 LTE-R 구축 완료
is_head socre: 0.9538815298700234

[본문 문장]
반대로 본문 문장에 가까울수록 is_head는 0에 가까운 값을 가집니다.
문장: LG유플러스가 서울시메트로9호선(주), 서울교통공사 9호선운영부문과 함께 서울 지하철 9호선 전 구간에 ‘LTE-R(철도통합무선망)’ 구축을 완료했다고 16일 밝혔다.
is_head socre: 0.0066271301405524275


'''
```

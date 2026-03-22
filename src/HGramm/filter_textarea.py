import pandas as pd
from typing import List, Dict
import os
from konlpy.tag import Mecab
from .preprocessing import clean_text
from copy import deepcopy
from collections import defaultdict
import pickle

from sentence_transformers import SentenceTransformer
import torch

from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
  '''
  코사인 유사도 계산
  
  :param A: np.array[float]. 코사인 유사도를 계산할 벡터
  :param B: np.array[float]. 코사인 유사도를 계산할 벡터. A, B는 차원이 같아야 한다.
  '''
  return dot(A, B)/(norm(A)*norm(B))

class HGramm:
    text_col = ''
    id_col = ''
    sent_df = None

    tokenizer = None
    llm = None

    BASE_DIR = ''

    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        return
    
    def set_mecab(self, mecab_path):
        self.tokenizer = Mecab(dicpath=mecab_path)
        return

    def set_llm(self, llm_path):
        # 모델 로드
        self.llm = SentenceTransformer(llm_path)

        #모델 gpu에 할당
        if torch.cuda.is_available():
            self.llm.to('cuda')
            print('gpu에 모델 할당 완료!')
        else:
            print('gpu를 감지하지 못했습니다.')
        return

    def _make_sentdf(self, doc_list:List[Dict]):
        '''
        doc 기준으로 정리된 데이터를 sent 기준으로 쪼갬
        '''
        data = []
        for doc in doc_list:
            for sent in doc[self.text_col]:
                id = doc[self.id_col]
                data.append({'sent':sent, self.id_col:id})
        self.sent_df = pd.DataFrame(data=data)
        return

    def get_textarea(self, doc_list:List[Dict], text_col:str, id_col:str, is_llm:bool=False):
        '''
        문서 리스트를 입력받아, 각 문서의 textarea를 List[tuple]로 반환합니다.
        textarea는 연속된 표제어 사이에 있는 연속된 본문으로 정의합니다. 

        각 문장의 표제어 여부는 HGramm으로 판정합니다.
          
        :param doc_list: 문서 리스트입니다. 각 문서는 dict형으로 주어집니다. (e.g. {'text':['문장1','문장2',..], 'id':'url1'})
        :param text_col: dict 문서 객체에서 문장들(List[Dict])이 담긴 key 이름입니다. {'text':['문장1','문장2',..], 'id':'url1'}에서 'text'를 의미합니다.
        :param id_col: dict 문서 객체에서 id가 담긴 key 이름입니다. id는 null이 없는 unique한 값이어야 합니다. {'text':['문장1','문장2',..], 'id':'url1'}에서 'id'를 의미합니다.
        '''
        # is_head 추가
        is_head = self.get_isHead(doc_list=doc_list, text_col=text_col, id_col=id_col)
        self.sent_df['is_head'] = is_head
        #문장 별 dataframe을 문서 별 dataframe으로 변환
        self.doc_df = self.sent2doc()
        if is_llm:
            # textarea 영역 선출
            self.doc_df['textarea_range'] = self.doc_df.apply(self._cal_textarea, axis=1)
            self.doc_df['textarea'] = self.doc_df.apply(self._extract_textarea, axis=1)
        else:
            # textarea 영역 선출
            self.doc_df['textarea_range'] = self.doc_df.apply(self._cal_textarea_nollm, axis=1)
            self.doc_df['textarea'] = self.doc_df.apply(self._extract_textarea, axis=1)

        return self.doc_df

    def _cal_textarea_nollm(self, row):
        doc = row[self.text_col]
        headh = row['is_head']
        meanPooling = self.getMeanPooling(is_head=headh)
        start_ta = self.get1to0(meanPooling)
        end_ta = self.get0to1(meanPooling, start=start_ta[0])

        # 만약 본문 영역으로 보이는 부분을 못 찾은 경우, [0,0] 반환
        if start_ta[0] == -1 or end_ta[0] == -1:
            return [0, -1]

        start_h = headh[start_ta[0]:start_ta[1]]
        end_h = headh[end_ta[0]:end_ta[1]]
        # textarea_range 추출
        textarea_range = self.getTextAreaRange_nollm(start_h, end_h, start_ta, end_ta)
        return textarea_range

    def getTextAreaRange_nollm(self,start_h, end_h, start_ta, end_ta):
        '''
        가장 headline일 확률이 높은 곳을 기준으로 본문-비본문 전환점을 찾는다
        모든 문장의 is_head가 0.25 미만이면 전부 본문으로 본다. 만약 is_head가 0.25보다 큰 지점이 여러 개 있을 경우, 가장 안쪽에 있는 문장을 전환점으로 본다.
        '''
        textarea = [-1, -1]
        for i, is_head in enumerate(start_h):
            if is_head > 0.25:
                textarea[0] = start_ta[0]+i+1
            elif i == len(start_h) -1 and textarea[0]==-1:
                textarea[0] = start_ta[0]

        for i, is_head in enumerate(end_h):
            if is_head>0.25:
                textarea[1] = end_ta[0]+i -1
                break
            elif i == len(end_h) - 1 and textarea[1]==-1:
                textarea[1] = end_h[1] -1

        return textarea

    def _cal_textarea(self, row):
        doc = row[self.text_col]
        headh = row['is_head']
        meanPooling = self.getMeanPooling(is_head=headh)
        start_ta = self.get1to0(meanPooling)
        end_ta = self.get0to1(meanPooling, start=start_ta[0])

        #만약 본문 영역으로 보이는 부분을 못 찾은 경우, [0,0] 반환
        if start_ta[0] == -1 or end_ta[0] == -1:
            return [0,0]
        
        #벡터 임베딩
        start_sents = [sent for sent in doc[start_ta[0]:start_ta[1]]]
        end_sents = [sent for sent in doc[end_ta[0]:end_ta[1]]]
        start_embeddings = self.llm.encode(start_sents)
        end_embeddings = self.llm.encode(end_sents)

        start_h = headh[start_ta[0]:start_ta[1]]
        end_h = headh[end_ta[0]:end_ta[1]]
        #textarea_range 추출
        textarea_range = self.getTextAreaRange(start_embeddings, end_embeddings, start_ta, end_ta, start_h, end_h)
        return textarea_range
    
    def getMeanPooling(self, is_head:List[dict], window_size=4):
        '''
        window sliding으로 연속된 4문장의 평균을 구한다.
        '''
        mean_pooling = []
        for i in range(window_size,len(is_head)):
            sum_tag = 0
            for j in range(i-window_size,i):
                sum_tag += is_head[j]
            mean_pooling.append(sum_tag/window_size)

        return mean_pooling
    
    def get1to0(self, pooling:List[float], window_size=4):
        '''
        비본문 영억에서 본문으로 바뀌는 위치를 찾는 함수.
        mean pooling 리스트에서 본문으로 바뀌는 변환점을 찾는다.

        margin: 변환점 앞뒤로 몇 칸을 LLM으로 확인할 건지 결정하는 변수. int. default=2
        '''
        checkpoint = -1
        for i in range(len(pooling)):
            if pooling[i] < 0.3:
                checkpoint = i 
                break
        start = checkpoint 
        end = checkpoint + window_size

        return start, end

    def get0to1(self, pooling:List[float], window_size=4, start = 0):
        '''
        본문 영억에서 비본문으로 바뀌는 위치를 찾는 함수. (역순 순회)
        mean pooling 리스트에서 본문으로 바뀌는 변환점을 찾는다.

        margin: 변환점 앞뒤로 몇 칸을 LLM으로 확인할 건지 결정하는 변수. int. default=2
        start: get1to0에서 얻은 본문 시작 구역 index
        '''
        if start == -1:
            return -1, -1
        checkpoint = -1
        # 중간에 소제목이 있을 경우, 이후 본문 영역이 이어지면 유예를 주기 위해 사용
        find_checkpoint_flag = 0
        for i in range(start, len(pooling)):
            if pooling[i] > 0.3:
                if find_checkpoint_flag == 0: 
                    checkpoint = i
                    find_checkpoint_flag = 1
                else:
                    # 전환점 의심 지점을 발견하고 바로 다시 전환점 의심 지점이 나오면, 처음으로 발견한 전환점을 반환
                    if 1 == i - checkpoint: 
                        break
                    else:
                        #계속 본문이 나오다가 새 전환점을 발견한 경우, 현재 지점을 새 전환점으로 여김.
                        checkpoint = i
        start = checkpoint 
        end = checkpoint + window_size

        return start, end    

    def getTextAreaRange(self, start_embeddings, end_embeddings, start_ta, end_ta, start_h, end_h):
        '''
        코사인 유사도가 가장 낮은 부분을 전환점이라고 여기고, 전환점을 반환한다.  
        반환하는 전환점은 [본문이 시작되는 부분, 본문이 끝나는 부분]을 doc 내 index 기준으로 반환한다.
        '''
        # cosine similarity가 가장 낮은 부분을 전환점이라고 여김.
        start_cos, end_cos = [], []
        start_cos_min, end_cos_min = 1, 1
        textarea = [0,-1]
        for i in range(1, len(start_embeddings)):
            start_cos.append(cos_sim(start_embeddings[i-1],start_embeddings[i]))
            if start_cos_min > start_cos[-1]:
                start_cos_min = start_cos[-1]
                textarea[0] = i 
        
        for i in range(1, len(end_embeddings)):
            end_cos.append(cos_sim(end_embeddings[i-1],end_embeddings[i]))
            if end_cos_min > end_cos[-1]:
                end_cos_min = end_cos[-1]
                textarea[1] = i-1

        # 만약 제일 낮은 cosine similarity가 0.7을 넘길 경우, is_head를 기준으로 자름.
        if start_cos_min > 0.7 : 
            if max(start_h) < 0.1: #모두 본문일 확률이 높을 경우
                textarea[0] = 0
            else:
                textarea[0] = start_h.index(max(start_h)) + 1

        if end_cos_min > 0.7 : 
            if max(end_h) < 0.1: #모두 본문일 확률이 높을 경우
                textarea[1] = len(end_h) -1
            else:
                textarea[1] = end_h.index(max(end_h)) -1
                for i, eh in enumerate(end_h):    # 표제어 텍스트가 여러개 있을 경우, 처음 등장하는 표제어를 전환점으로 삼음.
                    if eh > 0.5: 
                        textarea[1] = i -1
                        break
        
        # textarea 안의 index가 임베딩된 문장 list 내 index가 아니라 doc 안에서의 index가 되도록 수정
        textarea[0] += start_ta[0]
        textarea[1] += end_ta[0]
        return textarea

    
    def _extract_textarea(self, row):
        text = row[self.text_col]
        ta = row['textarea_range']

        return text[ta[0]:ta[1]+1]

    def get_isHead(self, doc_list:List[Dict], text_col:str, id_col:str):
        '''
        HGramm으로 각 문장의 표제어 확률을 구해 반환합니다. return 타입은 List[List[float]]입니다.
          
        :param doc_list: 문서 리스트입니다. 각 문서는 dict형으로 주어집니다. (e.g. {'text':['문장1','문장2',..], 'id':'url1'})
        :param text_col: dict 문서 객체에서 문장들(List[Dict])이 담긴 key 이름입니다. {'text':['문장1','문장2',..], 'id':'url1'}에서 'text'를 의미합니다.
        :param id_col: dict 문서 객체에서 id가 담긴 key 이름입니다. id는 null이 없는 unique한 값이어야 합니다. {'text':['문장1','문장2',..], 'id':'url1'}에서 'id'를 의미합니다.
        '''
        
        self.text_col = text_col
        self.id_col = id_col
        self._make_sentdf(doc_list=doc_list)
        
        # 전처리
        self._preprocess()
        # feature 추가
        self._get_features()
        # HGramm 표제어 예측
        result = self._predict()

        return result
    
    def _preprocess(self):
        '''
        sent_df의 텍스트를 전처리하는 코드
        '''
        # text cleaning
        self.sent_df['sent'] = self.sent_df['sent'].apply(clean_text)
        # 길이가 0인 문장 제거
        self.sent_df = self.sent_df[self.sent_df['sent']!=""]

        return
    

    def _get_features(self):
        '''
        sent_df에 is_head 예측에 필요한 컬럼을 추가합니다.
        '''
        # pos 값 추가
        self.sent_df = self._add_pos(self.sent_df)
        # pos를 바탕으로 파생변수 추가
        self.sent_df = self._addSentInfo(self.sent_df)
        # sentdf 후처리
        # 지나치게 토큰 수가 적은 문장 제거
        self.sent_df = self.sent_df[self.sent_df['num_token']>3]
        self.sent_df.reset_index(drop=True, inplace=True)
        # 예측에 필요한 column들만 남기기
        features = ['SN', 'SC', 'SY', 'SF', 'NNP', 'NNG', 'NNBC', 'JKB', 'VV', 'ETM', 'EC',
       'EF', 'JX', 'num_token', 'num_josa', 'num_eomi', 'num_noun',
       'num_symbol', 'num_rare_symbol', 'num_symbol_nSF', 'is_SF',
       'rate_stopwords', 'rate_symbol', 'rate_noun']
        
        self.sent_X = self.sent_df[features]

        return
    
    def _predict(self):
        # 모델 로드
        model_path = os.path.join(self.BASE_DIR, 'models/HGramm_gbm')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # 예측
        proba = model.predict_proba(self.sent_X)[:,1]
        return proba
    
    def _add_pos(self, df):
        res = deepcopy(df)
        sents = res['sent'].tolist()
        
        pos_data = []
        for sent in sents:
            tokens = self.tokenizer.pos(sent)
            pos_dict = defaultdict(int)
            
            for token, token_pos in tokens:
                pos_list = token_pos.split('+')
                for pos in pos_list:
                    pos_dict[pos] += 1

            pos_data.append(pos_dict)

        res.reset_index(drop=True, inplace=True)
        pos_df = pd.DataFrame(pos_data)
        res = pd.concat([res, pos_df], axis=1)
        res.fillna(0, inplace=True)
        return res
    
    def _addSentInfo(self, df:pd.DataFrame)->pd.DataFrame:
        '''
        입력으로 들어온 문장들의 품사 포함 정보를 반환하는 함수.  
        조사, 어미, 특수문자의 개수, 전체 형태소 수, 특수문자 비율, 조사 비율, 어미 비율을 각 문장에 대해 dict로 묶어 반환.
        '''        
        cols = df.columns
        # 토큰 수: url과 sent를 제외한 모든 컬럼의 값을 더해 계산
        token_cols = [col for col in cols if col not in [self.id_col,'sent']]
        # 조사: J로 시작하는 모든 태그
        josa_cols = [col for col in cols if col[0]=='J']
        # 어미: ETN을 제외하고 E로 시작하는 모든 태그 + VCP(긍정 지정사 ~이다) + VCN(부정 지정사 ~아니다) + XSV(동사 파생 접미사) + XSA(형용사 파생 접미사)  
        eomi_cols = ['EF','EC','ETM', 'VCP','VCN','XSV', 'XSA']
        # 접속사: MAJ(접속 조사) + JC(접속 부사)
        conj_cols = ['MAJ','JC']
        # 명사: NNP(고유 명사) + NNG(일반 명사) + XSN(명사 파생 접미사) + ETN (명사형 전성어미) + SH(한자) + SN(숫자)
        noun_cols = ['NNP','NNG','XSN','ETN', 'SH', 'SN']
        # 용언: V로 시작하는 모든 태그
        verb_cols = [col for col in cols if col[0]=='V']
        # 특수문자: S로 시작하는 모든 태그(외국어, 한자, 숫자 제외)
        symbol_cols = ['SF','SE','SSO','SSC','SC','SY','UNKNOWN','NA','UNA']
        # 희귀 특수문자
        rare_symbol_cols = ['UNKNOWN','NA','UNA', 'SY', 'SE']

        #현재 데이터 프레임에 없는 컬럼은 값을 0으로 셋팅해 추가
        for col in eomi_cols:
            if col not in cols: df[col] = 0
        for col in conj_cols:
            if col not in cols: df[col] = 0
        for col in noun_cols:
            if col not in cols: df[col] = 0
        for col in symbol_cols:
            if col not in cols: df[col] = 0
        for col in rare_symbol_cols:
            if col not in cols: df[col] = 0
        
        res = df.to_dict(orient='records')


        for i, row in enumerate(res):
            # 토큰 수 추가. url, sent를 제외한 모든 컬럼의 값을 더한다.
            res[i]['num_token'] = self.dict_sum(row, token_cols)
            #조사 수 추가
            res[i]['num_josa'] = self.dict_sum(row, josa_cols)
            #어미 수 추가
            res[i]['num_eomi'] = self.dict_sum(row, eomi_cols)
            #접속사 추가
            res[i]['num_conj'] = self.dict_sum(row, conj_cols)
            # 명사 수 추가
            res[i]['num_noun'] = self.dict_sum(row, noun_cols)
            # 용언 수 추가
            res[i]['num_verb'] = self.dict_sum(row, verb_cols)
            # 특수문자 수 추가
            res[i]['num_symbol'] = self.dict_sum(row, symbol_cols)
            # 희귀 특수문자 수 추가
            res[i]['num_rare_symbol'] = self.dict_sum(row, rare_symbol_cols)
            # SF 수를 제외한 특수문자 수 추가
            res[i]['num_symbol_nSF'] = res[i]['num_symbol'] - res[i]['SF']
            # SF가 1개 이상인지 여부
            res[i]['is_SF'] = 1 if row['SF']>0 else 0

            #불용어 비율 계산 ((조사+어미)/(전체 토큰 수))
            res[i]['rate_stopwords'] = (res[i]['num_eomi']+res[i]['num_josa']) / res[i]['num_token']
            # 특수문자 비율 추가
            res[i]['rate_symbol'] = res[i]['num_symbol']/res[i]['num_token']
            #불용어 수 대비 명사 수
            res[i]['rate_noun'] = res[i]['num_noun']/(res[i]['num_eomi']+res[i]['num_josa']+1)
            #전체 문장에서 외국어 비율
            res[i]['rate_foreign'] = res[i]['SL']/res[i]['num_token']

    
        return pd.DataFrame(res)

    def dict_sum(self, row:dict, keys:List[str])->int:
        num = 0
        for col in keys:
            num += row[col]

        return num


    def sent2doc(self):
        docs = []
        text = []
        is_head = []
        id_prev = self.sent_df.loc[0, self.id_col]
        for row in self.sent_df.to_dict(orient='records'):
            id = row[self.id_col]
            if id == id_prev:
                text.append(row['sent'])
                is_head.append(row['is_head'])
            else:
                doc = {self.text_col:text, self.id_col:id_prev, 'is_head':is_head}
                docs.append(doc)
                text = []
                is_head = []
                id_prev = id
        
        if text:
            doc = {self.text_col:text, self.id_col:id_prev, 'is_head':is_head}
            docs.append(doc)

        res = pd.DataFrame(docs)

        return res

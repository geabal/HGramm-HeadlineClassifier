import html
import re
import unicodedata

def clean_text(text):
    if not text:
        return ""

    # 1. HTML 엔티티 디코딩 (&amp; -> & 등)
    # &nbsp;는 이 과정에서 \xa0가 됩니다.
    text = html.unescape(text)

    # 2. 유니코드 정규화 (NFKC)
    # \xa0(특수 공백)를 일반 공백으로 바꾸고 유사 문자를 통합합니다.
    text = unicodedata.normalize('NFKC', text)

    # 3. 연속된 공백 하나로 합치기 및 양끝 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return ""
    else:
        return str(text)

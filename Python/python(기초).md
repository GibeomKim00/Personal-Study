## 객체(Object)
파이썬에서 모든 것(bool, 정수, 실수, 문자열, 데이터 구조, 함수, 프로그램)이 객체로 구현되어 있다.


## 변수(Variable)
객체에 이름을 붙여준 것
* 리터럴(Literal) 정수 VS 변수
<pre><code>3 # literal 정수(변경 불가)
a = 3 # 변수(변경 가능)
</code></pre>


## 클레스(Class)
객체의 정의를 의미한다.


## 문자열(string)
문자열은 불변하다.
```
name = 'Henny'
name[0] = 'P' # Error 발생
name.replace('H', 'P') # replace 함수를 이용
```
* 슬라이싱(Slicing)
```
a = '123456789'
a[9:0:-3] # [9,6,3] 출력(인덱스 0은 제외)
a[5:1:-1] # [5,4,3,2] 출력(인덱스 1은 제외)
a[1:10:-1] #인덱스 1부터 10까지(인덱스 10 제외) 역순으로 슬라이싱, 그러나 역순으로 슬라이싱 못하기에 빈 리스트 출력
a[:-3:-1] # [9,8] 출력(인덱스 -3 제외)
a[0:-3:-1] # 빈 리스트 출력
```
* 문자열 나누기: split()
```
a = 'get gloves,get mask,give cat'
a.split(,) 
# 출력값: ['get gloves', 'get mask', 'give cat']
# split함수에 인자를 주지 않을 시 공백 기준으로 나눈다.
```
* 문자열 결합하기: join()
```
a = ['Yeti', 'Bigfoot', 'Loch Ness Monster']
a_string = ', '.join(a)
# 출력값: Yeti, Bigfoot, Loch Ness Monster
```
* 이외 문자열 함수들
    * len(): 문자열 길이
    * startswith(): 문자열 시작 문자
    * endswith(): 문자열 끝 문자
    * find(): 찾고자 하는 인자값이 문자열의 어느 인덱스에 처음으로 나오는가?
    * rfind(): 어느 인덱스에 마지막으로 나오는가?
    * count(): 인자값이 문자열에 얼만큼 반복되는가?
    * isalnum(): 문자열이 글자와 숫자로만 이루어져 있는가?
    * strip(): 문자열 양끝에 인자값 존재 시 삭제
    * capitalize(): 문자열 첫 글자 대문자로 변환
    * title(): 모든 단어 첫 글자를 대문자로 변환
    * upper(): 글자 모두 대문자로 변환
    * lower(): 글자 모두 소문자로 변환
    * swapcase(): 대문자는 소문자로 소문자는 대문자로 변환
    * center(): 문자열을 중앙에 배치
    * ljust(): 왼쪽에 배치
    * rjust(): 오른쪽에 배치
    * replace(): 첫번째 인자를 두번째 인자로 변경
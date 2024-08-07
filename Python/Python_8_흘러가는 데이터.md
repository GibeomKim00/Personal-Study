# 흘러가는 데이터
## 파일 입출력
* 파일: 파일 이름(filename)으로 저장된 바이트 시퀀스다.
* 파일 열기: fileobj = open(filename, mode)

    * fileobj: open()에 의해 반환되는 파일 객체이다.
    * filename: 파일의 문자열 이름
    * mode: 파일 타입과 파일로 무엇을 할지 명시하는 문자열

* mode

    * mode의 첫 번째 글자는 작업을 명시한다.

        * r: 파일 읽기
        * w: 파일 쓰기(파일이 없으면 생성, 있으면 덮어쓴다.)
        * x: 파일 쓰기(파일이 존재하지 **않을** 경우에만 해당한다.)
        * a: 파일 추가하기(파일이 있으면 파일의 끝에서부터 쓴다.)
    * mode의 두 번째 글자는 파일 타입을 명시한다.

        * t(또는 공백): 텍스트 타입
        * b: 이진(binary) 타입

## 텍스트 파일
* write() - 텍스트 파일 쓰기
```
poem = '''There was a young lady named Bright'''
len(poem) # 출력값: 35

fout = open('relativity', 'wt')  # 파일 열기
fout.write(poem) # 출력값: 35, write() 함수는 파일에 쓴 바이트수를 반환한다.
print(poem, file=fout) # print() 함수로 파일을 만드는 방법
# write은 print와 다르게 줄바꿈 및 띄어쓰기를 하지 않는다.
fout.close() # 파일 닫기
```

* read(), readline(), readlines() - 텍스트 파일 읽기
```
fin = open('relativity', 'rt')
poem = fin.read() # 한 번에 전체 파일을 읽을 수 있다. => 메모리 낭비 주의
len(poem) # 출력값: 35

# 읽는 크기 제한하기
poem = ''
fin = open('relativity', 'rt')
chunk = 100
while True:
    fragment = fin.read(chunk) # 파일을 다 읽을 시 빈 문자열('') 반환
    if not fragment:
        break
    poem += fragment

fin.close()
```
```
# readline(): 파일을 라인 단위로 읽기
poem = ''
fin = open('relativity', 'rt')
while True:
    line = fin.readline()
    if not line:
        break
    poem += line

fin.close()
```
```
# iterator를 사용한 파일 읽기, iterator는 파일을 한 라인씩 읽어들인다.
poem = ''
fin = open('relativity', 'rt')
for line in fin:
    poem += line

fin.close()
```
```
# readlines(): 한 번에 모든 라인을 읽고, 한 라인으로 된 문자열들의 리스트 반환
fin = open('relativity', 'rt')
lines = fin.readlines()
fin.close()
len(lines) # 출력값: 1
```

## 이진 파일
* write() - 이진 파일 쓰기
```
bdata = bytes(range(0,256))
len(bdata) # 256

fout = oepn('bfile', 'wb') # 문자열 대신 바이트를 쓴다.
fout.write(bdata) # 출력값: 256, write() 함수는 파일에 쓴 바이트 수 반환
fout.close()
```

* read() - 이진 파일 읽기
```
fin = open('bfile', 'rb')
bdata = fin.read()
len(bdata) # 출력값: 256
fin.close()
```

## 파일을 위한 유용한 함수
* with() - 자동으로 파일 닫기
```
with open('relativity', 'wt') as fout:
    fout.write(poem)
```
* seek() - 파일 위치 찾기

    * seek() 함수 형식: seek(offset, origin)
    
        * origin이 0일 때(기본값), 시작 위치에서 offset 바이트 이동
        * origin이 1일 때, 현재 위치에서 offset 바이트 이동
        * origin이 2일 때, 마지막 위치에서 offset 바이트 이동
```
fin = open('bfile', 'rb')
fin.tell() # tell() 함수는 파일의 시작으로부터의 현재 offset을 바이트 단위로 반환한다.
# 출력값: 0

fin.seek(255) # seek() 함수는 다른 바이트 offset 위치를 이동할 수 있다.
# 출력값: 255

bdata = fin.read()
len(bdata) # 출력값: 1
bdata[0] # 출력값: 255
```

***seek() 함수는 이진 파일 함수의 위치를 옮길 때 유용하다. 텍스트 파일에 사용할 경우 아스키코드는(한 문자당 1바이트) 유용하지만 UTF-8에는(한 문자당 여러 바이트를 사용) 유용하지 않다.***

## 구조화된 텍스트 파일
1. CSV

    * CSV 파일을 한 번에 한 라인씩 읽어서, 콤마로 구분된 필드를 분리할 수 있다.
    * reader(), writer() 함수는 기본적으로 열은 콤마로 행은 줄바꿈 문자로 나눈다.
```
import csv
villains = [
    ['Doctor', 'No'],
    ['Rosa', 'Klebb'],
    ['Mister', 'Big']
]
with open('villains', 'wt') as fout: # 콘텍스트 매니저
    csvout = csv.writer(fout)
    csvout.writerows(villains)

with open('villains', 'rt') as fin:
    cin = csv.reader(fin)
    villains = [row for row in cin]

print(villains)
# 출력값: [['Doctor', 'No'], ['Rosa', 'Klebb'], ['Mister', 'Big']]

# 딕셔너리의 리스트로 csv 파일 읽기
with open('villains', 'rt') as fin:
    cin = fin.DictReader(fin, fieldnames=['first','last])
    villains = [row for row in cin]

print(villains)
# 출력값
[{'last': 'No', 'first':'Doctor'},
{'last':'Klebb', 'first':'Rosa'},
{'last':'Big', 'first':'Mister'}]

with open('villains', 'wt') as fout:
    cout = csv.DictWriter(fout, ['first','last'])
    cout.writeheader() # csv 파일 첫 라인에 열 이름을 쓰게 해준다.
    cout.writerows(villains)
```

2. XML

    * 마크업 형식으로 데이터를 구분하기 위해 **태그**를 사용한다.
    * XML 파싱(해석)을 위해 ElementTree 모듈을 사용한다.

3. HTML(Hypertext Markup Language)

    * 웹의 기본 문서 형식

4. JSON(JavaScript Object Notation)

    * json 모듈 사용
    * 이 모듈을 통해 JSON 문자열로 인코딩(dumps)하고, JSON 문자열을 다시 데이터로 디코딩(loads)할 수 있다.

5. YAML

    * 날짜와 시간 같은 데이터 타입을 많이 처리한다.
    * load(): YAML 문자열을 파이썬 데이터로 변경
    * dump(): 파이썬 데이터를 YAML 문자열로 변경

## 구조화된 이진 파일
1. 스프레드시트(spreadsheet)
2. HDF5(Hierarchical Data Format)

    * 아주 큰 데이터 집합에 대한 빠른 접근이 필요한 과학 분야에서 많이 사용된다.

## 관계형 데이터베이스(relative database)
* 다양한 종류의 데이터 간의 관계를 **테이블(table)** 형태로 표시하기 때문에 **관계형**이라고 한다.
* 기본키(primary key): 테이블에서 유일하며 데이터를 구분해준다.
* 외래키(foreign key): 서로 다른 테이블은 외래키를 통해 연관된다.
1. SQL

    * API 혹은 protocol이 아닌, 원하는 결과를 질의하는 **서술형 언어**다.
    * 관계형 데이터베이스의 보편적인 언어다.
    * SQL 질의(query): 클라이언트에서 데이터베이스 서버로 전송하는 텍스트 문자열이다.
    * SQL의 두 카테고리

        * DDL(Data Definition Language, 데이터 정의어): 테이블, 데이터베이스, 사용자에 대한 생성, 삭제, 제약조건, 권한을 다룬다.
        * DML(Data Manipulation Language, 데이터 조작어): 데이터의 조회, 삽입, 갱신, 삭제를 다룬다.

2. DB-API
    * API(Application Programming Interface): 어떤 서비스에 대한 접근을 얻기 위해 호출하는 함수들의 집합이다.
    * DB-API는 관계형 데이터베이스에 접근하기 위한 파이썬의 표준 API다.
    * 주된 함수들

        * connect(): 데이터베이스의 연결을 만든다.
        * cursor(): 질의를 관리하기 위한 **커서** 객체를 만든다.
        * execute(), executemany(): 데이터베이스에 하나 이상의 SQL 명령을 실행한다.
        * fetchone(), fetchmany(), fetchall(): 실행 결과를 얻는다.

3. SQLLite
    * 가볍고 좋은 오픈소스의 관계형 데이터베이스로 표준 파이썬 라이브러리로 구현되어 있다.
```
import sqlite 3
conn = sqlite3.connect('enterprise.db')
curs = conn.cursor()
curs.execute('''CREATE TABLE zoo (critter VARCHAR(20) PRIMARY KEY, count INT, damages FLOAT)''')

curs.execute('INSERT INTO zoo VALUES("duck", 5, 0.0)')

# placeholder를 통해 데이터를 안전하게 넣기
ins = 'INSERT INTO zoo (critter, count, damges) VALUES(?, ?, ?)'
curs.execute(ins, ('weasel', 1, 2000.0))

# 데이터베이스에 저장된 데이터 불러오기
curs.excute('SELECT * FROM zoo')
rows = curs.fetchall()

# 커서와 데이터베이스 닫기
curs.close()
conn.close()
```

4. MySQL
    * 오픈소스 관계형 데이터베이스이다.

5. PostgreSQL
    * 완전한 기능을 갖춘 오픈소스 관계형 데이터베이스이다.
    * MySQL보다 사용할 수 있는 고급 기능이 더 많다.

6. SQLAlchemy
    * 크로스 데이터베이스의 파이썬 라이브러리이다.
    * 여러 가지 수준에서 SQLAlchemy를 사용할 수 있다.
    
        * 엔진 레이어: SQLAlchemy의 가장 낮은 수준으로 DB-API보다 좀 더 많은 기능이 있다.
        * SQL 표현 언어: SQL과 파이썬의 중간 수준 연결로 엔진 레이어보다 더 다양한 SQL문을 처리한다.
        * ORM(Object-Relational Mapper)

## NoSQL 데이터 스토어
1. dbm 형식
    * **키-값** 저장 형식으로, 웹 브라우저와 같은 애플리케이션에 포함된다.
    * 키와 값은 바이트 로 저장된다.

2. Memcached
    * 인메모리 키-값의 캐시 서버다.
    * 데이터가 지속되지 **않아서**, 이전에 쓴 데이터가 사라질 수 있다.
    * 여러 memcached 서버를 동시에 연결할 수 있다.

3. Redis
    * 자료구조 서버로, 데이터를 디스크에 저장해 기존 데이터를 유지한다.
    * 문자열: 단일 값과 한 키는 Redis **문자열**이다.
    * 리스트: Redis의 리스트는 문자열만 포함할 수 있다.
    * 해시(hash): 파이썬의 딕셔너리와 비슷하지만 문자열만 포함할 수 있다.
    * 셋(set): 파이썬의 set과 유사하다.
    * 정렬된 셋(sorted set): 가장 많은 용도로 쓰이는 Redis의 데이터 타입이다.
    * 비트(bit): 대량의 숫자 집합을 공간-효율적인 방식으로 빠르게 처리한다.
    * 캐시와 만료: 모든 Redis의 키는 TTL(Time-To-Live), 즉 만료일(expiration date)을 가진다.
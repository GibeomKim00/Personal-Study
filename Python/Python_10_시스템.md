# 시스템
## 파일
1. open(): 생성하기
```
fout = open('oops.txt', 'wr')
fout.close()
```

2. exists(): 존재여부 확인하기
    * 상대 경로, 절대 경로 모두 사용 가능하다.
        * 절대 경로(/): **root**부터 해당 파일까지의 전체 경로
        * 상대 경로(./): 현재 파일의 위치를 기준으로 연결하려는 파일의 상대적인 경로
```
import os
os.path.exists('oops.txt') # true
os.path.exists('./oops.txt') # true, ./는 현재 위치를 나타낸다.
```

3. isfile(): 타입 확인하기
```
name = 'oops.txt'
os.path.isfile(name) # true
os.path.isdir(name) # false

os.path.isdir('.') # true, 하나의 점은 현재 디렉터리를 나타낸다.

# isabs() 함수는 인자가 절대 경로인지 확인한다.
os.path.isabs('name') # false
os.path.isabs('/big/fake/name') # true
```

4. copy(): 복사하기
```
import shutil
shutil.copy('oops.txt', 'ohno.txt') # 왼쪽을 오른쪽 파일에 복사
```

5. rename(): 이름 바꾸기
```
import os
os.rename('ohno.txt', 'ohwell.txt') # 오른쪽 인자의 이름으로 변경
```

6. link(), symlink(): 연결하기
    * **심벌릭 링크**(symbolic link)는 원본 파일을 새 이름으로 연결한다.
    * 하드 링크(hard link): 한 파일의 복사본을 만든다. 이때, 복사본의 i-node는 원본 i-node와 같으며 복사본을 통해 수정을 하면 원본도 바뀌며, 원본이 삭제되어도 복사본은 유지된다.
```
os.link('oops.txt', 'yikes.txt') # hard link 생성
os.path.isfile('yikes.txt') # true

os.path.islink('yikes.txt') # false, 파일이 심벌릭 링크인지 확인
os.symlink('oops.txt', 'jeepers.txt') # 심벌릭 링크 생성
os.path.islink('jeepers.txt') # true
```

7. chmod(): permission 바꾸기
```
os.chmod('oops.txt', 0o400)
```

8. abspath(): 절대 경로 얻기

9. realpath(): 심벌릭 링크 경로 얻기

10. remove(): 삭제하기

## 디렉터리
1. mkdir(): 생성하기
```
os.mkdir('poems')
os.path.exists('poems') # true
```

2. rmdir(): 삭제하기
```
os.rmdir('poems')
os.path.exists('poems') # false
```

3. listdir(): 콘텐츠 나열하기

4. chdir(): 현재 디렉터리 바꾸기

6. glob(): 일치하는 파일 나열하기

## 프로그램과 프로세스
* 하나의 프로그램을 실행할 때, 운영체제는 한 **프로세스**를 생성한다.
* 프로세스는 운영체제의 **커널(kernel)** 에서 시스템 리소스(CPU, 메모리, 디스크 공간) 및 자료구조를 사용한다.
```
import os
os.getpid() # 실행중인 파이썬 인터프리터에 대한 프로세스 ID를 가져온다.
os.getcws() # 현재 작업 디렉터리의 위치를 가져온다.

os.getuid() # 사용자 ID 출력
os.getgid() # 그룹 ID 출력
```

1. subprocess: 프로세스 생성하기(1)
    * getoutput(): 쉘에서 프로그램을 실행하여 생성된 결과를 얻을 수 있다.
    ```
    # data 프로그램의 결과를 얻어온다.
    import subprocess
    ret = subprocess.getoutput('data')
    ```
    * check_output(): 명령과 인자의 리스트를 취하며 문자열이 아닌 바이트 타입을 반환하며, 쉘을 사용하지 않는다.
    ```
    ret = subprocess.check_output(['data', '-u'])
    ```
    * getstatusoutput(): 프로그램의 종료 상태를 표시한다. 프로그램 상태 코드와 결과를 튜플로 반환한다.
    
    * call(): 결과가 아닌 상태 코드만 저장한다.

2. multiprocessing: 프로세스 생성하기(2)
    * 파이썬 함수를 별도의 프로세스로 실행하거나 한 프로그램에서 독립적인 여러 프로세스를 실행할 수 있다.
    ```
    import multiprocessing
    import os

    def do_this(what):
        whoami(what):

    def whoami(what):
        print("Process %s says: %s" % (os.getpid(), what))

    if __name__ == "__main__":
        whoami("I'm the main program")
        for n in range(4):
            p = multiprocessing.Process(target=do_this, args=("I'm function %s" % n,))
            p.start()
    # Process() 함수는 새 프로세스를 생성한 후 그곳에서 do_this() 함수를 실행한다.
    # 총 4개의 프로세스가 작동한다.
    ```

3. terminate(): 프로세스 죽이기
    ```
    import multiprocessing
    import time
    import os

    def whoami(name):
        print("I'm %s, in process %s" % (name, os.getpid()))

    def loopy(name):
        whoami(name)
        start = 1
        stop = 1000000
        for num in range(start, stop):
            print("\tNumber %s of %s. Honk!" % (num, stop))
            time.sleep(1)

    if __name__ == "__main__":
        whoami("main")
        p = multiprocessing.Process(target=loopy, args=("loopy",))
        p.start()
        time.sleep(5)
        p.terminate()
    ```

## 달력과 시간
1. datetime 모듈
    * date: 년, 월, 일
    * time: 시, 분, 초, 마이크로초
    * datetime: 날짜와 시간
    * timedelta: 날짜 또는 시간 간격
```
# date 객체
from datetime import date
halloween = date(2015, 10, 31)
halloween # 출력값: datetime.date(2015, 10, 31)
halloween.day # 출력값: 31
halloween.month # 출력값: 10
halloween.year # 출력값: 2015

halloween.isoformat() # 출력값: '2015-10-31'
now = date.today()
now # 출력값: datetime.date(2024, 7, 3)
```
```
# timedelta 객체
from datetime import timedelta
one_day = timedelta(days=1)
tomorrow = now + one_day
tomorrow # 출력값: datetime.date(2024, 7, 4)
```
```
# time 객체
from datetime import time
noon = time(12, 0, 0)
noon # 출력값: datetime.time(12,0)
noon.hour # 출력값: 12
noon.minute # 출력값: 0
noon.second # 출력값: 0
noon.microsecond # 출력값: 0
```
```
# datetime 객체
from datetime import datetime
some_day = datetime(2015, 1, 2, 3, 4, 5, 6)
some_day # 출력값: datetime.datetime(2015, 1, 2, 3, 4, 5, 6)

some_day.isoformat() # 출력값: '2015-01-02T03:04:05.000006'
now = datetime.now()
now # 출력값: (2024, 7, 3, 15, 25, 20, 6)
now.year # 출력값: 2024
now.hour # 출력값: 15
```
```
# date 객체와 time 객체 결합
from datetime import date, time, datetime
noon = time(12)
this_day = date.today()
noon_today = datetime.combine(this_day, noon)
noon_today # 출력값: datetime.datetime(2024, 7, 3, 12, 0)

noon_today.date() # 출력값: datetime.date(2024, 7, 3)
noon_today.time() # 출력값: datetime.time(12, 0)
```

2. time 모듈
 * 절대 시간: 어떤 시작점 이후 시간의 초를 세는 것
    * 유닉스 시간: 1970년 1월 1일 자정 이후 초를 세기 시작했고 이 값을 에포치(epoch)라고 한다.
```
import time
now = time.time()
now # 출력값: 1437085535.741846

# ctime(): epoch 값을 문자열로 반환
time.ctime(now) # 출력값: 'Thu Jul 16 22:34:04 2015'

# 각각의 날짜와 시간을 얻기 위해 time 모듈의 struct_time 객체를 사용할 수 있다.
# gmtime(): 시간을 UTC로 제공
time.localtime(now) # localtie은 시스템의 표준시간대로 제공한다.
# 출력값: time.struct_time(tm_year=2015, tm_mon=7, tm_mday=17, tm_hour=7, tm_min=25, tm_sec=35, tm_wday=4, tm_yday=198, tm_isdst=0
)

# mktime(): struct_time 객체를 epoch 초로 바꾼다.
tm = time.localtime(now)
time.mktime(tm) # 출력값: 1437085535.0
```

3. 날짜와 시간 읽고 쓰기
 * 앞에서 본 isoformat(), ctime()
 * strftime(): 날짜와 시간을 문자열로 반환한다.
    * datetime, date, time 객체에 메서드로 제공
    * time 모듈에서 함수로 제공
```
import time
fmt = "It's %A, %B %d, %Y, local time %I:%M:%S%p"
t = time.localtime()
t 
# 출력값: time.struct_time(tm_year=2015, tm_mon=7, tm_mday=18, tm_hour=22, tm_min=43, tm_sec=14, tm_wday=5, tm_yday=199, tm_isdst=0)
time.strftime(fmt, t)
"It's Saturday, July 01, 1900, local time 10:43:14PM"

# date 객체는 날짜 부분만 작동
from datetime import date
some_day = date(2015, 12, 12)
fmt = "It's %B %d, %Y, local time %I:%M:%S%p"
some_day.strftime(fmt)
# 출력값: "It's December 12, 2015, local time 12:00:00AM"

# time 객체는 시간 부분만 작동
from datetime import time
some_time = time(10, 35)
some_time.strftime(fmt)
# 출력값: "It's Monday, January 01, 1900, local time 10:35:00AM"
```

 * strptime(): 문자열을 날짜나 시간으로 변환
```
import time
fmt = "%Y-%m-%d"
time.strptime("2015 06 02", fmt) # 형식과 맞지 않기에 오류 발생

time.strptime("2015-06-02", fmt) # 정상 작동

time.strptime("2015-13-29", fmt) # 범위를 벗어나면(13) 오류 발생
```

 * locale 라이브러리
    * setlocale() 함수: 다른 월, 일 이름을 출력해준다.
        * 첫 번째 인자: 날짜와 시간을 위한 locale.LC_TIME
        * 두 번째 인자: 언어와 국가 약어가 결합된 문자열('ko-kr') 
# Web(World Wide Web)
* HTTP(Hypertext Transfer Protocol): 요청과 응답을 교환하기 위한 웹 서버와 클라이언트의 명세
* HTML(Hypertext Markup Language): 결과에 대한 표현 형식
* URL(Uniform Resource Locator): 고유의 해당 서버와 자원을 나타내는 방법

-> 웹 클라이언트가 HTTP로 URL을 요청하고, 서버로부터 HTML을 받는다.

## 파이썬 표준 웹 라이브러리
1. urllib
    * http는 모든 클라이언트-서버 HTTP 세부사항을 관리한다.

        * client는 클라이언트 부분을 관리한다.
        * server는 파이썬 웹 서버를 작성하는 데 도움을 준다.
        * cookies와 cookiejar는 사이트 방문자의 데이터를 저장하는 쿠기를 관리한다.
    * urllib은 http 위에서 실행된다.

        * request는 클라이언트 요청을 처리한다.
        * response는 서버 응답을 처리한다.
        * parse는 URL을 분석한다.
```
import urllib.request as ur
url = 'http://quotesondesign.com/wp-json/posts'
conn = ur.urlopen(url) # 다수의 메서드를 지닌 HTTPResponse 객체이다
data = conn.read() # 웹 페이지로부터 데이터를 읽어온다.
conn.status # HTTP 상태 코드
```
* HTTP 상태 코드

    * 1xx(조건부 응답): 서버는 요청을 받았지만, 클라이언트에 대한 몇 가지 추가 정보가 필요하다.
    * 2xx(성공): 성공적으로 처리되었다. 200 이외의 모든 성공 코드는 추가사항을 전달한다.
    * 3xx(redirection): resource가 이전되어 클라이언트에 새로운 URL을 응답해준다.
    * 4xx(client error): 404는 클라이언트 측에 문제가 있음을 나타낸다.
    * 5xx(server error): 500은 서버 에러를 나타낸다.

* 웹 서버는 원하는 포맷으로 데이터를 전송할 수 있으며, 일반적으로 HTML을 전송하지만, 포춘 쿠키에 대한 예제는 json 포맷이다.

    * application/json 문자열은 MIME(Multipurpose Internet Mail Extension) 타입으로, HTML을 위한 MIME 타입은 text/html이다.
```
print(conn.getheader('Content-Type))
# 출력값: application/json; charset=UTF-8
```

2. requests 모듈
```
import requests
url = 'http://quotesondesign.com/wp-json/posts'
resp = requests.get(url)
resp # 출력값: <Response [200]>
```

## 파이썬 웹 서버
1. 파이썬 웹 서버
    * 기본 포트 번호는 8000이지만, 다른 포트 번호를 지정할 수 있다.
        * python -m http.server 9999
```
# 순수한 파이썬 HTTP 서버 구현
$ python -m http.server
# 출력값: Serving HTTP on 0.0.0.0 port 8000 ...
# 0.0.0.0은 모든 TCP 주소를 의미한다. 그래서 서버가 어떤 주소를 가졌든 상관없이 그곳에 접근할 수 있다.
```
2. 프레임워크
    * 웹 서버는 HTTP WSGU의 세부사항을 처리한다. 반면 웹 **프레임워크**를 사용한다면 파이썬 코드를 통해 강력한 웹사이트를 만들 수 있다.
        * WSGI(Web Server Gateway Interface): 파이썬 웹 애플레케이션과 웹 서버 간의 범용적인 API
    * 웹 프레임워크는 클라이언트의 요청과 서버의 응답을 처리한다.
        * 라우트(route): URL을 해석하여 해당 서버의 파일이나 파이썬 코드를 찾아준다.
        * 템플릿(template): 서버 사이드의 데이터를 HTML 페이지에 병합한다.
        * 인증(authentication) 및 권한(authorization): 사용자 이름과 비밀번호, permission(허가)을 처리한다.
        * 세션(session): 웹사이트에 방문하는 동안 사용자의 임시 데이터를 유지한다.
    
    1. Bottle
        * Bottle 설치: $ pip install bottle
        * run() 함수 인자
            * debug=True: HTTP 에러 발생 시 디버깅 페이지 생성
            * reloader=True: 파이썬 코드 변경 시 변경된 코드를 다시 불러온다.
    ```
    # 테스트 웹 서버를 실행하는 코드(bottle1.py)
    from bottle import route, run

    @route('/') # route decorator를 사용하여 함수와 URL을 연결한다.
    def home():
        return "It isn't fanct, but it's my home page"

    run(host='localhost', port=9999)

    # 위 파이썬 코드 실행 후 브라우저 주소 창에 http://localhost:9999 입력하면 내용을 볼 수 있다.
    $ python bottle1.py
    ```
    ```
    # 홈페이지 요청 시 index.html 파일 내용 반환(bottle2.py)
    from bottle import route, run, static_file

    @route('/')
    def main():
        return static_file('index.html', root='.') # .는 현재 디렉터리를 의미

    run(host='localhost', prot=9999)
    ```
    ```
    # URL에 인자를 전달하여 사용하는 예제(bottle3.py)
    from bottle import route, run, static_file

    @route('/')
    def home():
        return static_file('index.html', root='.')

    @route('/echo/<thing>')
    def echo(thing):
        return "Say hello to my little friend: %s!" % thing

    run(host='localhost', post=9999)
    ```
    ```
    # requests와 같은 클라이언트 라이브러리 사용 예제(bottle_test.py)
    import requests

    resp = requests.get('http://localhost:9999/echo/Mothra')

    if resp.status_code == 200 and \
        resp.text == 'Say hello to my little friend: Mothral!':
        print('It worked! That almost never happens!')
    else:
        print('Argh, got this:', resp.text)

    # 실행
    $ python bottle_test.py
    # 터미널에서의 출력값: It worked! That almost never happens!
    ```

     2. Flask
        * Bottle 보다 더 많은 기능 제공
        * Flask 패키지에는 werkzeug WSGI 라이브러리와 jinja2 템플릿 라이브러리가 존재
        * Flask 설치: $ pip install flask
    ```
    # flask1.py
    from flask import Flask

    app = Flask(__name__, static_folder='.', static_url_path='')

    @app.route('/')
    def home():
        return app.send_satic_file('index.html')
    
    @app.route('/echo/<thing>)
    def echo(thing):
        return "Say hello to my little friend: %s" % thing

    app.run(port=9999, debug=True)
    ```
    * jinja2 이용 예제
    ```
    # templates 디렉터리 안에 flask2.html 파일
    <html>
        <head>
            <title>Flask2 Example</title>
        </head>
        <body>
            Say hello to my littel friend: {{ thing }}
        </body>
    </html>


    # flask2.py 파일(home() 함수 생략)
    from flask import Flask, render_template

    app = Flask(__name__)

    @app.route('/echo/<thing>')
    def echo(thing):
        return render_template('flask.html', thing=thing)

    app.run(port=9999, debug=True)


    # GET 매개변수로 인자 전달
    # URL 입력 방식: http://localhost:9999/echo?thing=Gorgo&place=place
    from flask import Flask, render_template, request

    app = Flask(__name__)

    @app.route('/echo/')
    def echo():
        thing = request.args.get('thing')
        place = request.args.get('place'
        )
        return render_template('flask3.html', thing=thing, place=place)

    app.run(port=9999, debug=True)
    ```

## 비파이썬 웹 서버
1. 아파치(apache)
    * mod_wsgi: 아파치 웹 서버에 최적화된 WSGI 모듈

2. 엔진엑스(nginx)
    * 안정성과 메모리를 적게 사용한다.
    * 파이썬 모듈이 없다.

## 기타 프레임워크
* 데이터베이스를 지원해주는 프레임워크
1. 장고(django)
2. web2py
3. pyramid
4. wheezy.web

## 웹 서비스와 자동화
1. webbrowser 모듈
```
import webbrowser
url = 'http://www.python.org/'
webbrowser.open(url) # 출력값: True
```
2. 웹 API와 REST
* API(Application Programming Interface)를 통해 데이터를 제공할 수 있다.
* REST(REpresentational State Transfer)는 웹 서비스에 접근할 수 있는 URL을 정의하여 **웹** 인터페이스만 제공한다.

3. HTML scrape하기: BeautifulSoup
    * 설치: $ pip install beautifulsoup4
```
def get_links(url):
    import requests
    from bs4 import BeautifulSoup as soup

    result = requests.get(url)
    page = result.text
    doc = soup(page)
    links = [element.get('href') for element in doc.find_all('a')]
    return links

if __name__ == '__main__':
    import sys
    for url in sys.argv[1:]:
        print('Links in', url)
        for num, link in enumerate(get_links(url), start=1):
            print(num, link)
        print() 
```
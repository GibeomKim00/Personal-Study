# 병행성과 네트워크
## 병행성
* 동시에 여러 개의 일을 하는 것
    * 동기(synchronous): 한 작업이 다른 작업을 따르는 것
    * 비동기(asynchronous): 작업들이 독립적인 것
* 컴퓨터가 일을 수행하면서 기다리는 경우는 I/O 바운드(대부분 여기에 해당) 또는 CPU 바운드(엄청난 계산이 필요할 때 발생) 때문이다.

### queue(큐)
* FIFO(First In First Out)
* **메시지**를 전달한다.
* 분산 작업 관리를 위한 큐의 경우 **작업 큐**(work queue, task queue,job queue)라고 한다.

### process
* 싱글 머신인 경우, 표준 라이브러리의 multiprocessing 모듈에 있는 Queue 함수를 이용해 큐를 구현할 수 있다.
```
import multiprocessing as mp

def washer(dishes, output):
    for dish in dishes:
        print('Washing', dish, 'dish')
        output.put(dish)

def dryer(input):
    while True:
        dish = input.get()
        print('Drying', dish, 'dish')
        input.task_done()

dish_queue = mp.JoinableQueue()
dryer_proc = mp.Process(target=dryer, args=(dish_queue,))
dryer_proc.daemon = True
dryer_proc.start()
dishes = ['salad', 'bread', 'entree', 'dessert']
washer(dishes, dish_queue)
dish_queue.join()
```

### thread
* 한 process 내에서 실행된다.
* process의 모든 자원에 접근 가능하다.
```
import threading

def do_this(What):
    whoami(what)

def whoami(what):
    print("Thread %s says: %s" % (threading.current_thread(), what))

if __name__ == "__main__":
    whoami("I'm the main program")
    for n in range(4):
        p = threading.Thread(target=do_this, args=("I'm function %s" % n,))
    p.start()
```

```
import threading, queue
import time

def washer(dishes, dish_queue):
    for dish in dishes:
        print("Washing", dish)
        time.sleep(5)
        dish_queue.input(dish)

def dryer(dish_queue):
    while True:
        dish = dish_queue.get()
        print("Drying", dish)
        time.sleep(10)
        dish_queue.task_done()

dish_queue = queue.Queue()
for n in range(2):
    dryer_thread = threading.Thread(target=dryer, args=(dish_queue,))
    dryer_thread.start()

dishes = ['salad', 'bread', 'entree', 'desert']
washer(dishes, dish_queue)
dish_queue.join()
```

### 그린 스레드와 gevent
* gevent 라이브러리: 이벤트 기반 프로그래밍이며, 명령 코드를 작성하면 이 조각들을 **코루틴**dmfh qusghksgksek.
    * 이벤트 기반(event-based) 프로그래밍: 중앙 **이벤트 루프**를 실행하고, 모든 작업을 조금씩 실행하면서 루프를 반복한다.
    * 코루틴(coroutine): 다른 함수와 서로 통신하여, 어느 위치에 있는지 파악하고 있는 제너레이터와 같다.

### twisted
* 비동기식 이벤트 기반 네트워킹 프레임워크다.

### asyncio

### Redis
* Redis 서버와 클라이언트가 TCP를 통해 네트워킹한다.
```
import redis
conn = redis.Redis()
print('Washer is startign')
dishes = ['salad', 'bread', 'entree', 'dessert']
for dish in dishes:
    msg = dish.encode('utf-8')
    conn.rpush('dishes', msg)
    print('Washed', dish)
conn.rpush('dishes', 'quit')
print('Washer is done')
```
```
import redis
conn = redis.Redis()
print('Dryer is starting')
while True:
    msg = conn.blpop('dishes')
    if not msg:
        break
    val = msg[1].decode('utf-8')
    if val == 'quit':
        break
    print('Dried', val)
print('Dishes are dried')
```

## 네트워크
### 네트워킹 애플리케이션 만드는 방법
1. 요청-응답 패턴(클라이언트-서버 패턴): 동기적이다.
2. push 또는 fanout 패턴
3. 발행(publisher)-구독(subscribe)

### 발행-구독 모델
* 큐가 아닌 브로드캐스트(broadcast)로 각 구독자 프로세스는 수신받고자 하는 메시지의 타입을 표시하고 각 메시지의 복사본은 타입과 일치하는 구독자에게 전송된다.

* 각 발행자는 단지 브로드캐스팅(broadcasting)만 할 뿐, 누가 구독하는지 알지 못한다.

1. Redis: 발행자는 토픽, 값과 함께 메시지를 전달하고, 구독자는 수신받고자 하는 토픽을 말한다.
```
# 발행자
import redis
import random

conn = redis.Redis()
cats = ['siamese', 'persian', 'maine coon', 'norwegian forest']
hats = ['stovepipe', 'bowler', 'tam-o-shanter', 'fedora']
for msg in range(10):
    cat = random.choice(cats) # 토픽
    hat = random.choice(hats) # 메시지
    print('Publish: %s wears a %s' % (cat hat))
    conn.publish(cat, hat)
```
```
# 한 구독자
import redis
conn = redis.Redis()

topics = ['maine coon', 'persian'] # 구독자가 받고 싶어하는 타입(토픽)
sub = conn.pubsub()
sub.subscribe(topic)
for msg in sub.listen(): # listen()은 딕셔너리를 반환한다.
    if msg['type'] == 'message': # 메시지의 타입이 'message'일 때 발행자가 보낸 기준에 일치한 것이다.
        cat = msg['channel'] # channel 키는 토픽을 포함
        hat = msg['data'] # data 키는 메시지를 포함
        print('Subscribe: %s wears a %s' % (cat, hat))
```

2. ZeroMQ
* 중앙 서버가 없으며, 각 발행자는 모든 구독자한테 메시지를 전달한다.
```
# 발행자
import zmq
import random 
import time

host = '*'
port = 6789
ctx = zmq.Context()
pub = ctx.socket(zmq.PUB) # zmq.PUB은 소켓 유형이다.
pub.bind('tcp://%s:%s' % (host, port))
cats = ['siamese', 'persian', 'maine coon', 'norwegian forest']
hats = ['stovepipe', 'bowler', 'tam-o-shanter', 'fedora']
time.sleep(1)
for msg in range(10):
    cat = random.choice(cats)
    cat_bytes = cat.encode('utf-8')
    hat = random.choice(hats)
    hat_bytes = hat.encode('utf-8')
    print('Publish: %s wears a %s' % (cat, hat))
    pub.send_multipart([cat_bytes, hat_bytes]) # 두 개의 바이트 문자열 전송
```
```
# 구독자
import zmq

host = '127.0.0.1'
port = 6789
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect('tcp://%s:%s' % (host, port))
topics = ['maine coon', 'persian']
for topic in topics:
    sub.setsocket(zmq.SUBSCRIBE, topic.encode('utf-8'))
while True:
    cat_bytes, hat_bytes = sub.recv_multipart()
    cat = cat_bytes.decode('utf-8')
    hat = hat_bytes.decode('utf-8')
    print('Subscribe: %s wears a %s' % (cat, hat))
```

### TCP/IP
* IP 계층: 네트워크 위치와 데이터 흐름의 패킷(packet)을 명시하는 계층(layer)이다. IP 계층에는 네트워크 위치 사이에서 바이틀르 이동하는 방법을 기술하는 두 가지 프로토콜(protocol)이 있다.
    * UDP(User Datagram Protocol): 짧은 데이터 교환에 사용된다. UDP 메시지는 응답 메시지(ACK)가 없어서 메시지가 목적지에 잘 도착했는지 확인할 수 없다.
    * TCP(Transmission Control Protocol): 수명이 긴 connection에 사용되며 TCP는 바이트 스트림이 **중복 없이 순서대로** 도착하는 것을 보장한다. 또한 핸드셰이크(handshake)를 통해 송신자와 수신자 사이의 컨넥션을 보장한다.

### 소켓
* UDP 연결
```
# 서버 프로그램
from datetime import datetime
import socket

server_address = ('local host', 6789) # (IP 주소, 포트)
max_size = 4096

print('Starting the server at', datetime.now())
print('Waiting for a client to call.')
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # 소켓 생성(IP 소켓 생성, UDP 사용)
server.bind(server_address) # 해당 IP와 포트에 도달하는 모든 데이터를 청취

data, client = server.recvfrom(max_size) # 서버에 들어오는 데이터그램을 기다림

print('At', datetime.now(), client, 'said', data)
server.sendto(b'Are you talking to me?', client)
server.close()
```
```
# 클라이언트 프로그램
import socket
from datetime import datetime

server_address = ('localhost', 6789)
max_size = 4096

print('Starting the client at', datetime.now())
client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client.sendto(b'Hey!', server_address)
data, server = client.recvfrom(max_size)
print('At', datetime.now(), server, 'said', data)
client.close()
```

* TCP 연결
```
# 클라이언트 프로그램
import socket
from datetime import datetime

address = ('localhost', 6789)
max_size = 1000

print('Starting the client at', datetime.now())
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(address)
client.sendall(b'Hey!')
data = client.recv(max_size)
print('At', datetime.now(), 'someone replied', data)
client.close()
```
```
# 서버 프로그램
import socket
from datetime import datetime

address = ('localhost', 6789)
max_size = 1000

print('Starting the server at', datetime.now())
print('Waiting for a client to call')
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(address)
server.listen(5) # 대기 중인 커넥션 최대 5

client, addr = server.accept() # 도착한 첫 번째 유효한 메시지를 얻는다.
data = client.recv(max_size)

print('At', datetime.now(), client, 'said', data)
client.sendall(b'Are you talking to me?')
client.close()
server.close()
```

### ZeroMQ
* ZeroMQ는 라이브러리로 ZeroMQ 소켓은 전체 메시지 교환, 커넥션 재시도, 송신자와 수신자 사이에 전송 타이밍이 맞지 않는 경우 데이터 보존을 위한 버퍼 사용과 같은 일을 수행한다.
* ZeroMQ의 경우 몇 가지 소켓의 타입과 패턴으로부터 네트워크를 구축한다.

1. 단일-응답 패턴: 가장 간단한 패턴으로 한 소켓이 요청하면, 다른 소켓에서 응답하는 동기적인 방식이다.
```
# 서버 코드
import zmq

host = '127.0.0.1'
port = 6789
context = zmq.Context()
server = context.socket(zmq.REP) # REP: 여러가지 소켓 타입 중 동기 응답에 해당
server.bind("tcp://%s:%s" % (host, port))
while True:
    # 클라이언트에서 다음 요청을 기다린다.
    request_bytes = server.recv()
    request_str = request_bytes.decode('utf-8')
    print("That voice in my head says: %s" % request_str)
    reply_str = "Stop saying %s" % request_str
    reply_bytes = bytes(reply_str, 'utf-8')
    server.send(reply_bytes)
```
```
# 클라이언트 코드
import zmq

host = '127.0.0.1'
port = 6789
context = zmq.Context()
client = context.socket(zmq.REQ) # REQ: 동기 요청
client.connect("tcp://%s:%s" % (host, port))
for num in range(1, 6):
    request_str = "message #%s" % num
    request_bytes = request_str.encode('utf-8')
    client.send(request_bytes)
    reply_bytes = client.recv()
    reply_str = reply_bytes.decode('utf-8')
    print("Sent %s, received %s" % (request_str, reply_str))
```

### Scapy
* 패킷을 분석하기 위한 유용한 라이브러리다.

### 여러가지 인터넷 서비스
1. DNS(Domain Name System): 분산된 데이터베이스를 통해 IP 주소를 이름으로 바꾸거나, 그 반대를 수행한다. 몇몇 DNS 함수는 저수준 socket 모듈에 있다.
    * gethostbyname(): 도메인 이름에 대한 IP 주소를 반환한다.
    * gethostbyname_ex(): 확장판으로 이름, 또 다른 이름의 리스트, 주소의 리스트를 반환한다.
    * getaddrinfo(): IP 주소를 검색한다. 또한 소켓을 생성하고 연결하기 위한 충분한 저보를 반환한다.

2. 파이썬 이메일 모듈
    * smtplib 모듈: SMTP(Simple Mail Transfer Protocol)를 통해 이메일 전송하기
    * email 모듈: 이메일 생성 및 파싱하기
    * poplib 모듈: POP3(Post Office Protocol 3)를 통해 이메일 읽기
    * imaplib 모듈: IMAP(Internet Message Access Protocol)을 통해 이메일 읽기

### 원격 프로세싱
1. 원격 프로시저 호출
    * RPC(Remote Procedure Call) 함수: 네트워크를 통해 원격에 있는 머신을 실행한다.
    * xmlpc 모듈: 표준 라이브러리로 교환 포멧으로 XML을 사용하여 RPC를 구현했다.
    ```
    # 서버에 함수 정의
    form xmlrpc.server import SimpleXMLRPCServer

    def double(num): # 서버에서 제공하는 함수
        return num * 2

    server = SimpleXMLRPCServer(("localhost", 6789))
    server.register_function(double, "double") # 클라이언트가 RPC를 통해 함수를 사용할 수 있도록 등록한다.
    server.serve_forever() # RPC 서버 구동
    ```
    ```
    # 클라이언트
    import xmlrpc.client

    # 클라이언트는 ServerProxy 함수를 이용해 서버에 연결한다
    proxy = xmlrpc.client.ServerProxy("http://localhost:6789/")
    num = 7
    result = proxy.double(num)
    print("Double %s is" % (result))
    ```

    * XML외에 인코딩에는 JSON, Protocol Buffer, MessagePack이 있다.
    ```
    # MessagePack을 이용한 서버 구현
    from msgpackrpc import Server, Address

    class Services():
        def double(self, num):
            return num * 2

    server = Server(Services())
    server.listen(Address("localhost", 6789))
    server.start()
    ```
    ```
    # 클라이언트
    from msgpackrpc import Client, Address

    client = Client(Address("localhost", 6789))
    num = 8
    result = client.call('double', num)
    print("Double %s is %s" % (num, result))
    ```

### 맵리듀스
* 맵리듀스(MapReduce)는 여러 머신에서 계산을 수행하여 결과를 수집한다.
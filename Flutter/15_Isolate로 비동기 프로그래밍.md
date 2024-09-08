# Isolate로 비동기 프로그래밍
### Isolate 소개
* Dart 애플리케이션은 main() 함수부터 실행되는데 이 메인 함수의 수행 흐름을 **main isolate**(또는 root isolate)라고 한다. 이 메인 아이솔레이트에서 Future나 Stream 등의 비동기 처리가 수행된다.

* main isolate 하위에 새로운 수행 흐름을 만들고 싶으면 새로운 isolate를 만들어야 한다. 새로운 아이솔레이트는 **spawn()** 함수로 만들며, 아이솔레이트에서 처리할 로직은 함수로 작성한다. 실행할 함수를 spawn() 함수의 매개변수로 지정하면 아이솔레이트가 시작될 때 실행된다.

* 아래 코드에서 onPress()는 메인 아이솔레이트에서 실행된다. 이 함수에서 spawn() 함수로 2개의 아이솔레이트를 만들었다. spawn() 함수의 첫 번째 매개변수는 실행한 함수이며, 두 번째 매개변수는 실행할 함수에 전달할 데이터이다. 아이솔레이트는 비동기로 작동하므로 메인 아이솔레이트에서 새로운 아이솔레이트를 실행하고 대기하지 않으며 각각의 아이솔레이트는 개별적으로 실행된다..
```
import 'dart:isolate';
...(생략)...
myIsolate1(var arg) {
    Future.delayed(Duration(seconds: 3), () {
        print('in myIsolate1...$arg');
    });
}

class MyApp extends StatelessWidget {
    myIsolate2(var arg) {
        Future.delayed(Duration(seconds: 2), () {
            print('in myIsolate2...$arg');
        });
    }

    void onPress() {
        print('onPress... before run isolate...');
        Isolate.spawn(myIsolate1, 10);
        Isolate.spawn(myIsolate2, 20);
        print('onPress... after run isolate...');
    }
}

# 출력값
onPress... before run isolate...
onPress... after run isolate...
in myIsolate2...20
in myIsolate2...10
```


### port로 데이터 주고받기
* 아이솔레이트는 thread 처럼 동작한다. 즉, 독립된 메모리에서 진행되며 이 메모리에 다른 아이솔레이트가 접근할 수 없다.

* 만약 아이솔레이트가 메모리를 공유하고 있었다면 아래 코드 메인 아이솔레이트에서 값을 변경할 때 다른 아이솔레이트도 변경된 값을 참조할 수 있었을 것이다.
```
# 독립된 메모리를 사용하는 아이솔레이트
String topData = 'hello';
myIsolate1(var arg) {
    Future.delayed(Duration(seconds: 2) {
        print('in myIsolate1...$arg, $topData');
    });
}

class MyApp extends StatelessWidget {
    String classData = 'hello';

    myIsolate2(var arg) {
        Future.delayed(Duration(seconds: 2) {
            print('in myIsolate2...$arg, $topData, $classData');
        });
    }

    void onPress() {
        print(
            'onPress... before run isolate... topData: $topData, classData: $classData');
        Isolate.spawn(myIsolate1, 10);
        Isolate.spawn(myIsolate2, 20)l
        topData = 'world';
        classData - 'world';
        print(
            'onPress... after run isolate... topData: $topData,
            classData: $classData');
    }
}

# 출력값
onPress... before run isolate... topData: hello, classData: hello
onPress... after run isolate... topData: world, classData: world
in myIsolate1...10, hello
in myIsolate2...20, hello, hello
```

* 아이솔레이트끼리 데이터를 주고 받기 위해서 **ReceivePort**와 **SendPort**를 이용한다. 각각을 통로로 생각하고 SendPort로 데이터를 전달하고, ReceivePort로 데이터를 받는다.

* ReceivePort와 SendPort를 이용하기 위해서 데이터를 받는 곳에서 ReceivePort를 만들고 이 ReceivePort를 이용해 SendPort를 만들어야 한다. 즉, ReceivePort와 SendPort 두 객체는 같은 포트에 있다고 표현하고 SendPort로 데이터를 전송하면 별도로 명시하지 않아도 이 포트를 만든 ReceivePort에 전달된다.

* 한 아이솔레이트에 여러 개의 ReceivePort, SendPort를 이용할 수 있다.

* SendPort는 ReceivePort의 sendPort 속성으로 만들고 아이솔레이트를 실행할 때 두 번째 매개변수로 전달한다. 아이솔레이트에서 SendPort로 전달한 데이터를 받으려면 ReceivePort의 **first**속성을 이용한다.

* first 속성은 처음으로 보낸 데이터를 받겠다는 의미이다. 또한 first로 데이터를 받은 후 포트는 자동으로 닫힌다.
```
# 데이터 받기
void onPress() async {
    ReceivePort receivePort = ReceivePort();

    # ReceivePort에 데이터를 전달할 sendPort를 준비해서 아이솔레이트 구동
    await Isolate.spawn(myIsolate, receivePort.sendPort);

    String data = await receivePort.first;
    print("main isolate... data read : " + data);
}
```

* 아이솔레이트에서 실행할 함수의 매개변수로 데이터를 보내는 **SendPort**를 받는다. 이 SendPort의 **send()** 함수를 이용해 데이터를 보내면 SendPort를 만든 ReceivePort에 전달된다.
```
# 데이터 보내기
myIsolate(SendPort sendPort) {
    Future.delated(Duration(seconds: 2), () {
        sendPort.send("hello world");
    })
}
```

1. listen() - 반복해서 데이터 주고받기
    * ReceivePort의 **listen()** 함수를 이용하면 데이터를 여러 번 주고받을 수 있다.
    ```
    # 1초마다 데이터 보내기
    void myIsolate2(SendPort sendPort) {
        int counter = 0;
        Timer.periodic(new Duration(seconds: 1), (Timer t) {
            sendPort.send(++counter);
        });
    }
    ```

    * 아이솔레이트에서 전달받은 데이터는 listen() 함수의 매개변수로 지정한 함수가 호출될 때 그 함수의 **매개변수로 전달**된다.
    ```
    # 반복해서 데이터 받기
    void onPressListen() async {
        ReceivePort receivePort = ReceivePort();
        Isolate isolate = await Isolate.spawn(myIsolate2,   receiveProt.sendPort);
        receivePort.listen((data) {
            print('receive : $data');
        });
    }
    ```

    * 포트를 닫기 위해서 ReceivePort의 **close()** 함수를 이용하면 된다.
    ```
    void onPressListen() {
        ReceivePort receivePort = ReceivePort();
        Isolate isolate = await Isolate.spawn(myIsolate2, receivePort.sendPort);
        receivePort.listen((data) {
            if (int.parse(data.toString()) > 5) {
                receivePort.close();
            } else {
                print('receive : $data');
            }
        });
    }
    ```

    * 구독 중인 아이솔레이트를 종료할 때는 **kill()** 함수를 이용한다.
    ```
    isolate.kill(priority: Isolate.immediate);
    ```

2. port를 여러 개 이용하기
    * 아래 코드 함수의 매개변수에 전달된 **SendPort**는 메인 아이솔레이트가 준비한 포트이다. 그러나 메인 아이솔레이트에서 전달하는 데이터를 받아야 할 때도 있다. 이때는 함수의 첫 줄에 새로운 **ReceivePort**를 만든다.
    ```
    myIsolate(SendPort mainPort) {
        ReceivePort isoPort = ReceivePort();
        mainPort.send({'port':isoPort.sendPort});
        isoPort.listen((message) {
            if (message['msg'] != 'bye') {
                int count = message['msg'];
                mainPort.send({'msg': count * count});
            } else {
                isoPort.close();
            }
        })
    }
    ```
    ```
    # main isolate
    void onPress() async {
        ReceivePort mainPort = ReceivePort();
        Isolate isolate = awiat Isolate.spawn(myIsolate, mainPort.sendPort);

        sendPort? isoPort;
        mainPort.listen((message) {
            if (message['port'] != null) {
                isoPort = message['port'];
            } else if (message['msg'] != null) {
                print('msg : ${message['msg']}');
            }
        });

        int count = 0;
        Timer.periodic(Duration(seoncds: 1), (timer) {
            count++;
            if (count < 6) {
                isoPort?.send({'msg':count});
            } else {
                isoPort?.send({'msg':'bye'});
                mainPort.close();
            }
        })
    }
    ```

3. compute() - 작업 완료 후 최종 결괏값 받기
    * 특정 작업을 아이솔레이트로 실행하고 최종 결과를 받을 때 이용한다. 즉, 데이터를 주고받는 구조가 아니라 모든 작업이 완료된 후 최종 결괏값을 받을 때 사용한다.

    * compute() 함수의 첫 번째 매개변수는 아이솔레이트에서 실행할 함수이며, 두 번째 매개변수는 함수에 전달할 데이터이다. 또한 **then()** 함수를 이용해 데이터를 구할 수 있다.

    * 아이솔레이트의 spawn() 함수를 사용하지 않았지만 내부적으로 아이솔레이트를 실행한다.
    ```
    int myIsolate(int no) {
        int sum = 0;
        for (int i = 0, i <= no; i++) {
            sleep(Duration(seconds: 1));
            sum += i;
        }
        return sum;
    }

    # compute() 함수로 최종 결과만 받기
    compute(myIsolate, 10).then((value) => print('result : $value'));
    ```
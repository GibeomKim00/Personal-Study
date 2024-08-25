# Future와 Stream으로 비동기 프로그래밍

### Future와 FutureBuilder
1. Future
    * **Future, Stream,await, async**는 모두 비동기 프로그래밍을 지원하는 기능이다. 
    * 비동기 프로그래밍이란 시간이 오래 걸리는 작업을 실행한 후 끝날 때까지 기다리지 않고 다른 작업을 실행하는 것이다. 비동기 프로그래밍의 반대말은 동기 프로그래밍으로 어떤 작업이 끝날 때까지 기다렸다가 그다음 작업을 수행하는 것이다.

    * 아래 코드를 보면 sum() 함수가 끝나야 'onPress bottom'이 출력된 것을 알 수 있다. 즉, onPress() 함수는 sum() 함수의 실행이 끝날 때까지 대기한다. 이처럼 시간이 오래 걸리는 작업(**네트워킹 또는 파일을 읽거나 쓰기**)은 앱의 성능을 떨어진다.
    ```
    # 동기 프로그래밍
    void sum() {
        var sum = 0;
        Stopwatch stopwatch = Stopwatch();
        stopwatch.start();
        for (int i = 0; i < 50000000; i++) {
            sum += 1;
        }
        stopwatch.stop();
        print("${stopwatch.elapsed}, result: $sum");
    }

    void onPress() {
        print('onPress top...');
        sum();
        print('onPress bottom...');
    }
    ```

    * Future는 비동기 프로그래밍을 위한 다트 언어에서 제공하는 클래스이며 미래에 발생할 데이터를 의미한다.

    * sum() 함수의 반환 타입을 Future로 선언했으며 **실행되자마자 Future 객체를 반환**한다. 이렇게 되면 sum() 함수를 호출한 곳은 호출하자마자 결과를 받으므로 sum() 호출문 그다음 줄을 바로 실행한다.

    * 오래 걸리는 작업은 Future 생성자의 매개변수로 지정한 함수에서 처리하며 이 함수에서 반환하는 값이 Future에 담기는 미래 데이터이다.

    * Future를 선언할 때 generic을 이용해 미래에 발생할 데이터 타입을 지정한다.
    ```
    Future<int> sum() {
        return Future<int>(() {
            var sum = 0;
            Stopwatch stopwatch = Stopwatch();
            stopwatch.start();
            for (int i = 0; i < 50000000; i++) {
                sum += 1;
            }
            stopwatch.stop();
            print("${stopwatch.elapsed}, result: $sum");
            return sum;     # 실제 데이터를 상자에 담기
        });
    }

    void onPress() {
        print('onPress top...');
        sum();
        print('onPress bottom...');
    }
    ```

2. FutureBuilder
    * Future는 미래에 발생하는 데이터이므로 바로 화면에 출력할 수가 없다. **FutureBuilder 위젯**은 결과가 나올 때까지 기다렸다가 화면에 출력해 준다.

    * FutureBuilder는 위젯이지만 자체 화면을 가지지 않는다. FutureBuilder가 출력하는 화면은 생성자 매개변수로 지정된 **AsyncWidgetBuilder**에 명시한다.

    * future 매개변수에 지정한 Future 객체에서 미래에 데이터가 발생하면 AsyncWidgetBuilder부분을 실행해 해당 데이터로 화면을 구성한다.
    ```
    # FutureBuilder 생성자
    const FutureBuilder<T>(
        { Key? key,
        Future<T>? future,
        T? initialData,
        required AsyncWidgetBuilder<T> builder}
    )
    ```
    * AsyncWidgetBuilder는 함수이며 이 함수의 반환 타입이 위젯이다. FutureBuilder가 이 위젯을 화면에 출력한다.

    * AsyncWidgetBuilder의 두 번째 매개변수 타입이 AsyncSnapshot이며 이곳에 Future 데이터를 전달해 준다.
    ```
    # AsyncWidgetBuilder 정의
    AsyncWidgetBuilder<T> = Widget Function(
        BuildContext context,
        AsyncSnapshot<T> snapshot
    )
    ```

    * AsyncSnapshot 객체의 **hasData** 속성으로 데이터가 있는지 판단할 수 있으며

    * AsyncWidgetBuilder의 반환 타입은 위젯이어야 하는데 Future 데이터를 이용하므로 실제 데이터가 발생하기까지 시간이 걸린다. 따라서 데이터 발생 전에 출력할 위젯을 별도로 명시해 줘야 하는데 아래 코드에서는 화면에 빙글빙글 돌아가는 원을 표현하는 CurcularProgressIndicator 위젯을 이용했다.
    ```
    # Future 데이터 출력하기
    body: Center(
        child: FutureBuilder(
            future: calFun();
            builder: (context, snapshot) {
                if (snapshot.hasData) {
                    return Text('${snapshot.data'});
                }
                return CircularProgressIndicator();
            }
        )
    )
    ```


### await와 async
* sum() 함수의 반환 타입은 Future이다. 반환받은 Future 객체로 sum() 함수에서 실제 발생한 데이터를 이용할 것처럼 보이지만 타입 정보만 출력된다.
```
# Future에 담은 데이터 가져오기
void onPress() {
    print('onPress top...');
    Future<int> future = sum()
    print('onPress future: $future');   # 출력값: onPress future: Instance of 'Future<int>'
    print('onPress bottom...');
}
```

* 실제 발생한 데이터를 받으려면 콜백 함수를 등록하고 데이터가 발생한 시점에 그 함수가 전달하도록 해주어야 한다.

1. then() 함수 사용하기
    * Future 객체의 then()을 이용하여 매개변수에 콜백 함수를 등록해 주었다. 이렇게 하면 Future에 데이터가 담기는 순간 콜백 함수가 호출된다.

    * 콜백 함수의 매개변수가 실제 발생한 데이터이다. 또한 Future 객체의 **catchError()** 함수로 Future를 반환한 곳에서 발생한 오류를 처리할 수도 있다.
    ```
    void onPress() {
        print('onPress top...');
        Future<int> future = sum();
        future.then((value) => print('onPress then... $value'));
        future.catchError((error) => print('onPress catchError...$error'));
        print('onPress bottom...');
    }
    ```

    * then()을 이용하면 코드가 복잡해질 수 있다.

    * delayed() 함수의 첫 번째 매개변수인 Duration() 객체에 지정한 값이 지나면 두 번째 매개변수에 지정한 함수가 실행된다.
    ```
    # 시간이 오래 걸리는 funA와 funB
    Future<int> funA() {
        return Future.delayed(Duration(seconds: 3), () {
            return 10;
        });
    }

    Future<int> funB(int arg) {
        return Future.delay(Duration(seconds: 2), () {
            return arg * arg;
        });
    }
    ```
    ```
    # funA의 실행 결과를 받아서 funB를 호출해야 하는 calFun
    Future<int> calFun() {
        return funA().then((aResult) {
            return funB(aResult);
        }).then((bResult) {
            return bResult;
        });
    }
    ```

1. await와 async 사용하기
    * await는 실행 영역에 작성하며 async는 선언 영역에 작성한다.

    * await는 한 작업의 처리 결과를 받아서 다음 작업을 처리해야 할 때 첫 번째 작업의 처리가 끝날 때까지 대기시키는 용도로 사용된다.

    * await는 꼭 async로 선언한 영역에 있어야 한다.

    * 함수를 async로 선언했다면 반환 타입은 꼭 **Future**여야 한다.
    ```
    Future<int> calFun() async {
        int aResult = await funA();
        int bResult = await funB(aResult);
        return bResult;
    }
    ```


### Stream과 StreamBuilder

1. Stream
    * Stream은 미래에 반복해서 발생하는 데이터를 다룰 때 사용하지만 비동기가 아닌 곳에서도 사용할 수 있다.

    * 비동기 관점에서 Future는 미래에 한 번 발생하는 데이터를 의미한다면, Stream은 미래에 반복해서 발생하는 데이터를 의미한다.
    ```
    # 데이터 한 번 반환
    Future<int> futureFun() async {
        return await Future.delayed(Duration(seconds: 1), () {
            return 10;
        });
    }

    void onPress() async {
        await futureFun().then((value) => print("result: $value"));
    }
    ```

    * 데이터를 여러 번 반환하고자 반환 타입을 Stream\<int>로 선언하였다. 즉, int 타입을 여러 번 반환하겠다는 의미이다.

    * Future 타입을 반환하는 비동기 함수를 만들 때는 async와 await를 이용하지만, Stream 타입을 반환하는 비동기 함수는 **async\*** 와 **await**를 이용한다.

    * yield 키워드를 통해 값을 하나씩 비동기적으로 반환할 수 있다. 이때, yield 타입은 Stream\<T>에서 T와 같아야 한다.

    * streamFun() 함수를 호출하는 곳에서는 함수가 반환하는 여러 개의 데이터를 받고자 for 문을 이용한다. 결국 데이터 개수만큼 for 문을 반복해서 실행한다.
    ```
    # 데이터 5번 반환
    Stream<int> streamFun() async* {
        for(int i = 1; i <= 5; i++) {
            await Future.delayed(Duration(seconds: 1));
            yield i;
        }
    }

    void onPress() async {
        await for (int value in streamFun()) {
            print("value: $value");
        }
    }
    ```

    * for 문을 사용하지 않는다면 listen() 함수로 여러 번 반환하는 데이터를 받을 수 있다.
    ```
    void onPress() {
        streamFun().listen((value) {
            print('value : $value');
        })
    }
    ```

2. 유용한 Stream 함수들
    * fromIterable() - lterable 타입 데이터 만들기
        * fromIterable()은 stream의 생성자이다. 이 생성자로 Stream 객체를 만들면서 매개변수로 List 같은 Iterable 타입의 데이터를 전달하면 Iterable의 데이터를 하나씩 순서대로 Stream 객체를 생성한다.
    ```
    var stream = Stream.fromIterable([1,2,3]);
    stream.listen((value) {
        print("value : $value");
    });
    ```

    * fromFuture() - Future 타입 데이터 만들기
        * fromFuture()는 Future 타입의 데이터를 Stream 객체로 만들어 주는 생성자이다.
    ```
    Future<int> futureFun() {
        return Future.delayed(Duration(seconds: 3), () {
            return 10;
        });
    }

    test4() {
        var stream = Stream.fromFuture(futureFun());
        stream.listen((value) {
            print("value : $value");
        });
    }
    ```

    * periodic() - 주기 지정하기
        * periodic()은 주기적으로 어떤 작업을 실행하는 Stream 객체를 만드는 생성자이다.

        * 첫 번째 매개번수는 주기를 표현하는 **Duration** 객체이며, 두 번째 매개변수는 주기적으로 실행하는 **함수**이다.

        * calFun() 함수의 호출 횟수가 calFun() 함수의 매개변수에 전달된다.
    ```
    int calFun(int x) {
        return x * x;
    }

    test1() async {
        Duration duration = Duration(seconds: 2);
        Stream<int> stream = Stream<int>.periodic(duration, calFun);
        await for (int value in stream) {
            print('value : $value');
        }
    }
    ```

    * take() - 횟수 지정하기
        * 데이터 발생 횟수를 지정할 때 사용한다.

        * periodic()와 take() 함수의 반환 타입 모두 Stream 객체이다.
    ```
    # 2초에 한 번씩 총 3번만 데이터가 발생
    int calFun(int x) {
        return x * x;
    }

    test1() async {
        Duration duration = Duration(seconds: 2);
        Stream<int> stream = Stream<int>.periodic(duration, calFun);
        stream = stream.take(3);
        await for (int value in stream) {
            print('value : $value');
        }
    }
    ```

    * takeWhile() - 조건 지정하기
        * 발생 조건을 설정할 때 사용한다.

        * 매개변수에 조건을 설정하는 **함수**를 지정하면 이 조건 함수에서 true를 반환할 때마다 데이터를 만들고 false를 반환하면 멈춘다.
    ```
    int calFun(int x) {
        return x * x;
    }

    test1() async {
        Duration duration = Duration(seconds: 2);
        Stream<int> stream = Stream<int>.periodic(duration, calFun);
        stream = stream.takeWhile((value) {
            return value < 20;
        });
        await for (int value in stream) {
            print('value : $value');
        }
    }
    ```

    * skip() - 생략 지정하기
        * skip() 함수에 지정한 횟수만큼만 생략하고 그 이후부터 데이터를 만든다.
    ```
    # 처음 2번은 데이터를 만들지 않는다.
    int calFun(int x) {
        return x * x;
    }

    test1() async {
        Duration duration = Duration(seconds: 2);
        Stream<int> stream = Stream<int>.periodic(duration, calFun);
        stream = stream.takeWhile((value) {
            return value < 20;
        });
        stream = stream.skip(2);
        await for (int value in stream) {
            print('value : $value');
        }
    }
    ```

    * skipWhile() - 생략 조건 지정하기
        * 매개변수에 지정한 하뭇에서 true가 반환될 때 데이터 발생을 생략하는 함수이다. 즉, skipWhile() 함수에서 false가 반환될 때까지 데이터 발생을 생략한다.
    ```
    int calFun(int x) {
        return x * x;
    }

    test1() async {
        Duration duration = Duration(seconds: 2);
        Stream<int> stream = Stream<int>.periodic(duration, calFun);
        stream = stream.take(10);
        stream = stream.skipWhile((value) {
            return value < 50;
        });
        await for (int value in stream) {
            print('value : $value');
        }
    }
    ```

    * toList() - List 타입으로 만들기
        * Stream으로 발생한 여러 데이터를 모아서 한 번에 List 타입으로 반환해 준다. 반환 타입은 **Future**이다.
    ```
    # 3번 발생한 데이터를 모아서 Future<List> 타입으로 만듬
    int calFun(int x) {
        return x * x;
    }

    test1() async {
        Duration duration = Duration(seconds: 2);
        Stream<int> stream = Stream<int>.periodic(duration, calFun);
        stream = stream.take(3);
        Future<List<int>> future = stream.toList();
        future.then((list) {
            list.forEach((value) {
                print('value : $value');
            });
        });
    }
    ```


3. StreamBuilder
    * Stream 객체를 화면에 출력할 때는 StreamBuilder 위젯을 이용한다.

    * StreamBuilder 생성자의 **stream** 매개변수에 반복해서 데이터를 발생시키는 Stream을 지정해 주면 데이터가 발생할 때마다 **builder** 매개변수에 지정한 함수가 호출된다. 이 함수의 두 번째 함수의 매개변수가 **AsyncSnapshot** 객체이며, **hadData** 속성으로 발생한 데이터가 있는지를 판단할 수 있다. 또한 **data** 속성으로 발생한 데이터를 받을 수 있다.
    ```
    body: Center(
        child: StreamBuilder(
            stream: test(),
            builder: (BuildContext context, AsyncSnapshot<int> snapshot) {
                if (snapshot.hasData) {
                    return Text('data : ${snapshot.data}');
                }
                return CircularProgressIndicator();
            }
        ))
    ```

    * AsyncSnapshot의 **connectionState** 속성을 이용하여 Stream이 데이터 발생이 끝난 건지 발생하고 있는지 아니면 대기하고 있는지를 판단할 수 있다.
        * ConnectionState.waiting: 데이터 발생을 기다리는 상태
        * ConnectionState.active: 데이터가 발생하고 있으며 아직 끝나지 않은 상태
        * ConnectionState.done: 데이터 발생이 끝난 상태
    ```
    Center(
        child: StreamBuilder(
            stream: test(),
            builder: (BuildContext context, AsyncSnapshot<int> snapshot) {
                if (snapshot.connnectionState == ConnectionState.done) {
                    return Text(
                        'Conpleted',
                        style: TextStyle(
                            fontSize: 30
                        )
                    );
                } else if (snapshot.connectionState == ConnectionState.waiting) {
                    return Text(
                        'Waiting For Stream',
                        style: TextStyle(
                            fontSize: 30
                        )
                    );
                }
                return Text(
                    'data : ${snapshot.data}'
                    style: TextStyle(
                        fontSize: 30
                    )
                );
            }
        )
    )
    ```


### 스트림 구독, 제어기, 변환기
1. StreamSubscription - 스트림 구독자
    * StreamSubscription은 Stream 데이터를 소비하는 구독자이다. 즉, Stream에서 반복해서 발생하는 데이터를 별도의 구독자로도 이용할 수 있다.

    * 지금까지 for 문이나 listen() 함수를 이용해 Stream에서 발생하는 데이터를 얻었는데, 이 listen() 함수의 반환 타입이 StreamSubscription이다.

    ```
    # 스트림 데이터 얻기
    var stream = Stream.fromIterable([1, 2, 3]);
    stream.listen((value) {
        print("value : $value");
    });
    ```

    * listen() 함수의 매개변수애는 데이터를 받는 기능 외에 오류나 데이터 발생이 끝났을 때 실행할 함수 등을 등록할 수 있다. **onError** 매개변수에 지정한 함수는 오류가 발생했을 때 호출되며, **onDone** 매개변수에 지정한 함수는 데이터 발생이 끝났을 때 호출된다.
    ```
    var stream = Stream.fromIterable([1, 2, 3]);
    stream.listen((value) {
        print("value : $value");
    },
    onError: (error) {
        print('error: $error');
    },
    onDone: () {
        print('stream done...');
    });
    ```

    * listen() 함수에 등록할 게 많다면 **StreamSubscription**을 이용할 수 있다.

    * listen() 함수에 null을 전달했다. 이는 listen() 함수의 매개변수로 별도의 데이터 처리를 명시하지 않겠다는 의미이다.

    * StreamSubscription 객체의 **cancel()**, **pause()**, **resume()** 함수로 데이터 구독을 취소하거나 잠시 멈추었다가 다시 구독할 수도 있다.
    ```
    var stream = Stream.fromIterable([1, 2, 3]);
    StreamSubscription subscription = stream.listen(null);

    subscription.onData((data) {
        print("value : $vdata");
    })

    subscription.onError((error) {
        print('error: $error');
    });

    subscription.onDone(() {
        print('stream done...');
    });
    ```


2. StreamController - 스트림 제어기
    * Stream이 여러 개인 경우 스트림 제어기를 이용하는 것이 좋다.

    * 하나의 스트림 제어기는 하나의 내부 스트림만 가질 수 있으며 스트림 선언 이후에도 스트림을 조작할 수 있게 해준다.
    ```
    var controller = StreamController();

    var stream1 = Stream.fromIterable([1, 2, 3]);
    var stream2 = Stream.fromIterable(['A', 'B', 'C']);

    stream1.listen((value) {
        controller.add(value);
    });

    stream2.listen((value) {
        controller.add(value);
    });

    controller.stream.listen((value) {
        print('$value');
    });
    ```

    * 스트림 제어기에 데이터를 추가하는 것은 꼭 스트림으로 발생하는 데이터뿐만 아니라 다른 데이터도 담을 수 있다.
    ```
    controller.stream.listen((value) {
        print('$value');
    });

    controller.add(100);
    controller.add(200);
    ```

    * 같은 스트림을 2번 이상 listen()으로 가져오면 오류가 발생한다.
    ```
    var stream1 = Stream.fromIterable([1, 2, 3]);
    stream1.listen((value) {print('listen1 : $value');});
    stream1.listen((value) {print('listen2 : $value');});       # 오류
    ```

    * 스트림 제어기를 이용하면 listen() 함수를 여러 번 호출할 수 있다. 그러려면 StreamController를 만들 때 **broadcast()**함수를 이용한다.
    ```
    var controller = StreamController.broadcast();
    controller.stream.listen((value) {print('listen1 : $value');});
    controller.stream.listen((value) {print('listen2 : $value');});
    controller.add(100);
    controller.add(200);
    ```

3. StreamTransformer - 스트림 변환기
    * 스트림으로 발생한 데이터를 변환하는 역할을 한다.
    ```
    var stream = Stream.fromIterable([1, 2, 3]);

    StreamTransformer<int, dynamic> transformer =
        StreamTransformer.fromHandlers(handleData: (value, sink) {
            print('in transformer...$value');
        });

    stream.transformer(transformer).listen(value) {
        print('in listen...$value');
    }
    ```

    * 위 코드는 Stream으로 발생한 3개의 데이터가 스트림 변환기에는 전달되었지만 listen()까지 전달되지는 못했다. listen() 함수까지 전달하기 위해서는 **fromHandlers()**의 매개변수에 지정한 함수의 두 번째 매개변수(sink)를 이용해야 한다.

    * **add()** 함수의 매개변수에 지정한 값이 listen()에 전달된다.
    ```
    var stream = Stream.fromIterable([1, 2, 3]);

    StreamTransformer<int, dynamic> transformer = StreamTransformer.fromHandlers(handleData:
    (value, sink) {
        print('int transformer...$value');
        sink.add(value * value);
    });

    stream.transform(transformer).listen((value) {
        print('in listen...$value');
    })

    # 결과
    in transformer...1
    in listen...1
    in transformer...2
    in listen...4
    in transformer...3
    in listen...9
    ```
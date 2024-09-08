# Provider 패키지 이용하기

### Provider 기본 개념
* Provider는 앱의 상태를 더 쉽게 관리해 준다.
    * Provider: 하위 위젯이 이용할 상태 제공
        * Provider: 기본 프로바이더
        * ChangeNotifierProvider: 상태 변경 감지 제공
        * MultiProvider: 여러 가지 상태 등록
        * ProxyProvider: 다른 상태를 참조해서 새로운 상태 생산
        * StreamProvider: 스트림 결과를 상태로 제공
        * FutureProvider: Future 결과를 상태로 제공

    * Consume: 하위 위젯에서 상태 데이터 이용
        * Provider.of(): 타입으로 프로바이더의 상태 데이터 흭득
        * Consumer: 상태를 이용할 위젯 명시
        * Selector: 상태를 이용할 위젯 그 위젯에서 이용할 상태 명시

* Provider 이용하기
    1. 상태 데이터 선언하기
        * Provider로 하위 위젯이 이용할 상태 데이터를 선언한다.
        * Provider는 위젯이다. 따라서 똑같은 상태를 이용하는 위젯의 상위에 Provider를 등록하고 프로바이더에 상태 데이터를 추가해서 하위 위젯이 이용하게 한다.
        * **Provider<int>.value()** 로 하위 위젯이 이용할 상태를 등록할 수 있다. **value()** 함수의 **value** 속성으로 데이터를 명시하면 이 데이터는 **child**에 명시한 위젯부터 그 하위의 위젯에서 이용할 수 있다.
        ```
        # 상태 데이터 선언
        Provider<int>.value(
            value: 10,
            child: SubWidget(),
        )
        ```

        * 하위 위젯에서 이용할 상태 데이터를 등록할 때 Provider.value() 생성자 외에 **Provider()** 생성자를 이용할 수도 있다. **create** 속성에 함수를 지정하여 이 함수에서 반환하는 값을 상태로 이용할 수 있다.
        ```
        Provider<int>(
            create: (context) {
                int sum = 0;
                for (int i = 0; i <= 10; i++) {
                    sum += 1;
                }
                return sum;
            },
            child: SubWidget(),
        )
        ```

    2. 상태 데이터 이용
        * 프로바이더의 상태를 이용하는 하위 위젯은 **Provider.of()** 함수로 상태 데이터를 얻을 수 있다. 제너릭 타입은 상위에서 프로바이더로 제공하는 상태 데이터의 타입이다.
        ```
        Widget build(BuildContext context) {
            final data = Provider.of<int>(context);
        }
        ```

### 다양한 Provider 이용하기
1. ChangeNotifierProvider - 변경된 상태를 하위 위젯에 적용하기
    * Provider에 등록된 상태 데이터는 값이 변경되더라도 하위 위젯이 **다시 빌드**하지 않으므로 변경 사항이 적용되지 않는다.
    * 변경된 상태 데이터를 하위 위젯에 적용하려면 **ChangeNotifierProvider**를 이용한다. ChangeNotifierProvider에 등록하는 상태 데이터는 **ChangeNotifier**를 구현해야 한다. 따라서 **int** 등 기초 타입의 상태는 등록할 수 없다.
    * 단순하게 상태 데이터가 변경되었다고 해서 하위 위젯을 다시 빌드하지 않으며, **notifyListeners()** 함수를  호출해 주어야 한다.
    ```
    class Counter with ChangeNotifier {
        int _count = 0;
        int get count => _count;

        void increment() {
            _count++;
            notifyListeners();
        }
    }
    ```

    * ChangeNotifierProvider는 자신에게 등록된 모델 클래스(Counter)에서 notifyListeners() 함수 호출을 감지해 child에 등록된 위젯을 다시 빌드해 준다.
    ```
    # 다시 빌드할 위젯 등록
    ChangeNotifierProvider<Counter>.value(
        value: Counter(),
        child: SubWidget(),
    )
    ```

2. MultiProvider - 멀티 프로바이더 등록하기
    * 여러 프로바이더를 한꺼번에 등록해서 이용할 때 하나의 프로바이더 위젯에 다른 프로바이더를 등록하여 계층 구조로 만들 수 있다.
    ```
    Provider<int>.value(
        value: 10,
        child: Provider<String>.value(
            value: "hello",
            child: ChangeNotifierProvider<Counter>.value(
                value: Counter(), 
                child:  SubWidget(),
            )
        )
    )
    ```

    * Provider를 계층 구조로 만들 수도 있지만 MultiProvider를 이용하여 더 쉽게 만들 수 있다. **providers** 속성에 여러 프로바이더를 등록할 수 있다.
    ```
    MultiProvider(
        providers: [
            Provider<int>.value(value: 10),
            Provider<String>.value(value: "hello"),
            ChangeNotifierProvider<Coutner>.value(value: Counter())
        ],
        child: SubWidget()
    )
    ```

    * 만약 등록한 Provider의 제너릭 타입이 같을 경우 마지막에 등록한 Provider를 하위 위젯이 이용한다.

3. ProxyProvider - 상태 조합하기
    * ProxyProvider에는 generic 타입을 2개 선언한다. 예로 ProxyProvider<A, B>로 선언한다면 A는 **전달받을 상태 타입**이며, B는 **A를 참조해서 만들 상태 타입**이다.

    * **update** 속성에 함수를 등록하는데, 이 함수에서 반환하는 값이 상태로 등록된다.

    * ProxyProvider<Counter, Sum>은 Counter 상태를 전달받아 Sum 상태를 등록하는 Provider이다. update에 등록한 함수에서 Sum 객체를 반환하므로 이 객체가 상태로 등록된다. 그런데 update 함수를 호출할 때 두 번째 매개변수에 Counter 객체가 전달되는데 결국 두 번째 매개변수로 전달된 상태를 참조하여 자신의 상태를 만들게 된다.
    ```
    # 상태 조합하기
    MultiProvider(
        providers: [
            ChangeNotifierProvider<Counter>.value(value: Counter()),
            ProxyProvider<Counter, Sum>(
                update: (context, model, sum) {
                    return Sum(model);
                }
            )
        ],
        child: SubWidget();
    )
    ```

    * ProxyProvider는 전달받을 상태 개수에 따라 ProxyProvider2 ~ ProxyProvider6까지 제공한다.
    ```
    ProxyProvider2<Counter, Sum, String>(
        update: (context, model1, model2, data) {
            return "${model1.count}, ${model2.sum}";
        }
    )
    ```

    * Provider에 등록한 상태 객체의 데이터가 변경되는 것이지, 객체가 새로 생성되지는 않는다. 하지만 ProxyProvider에 등록한 상태 객체는 데이터가 변경될 때마다 객체가 새로 생성될 수도 있다.
    * Counter 객체는 한 번만 객체가 생성되는 반면에 Sum 객체는 Counter 객체의 데이터가 변할 때마다 새로운 객체를 생성한다. 참조하는 상태(Counter)가 변경될 때마다 update에 지정한 함수가 **자동으로 호출**되며 변경된 값이 두 번째 매개변수로 전달된다.
    ```
    MultiProvider(
        providers: [
            ChangeNotifierProvider<Counter>.value(value: Counter()),
            ProxyProvider<Counter, Sum>(
                update: (context, model, sum) {
                    return Sum(model);
                }
            )
        ],
        child: SubWidget();
    )
    ```

    * update에 등록한 세 번째 매개변수로 **이전에 이용했던 상태 객체**를 전달하면 새로운 객체를 생성하지 않고 기존 객체를 이용하여 값만 변경된다.
    ```
    # 상태 객체 생성 판단하기
    ProxyProvider<Counter, Sum>(
        update: (context, model, sum) {
            if (sum != null) {  # 상탯값만 갱신
                sum.sum = model.count;
            } else {    # 새로운 객체 생성
                return Sum(model);
            }
        }
    )
    ```

4. FutureProvider - Future 데이터 상태 등록하기
    * 상태 데이터가 미래에 발생할 때 사용한다. **create**에 지정한 함수에서 Future 타입을 반환하면 미래에 발생하는 데이터를 상태로 등록한다.
    
    * 초기에 등록되는 상탯값은 **initialData**에 지정한다.
    ```
    FutureProvider<String>(
        create: (context) => Future.delayed(Duration(seconds:4), () => "world"),
        initialData: "hello"
    )
    ```

    * FutureProvider의 상태를 이용하는 하위 위젯
    ```
    var futureState = Provider.of<String>(context);
    return Column(
        children: <Widget>[
            Text('future : ${futureState}')
        ]
    );
    ```

5. StreamProvider 
    ```
    # 1초마다 1~5까지 5번 숫자를 만드는 스트림 함수
    Stream<int> streamFun() async* {
        for (int i = 1; i <= 5; i++) {
            await Future.delayed(Duration(seconds: 1));
            yield i;
        }
    }
    ```
    ```
    StreamProvider<int>(
        create: (context) => streamFun(),
        initialData: 0
    )
    ```
    ```
    var streamState = Provider.of<int>(context);
    return Column(
        children: <Widget>[
            Text('stream : ${streamState}')
        ]
    )
    ```


### Consumer와 Selector
1. Provider를 이용하는 위젯의 생명주기
    * Provider.of()로 상태를 이용하면 등록한 상탯값이 변경됐을 때 Provider의 하위에 추가한 위젯이 불필요하게 **다시 빌드**될 수 있다.

    * 아래 코드는 MyApp 위젯에 Provider로 상태를 등록했고, SubWidget1은 상위에 등록된 상태를 이용하고, SubWidget2는 이용하지 않는다는 가정이다.
        * 상태 데이터가 변경될 때 그 상태를 **이용**하는 위젯은 다시 빌드되어 변경된 값이 적용된다. 그러나 상태를 이용하지 않는 위젯은 다시 빌드되지 않는다.
    ```
    class SubWidget1 extends StatelessWidget {
        @override
        Widget build(BuildContext context) {
            var model1 = Provider.of<MyDataModel1>(context);
            ...(생략)...
        }
    }

    class SubWidget2 extends StatelessWidget {
        @override
        Widget build(BuildContext context) {
            ...(생략)...
        }
    }
    ```

    * 아래와 같은 상황일 때는 SubWidget2가 상태를 이용하지 않더라도 그의 부모 클래스인 HomeWidget이 상태를 이용하기에 상태가 변경될 때 HomeWidget과 SubWidget2 둘 다 다시 빌드된다. 이러한 비효율적인 상황을 막기 위해 **Consumer**나 **Selector**를 이용한다.
    ```
    class HomeWidget extends StatelessWidget {
        @override
        Widget build(BuildContext context) {
            var model1 = Provider.of<MyDataModel1>(context);
            return Column(
                children: <Widget>[
                    SubWidget1(),
                    SubWidget2()
                ]
            );
        }
    }

    class SubWidget1 extends StatelessWidget {
        @override
        Widget build(BuildContext context) {
            var model1 = Provider.of<MyDataModel1>(context)
            ...(생략)...
        }
    }

    class SubWidget2 extends StatelessWidget {
        @override
        Widget build(BuildContext context) {
            ...(생략)...
        }
    }
    ```

2. Consumer - 특정 타입만 빌드하기
    * Consumer를 이용하면 Provider.of()로 상태를 이용하는 것보다 편하게 코드를 작성할 수 있으며, 상탯값이 변경될 때 다시 빌드할 부분을 지정할 수 있다.

    * Consumer를 선언할 때 generic으로 어떤 타입의 상태를 이용할지 명시하면 이 상탯값이 변경될 때마다 **builder**에 지정한 함수가 자동으로 호출된다. 호출되는 함수의 두 번째 매개변수에 이용할 **상탯값이 전달**되므로 Provider.of()로 상태를 가져오지 않아도 된다.

    * HomeWidget에서 많은 Widget을 등록했는데 SubWidget1에서 상태를 이용해야 한다면 이 부분을 Consumer의 builder에 등록하면 된다.
    ```
    Column(
        children: [
            Consumer<MyDataModel1>(
                builder: (context, model, child) {
                    return SubWidget1(model);
                }
            ),
            SubWidget2(),
        ]
    )
    ```

    * builder 부분에 추가한 위젯에 다시 하위 위젯을 여러 개 추가할 수 있는데 어떤 하위 위젯은 다시 빌드되지 않게 할 수 있다.
    
    * 아래 코드는 SubWidget1을 다시 빌드할 때 SubWidget1_1은 함께 빌드하고, SubWidget1_2는 빌드하지 않는다. Comsumer의 **child** 속성에 다시 빌드하지 않을 위젯을 명시하면 된다.
    ```
    Consumer<MyDataModel1>(
        builder: (context, model, child) {
            return SubWidget1(model, child);
        },
        child: SubWidget1_2(),
    ),
    SubWidget2(),
    ...(생략)...

    class SubWidget1 extends StatlessWidget {
        MyDataModel1 model1;
        Widget? child;
        SubWidget1(this.model1, this.child);
        @override
        Widget build(BuildContext context) {
            return Column(
                children: [
                    SubWidget1_1(model1);
                    child!
                ]
            );
        }
    }
    ```

    * Consumer2 ~ Consumer6를 이용해 한 번에 여러 상태를 이용할 수 있다.
    ```
    Consumer2<MyDataModel1, MyDataModel2>(
        builder: (context, model1, model2, child) {
            ...(생략)...
        }
    )
    ```


3. Selector - 특정 데이터만 빌드하기
    * 상태의 타입뿐만 아니라 그 타입의 특정 데이터까지 지정하여 전달받거나 지정한 데이터가 변경될 때 다시 빌드할 수 있다.
    ```
    class MyDataModel with ChangeNotifier {
        int data1 = 0;
        int data2 = 10;
        ...(생략)...
    }
    ```

    * 위 클래스의 객체를 Provider 또는 Consumer로 등록하면 하위 위젯에서 이용할 수 있다. Consumer의 builder에 등록한 두 번째 매개변수에 generic 타입으로 선언한 상태 객체가 전달된다. 따라서 이 객체가 가지는 data1, data2값에 모두 전급 가능하며, 두 값이 변경되면 builder에 지정한 위젯이 다시 빌드된다.
    ```
    # Provider
    ChangeNotifierProvider<MyDataModel>.value(value: MyDataModel())

    # Consumer
    Consumer<MyDataModel>(
        builder: (context, model, child) {
            return Text('consumer, data1: ${model.data1}, data2: ${model.data2}');
        }
    )
    ```

    * 모든 데이터가 아니라 특정 데이터를 이용하고 싶을 때 Selector를 사용한다. Selector를 사용할 때 2개의 generic 타입을 지정해야 하는데 하나는 이용할 **상태 객체 타입**이며 다른 하나는 그 객체에서 이용할 **데이터 타입**이다. builder에 등록한 함수의 두 번째 매개변수로 그 상태 객체에서 이용할 데이터가 전달된다.
    ```
    Selector<MyDataModel, int>(
        builder: (context, data, child) {
            return Text('selector, data:${data}');
        },
        selector: (context, model) => model.data2,
    )
    ```

    * Selector를 이용할 때 꼭 **selector**로 함수를 지정해야 하는데, 이 함수에서 반환하는 데이터가 builder에 지정한 함수의 두 번째 매개변수로 전달된다.

    * Selector를 이용하면 해당 데이터가 변경될 때만 builder의 위젯을 다시 빌드한다.
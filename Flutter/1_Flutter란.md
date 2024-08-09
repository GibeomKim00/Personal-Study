# Flutter(cross platform framework)
### Flutter architecture
* 앱 개발 관점

    * 프레임워크(Dart)
    * 엔진(C/C++)
    * embedder: dart로 작성된 코드가 embedder를 통해 각 플랫폼에 맞게 동작할 수 있게 해준다.
* 웹 애플리케이션 개발 관점

    * 프레임워크(Dart)
    * 브라우저(JavaScript/C++)

### Flutter project 폴더 구성
* android: 안드로이드 앱 구성
* ios: iOS 앱 구성
* lib: 다트 파일
* test: 테스트 다트 파일
* lib/main.dart: 앱의 메인 다트 파일
* .gitignore: 깃에 업로드하지 않을 파일 등록
* pubspec.yaml: 플러터 프로젝트의 메인 환경 파일로 패키지나 리소스 폴더를 추가하는 작업이다.


### main.dart 파일
* import 구문

    * 플러터에서 제공하는 패키지 혹은 외부 패키지(여기서 패키지는 다트 파일을 의미)를 사용하기 위해 사용된다.
    * 개발자가 직접 작성한 다트 파일


* main 함수
    * 다트 엔진의 **진입점**(entry point)으로서 다트 엔진이 main()을 호출하면서 앱이 실행된다.
    * runApp(): 매개변수로 위젯을 지정(위젯이란 화면을 구성하는 클래스이다.)


* runApp 클래스(위젯 클래스)
    * 위젯 클래스는 StatelessWidget 또는 StatefulWidget 중 하나를 상속받아 작성한다.
    * build() 함수: 위젯은 화면 구상이 목적이기에 화면을 어떻게 구상할지 명시해준다.

        * MaterilaApp: 플러터에서 제공하는 위젯 클래스이며, 앱에 material 디자인을 작용해준다. 
        * MyHomePage: 사용자 정의 위젯 클래스에 해당한다.


* MyHomepage 클래스
    * StatefulWidget을 상속받는다.
    * StatefulWidget은 위젯의 화면 구성과 위젯에 출력되는 데이터 등을 별도의 State 클래스에 지정한다.
        * _MyHomeState가 State 클래스에 해당
    * StatefulWidget 클래스가 실행되면 createState() 함수가 자동으로 호출되며 이 함수에서 StatefulWidget을 위한 State 클래스 객체를 반환한다.


* _MyHomePageState 클래스
    * State을 상속받으며 build() 함수가 자동으로 호출되면서 이 함수에 구현된 위젯이 화면에 출력된다.
    * Scaffold는 appBar, body, floatingActionButtion 등으로 화면의 구성요소를 묶어 주는 위젯이다.
        * appBar: 화면 위쪽의 타이틀 바를 나타낸다.
        * body: 화면 중간에 text 위젯으로 문자열을 출력한다.
        * floatingActionButton:: 화면 오른쪽 아래에 둥근 버튼을 표시한다.


***main() -> MyApp -> MyHomePage -> _MyHomePageState 순으로 실행되었으며 화면을 구성하는 대부분은 _MyHomePageState의 build() 함수에 작성되었다.***

### 외부 패키지 사용하는 방법
1. pubspec.yaml
    * dependencies: 앱이 빌드되어 플랫폼에서 실행될 때도 필요한 패키지를 의미한다.
    * dev_dependencies: 앱을 개발할 때만 이용하는 패키지는 앱을 빌드할 때 포함할 필요가 없기에 이런 패키지를 등록한다.
    * 패키지 등록 후 화면 위에 **Pub get**을 클릭해 패키지를 내려받는다.

        * **Pub upgrade**: 패키지를 최신 버전으로 업그레이드
        * **Pub outdated**: 오래된 패키지 종속성 식별
        * **Flutter doctor**: 플러터 개발 환경 점검
2. android studio 터미널 이용
    * 터미널 창에 flutter pub add <패키지 이름>을 입력한다.
    * 등록된 패키지는 다트 파일에서 import 구문으로 불러와서 사용한다.
        * import 'package:english_words/english_words.dart';
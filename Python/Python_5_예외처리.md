# 에러 처리하기: try, except
```
short_list = [1,2,3]
position = 5
try:
    short_list[position]
except:
    print('Need a position between 0 and', len(short_list)-1, 'but got', position)
```
* 특정 예외 핸들러의 사용
   * 형태: except 예외 타입 as 이름
```
# IndexError: sequence에서 잘못된 위치를 입력할 때 발생
short_list = [1,2,3]
while True :
    value = input('Position [q to quit]? ')
    if value == 'q':
        break
    try:
        position = int(value)
        print(short_list[position])
    except IndexError as err:
        print('Bad index:', position)
    except Exception as other:
        print('Something else broke:', other)
```

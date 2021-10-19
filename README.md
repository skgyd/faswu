# FASWU

실시간 동영상에서의 인물 선별 송출 시스템

##프로젝트 설명(Introduction)

이 시스템은 다수의 인물을 인식하고 학습할 수 있으며 인식된 각 인물의 송출 여부를 사용자가 직접 선택할 수 있는데 이 모든 선택 송출 과정을 실시간으로 처리할 수 있다. 여기에서 제시한 기술은 다자간 화상 채팅이나 다자간 화상 회의에서 특정인의 프라이버시 보호를 위한 기술로 활용될 수 있다.


##설치방법(Installation)

1.프로젝트를 로컬디스크(C:)에 다운로드 후 파이참에서 실행합니다.

2. pip install -r requirements.txt 로 패키지 리스트로 설치합니다. 


##사용법 (Usage)

1. 다른 사용자와 연결하기위해 connect 버튼을 누릅니다.
2. 
3. 연결을 원하는 사용자의 ip와 port번호를 입력후 server 또는 client를 선택합니다.
4. 
5. 블라인드 처리를 해제하고 싶은 사용자가 있으면 learn버튼을 눌러서 학습시킵니다.
6. 
7. 다시 블라인드하여 가리고 싶으면 하단의 입력창에 user번호를 입력하여 블라인드를 해제합니다.


# 프로세스
![image](https://user-images.githubusercontent.com/51011817/137945630-36520328-8e7c-44a5-9d7e-d34e4c27aa0b.png)
![image](https://user-images.githubusercontent.com/51011817/137945789-af871212-20ea-4d3e-af4c-3dcf6b283877.png)

1.   프로그램 시작 시 화면에 보이는 모든 인물을 Unknown으로 인식

2.   송출하고 싶은 인물 학습 (연속된 사진 촬영)

3.    User9로 인식한 화면 (블라인드 해제)

4.   학습된 인물을 제외한 인물은 Unknown으로 블라인드 처리됨

5.   다중인물 인식

6.   사용자에게 송출될 인물의 자율성 부여 – user10 가리기

7.   user10 블라인드 처리됨



# 특허 및 논문
![image](https://user-images.githubusercontent.com/51011817/137946929-67a63f23-8964-49ae-b5e8-1078529c04b4.png)

[[KIPS 학술발표대회]실시간 동영상에서의 인물 선별 송출 시스템.pdf](https://github.com/skgyd/faswu/files/7374861/KIPS.pdf)



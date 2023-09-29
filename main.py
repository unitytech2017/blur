import streamlit as st
import cv2
import numpy as np

# Streamlit 앱 제목 설정
st.title("얼굴 블러 처리")

# 이미지 업로드
image = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])

if image is not None:
    # OpenCV로 이미지 읽기
    image_data = image.read()
    image_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # 얼굴 감지를 위한 Haar Cascade 분류기 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 이미지에 얼굴 블러 처리
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        face = cv2.GaussianBlur(face, (99, 99), 30)  # 블러 처리
        img[y:y + h, x:x + w] = face

    # 블러 처리된 이미지 출력
    st.image(img, caption="블러 처리된 이미지", use_column_width=True)

# Streamlit 앱 실행
if __name__ == '__main__':
    st.write("### 얼굴 블러 처리")
    st.write("이미지를 업로드하고 얼굴을 블러 처리하세요.")
    st.write("블러 처리된 이미지가 아래에 표시됩니다.")
    st.write("(Ctrl+C로 앱을 중지할 수 있습니다.)")
    st.image(img, caption="블러 처리된 이미지", use_column_width=True)
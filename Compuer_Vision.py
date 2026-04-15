import cv2
import numpy as np
import os
from datasets import load_dataset

# [심화 문제] 이상치 탐지 필터링 함수
def is_outlier(img_bgr, dark_threshold=50, min_area_ratio=0.05):
    # 명암 분석을 위해 흑백으로 변환
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. 너무 어두운 이미지 제거 (평균 밝기 기준)
    if np.mean(gray) < dark_threshold:
        return True
        
    # 2. 객체 크기가 너무 작은 이미지 제거
    # 1차 과제 조건인 cv2.threshold() 활용! Otsu 이진화로 배경과 전경(물체) 분리
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 외곽선(contour)을 찾아서 물체의 픽셀 면적 계산
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True # 잡힌 물체가 아예 없으면 이상치
        
    # 가장 큰 물체의 면적을 전체 이미지 면적과 비교
    largest_contour = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest_contour) / (img_bgr.shape[0] * img_bgr.shape[1])
    
    if area_ratio < min_area_ratio:
        return True # 물체가 전체 화면의 5%도 안 되면 버림
        
    return False # 위 조건 다 통과하면 정상 이미지!

# [기본 문제] AI 학습을 위한 전처리 및 증강 함수
def preprocess_and_augment(img_bgr):
    # 1. 크기 조정 (224x224 픽셀 고정)
    resized_img = cv2.resize(img_bgr, (224, 224))
    
    # 2. 노이즈 제거 (Gaussian Blur 필터 적용)
    blurred_img = cv2.GaussianBlur(resized_img, (5, 5), 0)

    # 3. 색상 변환 >> 흑백 이미지로 (Grayscale 적용)
    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    
    # 4. 정규화 (Normalize: 픽셀 값을 0~255로 선형 정규화)
    # 딥러닝 텐서에 넣을 땐 0~1로 하지만, 이미지로 저장해서 눈으로 보려면 0~255가 편함
    normalized_img = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
    
    # --- 데이터 증강 (Augmentation) ---
    # 5-1. 좌우 반전
    flipped_img = cv2.flip(normalized_img, 1)
    
    # 5-2. 회전 (중심축 기준으로 15도만 살짝 돌려봄)
    center = (112, 112) # 224의 절반
    rot_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated_img = cv2.warpAffine(normalized_img, rot_matrix, (224, 224))
    
    # 5-3. 색상 변화 (이미 흑백이니까 밝기와 대비를 조절하는 방식으로 적용)
    color_varied_img = cv2.convertScaleAbs(normalized_img, alpha=1.2, beta=20)

    # np.hstack을 통한 각 처리 과정을 한 img에 담음 (보기 편하게 가로로 길게 이어붙임)
    # [정규화본, 반전본, 회전본, 색상변화본]
    # final_display = np.hstack((normalized_img, flipped_img, rotated_img, color_varied_img))
    
    final_display = cv2.merge([normalized_img, flipped_img, rotated_img, color_varied_img])
    
    return final_display

def main():
    # 과제 제출용 디렉토리 생성 (안내서 제출 항목 이름에 맞춤)
    output_dir = r"C:\Users\chanw_uspvz1r\Desktop\Sample"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Hugging Face dataset...")
    # 로컬 경로 대신 안내서 예시에 있는 데이터셋 링크 직접 연결
    dataset = load_dataset("ethz/food101", split="train", streaming=True)

    saved_count = 0
    target_count = 10 # 처리된 이미지 5장 저장 목표

    for data in dataset:
        if saved_count >= target_count:
            break

        # PIL 이미지를 OpenCV 포맷(Numpy, RGB -> BGR)으로 변환
        pil_img = data['image']
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 이상치 탐지 (너무 어둡거나 작은 객체는 건너뜀)
        if is_outlier(img_bgr):
            continue

        # 전처리 및 데이터 증강 수행
        final_img = preprocess_and_augment(img_bgr)
        
        # 파일 저장
        file_path = os.path.join(output_dir, f"preprocessed_image_{saved_count+1}.jpg")
        cv2.imwrite(file_path, final_img)
        
        print(f"{file_path}에 저장 완료")
        saved_count += 1

    print("모든 전처리 작업 및 저장 완료")

if __name__ == "__main__":
    main()
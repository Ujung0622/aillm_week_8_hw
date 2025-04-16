# --------------------------------------------------
# [4] 시각화한 내용을 Steamlit에 배포하세요.
# 위에서 생성한 sunspots_for_prophet.csv를 다운로드 받아, 루트/data 아래에 넣어주세요.
# --------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from prophet import Prophet

# 페이지 설정
st.set_page_config(page_title="🌞 Sunspot Forecast", layout="wide")
st.title("🌞 Prophet Forecast with Preprocessed Sunspot Data")

# ----------------------------------
# [1] 데이터 불러오기
# ----------------------------------
# TODO: 'sunspots_for_prophet.csv' 파일을 불러오고, 'ds' 컬럼을 datetime 형식으로 변환하세요.

df = pd.read_csv('data/sunspots_for_prophet.csv')
df['ds'] = pd.to_datetime(df['ds'])
# 인덱스 정렬
df = df.sort_index()

st.subheader("📄 데이터 미리보기")
st.dataframe(df, use_container_width=True)

# ----------------------------------
# [2] Prophet 모델 정의 및 학습
# ----------------------------------
# TODO: Prophet 모델을 생성하고, 11년 주기 커스텀 seasonality를 추가한 후 학습하세요.

# Prophet 모델 생성
model = Prophet( yearly_seasonality=False)

# 11년 주기의 커스텀 seasonality 추가
model.add_seasonality(
    name='sunspot_cycle',
    period=11,
    fourier_order=5
)

# 모델 학습
model.fit(df)

# ----------------------------------
# [3] 예측 수행
# ----------------------------------
# TODO: 30년간 연 단위 예측을 수행하고, 결과를 forecast에 저장하세요.
# 30년 예측을 위한 미래 데이터프레임 생성 (연단위)
future = model.make_future_dataframe(periods=30, freq='Y')

# 예측 수행
forecast = model.predict(future)

# ----------------------------------
# [4] 기본 시각화
# ----------------------------------
st.subheader("📈 Prophet Forecast Plot")
# TODO: model.plot()을 사용하여 예측 결과를 시각화하세요.
# Prophet 예측 결과 시각화
fig1 = model.plot(forecast)
# 그래프 제목 설정
plt.title('Prophet Forecast Plot')

st.pyplot(fig1)

st.subheader("📊 Forecast Components")
# TODO: model.plot_components()를 사용하여 구성요소를 시각화하세요.

# trend와 sunspot_cycle만 표시하도록 설정
fig2 = model.plot_components(forecast)

st.pyplot(fig2)

# ----------------------------------
# [5] 커스텀 시각화: 실제값 vs 예측값 + 신뢰구간
# ----------------------------------
st.subheader("📉 Custom Plot: Actual vs Predicted with Prediction Intervals")

# TODO: 실제값, 예측값, 신뢰구간을 하나의 plot에 시각화하세요.
fig3, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["ds"], df["y"], label="Actual", color="blue", marker="o")
ax.plot(forecast["ds"], forecast["yhat"], label="Predicted", color="red", linestyle="--")
ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="red", alpha=0.1,
                label="Prediction Interval")
ax.legend(loc='upper right')
ax.grid(True)
ax.set_title('Sunspots: Actual vs. Predicted with Prediction Intervals')
ax.set_xlabel('Year')
ax.set_ylabel('Sunspot Activity')
st.pyplot(fig3)

# ----------------------------------
# [6] 잔차 분석 시각화
# ----------------------------------
st.subheader("📉 Residual Analysis (예측 오차 분석)")

# TODO: df와 forecast를 'ds' 기준으로 병합하여 residual 컬럼을 생성하세요.

merged = pd.merge(df, forecast, on='ds', how='outer')
merged['residual'] = merged['y'] - merged['yhat']

# TODO: residual 시계열을 시각화하세요.
fig4, ax2 = plt.subplots(figsize=(14, 4))
ax2.set_title('Residual Analysis (Actual - Predicted)')
ax2.set_xlabel('Year')
ax2.set_ylabel('Residual')
ax2.grid(True)

# 힌트:
ax2.plot(merged["ds"], merged["residual"],color="purple", marker="o", markersize=6, linewidth=2, label="Residual")
ax2.axhline(0, color="black", linestyle="--")
ax2.legend(loc='upper right')
st.pyplot(fig4)

# ----------------------------------
# [7] 잔차 통계 요약 출력
# ----------------------------------
st.subheader("📌 Residual Summary Statistics")
# TODO: merged["residual"].describe()를 출력하세요.
st.dataframe(merged["residual"].describe(), use_container_width=True)

import time, os
import logging
import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image

# from src.sound import sound
# from src.model import CNN
# from setup_logging import setup_logging
from keras.models import load_model


#file uplod
file_name=st.file_uploader("choose a .wav file")

 

# setup_logging()
logger = logging.getLogger('app')

loaded_model = load_model("emotion_predictor.h5")
labels = {
    0 : "Angry",
    1 : "Disgust",
    2 : "Fear",
    3 : "Happy",
    4 : "Neutral",
    5 : "Pleasant Surprise",
    6 : "Sad"
}

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = 40)
    mfcc_scaled_features = np.mean(mfcc.T, axis = 0)
    
    return mfcc_scaled_features

def get_prediction(filename, model):
    prediction = features_extractor(filename)
    prediction = prediction.reshape(1,-1)
    pred = model.predict(prediction)
    label = np.argmax(pred)
    return labels[label]
    

# def get_spectrogram(type='mel'):
#     logger.info("Extracting spectrogram")
#     y, sr = librosa.load(WAVE_OUTPUT_FILE, duration=DURATION)
#     ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     logger.info("Spectrogram Extracted")
#     format = '%+2.0f'
#     if type == 'DB':
#         ps = librosa.power_to_db(ps, ref=np.max)
#         format = ''.join[format, 'DB']
#         logger.info("Converted to DB scale")
#     return ps, format

def display(spectrogram, format):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')
    plt.title('Mel-frequency spectrogram')
    plt.colorbar(format=format)
    plt.tight_layout()
    st.pyplot(clear_figure=False)


def main():
    title = "Sentiment analysis"
    st.title(title)
    
    # image = Image.open(os.path.join(IMAGE_DIR, 'app_guitar.jpg'))
    # st.image(image, use_column_width=True)

    # if st.button('Record'):
    #     with st.spinner(f'Recording for {DURATION} seconds ....'):
    #         # sound.record()
    #     st.success("Recording completed")

   
    if st.button('Predict'):

        with st.spinner("Processing"):

            prediction = get_prediction(file_name, loaded_model)
        st.success("Prediction completed")
        st.write("### The emotion is **", prediction + "**")
        if prediction == 'N/A':
            st.write("Please Upload the file first")
        st.write("\n")
    st.audio(file_name, format="audio/wav", start_time=0)
    # Add a placeholder
    # if st.button('Display Spectrogram'):
    #     # type = st.radio("Scale of spectrogram:",
    #     #                 ('mel', 'DB'))
    #     if os.path.exists(WAVE_OUTPUT_FILE):
    #         spectrogram, format = get_spectrogram(type='mel')
    #         display(spectrogram, format)
    #     else:
    #         st.write("Please record sound first")

if __name__ == '__main__':
    main()
    # for i in range(100):
    #   # Update the progress bar with each iteration.
    #   latest_iteration.text(f'Iteration {i+1}')
    #   bar.progress(i + 1)
    #   time.sleep(0.1)
    
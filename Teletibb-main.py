import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
import cv2
from scipy import io as scio
import pandas as pd
from scipy import linalg
from scipy import signal
from scipy import sparse
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import joblib
# from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from scipy.signal import medfilt, welch
import pywt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, kurtosis
from tensorflow.keras.preprocessing.sequence import pad_sequences


def process_video(frames):
    """Calculates the average value of each frame."""
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    return np.asarray(RGB)


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal

def jade(X, m, Wprev):
    n = X.shape[0]
    T = X.shape[1]
    nem = m
    seuil = 1 / math.sqrt(T) / 100
    if m < n:
        D, U = np.linalg.eig(np.matmul(X, np.mat(X).H) / T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m, 1), ibl)
        W = np.matmul(np.diag(bl), np.transpose(U[0:n, k[n - m:n]]))
        IW = np.matmul(U[0:n, k[n - m:n]], np.diag(ibl))
    else:
        IW = linalg.sqrtm(np.matmul(X, X.H) / T)
        W = np.linalg.inv(IW)

    Y = np.mat(np.matmul(W, X))
    R = np.matmul(Y, Y.H) / T
    C = np.matmul(Y, Y.T) / T
    Q = np.zeros((m * m * m * m, 1))
    index = 0

    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = np.matmul(Yjk1 / math.sqrt(T), Y[ix, :].T / math.sqrt(
                        T)) - R[ix, jx] * R[lx, kx] - R[ix, kx] * R[lx, jx] - C[ix, lx] * np.conj(C[jx, kx])
                    index += 1
    # Compute and Reshape the significant Eigen
    D, U = np.linalg.eig(Q.reshape(m * m, m * m))
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem * m), dtype=complex)
    Z = np.zeros(m)
    h = m * m - 1
    for u in range(0, nem * m, m):
        Z = U[:, K[h]].reshape((m, m))
        M[:, u:u + m] = la[h] * Z
        h = h - 1
    # Approximate the Diagonalization of the Eigen Matrices:
    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    Bt = np.mat(B).H

    encore = 1
    if Wprev == 0:
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)
    # Main Loop:
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.mat([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp1 = np.matmul(g, g.H)
                temp2 = np.matmul(B, temp1)
                temp = np.matmul(temp2, Bt)
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                la = D[K]
                angles = vcp[:, K[2]]
                if angles[0, 0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0, 0] / 2)
                s = 0.5 * (angles[1, 0] - 1j * angles[2, 0]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.mat([[c, -np.conj(s)], [s, c]])  # Givens Rotation
                    V[:, pair] = np.matmul(V[:, pair], G)
                    M[pair, :] = np.matmul(G.H, M[pair, :])
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    temp = np.concatenate((temp1, temp2), axis=1)
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    A = np.matmul(IW, V)
    S = np.matmul(np.mat(V).H, Y)
    return A, S

def ica(X, Nsources, Wprev=0):
    nRows = X.shape[0]
    nCols = X.shape[1]
    if nRows > nCols:
        print(
            "Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print(
            'Warning - The number of soures cannot exceed number of observation channels.')
        print('The number of sources will be reduced to the number of observation channels ', Nsources)

    Winv, Zhat = jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)
    return W, Zhat

#Unsupervised models used

def POS_WANG(frames, fs):
    WinSec = 1.6
    RGB = process_video(frames)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP

def CHROME_DEHAAN(frames,FS):
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6

    RGB = process_video(frames)
    FN = RGB.shape[0]
    NyquistF = 1/2*FS
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')

    WinL = math.ceil(WinSec*FS)
    if(WinL % 2):
        WinL = WinL+1
    NWin = math.floor((FN-WinL//2)/(WinL//2))
    WinS = 0
    WinM = int(WinS+WinL//2)
    WinE = WinS+WinL
    totallen = (WinL//2)*(NWin+1)
    S = np.zeros(totallen)

    for i in range(NWin):
        RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
        RGBNorm = np.zeros((WinE-WinS, 3))
        for temp in range(WinS, WinE):
            RGBNorm[temp-WinS] = np.true_divide(RGB[temp], RGBBase)
        Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
        Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)

        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf-Alpha*Yf
        SWin = np.multiply(SWin, np.hanning(WinL))

        temp = SWin[:int(WinL//2)]
        S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
        S[WinM:WinE] = SWin[int(WinL//2):]
        WinS = WinM
        WinM = WinS+WinL//2
        WinE = WinS+WinL
    BVP = S
    return BVP

def ICA_POH(frames, FS):
    # Cut off frequency.
    LPF = 0.7
    HPF = 2.5
    RGB = process_video(frames)

    NyquistF = 1 / 2 * FS
    BGRNorm = np.zeros(RGB.shape)
    Lambda = 100
    for c in range(3):
        BGRDetrend = detrend(RGB[:, c], Lambda)
        BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)
    _, S = ica(np.mat(BGRNorm).H, 3)

    # select BVP Source
    MaxPx = np.zeros((1, 3))
    for c in range(3):
        FF = np.fft.fft(S[c, :])
        F = np.arange(0, FF.shape[1]) / FF.shape[1] * FS * 60
        FF = FF[:, 1:]
        FF = FF[0]
        N = FF.shape[0]
        Px = np.abs(FF[:math.floor(N / 2)])
        Px = np.multiply(Px, Px)
        Fx = np.arange(0, N / 2) / (N / 2) * NyquistF
        Px = Px / np.sum(Px, axis=0)
        MaxPx[0, c] = np.max(Px)
    MaxComp = np.argmax(MaxPx)
    BVP_I = S[MaxComp, :]
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
    BVP_F = signal.filtfilt(B, A, np.real(BVP_I).astype(np.double))

    BVP = BVP_F[0]
    return BVP



def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()
def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr



# Model selection
model = RandomForestRegressor(n_estimators=100, random_state=42)


def predict_heart_rate(age, gender, ica_output, chrome_output, pos_output, model):
    """
    Predicts the heart rate using the given inputs.

    Args:
    - age (int): Age of the person.
    - gender (str): Gender of the person ('M' or 'F').
    - ica_output (list): List of ICA signal values.
    - chrome_output (list): List of CHROME signal values.
    - pos_output (list): List of POS signal values.
    - model: Trained machine learning model.

    Returns:
    - predicted_hr (float): Predicted heart rate.
    """

    # Calculate features for ICA, CHROME, and POS signals
    ica_mean = np.mean(ica_output)
    ica_median = np.median(ica_output)
    ica_std = np.std(ica_output)

    chrome_mean = np.mean(chrome_output)
    chrome_median = np.median(chrome_output)
    chrome_std = np.std(chrome_output)

    pos_mean = np.mean(pos_output)
    pos_median = np.median(pos_output)
    pos_std = np.std(pos_output)

    # Create DataFrame with age, gender, and calculated features
    data = {
        'Age': [age],
        'Gender': [gender],
        'ICA_Output_Mean': [ica_mean],
        'ICA_Output_Median': [ica_median],
        'ICA_Output_Std': [ica_std],
        'CHROME_Output_Mean': [chrome_mean],
        'CHROME_Output_Median': [chrome_median],
        'CHROME_Output_Std': [chrome_std],
        'POS_Output_Mean': [pos_mean],
        'POS_Output_Median': [pos_median],
        'POS_Output_Std': [pos_std]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Gender'])
    # Encode gender using Label Encoding
    df['Gender'] = label_encoder.transform(df['Gender'])

    # Select features for prediction
    features = ['Age', 'ICA_Output_Mean', 'ICA_Output_Median', 'ICA_Output_Std',
                'CHROME_Output_Mean', 'CHROME_Output_Median', 'CHROME_Output_Std',
                'POS_Output_Mean', 'POS_Output_Median', 'POS_Output_Std',
                'Gender']

    # Extract features
    X_new = df[features]

    # Predict heart rate
    predicted_hr = model.predict(X_new)

    return predicted_hr[0]




##BloodPressure
def wavelet_features(signal, max_length=6):
    coeffs, _ = pywt.cwt(signal, np.arange(1, 128), 'morl')
    # Extract features from the coefficients
    mean_coeff = np.mean(coeffs, axis=1)
    median_coeff = np.median(coeffs, axis=1)
    max_coeff = np.max(coeffs, axis=1)
    min_coeff = np.min(coeffs, axis=1)
    skewness_coeff = skew(coeffs, axis=1)
    kurtosis_coeff = kurtosis(coeffs, axis=1)
    # Pad or truncate features to a fixed length
    padded_features = [mean_coeff, median_coeff, max_coeff, min_coeff, skewness_coeff, kurtosis_coeff]
    padded_features = [np.pad(f, (0, max_length - len(f)), mode='constant') if len(f) < max_length else f[:max_length] for f in padded_features]
    return np.array(padded_features)
def predict_blood_pressure(age, gender, ica_signal, chrome_signal, pos_signal, rf_sys, rf_dia):
    # Convert gender to numeric value
    gender_numeric = 0 if gender == 'F' else 1
    max_length= 2664
    # Combine signals into one array
    X_ica_padded = pad_sequences([ica_signal], maxlen=max_length, padding='post', truncating='post', dtype='float32')
    X_chrome_padded = pad_sequences([chrome_signal], maxlen=max_length, padding='post', truncating='post', dtype='float32')
    X_pos_padded = pad_sequences([pos_signal], maxlen=max_length, padding='post', truncating='post', dtype='float32')

    X_combined = np.stack([X_ica_padded, X_chrome_padded, X_pos_padded], axis=-1)

    # Apply filtering, normalization, baseline correction
    filtered_array = []
    for sample in X_combined:
        filtered_sample = []
        for signal in sample.T:
            # Apply median filtering
            filtered_signal = medfilt(signal, kernel_size=3)
            # Normalize the signal
            normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
            # Apply baseline correction if needed
            baseline_corrected_signal = normalized_signal - np.mean(normalized_signal)
            filtered_sample.append(baseline_corrected_signal)
        filtered_array.append(np.stack(filtered_sample, axis=-1))

    filtered_array = np.array(filtered_array)

    # Feature Engineering: Add statistical features to the input data
    statistical_features = np.hstack([
        np.mean(filtered_array[..., 0], axis=1).reshape(-1, 1),
        np.mean(filtered_array[..., 1], axis=1).reshape(-1, 1),
        np.mean(filtered_array[..., 2], axis=1).reshape(-1, 1),
        np.median(filtered_array[..., 0], axis=1).reshape(-1, 1),
        np.median(filtered_array[..., 1], axis=1).reshape(-1, 1),
        np.median(filtered_array[..., 2], axis=1).reshape(-1, 1),
        np.var(filtered_array[..., 0], axis=1).reshape(-1, 1),
        np.var(filtered_array[..., 1], axis=1).reshape(-1, 1),
        np.var(filtered_array[..., 2], axis=1).reshape(-1, 1),
        skew(filtered_array[..., 0], axis=1).reshape(-1, 1),
        skew(filtered_array[..., 1], axis=1).reshape(-1, 1),
        skew(filtered_array[..., 2], axis=1).reshape(-1, 1),
        kurtosis(filtered_array[..., 0], axis=1).reshape(-1, 1),
        kurtosis(filtered_array[..., 1], axis=1).reshape(-1, 1),
        kurtosis(filtered_array[..., 2], axis=1).reshape(-1, 1)
    ])

    # Compute frequency domain features
    frequency_features = []
    for sample in filtered_array:
        sample_features = []
        for signal in sample.T:
            _, psd = welch(signal, nperseg=256)
            sample_features.extend([
                np.mean(psd), np.median(psd), np.max(psd), np.min(psd), skew(psd), kurtosis(psd)
            ])
        frequency_features.append(sample_features)

    frequency_features = np.array(frequency_features)

    # Wavelet Transform features
    wavelet_features_list = [wavelet_features(seq) for seq in filtered_array]
    wavelet_features_array = np.array(wavelet_features_list)

    # Combine all features
    X_features = np.hstack([
        statistical_features,
        frequency_features, wavelet_features_array.reshape(wavelet_features_array.shape[0], -1),
        np.array([[age, gender_numeric]])
    ])

    # Predict systolic blood pressure
    sys_bp_pred = rf_sys.predict(X_features)[0]

    # Predict diastolic blood pressure
    dia_bp_pred = rf_dia.predict(X_features)[0]

    return sys_bp_pred, dia_bp_pred




def _load_and_preprocess_video_from_webcam():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Load Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create a frame window and counter placeholder
    FRAME_WINDOW = st.image([])
    frame_counter_placeholder = st.empty()

    # Initialize variables
    frames = []
    face_count = 0
    frame_counter = 0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Draw rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Assume only one face in the frame and crop the region
            x, y, w, h = faces[0]
            face_region = frame[y:y + h, x:x + w]

            # Resize the face region to the desired size
            face_region = cv2.resize(face_region, (224, 224))

            # Add the processed frame to the frames list
            frames.append(face_region)

            face_count += 1

            if face_count >= 900:
                break

        frame_counter += 1
        frame_counter_placeholder.write(f"Frame Counter: {face_count}/900")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    # Release the VideoCapture object
    cap.release()

    # Convert the list of frames to a single numpy array
    frames_np = np.array(frames, dtype=np.uint8)

    return frames_np, face_count




def main():
    st.title("TeleTibb")
    st.title("Blood Pressure Measurement Using RGB Imagery and Machine Learning")
    st.text("Muhammad Haseeb       Mair Ahmed          Sania Khan")
    

    # Get user input at the start
    age = st.number_input("Enter your age:")
    gender = st.selectbox("Select your gender:", ["M", "F"])

    st.text("Press 'Start' to begin capturing video from your webcam.")

    if st.button("Start"):
        st.text("Capturing video... Press 'Stop' to stop capturing.")

        frames_np, frame_counter = _load_and_preprocess_video_from_webcam()

        st.text(f"Captured {frame_counter} frames.")

        FS = 30  # Sample rate

        hr_values = []

        # Process face frames with POS_WANG function
        POSrPPG_signal = POS_WANG(frames_np, FS)
        hr = _calculate_fft_hr(POSrPPG_signal, FS)
        hr_values.append(hr)

        # Process face frames with CHROME_DEHAAN function
        CHROMErPPG_signal = CHROME_DEHAAN(frames_np, FS)
        hr = _calculate_fft_hr(CHROMErPPG_signal, FS)
        hr_values.append(hr)

        # Process face frames with ICA_POH function
        ICArPPG_signal = ICA_POH(frames_np, FS)
        hr = _calculate_fft_hr(ICArPPG_signal, FS)
        hr_values.append(hr)

        # st.text("Heart Rate (HR) Values:")
        # for i, hr in enumerate(hr_values, start=1):
        #     st.text(f"HR{i}: {hr}")

        # Load the model
        # loaded_model = load_model("your_model_file")
        loaded_model = joblib.load('random_forest_model.pkl')
        sys_model= joblib.load('best_rf_sys_model.pkl')
        dia_model= joblib.load('best_rf_dia_model.pkl')
        # Predict heart rate
        ICA_signal = ICArPPG_signal
        CHROME_signal = CHROMErPPG_signal
        POS_signal = POSrPPG_signal
        predicted_hr = predict_heart_rate(age=age, gender=gender, ica_output=ICA_signal, 
                                          chrome_output=CHROME_signal, pos_output=POS_signal, 
                                          model=loaded_model)
        bp= predict_blood_pressure(age=age, gender=gender, ica_signal= ICA_signal, chrome_signal=CHROME_signal , pos_signal= POS_signal, rf_sys=sys_model, rf_dia=dia_model)
        st.text("Predicted Heart Rate: " + str(predicted_hr))
        st.text("Predicted Systolic Blood Pressure: " + str(bp[0]))
        st.text("Predicted Diastolic Blood Pressure: " + str(bp[1]))

if __name__ == "__main__":
    main()

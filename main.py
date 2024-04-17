import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
import cv2
import playsound
import threading
import time
import pyaudio
import wave
from openai import OpenAI
import os
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI as ai
import yolov9
import re

os.environ["OPENAI_API_KEY"] = "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"
API_KEY = "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"
is_recording = False
bill = None
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)


def play_sound(sound_file):
    playsound.playsound(sound_file)


def STT(sound_file):
    client = OpenAI(api_key=API_KEY)
    audio_file = open(sound_file, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    audio_file.close()
    print(transcription.text)
    return transcription.text


def answer(chain, question):
    pattern = re.compile(r'\b(thanh toán|tính tiền|pay)\b')
    if pattern.search(question.lower()):
        return f"Hãy sắp xếp các sản phẩm trước camera, sau đó nhấn phím T"
    else:
        response = chain({"question": question})
        print("Question:" + question)
        print("Answer:" + response["result"])
        return response["result"]


def process_answer(chain, file_name):

    answer_text = answer(chain, STT(file_name))

    TTS(answer_text)

    play_sound("output.mp3")

    os.remove("output.mp3")


def TTS(text):
    client = OpenAI(api_key=API_KEY)
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text,
    ) as response:
        response.stream_to_file("output.mp3")


def record_audio(file_name, duration=5):
    global is_recording
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = duration

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    frames = []

    print("Recording...")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

        if not is_recording:
            break

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_name, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


def detect_object(imgs):
    global bill
    model = yolov9.load(
        "model/yolov9-c.pt",
        device="cpu",
    )

    model.conf = 0.25
    model.iou = 0.45
    model.classes = None

    results = model(imgs)

    results = model(imgs, size=640)

    predictions = results.pred[0]
    categories = predictions[:, 5]
    names = list(results.names.values())
    bill = [(names[int(c)], int((categories == c).sum())) for c in categories.unique()]
    print(bill)
    calculate_invoice(bill)


def calculate_invoice(items):
    total_cost = 0
    csv_file = "./data/office_products.csv"
    df = pd.read_csv(csv_file)
    total_cost = 0
    print("******************* HÓA ĐƠN *****************")
    print("Sản phẩm              Số lượng     Thành tiền")
    print("---------------------------------------------")
    for item_name, item_quantity in items:
        product = df[df["Sản phẩm(EN)"] == item_name]
        if not product.empty:
            available_quantity = product.iloc[0]["Số lượng"]
            name = product.iloc[0]["Sản phẩm"]
            if available_quantity >= item_quantity:
                # Tính toán chi phí và cập nhật số lượng trong DataFrame
                cost = product.iloc[0]["Giá"] * item_quantity
                total_cost += cost
                df.loc[df["Sản phẩm(EN)"] == item_name, "Số lượng"] -= item_quantity
                print(f"{name:20} {item_quantity:9} {cost:13}")
    print("---------------------------------------------")
    print(f"Tổng cộng: {total_cost}.000 vnđ")
    print("*********************************************")
    df.to_csv(csv_file, index=False)
    text = f"Tổng hóa đơn của quý khách là {total_cost}"
    TTS(text)
    play_sound("output.mp3")
    os.remove("output.mp3")
def BOT():
    flag = False
    time_since_zero_faces = None
    Bill = []
    cap = cv2.VideoCapture(0)
    sound_thread = None
    answer_thread = None
    loader = CSVLoader(
        file_path="data/office_products.csv",
        csv_args={
            "delimiter": ",",
            "fieldnames": ["Sản phẩm", "Sản phẩm(EN)", "Vị trí", "Giá", "Số lượng"],
        },
        encoding="utf-8",
    )
    index_creator = VectorstoreIndexCreator()
    docsearch = index_creator.from_loaders([loader])
    chain = RetrievalQA.from_chain_type(
        llm=ai(),
        chain_type="stuff",
        retriever=docsearch.vectorstore.as_retriever(),
        input_key="question",
    )
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40)
        )

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        if len(faces) > 0 and not flag:
            flag = True
            sound_thread = threading.Thread(
                target=play_sound, args=("audio/xinchao.mp3",)
            )
            sound_thread.start()
            time_since_zero_faces = None
        elif len(faces) == 0 and flag:
            if time_since_zero_faces is None:
                time_since_zero_faces = time.time()
            elif time.time() - time_since_zero_faces > 10:
                flag = False
                time_since_zero_faces = None

        cv2.imshow("Bot", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            global is_recording
            file_name = "audio/recorded_audio.wav"
            if is_recording:
                # Kết thúc ghi âm
                is_recording = False
                print("Stopping recording...")
                answer_thread = threading.Thread(
                    target=process_answer,
                    args=(
                        chain,
                        file_name,
                    ),
                )
                answer_thread.start()
            else:
                # Bắt đầu ghi âm
                is_recording = True
                recording_thread = threading.Thread(
                    target=record_audio, args=(file_name,)
                )
                recording_thread.start()
        elif key == ord("t"):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thread = threading.Thread(target=detect_object, args=(rgb_frame,))
            thread.start()
    cap.release()
    cv2.destroyAllWindows()


BOT()

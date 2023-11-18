# Importaciones
import pygame
import cv2
import os
from datetime import datetime
from ultralytics import YOLO
import pandas as pd
from PIL import Image

# Variables Globales y Configuraciones
SIZE = (WIDTH, HEIGHT) = (640, 480)
CLS_MODEL = '../resources/models/1014/weights/last.pt'
CLASS_NAMES = {
    0: 'good',
    1: 'texting',
    2: 'talking',
    3: 'radio',
    4: 'drink',
    5: 'behind',
    6: 'away',
    7: 'Unknown'
}
SEG_MODEL = 'yolov8s-seg.pt'
SEG_KWARG = {
    'classes': 0,
    'imgsz': WIDTH,
    'boxes': True,
    'save': False,
    'show_labels': False,
    'show_conf': False,
    'max_det': 1,
    'verbose': False
}
SAVE_DIR = './captures'
THRESHOLD = 0.7

# Definiciones de Funciones


def init_models(cls=CLS_MODEL, seg=SEG_MODEL):
    global img_seg_model, img_class_model
    img_seg_model = YOLO(seg)
    img_class_model = YOLO(cls)


def init_dataframe():
    global SAVE_DIR, df

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    try:
        df = pd.read_csv('data.csv')
    except:
        df = pd.DataFrame(
            columns=['image_name', 'class', 'confidence', 'time'])
        df.to_csv('data.csv', index=False)


def init():
    init_models()
    init_dataframe()


def pre_process(image, size=SIZE):
    if image is None:
        print(f"Unable to read image")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.resize(image, size)
    return image


def apply_mask(image, _kwarg=SEG_KWARG):
    global img_seg_model
    image = Image.fromarray(image)
    image = img_seg_model(image, **_kwarg)
    return image[0].plot(conf=False, labels=False, pil=True)


def log_distraction(cls, conf, image):
    global SAVE_DIR, df
    now = datetime.now()
    image_name = f"{SAVE_DIR}/{cls}_{now.strftime('%Y%m%d%H%M%S')}.jpg"
    cv2.imwrite(image_name, image)
    df.loc[-1] = [image_name, cls, conf, now.strftime('%Y-%m-%d %H:%M:%S')]
    df.index = df.index + 1
    df = df.sort_index()


def predict_distraction(image):
    global CLASS_NAMES, THRESHOLD, img_class_model
    img = pre_process(image)
    img = apply_mask(img)
    probs = img_class_model(img, verbose=False)[0].probs
    cls = probs.top1
    conf = probs.top1conf.item()
    cls = CLASS_NAMES[cls if conf >= THRESHOLD else 7]
    log_distraction(cls, conf, img)
    return img, cls


pygame.mixer.init()
current_sound = None


def play_sound_async(file_name):
    global current_sound
    if current_sound is not None:
        current_sound.stop()
    current_sound = pygame.mixer.Sound(file_name)
    current_sound.play()


def playAlert(cls):
    ruta = '../resources/audios/'
    if cls == 'texting':
        play_sound_async(ruta + 'Alerta_1.mp3')
    elif cls == 'talking':
        play_sound_async(ruta + 'Alerta_2.mp3')
    elif cls == 'radio':
        play_sound_async(ruta + 'Alerta_3.mp3')
    elif cls == 'drink':
        play_sound_async(ruta + 'Alerta_4.mp3')
    elif cls == 'behind':
        play_sound_async(ruta + 'Alerta_5.mp3')
    elif cls == 'away':
        play_sound_async(ruta + 'Alerta_6.mp3')


# Código Principal
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    times_per_second = 3
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait = fps // times_per_second
    colddown = fps // 1
    init()
    frames = 0
    cls = 'None'
    pastCls = 'None'
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames")
            break
        frame = cv2.resize(frame, SIZE)
        if frames % wait == 0:
            frame, cls = predict_distraction(frame)
            if cls != 'Unknown':
                if cls != pastCls:
                    playAlert(cls)

                pastCls = cls
        cv2.putText(frame, cls + ' - ' + str(frames), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        frames += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    df.to_csv('data.csv', index=False)

    def plot_distractions():
        data = pd.read_csv('data.csv')
        data['class'].value_counts().plot(kind='bar')
    plot_distractions()

import io
from django.http import JsonResponse, HttpResponseBadRequest
import requests
from mpmath.identification import transforms
from torch import classes
from torchvision import models
import torchvision.transforms as transforms
import pyrealsense2 as rs
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from django.http import HttpResponse, HttpResponseBadRequest
from PIL import Image
import math
from threading import Event
import multiprocessing as mp
from ultralytics import YOLO
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
import serial
import serial.tools.list_ports
import time

# python3 manage.py runserver 192.168.181.129:8000

executor = ThreadPoolExecutor()
ser = serial.Serial('COM3', 9600)



#Per testare lo stato in modo semplice la connessione all'api.
#Se restituisce 'Ok' allora c'è connessione e il server  online
def connection_api(request):
    if request.method == 'GET':
        try:
            return JsonResponse({'Status': 'Ok'})
        except requests.exceptions.RequestException:
            return HttpResponse('Error during API request')
        except ValueError:
            return HttpResponse('Invalid response format')
    else:
        return HttpResponse('Invalid request method')




#       **********            STOP Process Image           **********
stop_event = Event()
def stop_processing(request):
    if request.method == 'GET':
        stop_event.set()
        print("Stopped")
        return HttpResponse('Processing stopped')
    else:
        return HttpResponse('Invalid request method')




#       **********            Process Image           **********
async def process_image(request):
    if request.method == 'POST':
        print('POST requested')

        # Verifica se il file del modello è presente nella richiesta
        if 'model' not in request.FILES:
            model = None
        else:
            # Leggi i dati binari del file del modello
            model_file = request.FILES['model']
            model = torch.load(model_file)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", device)
        model.to(device)

        # Load the YOLOv8 model
        modelYolo = YOLO('yolov8n.pt')

        # Crea la coda dei lavori da elaborare in parallelo
        jobs = mp.Queue()

        # Crea i processi di elaborazione
        num_processes = mp.cpu_count() -4
        print("core: ", mp.cpu_count())
        processes = [mp.Process(target=process_job, args=(jobs, i, modelYolo, model, device)) for i in range(num_processes)]

        # Avvia i processi
        for process in processes:
            process.start()

        while not stop_event.is_set():

            try:
                result = await capture_image_async()
                if result is not None:
                    color_frame, depth_frame = result
                    bboxes = detect_person(color_frame, modelYolo)
                    if bboxes is not None:
                        bPredict, bbox_person_predicted, positive_prob = predict_image(bboxes, color_frame, model)
                        if bPredict is True:
                            x_coordinate, y_coordinate, distance = calculate_coordinates(depth_frame, bbox_person_predicted)
                            print(f"____________Coordinates: ({x_coordinate}, {y_coordinate}, {distance}) Probability: ", positive_prob, " ____________")

                            if find_arduino_port() is not None:
                                positive_prob = 0.5
                                Print_Serial(x_coordinate, y_coordinate, distance, positive_prob)


                    # Aggiungi il lavoro alla coda
                    jobs.put((color_frame, depth_frame))  # Passa le informazioni necessarie come una tupla
                else:
                    print("Failed to capture image.")
            except RuntimeError as e:
                print(f"Error processing job: {e}")


            # Se viene premuto il tasto 'q', interrompi il ciclo while
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Aggiungi un messaggio di terminazione per ogni processo
        for i in range(num_processes):
            jobs.put(None)

        # Attendi la terminazione dei processi
        for process in processes:
            process.join()

        # Chiudi tutte le finestre di OpenCV
        cv2.destroyAllWindows()

        # Ferma il profiler
        #profiler.disable()

        # Ritorna una HttpResponse con un messaggio di successo
        return HttpResponse('Image processing completed successfully')

    else:
        # Se la richiesta non è una POST, ritorna una HttpResponseBadRequest
        return HttpResponseBadRequest('Invalid request method')


def process_job(jobs, process_id, modelYolo, model, device):
    """
    Funzione eseguita da ogni processo per elaborare i lavori in parallelo.
    """

    async def async_process_job(jobs, process_id, modelYolo, model, device):
        while True:
            # Preleva un lavoro dalla coda
            job = jobs.get()

            # Se viene passato il segnale di terminazione, interrompi il ciclo
            if job is None:
                break

            bboxes = None

            try:
                print(f"Core: {process_id}")
                result = await capture_image_async()
                if result is not None:
                    color_frame, depth_frame = result
                    bboxes = detect_person(color_frame, modelYolo)
                    if bboxes is not None:
                        bPredict, bbox_person_predicted, positive_prob = predict_image(bboxes, color_frame, model)
                        if bPredict is True:
                            x_coordinate, y_coordinate, distance = calculate_coordinates(depth_frame,bbox_person_predicted)
                            print(f"____________MP Coordinates: ({x_coordinate}, {y_coordinate}, {distance})  Probability: ", positive_prob, " ____________")

                            if find_arduino_port() is not None:
                                positive_prob = 0.5
                                Print_Serial(x_coordinate, y_coordinate, distance, positive_prob)
                else:
                    print("Failed to capture image.")
            except RuntimeError as e:
                print(f"Error processing job: {e}")

    asyncio.run(async_process_job(jobs, process_id, modelYolo, model, device))


# Cerca la persona e crea la bbox di essa
def detect_person(frame, model):
    bbox = None
    person_bboxes = []

    # Use the YOLOv8 model to detect objects in the image
    results = model(frame)
    if len(results) > 0:
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.conf[0] > 0.5 and box.cls[0] == 0:
                    r = box.xyxy[0].astype(int)
                    bbox = (r[0], r[1], r[2], r[3])
                    person_bboxes.append(bbox)
                    print("Predicted!")

    if bbox is None:
        print("No person detected")

    return person_bboxes



def predict_image(bboxes, frame, model):
    # Define the transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize((640, 480)),  # Ridimensiona l'immagine a una risoluzione di 640x480
        transforms.ToTensor(),  # Converte l'immagine in un tensore PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza l'immagine
    ])

    # Inizializza le variabili per il bbox e la probabilità più alta
    max_bbox = None
    max_positive_prob = 0.0

    for bbox in bboxes:
        # Ritaglia l'immagine del tronco/schiena utilizzando il bbox rilevato
        x, y, w, h = bbox
        image = frame[y:y + h, x:x + w]

        # Converti l'immagine in un tensore PyTorch e applica le trasformazioni
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
        image = Image.fromarray(image)  # Convert the image to a PIL image
        image = transform(image).unsqueeze(0)  # Convert the image to a PyTorch tensor and apply transformations

        # Utilizza il modello per effettuare una predizione
        with torch.no_grad():
            model.eval()
            outputs = model(image)
            predicted = F.softmax(outputs, dim=1)

        # Ottieni il valore di probabilità della classe positiva (indice 1)
        positive_prob = predicted[0][1].item()

        # Aggiorna il bbox e la probabilità più alta se la probabilità attuale è maggiore
        if positive_prob > max_positive_prob:
            max_bbox = bbox
            max_positive_prob = positive_prob

    # Se la probabilità della classe positiva è maggiore della soglia specificata, mostra una finestra con il frame
    threshold = 0.009
    if max_positive_prob > threshold:
        bPredict = True
        return (bPredict, max_bbox, max_positive_prob)
    else:
        bPredict = False
        return (bPredict, None, None)



def calculate_coordinates(depth_image, bbox):

    # Ottiene le coordinate del centro del bounding box
    x_min, y_min, x_max, y_max = bbox
    x_center = int((x_min + x_max) / 2)
    y_center = int((y_min + y_max) / 2)

    # Ottiene la distanza dal pixel di interesse nel frame della profondità
    z_center = depth_image[y_center, x_center]

    # coordinate di profondità
    distance = math.sqrt(x_center ** 2 + y_center ** 2 + z_center ** 2)

    # Converte le coordinate in metri
    x = x_center
    y = y_center
    distance = distance / 1000.0

    # Ritorna la distanza e le coordinate della persona
    return x, y, distance










#       **********            UPDATE MODEL           **********

def update_model(request):
    photos_now = capture_4photos(request)
    cropped_images = []
    # Verifica se l'immagine è stata acquisita correttamente
    if photos_now is not None:
        print("Photos captured")
        for photo_now in photos_now:
            # Elabora l'immagine per rilevare la parte posteriore o inferiore del corpo
            cropped_images.append(crop_person(photo_now))


        # Verifica e assegna l'idice della foto per l'addestramento
        if 'indexPhoto' not in request.FILES:
            indexPhoto = 1
        else:
            # Leggi i dati binari del file del modello
            indexPhoto = int(request.FILES['indexPhoto'].read())  # Leggi l'indice come intero

        ## *** Model ***
        # Verifica se il file del modello è presente nella richiesta
        if 'model' not in request.FILES:
            model = None
            indexPhoto = 0
        else:
            # Leggi i dati binari del file del modello
            model = request.FILES['model'].read()

        # Verifica che il file del modello esista
        #model_path = os.path.join(settings.BASE_DIR, 'model.pth')
        if not model:
            # Se il file non esiste, crea un nuovo modello vuoto
            print("Create new model")
            model = create_new_model()
            model = retrain_method(model, cropped_images, indexPhoto)
        else:
            try:
                # Prova a caricare il file del modello esistente
                model = torch.load(io.BytesIO(model))
                model = retrain_method(model, cropped_images, indexPhoto)
            except (IOError, RuntimeError):
                # Se ci sono problemi con il file del modello esistente, crea un nuovo modello vuoto
                model = create_new_model()
                model = retrain_method(model, cropped_images, indexPhoto)

        # Serializza il modello come bytes usando io.BytesIO e torch.save
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        model_bytes = buffer.read()

        # Costruisci la risposta HTTP con l'allegato del file model.pth
        response = HttpResponse(content_type='application/octet-stream')
        response['Content-Disposition'] = 'attachment; filename="model.pth"'
        response.write(model_bytes)

        print("Model sent!")

        return response
    else:
        return HttpResponse("Failed to acquire image.")



def create_new_model():
    num_classes = 2
    # Carica una CNN pre-addestrata
    model = models.resnet18(pretrained=True)

    # Sostituisci l'ultimo strato completamente connesso per adattarlo al numero di classi del tuo problema
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    return model

def retrain_method(model, photos_now, indexPhoto):

    folder_number = (indexPhoto % 8) + 1
    print("Folder numeber: ", folder_number)
    photo_folder = f"C:\\Users\\MadSox\\Desktop\\FotoPersone\\{folder_number}/"  # Nome della cartella corrispondente
    photo_files = os.listdir(photo_folder)  # Elenco dei file nella cartella
    photo_files.sort()  # Ordinamento dei file nella cartella
    incorrect_photo_number = 0

    try:
        for photo_file in photo_files:
            print("***  Start with another incorrect photo. N: ", incorrect_photo_number, "  ***")
            for index in range(len(photos_now)):
                photo_path = os.path.join(photo_folder, photo_file)
                photo_incorrect = Image.open(photo_path)

                incorrect_photo_number += 1

                model = retrain_model(model, photos_now[index], photo_incorrect)

                print("Data augmentation")
                transformed_images_corrects = data_augmentation(photos_now[index])
                transformed_images_incorrects = data_augmentation(photo_incorrect)
                for transformed_images_correct in transformed_images_corrects:
                    for transformed_images_incorrect in transformed_images_incorrects:
                        model = retrain_model(model, transformed_images_correct, transformed_images_incorrect)


                print("photo brighter")
                for i in range(1, 3):
                    bright_image = change_brightness(photos_now[index], i)
                    model = retrain_model(model, bright_image, photo_incorrect)
                print("photo less bright")
                for i in range(2, 0, -1):
                    bright_image = change_brightness(photos_now[index], i)
                    model = retrain_model(model, bright_image, photo_incorrect)

        print("*** Finished with training  ****")

    except:
        print("Error: Photo file not found")
        pass
    return model




def retrain_model(model, photo_now, photo_incorrect):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    model.to(device)

    # Define the transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Converti l'immagine in un tensore PyTorch e applica le trasformazioni
    image = cv2.cvtColor(photo_now, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
    image = Image.fromarray(image)  # Convert the image to a PIL image
    img_tensor_now = transform(image).unsqueeze(0)  # Convert the image to a PyTorch tensor and apply transformations
    # Etichetta positivo
    dataset_now = torch.utils.data.TensorDataset(img_tensor_now, torch.tensor([1]))  # Assign label 1 to correct imag

    image_incorrect = np.asarray(photo_incorrect)
    image_incorrect = cv2.cvtColor(image_incorrect, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
    image_incorrect = Image.fromarray(image_incorrect)  # Convert the image to a PIL image
    img_tensor_incorrect = transform(image_incorrect).unsqueeze(0)  # Convert the image to a PyTorch tensor and apply transformations
    # Etichetta negativo
    dataset_incorrect = torch.utils.data.TensorDataset(img_tensor_incorrect, torch.tensor([0]))  # Assign label 0 to incorrect image

    # Combine the datasets for training
    combined_dataset = torch.utils.data.ConcatDataset([dataset_now, dataset_incorrect])

    # Training the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=1, shuffle=True)
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1} loss: {running_loss / len(train_loader)}")

    return model



def crop_person(image):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    cropped_image = None
    person_count = 0

    # Use the YOLOv8 model to detect objects in the image
    results = model(image)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            if box.conf[0] > 0.5 and box.cls[0] == 0:
                r = box.xyxy[0].astype(int)
                cropped_image = image[r[1]:r[3], r[0]:r[2]]
                person_count += 1
                if person_count > 1:
                    print("Error: More than one person detected")
                    return None


    # Resize images to the same size
    # Check if cropped_image has a valid value before using it
    if cropped_image is not None:
        # Resize images to the same size
        cropped_image = cv2.resize(cropped_image, (224, 225))

        print("Image is good")
        return cropped_image

    else:
        print("Error: No valid bounding boxes found")
        raise Exception('Unable to detect body')



#       **********            UTILITIES           **********

async def capture_image_async():
    while True:
        try:
            async for result in capture_image():
                return result
        except RuntimeError as e:
            if "Device or resource busy" in str(e):
                # Attendi per un breve periodo di tempo e riprova
                print("Delay")
                await asyncio.sleep(0.04)
            else:
                # Gestisci altri errori in modo appropriato
                raise


async def capture_image():
    # Verifica la presenza di una fotocamera Intel RealSense collegata tramite USB
    ctx = rs.context()
    devices = ctx.query_devices()
    if devices.size() > 0:
        # Ottieni l'immagine dalla fotocamera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        #config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

        # Aggiunta di un filtro di decimazione per ridurre la risoluzione
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 2)

        # Aggiunta di un filtro di riduzione del rumore
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)

        # Aggiunta di un filtro di smoothing temporale
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal.set_option(rs.option.filter_smooth_delta, 20)

        # Aggiunta dell'autocalibrazione
        #config.enable_stream(rs.stream.gyro)
        #config.enable_stream(rs.stream.accel)
        #config.enable_stream(rs.stream.pose)

        pipeline.start(config)

        # Applicazione dei filtri
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)

        # Crea un buffer per i frame
        color_frames = []
        depth_frames = []

        try:
            while True:
                loop = asyncio.get_event_loop()
                frames = await loop.run_in_executor(executor, lambda: asyncio.to_thread(pipeline.wait_for_frames, timeout_ms=1500))
                frames = await frames  # Await the coroutine to get the frames object
                color_frame = frames.get_color_frame()
                color_frame = np.asarray(color_frame.get_data())

                depth_frame = frames.get_depth_frame()
                depth_frame = decimation.process(depth_frame)
                depth_frame = depth_to_disparity.process(depth_frame)
                depth_frame = spatial.process(depth_frame)
                depth_frame = temporal.process(depth_frame)
                depth_frame = disparity_to_depth.process(depth_frame)
                depth_frame = get_resized_depth_frame(color_frame, depth_frame)


                color_frames.append(color_frame)
                depth_frames.append(depth_frame)

                if len(color_frames) > 0 and len(depth_frames) > 0:
                    await asyncio.sleep(0.1)
                    yield color_frames.pop(0), depth_frames.pop(0)



        finally:
            pipeline.stop()
        #print("Intel - USB")
    else:
        print("No camera detected!")







def capture_4photos(request):
    # Verifica la presenza di una fotocamera Intel RealSense collegata tramite USB
    ctx = rs.context()
    devices = ctx.query_devices()
    if devices.size() > 0:
        # Ottieni l'immagine dalla fotocamera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Aggiunta di un filtro di riduzione del rumore
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)

        # Aggiunta di un filtro di smoothing temporale
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal.set_option(rs.option.filter_smooth_delta, 20)

        pipeline.start(config)

        # Crea un buffer per i frame
        color_frames = []

        try:
            for i in range(4):
                frames = pipeline.wait_for_frames(timeout_ms=1500)
                color_frame = frames.get_color_frame()
                color_frame = np.asarray(color_frame.get_data())

                color_frames.append(color_frame)

            return color_frames

        finally:
            pipeline.stop()
    else:
        print("No camera detected!")












def get_resized_depth_frame(color_image, depth_frame):
    """
    Resize the depth frame to match the size of the color image
    """
    depth_image = np.asanyarray(depth_frame.get_data())

    # Resize the depth image to match the size of the color image
    resized_depth_image = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    return resized_depth_image






def start_video():
    # Verifica la presenza di una fotocamera Intel RealSense collegata tramite USB
    ctx = rs.context()
    devices = ctx.query_devices()
    if devices.size() > 0:
        # Ottieni lo streaming video dalla fotocamera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        print("Start streaming: Intel - USB")
        return(pipeline)
    else:
        print("Error with Intel")
        return HttpResponse("Failed to acquire video stream.")








def data_augmentation(image):
    # Conversione da PIL a cv2
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    transformed_images = []

    # Rotazione dell'immagine
    angle = 10
    rows, cols, _ = image.shape
    M_rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M_rotation, (cols, rows))
    transformed_images.append(rotated_image)

    # Ridimensionamento dell'immagine
    # scale_percent = 110
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # transformed_images.append(resized_image)

    # Traslazione dell'immagine
    # x_translation = 20
    # y_translation = 20
    # M_translation = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    # translated_image = cv2.warpAffine(image, M_translation, (cols, rows))
    # transformed_images.append(translated_image)

    # Riduzione della saturazione
    transformed_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    transformed_image[:, :, 1] = 0.5 * transformed_image[:, :, 1]
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_HSV2RGB)
    transformed_images.append(transformed_image)

    # Applicazione del filtro di sfocatura
    transformed_image = cv2.GaussianBlur(image, (5, 5), 0)
    transformed_images.append(transformed_image)

    return transformed_images




def change_brightness(image, brightness_factor):
    """
    Modifica la luminosità dell'immagine
    :param image: l'immagine da modificare
    :param brightness_factor: il fattore di luminosità, compreso tra 0 e 1
    :return: l'immagine modificata
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * brightness_factor
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * brightness_factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image









def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description:
            return port.device
    return None



def Print_Serial(x_coordinate, y_coordinate, distance, positive_prob):


    distance = round(distance, 2)
    positive_prob = positive_prob * 100
    positive_prob = round(positive_prob, 2)

    print(f"{x_coordinate},{y_coordinate},{distance},{positive_prob}\n")

    x_coordinate = 0.640
    distance = 0.054
    positive_prob = 5.403

    # Formatta i dati come una stringa separata da virgole
    data = f"{x_coordinate},{distance},{positive_prob}\n"

    # Invia i dati ad Arduino
    ser.write(data.encode())

    # Attendi un breve ritardo prima di inviare il prossimo set di dati
    time.sleep(0.01)

    return None


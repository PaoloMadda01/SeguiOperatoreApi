import io
from django.http import JsonResponse, HttpResponseBadRequest
import requests
from mpmath.identification import transforms
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

# python3 manage.py runserver 192.168.181.129:8000


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
        model.to(device)

        # Load the YOLOv8 model
        model = YOLO('yolov8n.pt')

        # Crea la coda dei lavori da elaborare in parallelo
        jobs = mp.Queue()

        # Crea i processi di elaborazione
        num_processes = mp.cpu_count() - 1
        processes = [mp.Process(target=process_job, args=(jobs, i, model, device)) for i in range(num_processes)]

        # Avvia i processi
        for process in processes:
            process.start()

        while not stop_event.is_set():

            try:
                frame = await capture_image_async()
                bbox = detect_person(frame, model)
                if bbox is not None:
                    x_coordinate, y_coordinate, distance = calculate_coordinates(frame, bbox)
                    print(f"____________Coordinates: ({x_coordinate}, {y_coordinate}, {distance}____________")

            except RuntimeError as e:
                print(f"Error processing job: {e}")

            # Aggiungi il lavoro alla coda
            jobs.put((frame))  # Passa le informazioni necessarie come una tupla

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

        # Ritorna una HttpResponse con un messaggio di successo
        return HttpResponse('Image processing completed successfully')

    else:
        # Se la richiesta non è una POST, ritorna una HttpResponseBadRequest
        return HttpResponseBadRequest('Invalid request method')


def process_job(jobs, process_id, model, device):
    """
    Funzione eseguita da ogni processo per elaborare i lavori in parallelo.
    """

    async def async_process_job(jobs, process_id, model, device):
        while True:
            # Preleva un lavoro dalla coda
            job = jobs.get()

            # Se viene passato il segnale di terminazione, interrompi il ciclo
            if job is None:
                break

            bbox = None
            try:
                print(f"Core: {process_id}")
                frame = await capture_image_async()
                bbox = detect_person(frame, model)
                if bbox is not None:
                    x_coordinate, y_coordinate, distance = calculate_coordinates(frame, bbox)
                    print(f"____________MP Coordinates: ({x_coordinate}, {y_coordinate}, {distance}____________")

            except RuntimeError as e:
                print(f"Error processing job: {e}")

    asyncio.run(async_process_job(jobs, process_id, model, device))








def predict_image(bbox, frame, model):
    # Define the transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize((640, 480)),  # Ridimensiona l'immagine a una risoluzione di 640x480
        transforms.ToTensor(),  # Converte l'immagine in un tensore PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza l'immagine
    ])

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

    # Se la probabilità della classe positiva è maggiore della soglia specificata, mostra una finestra con il frame
    threshold = 0.0000009
    if positive_prob > threshold:
        print("Predicted!")
        x_coordinate, y_coordinate, distance = calculate_coordinates(image, bbox)
        print(f"Coordinates: ({x_coordinate}, {y_coordinate}, {distance}")

    return(x_coordinate, y_coordinate, distance)



def calculate_coordinates(image, bbox):
    # Acquisisce il frame della profondità dall'immagine
    depth_frame = get_depth_frame(image)

    # Ottiene le coordinate del centro del bounding box
    x_min, y_min, x_max, y_max = bbox
    x_center = int((x_min + x_max) / 2)
    y_center = int((y_min + y_max) / 2)

    # Ottiene la distanza dal pixel di interesse nel frame della profondità
    index_1 = depth_frame[0][1]
    index_2 = depth_frame[0][2]
    # coordinate di profondità
    z1 = 0  # distanza dal sensore di profondità corrispondente al pixel (0, 0)
    z2 = np.where(depth_frame[0] == index_1)[0][0]  # distanza dal sensore di profondità corrispondente al pixel (0, 1)
    z3 = np.where(depth_frame[0] == index_2)[0][0]  # distanza dal sensore di profondità corrispondente al pixel (0, 2)

    # distanza interpolata
    distance = math.sqrt((z2 - z1)**2 + (x_center**2 + y_center**2) + (z3 - z1)**2)

    # Calcola le coordinate della persona utilizzando il sensore ad infrarossi
    # per ottenere la distanza tra la telecamera e il pixel di interesse
    #x, y, z = rs.rs2_deproject_pixel_to_point(
    #    depth_frame.profile.as_video_stream_profile().intrinsics,  # parametri della fotocamera
    #    [x_center, y_center],  # coordinate del centro del bounding box
    #    distance)  # distanza dal pixel di interesse

    # Converte le coordinate in metri
    x = x_center / 1000.0
    y = y_center / 1000.0
    distance = distance / 1000.0

    # Ritorna la distanza e le coordinate della persona
    return x, y, distance




# Cerca la persona e crea la bbox di essa
def detect_person(frame, model):
    bbox = None

    # Use the YOLOv8 model to detect objects in the image
    results = model(frame)
    if len(results) > 0:
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.conf[0] > 0.5:
                    r = box.xyxy[0].astype(int)
                    bbox = (r[0], r[1], r[2], r[3])
                    print("Predicted!")
                    break

    if bbox is None:
        print("No person detected")

    return bbox








#       **********            UPDATE MODEL           **********

def update_model(request):
        photo_now = capture_image(request)

        # Verifica se l'immagine è stata acquisita correttamente
        if photo_now is not None:

            # Elabora l'immagine per rilevare la parte posteriore o inferiore del corpo
            cropped_image = crop_person(photo_now)


            ## *** Model ***
            # Verifica se il file del modello è presente nella richiesta
            if 'model' not in request.FILES:
                model = None
            else:
                # Leggi i dati binari del file del modello
                model = request.FILES['model'].read()

            # Verifica che il file del modello esista
            #model_path = os.path.join(settings.BASE_DIR, 'model.pth')
            if not model:
                # Se il file non esiste, crea un nuovo modello vuoto
                print("Create new model")
                model = create_new_model()
                model = retrain_method(model, cropped_image)
            else:
                try:
                    # Prova a caricare il file del modello esistente
                    model = torch.load(io.BytesIO(model))
                    model = retrain_method(model, cropped_image)
                except (IOError, RuntimeError):
                    # Se ci sono problemi con il file del modello esistente, crea un nuovo modello vuoto
                    model = create_new_model()
                    model = retrain_method(model, cropped_image)

            # Serializza il modello come bytes usando io.BytesIO e torch.save
            buffer = io.BytesIO()
            torch.save(model, buffer)
            buffer.seek(0)
            model_bytes = buffer.read()

            # Costruisci la risposta HTTP con l'allegato del file model.pth
            response = HttpResponse(content_type='application/octet-stream')
            response['Content-Disposition'] = 'attachment; filename="model.pth"'
            response.write(model_bytes)

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

def retrain_method(model, photo_now):
    model = retrain_model(model, photo_now)
    for i in range(1, 6):
        model = retrain_model(model, photo_now)
    print("photo flipped 1")
    #for i in range(1, 6):
        # flipped_image = cv2.flip(photo_now, i)
        # model = retrain_model(model, flipped_image)
    print("flip photo 2")
    #for i in range(5, 0, -1):
        # flipped_image = cv2.flip(photo_now, i)
        # model = retrain_model(model, flipped_image)
    print("photo brighter")
    for i in range(1, 3):
        bright_image = change_brightness(photo_now, i)
        model = retrain_model(model, bright_image)
    print("photo less bright")
    for i in range(2, 0, -1):
        bright_image = change_brightness(photo_now, i)
        model = retrain_model(model, bright_image)

    return model




def retrain_model(model, photo_now):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Convert the image from BGR to RGB
    image = photo_now[:, :, [2, 1, 0]]  # Swap the order of the channels from BGR to RGB

    # Convert the image from numpy array to PIL image
    image = Image.fromarray(image)

    # Convert the image to a PyTorch tensor and apply transformations with cv2
    # Apply the transformations to the image
    img_tensor = transform(image).unsqueeze(0)
    # Crea un oggetto di tipo TensorDataset utilizzando l'immagine img_tensor come input e un valore costante 0 come output.
    dataset = torch.utils.data.TensorDataset(img_tensor, torch.tensor([0]))

    # Addestramento del modello
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for epoch in range(5):
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
            if box.conf[0] > 0.5:
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
        print("Try with YOLOv8 - full body")
        return cropped_image

    else:
        print("Error: No valid bounding boxes found")
        raise Exception('Unable to detect body')



#       **********            UTILITIES           **********

async def capture_image_async():

    while True:
        try:
            frame = capture_image(request=None)
            return frame
        except RuntimeError as e:
            if "Device or resource busy" in str(e):
                # Attendi per un breve periodo di tempo e riprova
                await asyncio.sleep(0.01)
            else:
                # Gestisci altri errori in modo appropriato
                raise


def capture_image(request):
    # Verifica la presenza di una fotocamera Intel RealSense collegata tramite USB
    ctx = rs.context()
    devices = ctx.query_devices()
    if devices.size() > 0:
        # Ottieni l'immagine dalla fotocamera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        photo_now = np.asarray(color_frame.get_data())
        pipeline.stop()
        #print("Intel - USB")
    else:
        # Verifica se le immagini sono presenti nella richiesta
        if 'photo' in request.FILES:
            # Leggi l'immagine dalla richiesta
            photo = request.FILES['photo'].read()
            nparr = np.frombuffer(photo, np.uint8)
            photo_now = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print("Webcam")
        else:
            return HttpResponse("Failed to acquire image.")
    return photo_now

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


def get_depth_frame(frame):
    """
    Get the depth frame from the input frame using the RealSense depth sensor.

    Args:
        frame: the input frame.

    Returns:
        The depth frame.
    """
    # Create a pipeline object for the RealSense depth sensor
    pipeline = rs.pipeline()

    # Create a configuration object for the pipeline
    config = rs.config()

    # Enable the depth stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the pipeline
    profile = pipeline.start(config)

    try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Convert the depth frame to a numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Resize the depth image to match the size of the color image
        resized_depth_image = cv2.resize(depth_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        return resized_depth_image

    finally:
        # Stop the pipeline
        pipeline.stop()



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



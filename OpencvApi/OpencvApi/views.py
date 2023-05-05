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

def process_image(request):
    if request.method == 'POST':
        print('POST requested')

        # Verifica se il file del modello è presente nella richiesta
        if 'model' not in request.FILES:
            model = None
        else:
            # Leggi i dati binari del file del modello
            model_file = request.FILES['model']
            model = torch.load(model_file)

        # Define the transformations to be applied to the image
        transform = transforms.Compose([
            transforms.Resize((640, 480)),  # Ridimensiona l'immagine a una risoluzione di 640x480
            transforms.ToTensor(),  # Converte l'immagine in un tensore PyTorch
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizza l'immagine
        ])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)


        while True:

            frame = capture_image(request=None)

            # Rileva la persona nell'immagine
            bbox = detect_person(frame)

            if bbox is not None:
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

                # Stampa i valori di probabilità predetti
                #print(f"Predicted probabilities: {predicted}")

                # Se la probabilità della classe positiva è maggiore della soglia specificata, mostra una finestra con il frame
                threshold = 0.0009
                if positive_prob > threshold:
                    #print("Predicted!")
                    x_coordinate, y_coordinate, distance = follow_person(image, bbox)
                    # Print the distance and the coordinates of the person
                    print(f"Coordinates: ({x_coordinate}, {y_coordinate}, {distance})")

            # Se viene premuto il tasto 'q', interrompi il ciclo while
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Rilascia le risorse
        #pipeline.stop()
        # Chiudi tutte le finestre di OpenCV
        cv2.destroyAllWindows()

        # Ritorna una HttpResponse con un messaggio di successo
        return HttpResponse('Image processing completed successfully')

    else:
        # Se la richiesta non è una POST, ritorna una HttpResponseBadRequest
        return HttpResponseBadRequest('Invalid request method')



def follow_person(image, bbox):
    # Get the center point of the bounding box
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    # Get the dimensions of the image
    height = image.shape[3]
    width = image.shape[2]
    channels = image.shape[1]

    # Calculate the x and y coordinates of the center point of the bounding box as a percentage of the image size
    x_percent = x_center / width
    y_percent = y_center / height

    # Calculate the distance and the coordinates of the person
    x_coordinate, y_coordinate, distance = calculate_coordinates(image, bbox)

    return(x_percent, y_percent, distance)

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
def detect_person(frame):
    # Carica il modello YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Utilizza il modello YOLOv5 per rilevare la parte posteriore della persona
    results = model(frame, size=640)

    # Inizializza le liste per le scatole rilevate, le confidenze e le classi
    boxes = []
    confidences = []
    class_ids = []

    # Imposta i parametri di confidenza e non massima soppressione
    conf_threshold = 0.5
    iou_threshold = 0.4

    # Analizza gli output della rete YOLOv5

    for detection in results.xyxy[0]:
        # Ottiene le informazioni sulla classe, la confidenza e le coordinate della scatola
        class_id = detection[-1]
        confidence = detection[-2]
        if confidence > conf_threshold and class_id == 0:  # Class ID 0: persona
            x, y, w, h = detection[:4].cpu().numpy().astype(int)  # Converte il tensore in un array NumPy e utilizza il metodo astype
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    # Applica la non massima soppressione per rimuovere le sovrapposizioni
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    # Crea il bounding box per la persona rilevata
    if len(indices) > 0:
        #print("Person detected")
        i = indices[0]
        x, y, w, h = boxes[i]
        bbox = (x, y, x + w, y + h)
    else:
        bbox = None
        print("No person detected")

    return bbox





#       **********            UPDATE MODEL           **********

def update_model(request):
        photo_now = capture_image(request)

        # Verifica se l'immagine è stata acquisita correttamente
        if photo_now is not None:
            # Ridimensiona l'immagine
            photo_now = cv2.resize(photo_now, (640, 480))

            # Elabora l'immagine per rilevare la parte posteriore o inferiore del corpo
            cropped_image = crop_back(photo_now)


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


def crop_back(image):
    # Carica il modello YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Utilizza il modello YOLOv5 per rilevare la parte posteriore della persona
    results = model(image, size=640)

    # Estrae le coordinate del bounding box della parte posteriore della persona
    bboxes = results.xyxy[0].cpu().numpy()
    for bbox in bboxes:
        if bbox[5] == 0:  # Se l'oggetto rilevato è una persona
            x, y, w, h = bbox[:4].astype(int)
            cropped_image = image[y:y + h, x:x + w]

            # Adatta le immagini alla stessa dimensione
            cropped_image = cv2.resize(cropped_image, (224, 225))

            print("Image is good")
            print("Try with YOLOv5 - full body")

            return cropped_image


    else:
        # Restituisci un errore se sono state trovate più o nessuna parte
        print("Unable to detect back or lower body")
        raise Exception('Unable to detect back or lower body')





#       **********            UTILITIES           **********
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



import io
from django.http import JsonResponse, HttpResponseBadRequest
import requests
from django.http import HttpResponse
import numpy as np
import cv2
from mpmath.identification import transforms
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import pyrealsense2 as rs
import numpy as np


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
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Acquisisce lo streaming video dalla fotocamera o dal dispositivo di acquisizione video
        cap = cv2.VideoCapture(0)

        while True:
            # Leggi il frame dallo streaming video
            ret, frame = cap.read()

            if ret:
                # Rileva la persona nell'immagine
                bbox = detect_person(frame)

                if bbox is not None:
                    # Ritaglia l'immagine del tronco/schiena utilizzando il bbox rilevato
                    x, y, w, h = bbox
                    image = frame[y:y + h, x:x + w]

                    # Converti l'immagine in un tensore PyTorch e applica le trasformazioni
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
                    image = Image.fromarray(image)  # Convert the image to a PIL image
                    image = transform(image).unsqueeze(
                        0)  # Convert the image to a PyTorch tensor and apply transformations

                    # Acquisisce il frame della profondità dall'immagine
                    depth_frame = get_depth_frame(frame)

                    # Utilizza il modello per effettuare una predizione
                    with torch.no_grad():
                        model.eval()
                        outputs = model(image)
                        predicted = F.softmax(outputs, dim=1)

                    # Ottieni il valore di probabilità della classe positiva (indice 1)
                    positive_prob = predicted[0][1].item()

                    # Stampa i valori di probabilità predetti
                    print(f"Predicted probabilities: {predicted}")

                    # Se la probabilità della classe positiva è maggiore della soglia specificata, mostra una finestra con il frame
                    threshold = 0.8
                    if positive_prob > threshold:
                        cv2.imshow('frame', frame)
                        follow_person(image, bbox, depth_frame)

                # Se viene premuto il tasto 'q', interrompi il ciclo while
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Rilascia le risorse
        cap.release()
        # Chiudi tutte le finestre di OpenCV
        cv2.destroyAllWindows()

        # Ritorna una HttpResponse con un messaggio di successo
        return HttpResponse('Image processing completed successfully')

    else:
        # Se la richiesta non è una POST, ritorna una HttpResponseBadRequest
        return HttpResponseBadRequest('Invalid request method')


def follow_person(image, bbox, depth_frame):
    # Get the center point of the bounding box
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    # Get the dimensions of the image
    height, width = image.shape

    # Calculate the x and y coordinates of the center point of the bounding box as a percentage of the image size
    x_percent = x_center / width
    y_percent = y_center / height

    # Calculate the distance and the coordinates of the person
    distance, (x_coordinate, y_coordinate, z_coordinate) = calculate_distance_and_coordinates(depth_frame, bbox)

    # Print the distance and the coordinates of the person
    print(f"Distance: {distance}")
    print(f"Coordinates: ({x_coordinate}, {y_coordinate}, {z_coordinate})")

    # Move the robot to follow the person
    return(x_percent, y_percent, distance)




def calculate_distance_and_coordinates(depth_frame, bbox):
    # Ottiene i dati di profondità dall'immagine di profondità
    depth_data = depth_frame.get_data()

    # Ottiene le coordinate del centro del bounding box
    x_min, y_min, x_max, y_max = bbox
    x_center = int((x_min + x_max) / 2)
    y_center = int((y_min + y_max) / 2)

    # Calcola la distanza media delle zone all'interno del bounding box
    distances = []
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            dist = depth_data[y][x] / 10.0  # Converte la distanza in cm
            if dist != 0:
                distances.append(dist)

    avg_distance = sum(distances) / len(distances)

    # Calcola le coordinate della persona
    # utilizzando il sensore ad infrarossi per ottenere la distanza
    x, y, z = rs.rs2_deproject_pixel_to_point(
        depth_frame.profile.as_video_stream_profile().intrinsics,  # parametri della fotocamera
        [x_center, y_center],  # coordinate del centro del bounding box
        avg_distance)  # distanza media

    # Converte le coordinate in metri
    x = x / 1000.0
    y = y / 1000.0
    z = z / 1000.0

    # Ritorna la distanza media e le coordinate della persona
    return avg_distance, (x, y, z)


def detect_person(frame):
    # Scarica il modello YOLOv3 pre-addestrato
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # Imposta i parametri di confidenza e non massima soppressione
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Estrae le dimensioni del frame
    height, width, _ = frame.shape

    # Crea un blob dall'immagine per l'input del modello
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    # Passa il blob attraverso la rete
    net.setInput(blob)

    # Ottiene gli output dal layer dell'output della rete YOLOv3
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # Inizializza le liste per le scatole rilevate, le confidenze e le classi
    boxes = []
    confidences = []
    class_ids = []

    # Analizza gli output della rete YOLOv3
    for output in outputs:
        for detection in output:
            # Ottiene le informazioni sulla classe, la confidenza e le coordinate della scatola
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 0:  # Class ID 0: persona
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype(int)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Applica la non massima soppressione per rimuovere le sovrapposizioni
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Crea il bounding box per la persona rilevata
    bbox = None
    if len(indices) > 0:
        i = indices[0][0]
        x, y, w, h = boxes[i]
        bbox = (x, y, x + w, y + h)

    return bbox

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







def update_model(request):
    if request.method == 'POST':
        # Verifica se le immagini sono presenti nella richiesta
        if 'photoNow' not in request.FILES:
            return HttpResponseBadRequest('Problems with your request')
        # Leggi i dati binari dell'immagine
        photo_now = request.FILES['photoNow'].read()
        # Verifica se il file del modello è presente nella richiesta
        if 'model' not in request.FILES:
            model = None
        else:
            # Leggi i dati binari del file del modello
            model = request.FILES['model'].read()

        # Controlla che l'immagine sia stata correttamente ricevuta
        if not photo_now:
            return HttpResponseBadRequest('Problems with your photo')

        # Decodifica l'immagine utilizzando cv2.imdecode
        nparr_now = np.frombuffer(photo_now, np.uint8)
        img_now = cv2.imdecode(nparr_now, cv2.IMREAD_COLOR)

        # Ritaglia l'immagine del volto
        photo_now = crop_back(img_now)

        # Verifica che il file del modello esista
        #model_path = os.path.join(settings.BASE_DIR, 'model.pth')
        if not model:
            # Se il file non esiste, crea un nuovo modello vuoto
            print("Create new model")
            model = create_new_model()
            model = retrain_method(model, photo_now)
        else:
            try:
                # Prova a caricare il file del modello esistente
                model = torch.load(io.BytesIO(model))
                model = retrain_method(model, photo_now)
            except (IOError, RuntimeError):
                # Se ci sono problemi con il file del modello esistente, crea un nuovo modello vuoto
                model = create_new_model()
                model = retrain_method(model, photo_now)

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
    for i in range(1, 21):
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
    for i in range(1, 6):
        bright_image = change_brightness(photo_now, i)
        model = retrain_model(model, bright_image)
    print("photo less bright")
    for i in range(5, 0, -1):
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
    # Utilizza il classificatore di Haar per rilevare la parte posteriore della persona
    back_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    backs = back_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Verifica se è stata trovata una sola parte posteriore
    if len(backs) == 1:
        # Ritaglia l'immagine per avere solo la parte posteriore
        (x, y, w, h) = backs[0]
        cropped_image = image[y:y + h, x:x + w]

        # Adatta le immagini alla stessa dimensione
        cropped_image = cv2.resize(cropped_image, (224, 225))

        print("Image is good")

        return cropped_image

    # Utilizza il classificatore di Haar per rilevare la parte inferiore del corpo
    lower_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')
    lowers = lower_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Verifica se è stata trovata una sola parte inferiore
    if len(lowers) == 1:
        # Ritaglia l'immagine per avere solo la parte inferiore
        (x, y, w, h) = lowers[0]
        cropped_image = image[y:y + h, x:x + w]

        # Adatta le immagini alla stessa dimensione
        cropped_image = cv2.resize(cropped_image, (224, 225))

        print("Image is good")

        return cropped_image

    # Utilizza il classificatore di Haar per rilevare la parte superiore del corpo
    lower_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_upperbody.xml')
    lowers = lower_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Verifica se è stata trovata una sola parte inferiore
    if len(lowers) == 1:
        # Ritaglia l'immagine per avere solo la parte inferiore
        (x, y, w, h) = lowers[0]
        cropped_image = image[y:y + h, x:x + w]
        # Adatta le immagini alla stessa dimensione
        cropped_image = cv2.resize(cropped_image, (224, 225))

        print("Image is good")

        return cropped_image

    # Utilizza il classificatore di Haar per rilevare la parte superiore del corpo
    lower_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_neck.xml')
    lowers = lower_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Verifica se è stata trovata una sola parte inferiore
    if len(lowers) == 1:
        # Ritaglia l'immagine per avere solo la parte inferiore
        (x, y, w, h) = lowers[0]
        cropped_image = image[y:y + h, x:x + w]
        # Adatta le immagini alla stessa dimensione
        cropped_image = cv2.resize(cropped_image, (224, 225))

        print("Image is good")

        return cropped_image

    # Utilizza il classificatore di Haar per rilevare la parte superiore del corpo
    lower_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_head_and_shoulders.xml')
    lowers = lower_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Verifica se è stata trovata una sola parte inferiore
    if len(lowers) == 1:
        # Ritaglia l'immagine per avere solo la parte inferiore
        (x, y, w, h) = lowers[0]
        cropped_image = image[y:y + h, x:x + w]
        # Adatta le immagini alla stessa dimensione
        cropped_image = cv2.resize(cropped_image, (224, 225))

        print("Image is good")

        return cropped_image

    # Utilizza il classificatore di Haar per rilevare l'intera persona
    full_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_head.xml')
    fulls = full_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Verifica se è stata trovata l'intera persona
    if len(fulls) == 1:
        # Ritaglia l'immagine per avere solo la persona
        (x, y, w, h) = fulls[0]
        cropped_image = image[y:y + h, x:x + w]

        # Adatta le immagini alla stessa dimensione
        cropped_image = cv2.resize(cropped_image, (224, 225))

        print("Image is good")

        return cropped_image

    else:
        # Restituisci un errore se sono state trovate più o nessuna parte
        print("Unable to detect back or lower body")
        raise Exception('Unable to detect back or lower body')


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

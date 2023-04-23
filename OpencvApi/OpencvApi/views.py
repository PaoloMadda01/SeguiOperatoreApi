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
            model_file = request.FILES['model']
            model = torch.load(model_file)

        # Controlla che l'immagine sia stata correttamente ricevuta
        if not photo_now:
            return HttpResponseBadRequest('Problems with your photo')

        # Decodifica l'immagine utilizzando cv2.imdecode
        nparr_now = np.frombuffer(photo_now, np.uint8)
        img_now = cv2.imdecode(nparr_now, cv2.IMREAD_COLOR)

        # Ritaglia l'immagine del volto
        photo_now = crop_face(img_now)

        # Verifica che il file del modello esista
        if model:
            # Define the transformations to be applied to the image
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            # Convert the image from BGR to RGB
            image = np.asarray(photo_now)[:, :, [2, 1, 0]]  # Swap the order of the channels from BGR to RGB

            # Convert the image from numpy array to PIL image
            image = Image.fromarray(image)

            # Convert the image to a PyTorch tensor and apply transformations with cv2
            # Apply the transformations to the image
            img_tensor = transform(image).unsqueeze(0)

            # Utilizza il modello per effettuare una predizione
            with torch.no_grad():
                model.eval()
                outputs = model(img_tensor)
                predicted = F.softmax(outputs, dim=1)

            # Ottieni il valore di probabilità della classe positiva (indice 1)
            positive_prob = predicted[0][1].item()

            # Stampa i valori di probabilità predetti
            print(f"Predicted probabilities: {predicted}")

            # Se la probabilità della classe positiva è maggiore della soglia specificata, ritorna 1, altrimenti 0
            threshold = 0.8
            if positive_prob > threshold:
                score = 1
            else:
                score = 0

            print(f"Score: {score} (positive probability: {positive_prob})")

            # Restituisci la risposta di quanto si assomigliano i visi
            return HttpResponse(str(score))

        else:
            # Se il modello non è presente nella richiesta, ritorna un errore
            return HttpResponseBadRequest('Model not found')

    else:
        # Se la richiesta non è una POST, ritorna una HttpResponseBadRequest
        return HttpResponseBadRequest('Invalid request method')


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
        photo_now = crop_face(img_now)

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


def crop_face(image):
    # Usa il classificatore di Haar per rilevare il volto
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Verifica se è stato trovato un solo volto
    if len(faces) == 1:
        # Ritaglia l'immagine per avere solo il volto
        (x, y, w, h) = faces[0]
        cropped_image = image[y:y + h, x:x + w]

        # Adatta le immagini alla stessa dimensione
        cropped_image = cv2.resize(cropped_image, (224, 225))

        print("Image is good")

        return cropped_image
    else:
        # Restituisci un errore se sono stati trovati più o nessun volto
        print("Unable to detect face or multiple faces detected")
        raise Exception('Unable to detect face or multiple faces detected')




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

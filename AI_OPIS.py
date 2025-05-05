import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# PRZYGOTOWANIE DANYCH
train_data = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,         #NORMALIZACJA PIKSELI NP. SZRY 70/255 TO 0.27
    validation_split=0.2    # 20% DANYCH ZOSTANIE PRZEZNACZONE NA WALIDACJE(CZYLI CO 1 EPOKE/CYKL BEDZIE TESTOWAL SWOJE UMIEJETNOSCI NA 20%) A 80% NA TRENING
    )

#GENERATOR DO DANYCH TRENINGOWYCH
train_generator = train_data.flow_from_directory(
    './data/train',         #SCIEZKA DO FOLDERU Z DANYMI TRENINGOWYMI
    target_size=(150, 150), #KAZDY OBRAZ SKALOWANY JEST DO 150X150 PIKSELI
    batch_size=32,             #GENERATOR BEDZIE LADOWAL PO 32 OBRAZY
    class_mode='binary',    #KLASYFIKACJA BINARNA, MODEL ROZROZNIA 2 TYPY 0(KOT) I 1(PIES)
    subset='training'       #GENERATOR LADUJE DANE TYLKO PRZEZNACZONE NA TRENING
    )

#GENERATOR DO DANYCH WALIDACYJNYCH
validation_generator = train_data.flow_from_directory(
    './data/train', 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='binary', 
    subset='validation'     #GENERATOR LADUJE DANE TYLKO PRZEZNACZONE NA WALIDACJE(CZYLI CO 1 EPOKE/CYKL BEDZIE TESTOWAL SWOJE UMIEJETNOSCI NA 20%)
    )


#MODEL
model = tf.keras.Sequential([   #Sequential MODEL SKLADA SIE Z WARSTW, ULOZONE JEDNA PO DRUGIEJ
    
    # 1 WARSTWA 32 FILTRY(KAZDY 3X3), KAZDY FILTR PRZESUWA SIE PO OBRAZIE I SZUKA WZORCOW(USZY, OCZY) I OBLICZA WARTOSC CECHY(CZY FRAGMENT PASUJE DO WZORCA)
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),   #'relu' ZOSTAWIA TYLKO WARTOSCI ISTOTNE
    tf.keras.layers.MaxPooling2D(2, 2),

#[3, 1, 2, 0]  
#[4, 5, 1, 3]    MaxPooling2D
#[0, 2, 3, 1]    ------------>  [5, 3]  
#[7, 8, 6, 4]                   [8, 6]  

#MaxPooling2D ZMNIEJSZA OBRAZ DO 2X2 BO ZOSTAWIA 2 NAJWIEKSZE WARTOSCI, DZIEKI TEMU MODEL DZIALA SZYBCIEJ


     # 2 WARSTWA (IM WIECEJ WARSTW TYM LEPIEJ ROZPOZNAJE)
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    
    tf.keras.layers.Flatten(),      #SPLASZCZA DANE NP DO 1,72
    
    tf.keras.layers.Dense(1, activation='sigmoid') #Dense(1) CZYLI 1 NEURON ZWRACA POJEDYNCZA WARTOSC, PRZYKLAD PREDYKCJI "0.12 = PRAWDOPODOBNIE KOT"
])

#KOMPILOWANIE
model.compile(
    loss='binary_crossentropy', #FUNKCJA STRATY, OKRESLA JAK BARDZO MODEL SIE MYLI (MODLE POWIEDZIAL 0.8=PIES A TO BYL KOT 0)
    optimizer='adam',           #OPTYMALIZATOR UCZY MODEL POPRZEZ POPRAWIANIE WAG NEURONOW
    metrics=['accuracy']        #LICZY % PRZYKLADOW SKLASYFIKOWANYCH POPRAWNIE
    )


#TRENING
model.fit(
    train_generator, #DANE TRENINGOWE
    epochs=15,         #ILE RAZY MODEL PRZEJDZIE PRZEZ CALY ZBIOR DANYCH
    validation_data=validation_generator        #OBRAZY KTORYCH MODEL NIE WIDZI PODCZAS NAUKI ALE SPRAWDZA SWOJA SKUTECZNOSC PO KAZDEJ EPOCE
    )


#TESTOWANIE
def test_model(img_path):

    #WCZYTANIE OBRAZU
    img = image.load_img(img_path, target_size=(150, 150))      #SKALUJE OBRAZ TESTOWY DO 150X150 TAK JAK W DANYCH TRENINGOWYCH
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0     #ZMIENIA OBRAZ NA TABLICE LICZB (MACIERZ PIKSELI) I NORMALIZUJE PRZEZ /255

    #PREDYKCJA
    prediction = model.predict(img_array)       #ANALIZUJE I PRZEWIDUJE NP BLISKO 0 = KOT

    if prediction < 0.5:
        print('To jest kot')
    else:
        print('To jest pies')


#OBRAZ DO TESTU
test_model('./test/3.jpg')

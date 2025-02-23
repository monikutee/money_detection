# Annotavimo API su Deep Learning Modeliu

Šis projektas yra sukurta naudojant FastAPI ir leidžia atlikti paveikslėlių anotavimą, naudojant giliojo mokymosi modelį objektų aptikimui. API priima įkeltą paveikslėlį, vykdo inferenciją (objektų aptikimą), anotuoja aptiktus objektus (prideda ribojančius stačiakampius ir tekstines etiketes) bei grąžina anotetą paveikslėlį JPEG formatu.

# Naudojamos Technologijos ir Bibliotekos

- FastAPI – modernus, greitas ir efektyvus web API kūrimo karkasas Python kalba.
- OpenCV (cv2) – atvirojo kodo kompiuterinės vizijos biblioteka, naudojama paveikslėlių įkėlimui, dekodavimui ir apdorojimui.
- NumPy – skaitmeninių duomenų apdorojimo biblioteka, naudojama baitų konvertavimui į masyvą.
- StreamingResponse – FastAPI klasė, leidžianti srautiniu būdu grąžinti užkoduotus paveikslėlius.
- Inference modelis – įkeltas su get_model funkcija iš inference modulio, skirtas atlikti paveikslėlio inferenciją.
Supervision (sv) – biblioteka, naudojama aptikčių rezultatų apdorojimui bei paveikslėlių anotavimui (bounding box ir label).

# Naudojamas modelis

https://universe.roboflow.com/education-ubaoy/geldbetrage-erkennen/model/2

Modelio paskirtis:
Modelis skirtas objektų aptikimui, konkrečiai – piniginių sumų atpažinimui iš paveikslėlių. Tai reiškia, kad jis geba aptikti ir lokalizuoti paveikslėlyje esančius objektus, susijusius su pinigais (pvz., banknotus, monetas ar jų skaitines reprezentacijas).


Technologija ir našumas:
Puslapyje nurodyta, kad naudojama YOLO-NAS (Accurate) architektūra. Modelis pasižymi labai aukškais našumo rodikliais –

```
mAP (vidutinis tikslumas): 99.1%
Precision (tikslumas): 90.4%
Recall (atgavimo rodiklis): 98.2%
```

# Kaip Naudoti

Paleidimas:
Paleiskite aplikaciją naudojant pvz., Uvicorn:
```bash
uvicorn main:app --reload
```


Užklausos siuntimas:
Siųskite POST užklausą į /annotate endpointą, pridėdami paveikslėlį kaip failą.

Rezultatas:
Gaunamas anotuotas paveikslėlis JPEG formatu, kuriame pažymėtos aptiktos objektų ribos ir jų etiketės.
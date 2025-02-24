# Eur Aptikimo Projektas

Šis projektas skirtas aptikti Euro pinigus (banknotus ir monetas) nuotraukose, naudojant YOLOS modelį ir Hugging Face Transformers biblioteką. Projekto metu buvo panaudoti du Roboflow datasetai:

- **Banknotų datasetas:**  
  [Roboflow - Kupirom](https://universe.roboflow.com/capstone-cz0rh/eur-e3z3g)
- **Monetų datasetas:**  
  [Roboflow - Euro Coin Detector](https://universe.roboflow.com/aag-oqasj/euro-coin-detector/dataset/6)

Be pradinio datasetų turinio, į projektą buvo papildyta naujomis nuotraukomis bei kai kurių kategorijų anotacijos, nes kai kurių kategorijų trūko.

Modelis taip pat patalpintas i huggingface su gradio interface:
https://huggingface.co/spaces/monikutee/money_detection

## Turinio sąrašas

- [Aprašymas](#aprašymas)
- [Duomenų paruošimas](#duomenų-paruošimas)
- [Duomenų augmentacija](#duomenų-augmentacija)
- [Diegimo instrukcijos](#diegimo-instrukcijos)
- [Naudojimo instrukcijos](#naudojimo-instrukcijos)
- [Iškilusios problemos](#iškilusios-problemos)

## Aprašymas

Šio projekto tikslas – sukurti sistemą, kuri aptiktų Euro banknotus ir monetas nuotraukose. Naudojami du skirtingi datasetai iš Roboflow, kuriuose yra nuotraukos su kupiuromis bei monetomis. Modelis aptinka ir suskaičiuoja aptiktus objektus, o rezultatai pateikiami tiek rankiniame būdu testuojant programą (išsaugotu apdorotu paveikslėliu), tiek per API (HTTP atsakyme grąžinant apdorotą paveikslėlį ir aptiktų banknotų bei monetų skaičių).


YOLOS (You Only Look One-Series) yra transformerių pagrindu sukurtas objektų atpažinimo modelis, įkvėptas DETR (Detection Transformer) architektūros. Pagrindiniai bruožai:

✅ Transformeriai vietoje CNN – skirtingai nei tradiciniai objektų aptikimo modeliai (pvz., Faster R-CNN ar YOLO), YOLOS naudoja transformerių dėmesio mechanizmą objektų aptikimui.

✅ Vieno etapo modelis – YOLOS atlieka objektų aptikimą vienu veiksmu be papildomų region proposal etapų.

✅ Puikiai apdoroja įvairaus dydžio objektus – transformerių dėka modelis gali pastebėti įvairaus dydžio objektus vienoje nuotraukoje.

✅ Silpnesnis nei YOLOv8 – palyginus su naujesniais YOLO modeliais, YOLOS yra lėtesnis ir reikalauja daugiau skaičiavimo resursų, bet gali geriau veikti esant sudėtingoms scenoms.



## Duomenų paruošimas

Visiems šio projekto naudojami paveikslėliai buvo paruošti šiais žingsniais:
- **Automatinis orientacijos nustatymas:** Paveikslėlių pikselių orientacija buvo automatiškai suderinta, kartu su EXIF duomenų pašalinimu.
- **Resize:** Nuotraukos buvo ištemptos iki 1000x400 dydžio.
- **Auto-contrast:** Taikytas kontrasto didinimas (contrast stretching), siekiant pagerinti vaizdo kokybę.

## Duomenų augmentacija

Kiekvienam pradiniam paveikslėliui buvo sukurta 3 papildomos versijos, naudojant šiuos transformacijos žingsnius:
- **Horizontalus flip:** 50% tikimybė.
- **Verticalus flip:** 50% tikimybė.
- **90° pasukimai:** Lygiai vienoda tikimybė pasirinkti: neiškeitimą, pasukimą pagal laikrodžio rodyklę, prieš laikrodžio rodyklę arba apverstą variantą.
- **Random crop:** Atsitiktinis paveikslėlio apkarpymas nuo 0% iki 22%.
- **Random rotation:** Atsitiktinis pasukimas nuo -45° iki +45°.
- **Random shear:** Atsitiktinis horizontalus ir vertikalus ištempimas nuo -11° iki +11°.
- **Random exposure adjustment:** Atsitiktinis apšvietimo koregavimas nuo -12% iki +12%.
- **Salt and pepper noise:** Triukšmas, paveikslėlio pikseliuose, pritaikytas 0,97% pikselių.

Papildomai kai kurių datasetų kategorijos buvo papildomai anotuojamos, nes pastebėta, kad kai kurių kategorijų trūksta.

## Diegimo instrukcijos

1. **Suklonuokite repozitoriją:**

```bash
git clone https://github.com/monikutee/money_detection.git
cd money_detection
```

2. **Sukurkite virtualią aplinką ir aktyvuokite ją:**

```bash
python -m venv venv
# Linux/MacOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

3. **Įdiekite priklausomybes:**

```bash
pip install -r requirements.txt
```

## Naudojimo instrukcijos

**Rankinis testavimas**

Norėdami patikrinti modelio veikimą su konkrečiu paveikslėliu, paleiskite skriptą su --manual parametru:

```bash
python your_script.py --manual path/to/your/test.jpg --output annotated_image.jpg
--manual – nurodo, kad paleidžiamas rankinis testavimas.
--output – kelias, kur bus išsaugotas apdorotas paveikslėlis.
```

**API naudojimas**
Paleiskite API serverį:

```bash
python your_script.py --host 0.0.0.0 --port 8000
```

Serveris bus pasiekiamas adresu: http://localhost:8000

Paleisti index.html failą ir lokaliai pasinaudoti endpointu:

```bash
open ./index.html
```


**Endpointas:** /detect

Šis endpointas priima POST užklausas su paveikslėlių failais ir grąžina:

Apdorotą paveikslėlį kaip JPEG.
HTTP antraštėse pateiktus aptiktų banknotų (bill_count) ir monetų (coin_count) skaičius.


## Iškilusios problemos


- Per pirmą treniravimą buvo pastebėta, kad modelis nepritaikė teisingai kategorijų arba neteisingai ženklino objektus. 

    Pakeitimų santrauka:
    
    | Problema  | Pakeitimas | Tikslas
    | ------------- | ------------- | ------------- |
    | Per didelis mokymosi greitis  | learning_rate=5e-5 ->	learning_rate=1e-5  | Stabilizuoti mokymąsi |
    | Nestabilūs gradientai  | max_grad_norm=None ->	max_grad_norm=1.0  | Išvengti didelių svorio pokyčių  |
    | Overfitting | weight_decay=0.0	-> weight_decay=0.01 | Geresnis bendrasis modelio veikimas  |
    | Mažos partijos sukelia triukšmą  | gradient_accumulation_steps=1 -> gradient_accumulation_steps=2  | Stabilizuoti treniravimą  |
    | Per trumpas treniravimas  | num_train_epochs=3 ->	num_train_epochs=5  | Užtikrinti, kad modelis išmoktų geriau  |
    | Per lėtas mokymasis pradžioje  | warmup_ratio=0.0	-> warmup_ratio=0.1  | 	Užtikrinti sklandžią pradžią  |

- Exportuoti duomenys iš Roboflow turejo blogai surikiuotas kategorijas (label), ilgai užtruko debugginimas problemos dėl ko modelis keistai veikia, buvo parašytas check_splits, kad įsitikinti ar anotacijų skaičius atitinka paveikslėlių kiekį.

- Naudojamas git lfs nes modelio failai netelpa i githuba, ilgas procesas tiek modelio apmokymo tiek įkėlimo i githubą viso projekto.

- Pirmą karta modelis buvo mokomas 3h valandas, antrą - 9h, tačiau norimas rezultatas nepasiektas. Reikia daugiau laiko hiperparametrų pasirinkimui, treniravimui ir stebėjimui.

- Buvo bandyta padeploint į render visą api, tačiau nemokamoje versijoje trūksta resursų 'Ran out of memory (used over 512MB) while running your code.'. Dėl to buvo priimtas sprendimas patalpinti į hugging face.






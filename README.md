<div align="center">
# 🏙️ Apartment Price Prediction
### 🏠 Estimarea prețurilor apartamentelor din București cu Machine Learning
</div>

## 📖 Despre proiect

**Apartment Price Prediction** este o aplicație Python care estimează prețurile apartamentelor din București pe baza caracteristicilor acestora.

Proiectul utilizează **8 modele de Machine Learning și Deep Learning**, antrenate, optimizate și comparate pentru identificarea celui mai performant model.

## 🧩 Etapele dezvoltării proiectului

**1. 📥 Colectarea datelor**  
Utilizarea unui dataset de pe Kaggle, cu informații despre apartamente din București.

**2. 🧹 Preprocesarea datelor**  
Curățarea datelor, eliminarea duplicatelor și pregătirea caracteristicilor pentru antrenare.

**3. 📊 Analiza exploratorie (EDA)**  
Vizualizarea distribuției prețurilor, diferențelor dintre sectoare și relației dintre suprafață și preț.

**4. 🤖 Antrenarea modelelor**  
Implementarea a 8 algoritmi de Machine Learning și Deep Learning, folosind seturi separate pentru antrenare, validare și testare.

**5. ⚙️ Optimizarea hiperparametrilor**  
Testarea mai multor configurații pentru îmbunătățirea performanțelor modelelor.

**6. 📈 Evaluarea și compararea modelelor**  
Analizarea metricilor MAE, RMSE, R², predicțiilor în ±10%, timpului de antrenare și diferențelor dintre antrenare și testare.

**7. 🏆 Selectarea modelului câștigător**  
Alegerea automată a modelului cu cea mai mică eroare MAE pe setul de validare.

**8. 💻 Predicția interactivă**  
Dezvoltarea unui formular în Jupyter Notebook pentru estimarea prețului unui apartament pe baza caracteristicilor introduse.

## 🏆 Modelul câștigător

<div align="center">

### 🥇 Deep Neural Network

**Selectat pe baza performanței pe setul de validare**

</div>

## 💶 Exemplu de predicție

| Caracteristică | Valoare |
|---|---|
| Suprafață | 75 m² |
| Camere | 3 |
| Etaj | 4 / 10 |
| Sector | 3 |
| Scor | 5 |

<div align="center">
### 💰 Preț estimat: 93.071 €
*Model utilizat: Deep Neural Network*
</div>

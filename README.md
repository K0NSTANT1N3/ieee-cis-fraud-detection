# IEEE-CIS Fraud Detection - საბოლოო ანგარიში

## პროექტის მიმოხილვა

### Kaggle კონკურსის აღწერა
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) კონკურსი მოიცავს Vesta Corporation-ის რეალური ელექტრონული კომერციის ტრანზაქციების მონაცემებს. მიზანია - გამოვავლინოთ თაღლითური ტრანზაქციები (isFraud=1). მონაცემები შედგება ტრანზაქციისა და იდენტიფიკაციის ფაილებისგან, სულ 590,540 სასწავლო მწკრივით და 506,691 სატესტო მწკრივით.

### მიდგომა
პირველადი ანალიზისა და 1-ლი ადგილის გამარჯვებული გუნდის ([cdeotte](https://www.kaggle.com/cdeotte)) გამოცდილების საფუძველზე გავიგეთ, რომ **ჩვენ არ ვპროგნოზირებთ თაღლითურ ტრანზაქციებს — ვპროგნოზირებთ თაღლითურ კლიენტებს (საკრედიტო ბარათებს)**. ერთხელ რომ მომხმარებელს დაუფიქსირდება თაღლითობა, მისი ყველა შემდგომი ტრანზაქცია ხდება `isFraud=1`. ამიტომ საჭიროა კლიენტის იდენტიფიკაცია და არა ცალკეული ტრანზაქციის ანალიზი.

---

## რეპოზიტორიის სტრუქტურა

```
ieee-cis-fraud-detection
├── data
│   ├── processed
│   │   ├── test_clean.parquet
│   │   ├── X_clean.parquet
│   │   └── y.csv
│   └── raw
│       ├── ieee-fraud-detection.zip
│       ├── sample_submission.csv
│       ├── test_identity.csv
│       ├── test_transaction.csv
│       ├── train_identity.csv
│       └── train_transaction.csv
├── download_data.py
├── kagglehub
├── notebooks
│   ├── data_cleaning.ipynb
│   ├── general_preprocessing
│   │   ├── data_cleaning.ipynb
│   │   ├── eda.ipynb
│   │   └── second_eda.ipynb
│   ├── inference
│   │   └── catboost_inference.ipynb
│   ├── logistic_regression
│   │   └── basic.ipynb
│   └── tree
│       ├── catboost_experiment.ipynb
│       ├── catgboost_experiment.ipynb
│       ├── tree_experiment.ipynb
│       ├── xgboost_experiment2.ipynb
│       └── xgboost_experiment.ipynb
└── README.md

```

### ფაილების განმარტება
- **model_experiment_*.ipynb** — ყოველი მოდელისთვის ცალკე notebook, რომელიც შეიცავს: Cleaning, Feature Engineering, Feature Selection და Training სექციებს, Heading-ებით გამოყოფილს
- **model_inference.ipynb** — საუკეთესო მოდელის ჩამოტვირთვა Model Registry-დან და submission-ის გენერაცია
- **README.md** — პროექტის სრული აღწერა ქართულ ენაზე

---

## EDA - მონაცემების ანალიზი

### კლასების დისბალანსი
```
isFraud = 0 (ლეგიტიმური): 96.5%
isFraud = 1 (თაღლითური):   3.5% (20,663 ტრანზაქცია)
კლასების თანაფარდობა: 27.6:1
```
**დასკვნა:** მძიმე კლასობრივი დისბალანსი — საჭიროა `scale_pos_weight` პარამეტრი მოდელში.

### გამოვლენილი მთავარი პატერნები
- **კლიენტის პატერნი:** მრავალი ტრანზაქციის მქონე კლიენტების 96.9% ყოველთვის ლეგიტიმურია ან ყოველთვის თაღლითი. შერეული სტატუსი მხოლოდ 0.2%-ს აქვს.
- **TransactionAmt:** 99% ტრანზაქცია მცირე თანხისაა. განაწილება ძლიერ გადახრილია.
- **C სვეტები:** C1-C14 მაღალ კორელაციაშია ერთმანეთთან (r>0.95).
- **V სვეტები:** Vesta-ს საკუთრებრივი ფიჩერები, 300+ სვეტი, ბევრს >50% missing აქვს.
- **Identity სვეტები:** id_* სვეტების >90% missing-ია, რადგან ბევრ ტრანზაქციას არ ახლავს პირადობის მონაცემები.

---

## Feature Engineering

### გამოყენებული მიდგომები

#### 1. TransactionAmt დამუშავება
```python
# Log Transform — გადახრილი განაწილების გასწორება
df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
# მრგვალი თანხები — სხვადასხვა თაღლითობის პატერნი
df["TransactionAmt_is_round"] = (df["TransactionAmt"] % 1 == 0)
# ცენტები — .99, .00 ნიმუშები
df["cents"] = df["TransactionAmt"] % 1
```
**შედეგი:** log transform-მა გააუმჯობესა მოდელის სიზუსტე, რადგან XGBoost/CatBoost უკეთ მუშაობს ნორმალიზებულ განაწილებასთან.

#### 2. დროის ფიჩერები
```python
df["TransactionDay"]     = df["TransactionDT"] / 86400
df["TransactionHour"]    = (df["TransactionDT"] / 3600) % 24
df["TransactionWeekday"] = df["TransactionDay"] % 7
```
**შედეგი:** დღის საათმა გააუმჯობესა შედეგი — თაღლითობა უფრო ხშირია გარკვეულ საათებში.

#### 3. D სვეტების ნორმალიზაცია (1-ლი ადგილის მიდგომა)
```python
# D1 = დღეები ბარათის გახსნიდან
# D1n = ბარათის გახსნის დღე (კლიენტზე თითქმის მუდმივი)
df["D1n"] = df["TransactionDay"] - df["D1"]
```
**შედეგი:** D1n თითქმის მუდმივია ერთი კლიენტისთვის — UID-ის კომპონენტად გამოდგება.

#### 4. Email Provider ამოღება
```python
df["P_email_provider"] = df["P_emaildomain"].str.split(".").str[0]
# gmail, yahoo, hotmail — განსხვავებული თაღლითობის განაკვეთი
```

#### 5. Card1 Frequency Encoding
```python
card1_freq = X["card1"].value_counts()
df["card1_freq"] = df["card1"].map(card1_freq)
```
**შედეგი:** ბარათის გამოყენების სიხშირე ძლიერი სიგნალია — იშვიათი ბარათები უფრო საეჭვოა.

#### 6. C სვეტების აგრეგაცია
```python
c_cols = ["C1","C2",...,"C14"]
df["C_mean"] = df[c_cols].mean(axis=1)
df["C_std"]  = df[c_cols].std(axis=1)
df["C_max"]  = df[c_cols].max(axis=1)
```
**შედეგი:** C სვეტები მაღალ კორელაციაშია, ამიტომ მათი დასუფთავება 4 მახასიათებლად ამცირებს სიჭარბეს.

#### 7. UID ფიჩერი (1-ლი ადგილის "Magic" ფიჩერი)
```python
# კლიენტის უნიკალური იდენტიფიკატორი
uid = card1 + "_" + addr1 + "_" + P_emaildomain
uid2 = card1 + "_" + addr1
```
**შედეგი:** UID კლიენტს ადენტიფიცირებს. UID-ით გაკეთებული აგრეგაციები ყველაზე ძლიერი ფიჩერები გახდა.

#### 8. Train+Test კომბინირება აგრეგაციებისთვის
```python
combined = pd.concat([train, test])
# TransactionAmt, D9, D11 სტატისტიკები card1 და uid2-ით
combined["TransactionAmt_card1_mean"] = combined.groupby("card1")["TransactionAmt"].transform("mean")
```
**მიზეზი:** სატესტო მონაცემების ბარათებიც უნდა იღებდნენ სწორ სტატისტიკებს — სხვაგვარად NaN იქნება.

---

## Feature Selection

### გამოყენებული მიდგომები

#### მიდგომა 1: Missing Values ზღვარი
| ზღვარი | წაშლილი სვეტები | შედეგი |
|--------|---------------------|--------|
| >50%   | 214 სვეტი           | ძალიან აგრესიული — კარგი სიგნალების დაკარგვა |
| >90%   | 12 სვეტი            | ოპტიმალური — მხოლოდ ნამდვილად ცარიელი სვეტები |

**გამოცდილება:** 50%-იანი ზღვარი ძალიან მკაცრი იყო. 90%-ზე გადასვლა შედეგს ამჯობინებდა.

#### მიდგომა 2: Near-Zero Variance
```python
low_var = [c for c in num_cols if X[c].std() < 0.01]
# V1, V305, C_min — გამოდევნილია
```

#### მიდგომა 3: მაღალი კორელაცია
| ზღვარი | გამოდევნილი | შედეგი |
|--------|-------------|--------|
| r > 0.95 | 130 სვეტი | ძალიან აგრესიული — შედეგი გაუარესდა |
| r > 0.99 | ნაკლები   | უკეთესი ბალანსი |

**გამოცდილება:** 0.95-იანი ზღვარი ბევრ სასარგებლო V სვეტს შლიდა. XGBoost/CatBoost კორელირებულ ფიჩერებს თავად უმკლავდება.

#### მიდგომა 4: High Cardinality
```python
# DeviceInfo — 1786 უნიკალური მნიშვნელობა — გამოდევნა
# card_identity — მაღალი cardinality, overfitting-ის წყარო
high_card = [c for c in cat_cols if X[c].nunique() > 200]
```

#### მიდგომა 5: CatBoost Feature Importance (საუკეთესო მიდგომა)
```
threshold=0.0:  432 ფიჩერი → AUC=0.9379
threshold=0.01: 248 ფიჩერი → AUC=0.9372
threshold=0.05: 166 ფიჩერი → AUC=0.9371
threshold=0.1:  134 ფიჩერი → AUC=0.9368
threshold=0.5:   51 ფიჩერი → AUC=0.9387  ← საუკეთესო
```
**დასკვნა:** 51 ყველაზე მნიშვნელოვანი ფიჩერი (importance >= 0.5) სჯობდა 432 ფიჩერს!

**Top 20 ფიჩერი CatBoost-ის მიხედვით:**
```
C13          5.50    card1_freq   4.69
C14          3.86    D1n          3.73
card2        3.51    card1        3.21
addr1        2.91    C1           2.51
C6           2.02    dist1        1.73
```

---

## Training - მოდელების სწავლება

### მოდელი 0: Logistic Regression (ზოგადი Baseline)

მოცემული მოდელი გავწვრთენი თითქმის მთლიან მონაცემებზე (50% > Nan გადავყარე, მაღალი კორელაციის სვეტები გადავაგდე და ასშ)

**შედეგები:**
```
Kaggle Private: 0.82
Kaggle Public:  0.86
```
**ანალიზი:** თითქმის ვერაფერი ვერ ისწავლა. აქედან მივხვდი რომ ძალიან დებილი მოდელიც კი 0.8 ზე მაღალ ქულას იღებს

### მოდელი 1: Decision Tree (ხის Baseline)

**კონფიგურაცია:**
```python
DecisionTreeClassifier(max_depth=6, min_samples_leaf=100, class_weight="balanced")
```

**შედეგები:**
```
Kaggle Private: 0.88
Kaggle Public:  0.92 
```

**ანალიზი:** სწრაფი და მარტივი baseline. ძალიან კარგი ქულით. მთელი მომდევნო მოდელების ტრენინგში თითქმის ყველაფერი ვცადე რაც კი შეიძლებოდა და ვერანაირად ვერ გავაუმჯობესე. Public score გაცილებით მეტია ვიდრე private. რაც ალბათ overfit ში წასვვლის ბრალია. მოგვიანებით მივხვდი რომ private სატესტო ბაზაში სავარაუდოდ ისეთი მომხმარებლები არიან რომლებიც ტრეინში არ შემხვედრიან საერთოდ. ახლაც მიჭირს იმის მივედრა თუ რატომ დადო გადაწყვეტილების ხემ უკეთესი შედეგი ვიდრე xgboost მა და lightbm.

---

### მოდელი 2: XGBoost (ბაზისური)

**კონფიგურაცია:**
```python
XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=27.58)
```

**შედეგები:**
```
Validation AUC:  0.9329
Kaggle Public:   0.9170
Kaggle Private:  0.8847
Gap:             0.032  ← overfitting-ის ნიშანი
```

**ანალიზი:** Random split-მა გამოიწვია validation-ის გაბერვა. მოდელმა "დაიხსომა" ტრანზაქციები სამომავლო დროიდან.

**Hyperparameter Search შედეგები (RandomizedSearchCV, 10 trial):**
```
Rank 1: AUC=0.9409 — subsample=0.9, reg_lambda=0.5, reg_alpha=0.5,
                       n_estimators=300, min_child_weight=50, max_depth=6,
                       learning_rate=0.1, colsample_bytree=0.6
Rank 2: AUC=0.9267 — subsample=0.6, reg_lambda=5.0, reg_alpha=0.5
Rank 3: AUC=0.9145 — subsample=0.8, reg_lambda=2.0, reg_alpha=0.1
...
Rank 10: AUC=0.8499
```

**Kaggle შედეგი (Rank 1 პარამეტრებით):**
```
Kaggle Public:  0.9170
Kaggle Private: 0.8847
```

---

### მოდელი 3: XGBoost + Feature Selection (Feature Importance)

**მიდგომა:** XGBoost-ის სწავლება ყველა ფიჩერზე → importance threshold-ით 184 ფიჩერის შერჩევა → გადასწავლება, ასევე XGBoost-ში ჰიპერპარამეტრების გადარჩევა (მოვსინჯე 30 მდე ვარიანტი ჰიპერპარამეტრებისა და AUC მეტრიკით ვწყვეტდი უკეთეს მოდელს).

**შედეგები:**
```
Validation AUC (ყველა ფიჩერი):     0.9627
Validation AUC (184 ფიჩერი):       0.9656
Kaggle Public:                       0.8889  ← გაუარესდა!
Kaggle Private:                      0.8450  ← გაუარესდა!
```

**რატომ გაუარესდა?** Feature selection random validation-ზე გავაკეთე, რომელიც leaky იყო. შერჩეული ფიჩერები კარგად მუშაობდა validation-ზე მაგრამ ვერ გენერალიზდებოდათქო ვიფიქრე, თუმცა ზუსტად მაინც ვერ ვხვდები ასეთი ცუდი შედეგი რატომ მივიღე, სავარაუდოდ ყველა feature ზე ტრენინგი უკეთესია და ხმაურიანი სვეტები არ გვხვდებოდა. აქ C ფიჩერები (რადგან 1-8 მდე მაღალ კორელაციაში იყო[იხილეთ EDA] ამტიომ გავაერთიანე საშუალოსა და მედიანის სვეტში, ეს მეთოდი ხშირად მაქვს სხვა მოდელებშიც გამოყენებული)


---

### მოდელი 4: LightGBM + UID ფიჩერი

**მიდგომა:** UID ფიჩერი (card1+addr1+D1n) და C/M სვეტების UID-ით აგრეგაცია.
როდესაც ასეთ საშინელ შედეგებს ვიღებდი ხელმეორედ ჩავატარე EDA და ვნახე რომ ერთხელ თაღითობის ჩამდენი მომხმარებელი ყველა გადარიცხვაზე თაღლითობდა. (ზუსტი მონაცემები second_eda.ipynb ფაილში მაქვს)

**კონფიგურაცია:**
```python
LGBMClassifier(scale_pos_weight=27.58, time_based_split=True)
```

**შედეგები:**
```
Validation AUC:  ~0.87 (time-based split)
Kaggle Public:   0.8800
Kaggle Private:  0.8600
```

**რატომ გაუარესდა?** UID აგრეგაციები leaky იყო — train-ის სტატისტიკები გამოვიყენეთ train-ის სასწავლებლად. სავარაუდოდ სატესტო კლიენტები train-ში არ გვინახავს, ამიტომ UID-ზე დაფუძნებული ფიჩერები NaN-ებს იძლეოდა.
აქვე მივხვდი რომ დროის მიხედვით გასპლიტვა ძალიან კარგად მეხმარებოდა ჩემივე განსხვავებულად დატრენინგებული მოდელების ერთმანეთთან შესადარებლად რადგან დროის ფაქტორი პირდაპირკავშირში იყო თაღლითობის მსვლელობასთან,
მაშასადამე private და public ქულები ერთმანეთთან უფრო ახლოს იყო მონაცემების დროის მიხედვით გახლეჩვისას, თუმცა ვინაიდან მოდელი უახლეს მონაცემებზე არ იწვრთნებოდა ამის გამო უჭირდა განზოგადება და ტესტზე ცუდ შედეგებს იძლეოდა.

---

### მოდელი 5: XGBoost კლავ

**მიდგომა:** train+test კომბინირება აგრეგაციებისთვის, სპეციფიური V სვეტები, frequency encoding.

**ფიჩერები:**
```python
# Frequency Encoding
card1_FE, addr1_FE, card2_FE, uid_FE, uid2_FE

# Aggregations
TransactionAmt_card1_mean/std
TransactionAmt_uid2_mean/std
D9_card1_mean/std, D11_uid2_mean/std
```

**შედეგები:**
```
CV AUC (3-fold):  0.8733
Validation AUC:   0.9111
Kaggle Public:    0.8720
Kaggle Private:   0.8370  ← კიდევ გაუარესდა
```

**რატომ გაუარესდა?** Time-based validation-ი უფრო მკაცრი validation-ია. CV AUC 0.87 vs Validation AUC 0.91 — კვლავ overfitting.

---

### მოდელი 6: CatBoost + Feature Importance Selection (საუკეთესო რასაც ინფერენსშიც ვიყენებ)

**CatBoost-ის უპირატესობები:**
- კატეგორიულ ფიჩერებს native-ად ამუშავებს (OrdinalEncoder საჭირო არ არის)
- Ordered Boosting — overfitting-ის შემცირება
- Early Stopping — ავტომატურად ჩერდება სწავლება

**კონფიგურაცია:**
```python
# Step 1: სრული სწავლება importance-ის მისაღებად
CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                   subsample=0.9, scale_pos_weight=27.58,
                   early_stopping_rounds=50)
# Val AUC (ყველა ფიჩერი): 0.9362

# Step 2: Feature Importance Threshold ტესტირება
threshold=0.5 → 51 ფიჩერი → AUC=0.9387  ← საუკეთესო

# Step 3: საბოლოო მოდელი 51 ფიჩერით
CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                   subsample=0.9, scale_pos_weight=27.58,
                   early_stopping_rounds=50)
```

**სწავლების პროგრესი:**
```
Iteration 0:   AUC=0.8175
Iteration 100: AUC=0.8946
Iteration 200: AUC=0.9085
Iteration 300: AUC=0.9198
Iteration 400: AUC=0.9291
Iteration 499: AUC=0.9353
```

**საბოლოო შედეგები:**
```
Validation AUC:  0.9353
Validation Precision: 0.2205
Validation Recall:    0.8285
Validation F1:        0.3483
Kaggle Public:        0.9200
Kaggle Private:       0.8960  ← საუკეთესო Private Score!
```

აქ ტრეინის დროს მონაცემების გახლეჩვა ისევ რანდომულად გადავწყვიტე დროის მიხედვით დაყოფის მაგივრად. ასევე ცდუნება მქონდა რომ აქაც UID მიხედვით მეცადა პროგნოზების გაკეთება თუმცა ყველა ასეთმა წინა მცდელობებმა დამანახა რომ უკეთეს შედეგებს არ ვიღებდი ამიტომ ამ მეთოდით აღარ დავატრენინგე.


---

## მოდელების შედარება

| მოდელი | Val AUC | Public | Private | Public-Private Gap |
|--------|---------|--------|---------|-------------------|
| Decision Tree | 0.8413 | ~0.84 | - | - |
| XGBoost Baseline | 0.9329 | 0.9170 | 0.8847 | 0.032 |
| XGBoost + FE Selection | 0.9656 | 0.8889 | 0.8450 | 0.044 |
| LightGBM + UID | ~0.87 | 0.8800 | 0.8600 | 0.020 |
| XGBoost Magic | 0.9111 | 0.8720 | 0.8370 | 0.034 |
| **CatBoost + FI** | **0.9353** | **0.9200** | **0.8960** | **0.024** |

---

## მთავარი გაკვეთილები და წარუმატებლობები

### 1. Random Split vs Time-based Split
**რა ვცადეთ:** Random 80/20 split validation-ისთვის.
**პრობლემა:** ტრანზაქციები დროში თანმიმდევრულია. Random split-ი "მომავლის" ინფორმაციას ათავსებს სასწავლო სეტში, ამიტომ validation AUC ბევრად მაღალია ვიდრე Kaggle Private Score.
**გამოსავალი:** Time-based split — სწავლება პირველ 80%-ზე, validation ბოლო 20%-ზე.

### 2. Aggressive Feature Selection-ი ზიანს აყენებს
**რა ვცადეთ:** კორელაციის ზღვარი r>0.95, missing ზღვარი >50%.
**პრობლემა:** 130 V სვეტი გამოვდევნეთ. V სვეტები Vesta-ს ძვირფასი ფიჩერებია — მათი მოჭრა სიგნალს კლავს.
**გამოსავალი:** ნაკლებად აგრესიული ზღვრები (r>0.99, missing>90%). XGBoost/CatBoost კორელირებულ ფიჩერებს თავადვე უმკლავდება.

### 3. UID Leakage
**რა ვცადეთ:** UID-ის მიხედვით C/M სვეტების აგრეგაცია train-ის სტატისტიკებიდან.
**პრობლემა:** Train-ის სტატისტიკები train-ის სასწავლებლად გამოვიყენეთ (data leakage). ბევრი სატესტო კლიენტი train-ში არ გვხვდება.
**გამოსავალი:** Train+Test კომბინირება აგრეგაციებამდე (ძალიან ცუდი მეთოდი იყო ამას აღარასდროს ვცდი).

### 4. Card Aggregates Leakage
**რა ვცადეთ:** card1_amt_mean — ბარათის საშუალო ტრანზაქციის თანხა.
**პრობლემა:** მთლიანი train-ის სტატისტიკები გამოვიყენეთ, ამიტომ validation AUC გაიბერა.
**გამოსავალი:** Median-ის გამოყენება Mean-ის ნაცვლად (უფრო robust) და train-only სტატისტიკები.

### 5. Feature Selection-ი Leaky Validation-ზე
**რა ვცადეთ:** Feature Importance threshold-ით 184 ფიჩერის შერჩევა (XGBoost-ზე).
**პრობლემა:** Validation AUC 0.9656 მივიღე, მაგრამ Kaggle Public 0.8889 დამიჯდა. Feature selection-ი leaky validation-ზე მოვახდინე.
**გამოსავალი:** CatBoost-ზე Feature Importance — უფრო სტაბილური შედეგი.

---

## MLflow Tracking

### ექსპერიმენტების სტრუქტურა DagsHub-ზე

**ბმული:** https://dagshub.com/kende23/ieee-cis-fraud-detection

```
Experiments:
├── tree_experiments
│   └── decision_tree_baseline
├── xgboost_experiments
│   ├── XGBoost_Cleaning
│   ├── XGBoost_Feature_Engineering
│   ├── XGBoost_Feature_Selection
│   ├── XGBoost_Trial_1 ... XGBoost_Trial_10
│   └── XGBoost_Training_Best
├── XGBoost_Magic
│   ├── XGBoost_Magic_Cleaning
│   ├── XGBoost_Magic_Feature_Engineering
│   ├── XGBoost_Magic_Feature_Selection
│   ├── XGBoost_Magic_Trial_1 ... Trial_10
│   └── XGBoost_Magic_Best
├── LightGBM_Training
│   ├── LightGBM_Cleaning
│   ├── LightGBM_Feature_Engineering
│   ├── LightGBM_Feature_Selection
│   ├── LightGBM_Trial_1 ... Trial_10
│   └── LightGBM_Training_Best
└── CatBoost_Training
    ├── CatBoost_Cleaning
    ├── CatBoost_Feature_Engineering
    ├── CatBoost_Feature_Selection
    └── CatBoost_Training_Best
```

### ლოგირებული მეტრიკები
- **val_auc** — Validation AUC (ROC-AUC)
- **val_precision** — Precision სიზუსტე
- **val_recall** — Recall სისრულე
- **val_f1** — F1 Score
- **cv_auc_mean** — Cross-Validation საშუალო AUC
- **cv_auc_std** — Cross-Validation სტანდარტული გადახრა
- **cv_rank** — Trial-ის რანგი

### ლოგირებული პარამეტრები
- Cleaning: missing_threshold, variance_threshold, dropped counts
- Feature Engineering: დამატებული ფიჩერები, cols_before/after
- Feature Selection: corr_threshold, final_feature_count, importance_threshold
- Training: ყველა hyperparameter, scale_pos_weight, validation_strategy

### Artifacts
- `dropped_columns.txt` — გამოდევნილი სვეტების სია
- `final_features.txt` — საბოლოო ფიჩერების სია
- `feature_importances.csv` — ფიჩერების მნიშვნელობა
- Model Pipeline — Model Registry-ში შენახული

### Model Registry
```
XGBoost_Fraud_Pipeline        — Version 1
XGBoost_Magic_Pipeline        — Version 1
LightGBM_Fraud_Pipeline       — Version 1
CatBoost_Fraud_Pipeline       — Version 1  ← საუკეთესო (Private: 0.8960)
```

---

## საბოლოო მოდელი და Inference

### რატომ CatBoost?
1. **Native categorical support** — M სვეტები, card ტიპები, email provider-ები ბუნებრივად მუშავდება. (ჩემი დაკვირვებიდან გამომდინარე ნაცადმა მეთოდებმა სამწუხაროდ უარესი შედეგები დადო ამიტომ მოდელს დავამუშავებინე კატეგორიულები)
3. **Early Stopping** — ავტომატურად ჩერდება სწავლება, ზედმეტი iteration-ების გარეშე
4. **საუკეთესო Kaggle Private Score** — 0.8960 (სხვა მოდელებს 0.84-0.88 ჰქონდათ)
5. **ყველაზე მცირე Public-Private Gap** — 0.024 (სხვებს 0.032-0.044 ჰქონდათ)

### Inference პროცესი
```python
# 1. Model Registry-დან ჩამოტვირთვა
model = mlflow.sklearn.load_model("models:/CatBoost_Fraud_Pipeline/1")

# 2. იგივე Feature Engineering
# 3. 51 ფიჩერის შერჩევა
# 4. Categorical სვეტების string-ად გადაყვანა
# 5. Prediction
test_probs = model.predict_proba(test[selected_features])[:, 1]
```

### საბოლოო შედეგები Kaggle-ზე
```
CatBoost (საუკეთესო მოდელი):
  Public Score:  0.9200
  Private Score: 0.8960
```

---

## დასკვნა

ამ პროექტში გამოვცადეთ 5 სხვადასხვა მოდელი და მრავალი მიდგომა Feature Engineering-სა და Feature Selection-ში. მთავარი გაკვეთილები:

1. **Validation Strategy** — რანდომულად გახლიჩვა და დროის მიხედვით გახლიჩვა 
2. **ფიჩერების რაოდენობა** — 51 სწორი ფიჩერი 432-ს სჯობს CatBoost ში თუმცა XGBoost ში პირიქით, ბევრი ფიჩერი უკეთეს შედეგებს იძლეოდა ვიდრე სელექციით შემცირებული რაოდენობა
3. **CatBoost კატეგორიულ მონაცემებზე კარგად მუშაობს** — native categorical handling უფრო ზუსტი იყო ვიდრე ხელით დამუშავებული კატეგორიულები
4. **Feature Leakage ძნელი შესამჩნევია** — Validation AUC 0.96 Kaggle-ზე 0.89-ად იქცევა
5. **UID მიდგომა** — ვპროგნოზირებდი კლიენტებს, ტრანზაქციების მაგივრად







import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Herbal medicine dataset
data = {
    "symptoms": [
        "malaria", "malaria, immune booster", "BPH", "BPH", "urine retention",
        "hypertension", "hypertension", "nerve disorder", "numbness", "poor circulation",
        "typhoid fever", "arthritic pain", "NSAID pain", "muscle pain", "arthritic pain",
        "nausea, heartburn", "ulcer, H. pylori", "anemia", "asthma", "cough", "bronchitis",
        "bacterial infection", "bacterial infection", "STI infection", "antibiotic",
        "menstrual pain", "fibroids", "female infertility", "cognitive ability", "cognitive ability",
        "male vitality", "aphrodisiac", "male vitality", "male vitality", "increase sperm production",
        "diabetes type II", "diabetes type II", "diabetes type II", "diabetes type II", "diabetes type II",
        "immune booster", "sedative", "palpitations", "liver disease", "hepatitis B", "waist pain", "immune booster"
    ],
    "medicine": [
        "Malaplus", "Ampoforte", "Croton Plux", "Prostacare mixture", "Uriflow",
        "Kin Tablet", "China Tea(Shanfa Tea)", "Nervon", "Numblex", "Bediako Garlic",
        "Salmophi", "Painaplus", "Pain Capsules", "NPK", "Cerrapac",
        "Dystomis", "Ulcer Care", "Mioko", "Resma", "Cough Mixture", "Cardioplus",
        "Azatrac", "Bactin D", "BactinPlus", "Kantinka BA",
        "Femifort", "Myovite", "Shanfa Ferticare", "Cerecare mixture", "China Cerecare Capsules",
        "Mmerima Mma Mixture", "Mmerima Mma Capsules", "Mmerima Mma Super", "Aphro Powder", "China Aphro Powder",
        "Diamed", "Diatonic", "Bridelia Tea", "Xioke", "Dia Capsules",
        "Immuno Care Booster", "Insomix", "Fefe Powder", "Heptonica", "Hep B Mixture", "Cameron", "Neem tincture"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Text feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["symptoms"])
y = df["medicine"]

# Train a simple classifier
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model and vectorizer
joblib.dump(model, "herbal_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")

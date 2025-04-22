from sklearn.preprocessing import LabelEncoder

def encode_labels(labels):
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return encoded, encoder

label_mapping = {
    0: "Incident",
    1: "Request",
    2: "Problem",
    3: "Change"
}

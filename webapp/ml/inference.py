from joblib import load

def infer(model_pth, data):
    
    model = load(model_pth)
    
    y= model.predict(data)
    
    return y


import torch
import numpy as np
from scipy.stats import multivariate_normal
import math
from sklearn.preprocessing import StandardScaler

def smooth_step(x, x0=0, x1=1):
    if (x0 >x1):
        print('Error switch function')
        return -1
    if not isinstance(x, np.ndarray):
        x = np.array([x])
    x = (x-x0)/(x1-x0)
    x = np.where(x>1,1,x)
    x = np.where(x<0,0,x)
    boolx = np.where(np.logical_and(x>0,x<1))
    x[boolx] = 0.5*(np.tanh((math.pi*x[boolx]-1)/(2*np.sqrt(x[boolx]*(1-x[boolx]))))+1)
    
    return x


def get_confidence_inference ( x, classifier, percentiles, mean, cov, scale_mean, scale_cov):
     
    classifier.eval()
    
    y_hat, latent_x = classifier(x)  
    y_hat = torch.argmax(y_hat, axis=1).cpu().numpy()
    
    confidence_dict = {}
    
    for key in np.unique(y_hat):
        
        confidence = np.ones(len(y_hat))
        for l in range (len(latent_x)):
            scale_latent_x = (latent_x[l].detach().numpy() - scale_mean[str(key)][l])/scale_cov[str(key)][l]
            logpdf_key = multivariate_normal.logpdf(scale_latent_x, mean=mean[str(key)][l], cov=cov[str(key)][l], allow_singular=True)
            confidence = confidence * smooth_step(logpdf_key, percentiles[0][str(key)][l], percentiles[1][str(key)][l])
        confidence_dict[str(key)] = confidence
        
    confidence = np.array([confidence_dict[str(label)][i] for i, label in enumerate(y_hat)])
    
    return y_hat, confidence


def get_percentiles_inference (classifier, x_dict_train, ood, q):
    
    classifier.eval()
    
    percentile_0 = {}
    percentile_1 = {}
    mean = {}
    cov = {}
    scale_mean = {}
    scale_std = {}

    for key in x_dict_train.keys():
        if key == str(ood):
            continue
        
        percentile_0 [key] = []
        percentile_1 [key] = []
        mean [key] = []
        cov [key] = []
        scale_mean [key] = []
        scale_std [key] = []
    
        dataset_key = torch.utils.data.TensorDataset(x_dict_train[key], float(key)*torch.ones([len(x_dict_train[key])]))
        loader_key = torch.utils.data.DataLoader(dataset_key, batch_size=len(dataset_key))
        x,y = next(iter(loader_key))
        y_hat, latent_x = classifier(x)   
        y_hat = torch.argmax(y_hat, axis=1)
        
        for l in range (len(latent_x)):  
            scaler = StandardScaler()
            confidence_set = y_hat==y
            hidden_values_train = scaler.fit_transform(latent_x[l][confidence_set].detach().numpy())
            
            scale_mean_key = scaler.mean_
            scale_std_key = scaler.scale_
            mean_key = np.mean(hidden_values_train,axis=0)
            cov_key = np.cov(hidden_values_train, rowvar=False, bias=True)
            logpdf_train_key = multivariate_normal.logpdf(hidden_values_train, mean=mean_key, cov=cov_key, allow_singular=True)

            scale_mean [key].append(scale_mean_key)
            scale_std [key].append(scale_std_key)
            percentile_0[key].append(np.percentile(logpdf_train_key, q[0]))
            percentile_1[key].append(np.percentile(logpdf_train_key, q[1]))
            mean[key].append(mean_key)
            cov[key].append(cov_key)
    
    return [percentile_0, percentile_1], mean, cov, scale_mean, scale_std
            
    
def get_confidence_mcdropout (x, classifier, n_models):

    classifier.train()
    
    results_models = []
    
    for i in range (n_models):
        y_hat, _ = classifier(x)        
        results_models.append(torch.argmax(y_hat, dim=1).cpu().numpy())

    prediction = []
    confidence = []
    for col in np.array(results_models).T:
        hist = np.histogram(col, bins=np.arange(11))
        pred = np.argmax(hist[0])
        prediction.append(pred)
        confidence.append(hist[0][pred]/n_models)

    confidence=np.array(confidence)
    prediction = np.array([prediction]).reshape(-1)
    
    return prediction, confidence


def get_confidence_ensemble (x, classifiers):
    
    results_models = []
    
    for classifier in classifiers:
        classifier.eval()
        y_hat, _ = classifier(x)        
        results_models.append(torch.argmax(y_hat, dim=1).cpu().numpy())

    prediction = []
    confidence = []
    for col in np.array(results_models).T:
        hist = np.histogram(col, bins=np.arange(11))
        pred = np.argmax(hist[0])
        prediction.append(pred)
        confidence.append(hist[0][pred]/len(classifiers))

    confidence=np.array(confidence)
    prediction = np.array([prediction]).reshape(-1)
    
    return prediction, confidence
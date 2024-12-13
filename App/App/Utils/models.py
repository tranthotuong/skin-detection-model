import os
import tensorflow as tf
from tensorflow.keras import Model
from Utils.Mdel.Model_SA.IRv2_SA import IRv2_SA_model
from Utils.Mdel.Model_SA.ResNet50_SA import ResNet50_SA_model

# from Mdel.Model_SA.IRv2_SA import IRv2_SA_model
# from Mdel.Model_SA.ResNet50_SA import ResNet50_SA_model



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def get_model():
    models_path= r'E:/MSE/Capstone/Project/skin-detection-model/App/App/Models'
    targetnames = []
    model_lib={}

    for targetname in os.listdir(models_path):
        targetname_path = os.path.join(models_path, targetname)
        if os.path.isdir(targetname_path):
            targetnames.append(targetname)

    os.path.join(models_path) 
    for model in targetnames:
        model_dir = os.path.join(models_path, model)
        model_list=[]
        weight_list=[]
        link_model=''

        for folder in os.listdir(model_dir):
            folder_path = os.path.join(model_dir, folder)

            for file in  os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and file.endswith('.keras'):
                    link_model = file_path.replace('\\',"/")
                    model_list.append(file)
                if os.path.isfile(file_path) and file.endswith('.hdf5'):
                    weight_list.append(file)

        weight_sa= []
        weight_nor= []
        for weight in weight_list:
            if weight.__contains__('_SA'):
                weight_sa.append(weight)
            else:
                weight_nor.append(weight)
            
        for model_ in model_list:
            if model_.__contains__('_SA'):
                model_lib[model_]=[f'{link_model}',weight_sa]
            else:
                model_lib[model_]=[f'{link_model}',weight_nor]

    return model_lib


def load_model():
    # Load your model here
    models_lib = get_model()
    load_models ={}
    for model, values in models_lib.items():
        link_model = values[0].split('/Model/')[0]
        weight = values[1]
        weight = str(weight).replace("[","'").replace("]","'").replace("'","").strip()
        if model.__contains__('_SA'):
            if model.__contains__('ResNet50_SA'):
                model_predict = ResNet50_SA_model()
            else:
                model_predict = IRv2_SA_model()
        else:
            model_predict = tf.keras.models.load_model(link_model+'/Model/'+model)
        weight_model_link = link_model+'/Weights/'+weight
        model_predict.load_weights(weight_model_link)
        load_models[model]= model_predict

    return load_models 
obp_minus = {'Age': 57.54125526479321, 'NDrugs': 1.2002807889709775, 'DBP': 86.90116918686277,
             'CVD': 0.11521554005845934, 'Diabetes': 0.18062556100255472, 'TOD': 0.09569840502658289,
             'SBP': 147.6058735529011, 'Smoker': 0.1582545052820548, 'Triglycerides': 131.51651778894472,
             'Sex': 1.4735436949066723}

obp_slash = {'Age': 13.870964370105686, 'NDrugs': 1.316767063463286, 'DBP': 11.322149124518676,
             'CVD': 0.31928188076916747, 'Diabetes': 0.3847076392730811, 'TOD': 0.2941771920152053,
             'SBP': 18.44494275931233, 'Smoker': 0.36497947454615437, 'Triglycerides': 80.10777333958275,
             'Sex': 0.4992995733232793}

abp_minus = {'Age': 57.58512278763608, 'NSBP': 119.34414554995514, 'NDrugs': 1.2087735045685746,
             'DDBP': 79.52172132845403, 'CVD': 0.11537664848443002, 'Diabetes': 0.18175331998434946,
             'DHR': 74.77752020149323, 'AbdominalCircumference': 98.50446705714054, 'Smoker': 0.15862275311284493,
             'Sex': 1.4746024074201938}

abp_slash = {'Age': 13.854377752299175, 'NSBP': 15.279281414145757, 'NDrugs': 1.3175189519078263,
             'DDBP': 10.565278286972694, 'CVD': 0.31947594192510065, 'Diabetes': 0.38564109047016265,
             'DHR': 11.118542073968058, 'AbdominalCircumference': 11.948154159939778, 'Smoker': 0.36532393202163255,
             'Sex': 0.4993545456798708}

obp_model_params = {'hidden_layer_dim': (32, 64, 32), 'learning_rate': 0.01,
                    'loss': 'binary_crossentropy', 'optimizer': 'adam'}

abp_model_params = {'hidden_layer_dim': (32, 64, 32), 'learning_rate': 0.001,
                    'loss': 'binary_crossentropy', 'optimizer': 'adam'}

obp_model_path = 'models/.office10_keras1357.h5'
abp_model_path = 'models/.ambulatory10_keras1357.h5'

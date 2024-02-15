import pandas as pd
from risk_calculator import predict_obp, predict_abp

if __name__ == '__main__':

    # Load CSV file
    file_path = 'data/example.csv'
    df = pd.read_csv(file_path)

    obp_predictions = []
    abp_predictions = []

    for index, row in df.iterrows():

        # Extract input values for each model prediction

        input_values_obp = row[['Age', 'SBP', 'NDrugs', 'Sex', 'TOD', 'CVD', 'Diabetes', 'Smoker', 'DBP',
                                'Triglycerides']].to_dict()

        input_values_abp = row[['Age', 'NSBP', 'NDrugs', 'Sex', 'DDBP', 'CVD', 'Diabetes', 'Smoker', 'DHR',
                                'AbdominalCircumference']].to_dict()

        # Predict OBP
        obp_predictions.append(predict_obp(input_values_obp))

        # Predict ABP
        abp_predictions.append(predict_abp(input_values_abp))

        # Print results

        print('Row %i\n' % index)
        print('Input values for OBP model:')
        print(input_values_obp)
        print('Predicted OBP: %.2f\n' % obp_predictions[-1])
        print('Input values for ABP model:')
        print(input_values_abp)
        print('Predicted ABP: %.2f\n' % abp_predictions[-1])


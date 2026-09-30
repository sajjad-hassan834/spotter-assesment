# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

## Solution Run Instructions

To reproduce the model predictions and generate the report, follow these steps:

1. **Install dependencies:**
```bash
python -m pip install -r requirements.txt
```

2. **Run the Model Pipeline:**
This script loads data, engineers features, trains the LightGBM model, and predicts on the validation and December datasets.
```bash
python model_pipeline.py
```

3. **Validate and Generate the December Chart:**
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

4. **Generate the PDF Report:**
```bash
python generate_report.py
```
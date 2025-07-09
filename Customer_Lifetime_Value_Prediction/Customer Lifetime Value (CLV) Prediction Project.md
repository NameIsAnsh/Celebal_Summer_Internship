# Customer Lifetime Value (CLV) Prediction Project

## Overview
This project focuses on predicting Customer Lifetime Value (CLV) using historical transaction data. CLV is a crucial metric for businesses to understand the long-term value of their customers, enabling more effective marketing strategies, resource allocation, and customer relationship management.

This project was developed as a major project for training completion, demonstrating skills in data analysis, machine learning, and practical application of CLV models.

## Dataset
The dataset used in this project is `online_retail_II.xlsx`, which contains transactional data from an online retail store.

## Project Structure
- `online_retail_II.xlsx`: The raw dataset.
- `data_exploration.py`: Script for initial data understanding and exploratory data analysis.
- `data_preprocessing.py`: Script for cleaning the data and engineering RFM (Recency, Frequency, Monetary) features.
- `clv_model.py`: Script for building and training the CLV models (BG/NBD and Gamma-Gamma).
- `model_evaluation.py`: Script for evaluating the models and visualizing the results.
- `rfm_data.csv`: Intermediate file containing the processed RFM data.
- `clv_predictions.csv`: Output file containing CLV predictions for each customer.
- `frequency_recency_matrix.png`: Visualization of expected future purchases.
- `probability_alive_matrix.png`: Visualization of customer probability of being alive.
- `clv_distribution.png`: Visualization of the distribution of predicted CLV.
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: Git ignore file.

## Installation
To run this project, you need Python 3.x and the following libraries. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage
1. Place the `online_retail_II.xlsx` file in the project root directory.
2. Run the scripts in the following order:
   ```bash
   python data_exploration.py
   python data_preprocessing.py
   python clv_model.py
   python model_evaluation.py
   ```

## Models Used
- **BG/NBD (Beta-Geometric/Negative Binomial Distribution) Model**: Used to model customer purchasing behavior (Recency and Frequency) and predict future transactions.
- **Gamma-Gamma Model**: Used to model the monetary value of customer transactions, assuming that monetary value and transaction frequency are independent.

## Results and Visualizations
The `model_evaluation.py` script generates several plots to help understand the model's performance and customer segments:
- **Frequency-Recency Matrix**: Shows the expected number of future purchases based on customer recency and frequency.
- **Probability of Being Alive Matrix**: Illustrates the probability of a customer being 'alive' (still active) based on their purchasing behavior.
- **CLV Distribution**: A histogram showing the distribution of the predicted Customer Lifetime Values across the customer base.

## Future Work
- Incorporate more advanced features like customer demographics or product categories.
- Implement more sophisticated CLV models (e.g., deep learning-based models).
- Develop a web application for interactive CLV prediction and visualization.
- Conduct A/B testing based on CLV segments.

## Acknowledgements
This project utilizes the `lifetimes` library for CLV modeling, which provides robust implementations of probabilistic models.

## License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute.



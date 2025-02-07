# Fraud Detection for E-commerce and Bank Transactions

## Overview
This project focuses on improving the detection of fraud cases for e-commerce transactions and bank credit transactions. Developed at **Adey Innovations Inc.**, a leader in the financial technology sector, the project aims to create accurate and robust fraud detection models tailored to the unique challenges posed by different types of transaction data. Key enhancements include geolocation analysis and transaction pattern recognition to bolster fraud detection capabilities.

## Business Need
Fraud detection is critical for enhancing transaction security. By leveraging advanced machine learning models and comprehensive data analysis, Adey Innovations Inc. seeks to:

- Identify fraudulent activities with higher accuracy
- Prevent financial losses
- Foster trust with customers and financial institutions
- Enable real-time monitoring and reporting
- Improve response times and reduce risks

## Project Scope
This project encompasses the following key tasks:

1. **Data Analysis and Preprocessing**
2. **Feature Engineering** to uncover fraud patterns
3. **Model Development and Training** using machine learning algorithms
4. **Model Evaluation** to ensure performance and reliability
5. **Deployment** for real-time fraud detection
6. **Monitoring and Continuous Improvement** for sustained effectiveness

## Datasets
### 1. **Fraud_Data.csv** (E-commerce Transactions)
- **user_id**: Unique identifier for the user
- **signup_time**: Timestamp of user registration
- **purchase_time**: Timestamp of the transaction
- **purchase_value**: Transaction amount in dollars
- **device_id**: Device identifier
- **source**: Traffic source (e.g., SEO, Ads)
- **browser**: Browser used (e.g., Chrome, Safari)
- **sex**: Gender of the user (M/F)
- **age**: User’s age
- **ip_address**: IP address of the transaction
- **class**: Target variable (1 = Fraudulent, 0 = Non-fraudulent)

### 2. **IpAddress_to_Country.csv** (IP Mapping)
- **lower_bound_ip_address**: Lower IP range
- **upper_bound_ip_address**: Upper IP range
- **country**: Associated country

### 3. **creditcard.csv** (Bank Transactions)
- **Time**: Seconds since the first transaction
- **V1 to V28**: Anonymized features (PCA-transformed)
- **Amount**: Transaction amount in dollars
- **Class**: Target variable (1 = Fraudulent, 0 = Non-fraudulent)

## Learning Outcomes
Upon completion, you will gain proficiency in:

- **Machine Learning Model Deployment** using Flask
- **Containerization** with Docker
- **REST API Development** for ML models
- **API Testing and Validation**
- **End-to-End Deployment Pipelines**
- **Scalable and Portable ML Solutions**
- **Dashboard Development** using Dash

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/fraud-detection.git
   cd fraud-detection
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## API Endpoints
- `POST /predict` - Predict fraud based on transaction data
- `GET /health` - Check API health status

## Deployment
- **Local Deployment**: Run with Flask server
- **Docker Deployment**:
  ```bash
  docker build -t fraud-detection .
  docker run -p 5000:5000 fraud-detection
  ```

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request





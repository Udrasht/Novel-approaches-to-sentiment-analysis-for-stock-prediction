## How to run code

First go to src folder
## 1 First run the below 2 files:

    python webscraping_financials.py

This will generate financial information for each ticker and store them in csv file

    python extract_news2.py

This will generate news title and summary for each ticker and store them in csv file

## 2 Run the below file:

    python extract_ohlc_data.py

This file will import indicators.py, this file will load ohlcv, news and financials data
and combine them to form final data which is stored in csv file

## 3 Then run the below file:

    python save_processed_data.py

This file will process the combined data, like applying google sentence encoder, normalization, pca, sentimental analysis using textblob for each ticker and store them in csv file

## 4 Then run the below file to generate excel file :

    python build_metrics_sheet.py

## 5 For Running ML Algorithms, run the below file:

    python apply_model.py

Specify the ML algorithms to run here, this file will import all the ML algorithms python file
and run the ML algorithm on processed data and will save the result in the excel file that was generated
by build_metrics_sheet.py

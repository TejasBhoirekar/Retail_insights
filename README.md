# Retail Business Insights

### Overview
**Retail Insights** is a data-driven project that analyzes transactional data to identify customer segments using **RFM Analysis** and **KMeans Clustering**.  
It includes an interactive **Streamlit Dashboard** that enables live customer segmentation, visualization, and business insights.

-------------------------------------------------------------------------------------------------------------------------------------------------

## Project Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/TejasBhoirekar/Retail_insights.git
cd Retail_insights
```
### Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the Streamlit App
```bash
streamlit run app.py
```
### Upload Your Dataset
Upload the file retail_dataset.csv when prompted in the dashboard interface.

------------------------------------------------------------------------------------------------------------------------------------------------

## Assumptions Made
- Each transaction in the dataset represents one invoice line item.  
- Cancelled invoices are identified by InvoiceNo starting with 'C' and are excluded.  
- CustomerID uniquely identifies a customer.  
- Revenue is computed as Quantity × UnitPrice.  
- The project assumes the dataset covers at least a few months of data to enable meaningful segmentation.  

------------------------------------------------------------------------------------------------------------------------------------------------

## Technologies Used
- Programming Language: Python  
- Data Processing: Pandas, NumPy  
- Data Visualization: Matplotlib  
- Machine Learning: Scikit-learn (KMeans)  
- Dashboard: Streamlit  
- Development Environment: Jupyter Notebook, VS Code  
- Version Control: Git, GitHub  

------------------------------------------------------------------------------------------------------------------------------------------------

## Approach Summary

### Data Cleaning
- Removed duplicates, missing CustomerIDs, and cancelled transactions.  
- Ensured positive quantities and valid unit prices.  
- Converted InvoiceDate to datetime (YYYY-MM-DD HH:MM:SS) format.  
- Created a new Revenue column = Quantity × UnitPrice.  

------------------------------------------------------------------------------------------------------------------------------------------------

### Exploratory Data Analysis (EDA)
- Identified top-selling products, countries with highest sales, and monthly sales trends.  
- Calculated key business metrics:  
  - Average Order Value (AOV)  
  - Purchase Frequency  
  - Revenue per Customer  

------------------------------------------------------------------------------------------------------------------------------------------------

### RFM Analysis
- Computed Recency, Frequency, and Monetary values for each customer.  
- Applied quantile-based scoring (1–5) for each metric.  
- Combined scores into RFM_Sum and categorized customers into:  
  - Loyal  
  - Potential  
  - At Risk  
  - Need Attention  

------------------------------------------------------------------------------------------------------------------------------------------------

### KMeans Clustering
- Standardized RFM data using StandardScaler.  
- Used the Elbow Method to determine the optimal number of clusters.  
- Performed clustering and visualized results:  
  - Recency vs Monetary  
  - Frequency vs Monetary  

------------------------------------------------------------------------------------------------------------------------------------------------

### Business Insights
- Loyal customers bring the most consistent revenue.  
- Potential customers show high growth opportunities.  
- At-risk customers can be reactivated with targeted campaigns.  
- Recommended marketing strategies for retention and upselling.

-----------------------------------------------------------------------------------------------------------------------------------------------  
  ## Screenshots

  <img width="2929" height="994" alt="image" src="https://github.com/user-attachments/assets/4ad0fd70-3022-4cea-89d4-aec211bdaa9b" />

  <img width="2905" height="1283" alt="image" src="https://github.com/user-attachments/assets/44b6aaff-88d3-46f6-9967-4be86f668397" />

  <img width="2893" height="1147" alt="image" src="https://github.com/user-attachments/assets/6b3dea9e-9703-4982-b21a-4a74ac839ae8" />

  <img width="2908" height="779" alt="image" src="https://github.com/user-attachments/assets/27af931f-5514-4d91-b54c-5bae3de1e5a6" />

  <img width="2898" height="869" alt="image" src="https://github.com/user-attachments/assets/cef9c331-06ff-4eac-afc6-56311a37f627" />








# Submissions for coding test -- Using stock fundamental data from SEC filings to understand fund holding changes

## Data

fund:

fund: we choosed 2 long-only hedge fund: Lone Pine Capital, Greenwich, Conn.(LPC) and Baupost Group, Boston (BG) as our focus of study
hoilding: we crawed the quarterly holdings on SEC 13F-hr filings from 2013 - 2019

investee:

fundamentals: we got information about the listed companies from SEC 10Q filings, including: 

              revenues_x: revenues
              
              op_income_x:operating income
              
              net_income_x:	net_income
              
              eps_basic_x: EPS basic
              
              eps_diluted_x:EPS diluted
              
              dividend_x: dividend
              
              assets_x: assets
              
              cur_assets_x: current assets
              
              cur_liab_x: current liabilities
              
              cash_x: cash
              
              equity_x: equity
              
              cash_flow_op_x: cash flow from operation
              
              cash_flow_inv_x: cash flow investment
              
              cash_flow_fin_x: cash flow financing

## Description

* **Holding**: Please check the `Holding` folders for the py file that crawls all the holdings information.
* **Fundamental**: Please check the `Fundamental` folders for that crawls all the fundamental information.
* **Analysis**: Please check the `Analysis` folder for the analysis.
* **Data**: Relevant data is stored in `Data` folder.

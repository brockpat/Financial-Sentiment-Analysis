# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:35:55 2026

@author: patri
"""

import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


def prepare_handcrafted_test_data(data_path):
    
    # Create test statements
    statements = [
        #____________________Positives________________
        "The central bank decided to pause rate hikes as inflation cools faster than expected.",
        "Quarterly earnings surpassed analyst estimates, driven by robust consumer demand.",
        "Unemployment reached a record low this morning, signaling a resilient labor market.",
        "The index surged to an all-time high following the successful merger announcement.",
        "Investors are optimistic as the new fiscal policy aims to reduce corporate tax burdens.",
        "Sovereign credit ratings were upgraded due to improved debt-to-GDP ratios.", 
        "The startup closed a Series B funding round, securing $50M for global expansion.",
        "Manufacturing PMI rose above 50, indicating expansion in the industrial sector.",
        "Consumer confidence has rebounded sharply after months of stagnation.", 
        "The yield curve is normalizing, easing fears of an imminent recession.", 
        "Unemployment declined to 10%", 
        "The merger arbitrage spread narrowed to 1.5% following the DOJ’s decision not to issue a second request.",
        "The Treasury auction saw a record 2.8x bid-to-cover ratio with heavy participation from indirect bidders.",
        "Forward P/E ratios for the S&P 500 have re-rated lower while consensus EPS estimates for 2027 remain unchanged.",
        "The ECB signals a 'pivot toward data-dependency' as upside risks to the inflation outlook begin to diminish.",
        "Weekly distillate inventories saw a larger-than-expected draw despite refinery utilization hitting 96%.",
        "The company reported steady margin improvement despite a challenging macro environment.",
        "Equity markets edged higher as investors digested mixed but generally stable economic data.",
        "Retail sales showed modest growth, slightly exceeding expectations.",
        "The firm reaffirmed its full-year guidance, signaling confidence in its outlook.",
        "Capital inflows into emerging markets picked up gradually over the past quarter.",
        "The latest inflation print came in marginally below forecasts, offering some relief to policymakers.",
        "Operating cash flow improved slightly compared to the previous quarter.",
        "Analysts noted a cautiously optimistic tone during the earnings call.",
        
        # Long paragraphs
        "The currency stabilized after recent volatility, supported by central bank comments.",
        "While growth remains uneven, leading indicators suggest a slow but positive trend.", 
        "The company delivered a standout quarter, with revenue and earnings both comfortably exceeding Wall Street expectations. Executives pointed to broad-based strength across its core divisions, particularly in higher-margin segments, while cost discipline further supported profitability. Management also raised its full-year guidance, citing sustained demand and a healthy order backlog, reinforcing investor confidence in the firm’s growth trajectory.",
        "Shares of the firm rose modestly after it reported results that, while not spectacular, reflected steady operational execution in a challenging environment. Revenue growth remained intact, supported by incremental pricing gains and stable customer retention. Analysts highlighted the company’s ability to protect margins despite ongoing cost pressures, suggesting a resilient underlying business model.",
        "The company’s latest earnings release underscored improving fundamentals, as sequential growth accelerated and key performance indicators trended in the right direction. Management emphasized early signs of traction in newer markets, which could provide an additional tailwind over the coming quarters. While uncertainties remain, the tone of the report was broadly constructive.",
        
        # Tricky to classify
        "In a sign of growing financial flexibility, the company announced a share buyback program alongside its quarterly results. Free cash flow generation improved meaningfully, enabling the firm to return capital to shareholders without compromising its investment plans. Market participants viewed the move as a signal of confidence from management in the company’s medium-term outlook.",
        "The company reported a 7% decline in quarterly revenue, reflecting ongoing weakness in its legacy segments; however, results still came in ahead of consensus estimates as cost controls proved more effective than expected. Management highlighted accelerating demand in its newer business lines and reiterated its full-year guidance, suggesting that the recent softness may be temporary rather than structural.",
        "While margins compressed modestly due to higher input costs, the firm delivered earnings that exceeded Wall Street forecasts, supported by resilient pricing and stable volumes. Executives acknowledged near-term uncertainties but pointed to a strengthening order pipeline, leading several analysts to describe the update as incrementally encouraging despite lingering risks.",
        "The latest report showed uneven performance across regions, with notable declines in Europe offset by stronger-than-anticipated growth in North America. Although management adopted a cautious tone regarding the macroeconomic backdrop, it emphasized continued progress on efficiency initiatives and maintained its capital return plans, reinforcing a broadly constructive outlook.",
        "Free cash flow came in slightly below expectations, driven in part by temporary working capital pressures; nonetheless, the company reduced its debt levels and announced a modest share repurchase program. Despite some mixed underlying metrics, investors appeared reassured by management’s confidence in medium-term growth prospects and its disciplined approach to capital allocation.",
                
        #____________________Negatives________________
        "The tech giant announced a massive layoff affecting 15% of its global workforce.",
        "Inflation surged to a 40-year high, eroding the purchasing power of households.", 
        "The bankruptcy filing sent shockwaves through the regional banking sector.", 
        "GDP growth slowed to a crawl in Q3, missing even the most pessimistic forecasts.", 
        "Rising interest rates are putting significant downward pressure on the housing market.", 
        "The trade deficit widened significantly this month, weighing on the national currency.", 
        "Several supply chain bottlenecks are expected to stifle production through year-end.", 
        "The company’s stock plummeted after the CEO resigned amidst a fraud investigation.", 
        "Default rates on high-yield bonds are beginning to tick upward.", 
        "Market volatility spiked as geopolitical tensions escalated in the region.", 
        "Unemployment rose to 10%.",
        "The company’s 'Days Sales Outstanding' (DSO) increased by 12 days despite achieving record top-line revenue.",
        "The yield curve inversion deepened today as the 2-year/10-year spread reached its widest level in decades.",
        "The ISM New Orders sub-index slipped into contraction territory while the Prices Paid component remains elevated.",
        "Credit Default Swaps (CDS) for the European banking sector widened by 50 basis points in thin holiday trading.",
        "The Chief Accounting Officer’s sudden resignation for 'personal reasons' comes weeks before the 10-K filing.", 
        "The company reported a slight decline in revenue compared to the previous quarter.",
        "Economic growth came in just below expectations, raising mild concerns among analysts.",
        "Investor sentiment weakened due to somewhat uncertain policy signals.",
        "Margins contracted modestly due to rising input costs.",
        "The outlook was revised slightly downward, reflecting softer demand conditions.",
        "Trading volumes were thinner than usual, indicating reduced market participation.",
        "Housing starts dipped marginally, hinting at cooling momentum in the sector.",
        "The firm experienced minor delays in its product rollout schedule.",
        
        # Long Paragraphs
        "Inflation remains sticky, complicating the central bank’s policy path.",
        "Export data showed a small but notable decline amid weakening global demand.",
        "The company reported disappointing quarterly results, with both revenue and profit falling short of analyst expectations. Management cited weakening demand and persistent cost pressures, particularly in its core markets. The firm also lowered its full-year outlook, raising concerns about its ability to navigate an increasingly challenging economic environment.",
        "Shares slipped after the company posted results that pointed to slowing momentum. While revenues were largely in line with forecasts, margins came under pressure due to higher input costs and operational inefficiencies. Analysts noted that the lack of clear catalysts could weigh on the stock in the near term.",
        "The latest earnings report revealed emerging cracks in the company’s growth story, as key segments underperformed and forward-looking indicators softened. Management acknowledged increased uncertainty and refrained from providing detailed guidance, a move that some investors interpreted as a sign of limited visibility into future performance.",
        
        # Tricky to classify
        "The firm’s financial position appears to be deteriorating gradually, with declining cash flow and rising leverage levels drawing scrutiny from market participants. Although no immediate liquidity concerns were flagged, analysts warned that continued weakness could constrain strategic flexibility going forward.",
        "The company posted a 12% increase in revenue, surpassing analyst expectations; however, the gain was largely driven by one-off factors, while core demand showed signs of deterioration. Margins declined ավելի sharply than anticipated, and management lowered its full-year outlook, citing reduced visibility into future orders.",
        "Earnings came in slightly ahead of forecasts, but the quality of the beat was called into question as it relied heavily on cost-cutting rather than sustainable growth. Executives struck a cautious tone on the outlook, pointing to softening demand trends and signaling that further margin pressure could lie ahead.",
        "Although the firm maintained its guidance and highlighted stable near-term performance, several key indicators suggested underlying weakness, including declining order volumes and rising inventory levels. Analysts noted that the absence of downward revisions may reflect limited visibility rather than confidence, leaving the broader picture tilted to the downside.",
        "The company announced a new strategic initiative aimed at revitalizing growth, which was initially welcomed by investors; nevertheless, its latest results revealed slowing momentum across core segments and a continued deterioration in cash flow. Management acknowledged the challenges but offered few concrete details on how or when conditions might improve, adding to market uncertainty.",
        ]
    

    sentiment = [1]*34 + [0]*34
    
    # 1. Check if we already embedded the test data
    if os.path.exists(data_path):
        print(f"Hand-crafted test dataset already exists. Loading from {data_path}...")
        df_test = pd.read_pickle(data_path)
        return df_test
        
    # 2. If not, fetch embeddings from OpenAI
    print("Test dataset not found. Embedding hand-crafted test data via OpenAI API...")
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=statements,
        dimensions=256
    )
    
    test_embeddings = [data.embedding for data in response.data]
    
    # 3. Save into a DataFrame to prevent future API costs
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    df_test = pd.DataFrame({
        "statement": statements, 
        "sentiment": sentiment, 
        "embedding": test_embeddings
    })
    
    df_test.to_pickle(data_path)
    print(f"Hand-crafted test data saved to {data_path}")
    
    return df_test

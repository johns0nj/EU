import pandas as pd

# 创建数据列表
data = {
    'Name': [
        'ASML Holding', 'LVMH Moet Hennessy Louis V', 'SAP', 'TotalEnergies', 'Siemens', 'L\'Oreal',
        'Prosus', 'Schneider Electric', 'Air Liquide', 'Sanofi', 'Allianz', 'Airbus',
        'Deutsche Telekom', 'Hermes International', 'Anheuser-Busch InBev', 'Iberdrola', 'VINCI',
        'BNP Paribas', 'Volkswagen', 'Banco Santander', 'AXA', 'EssilorLuxottica', 'Safran',
        'Muenchener Rueckversicher', 'Stellantis', 'Enel', 'BBVA', 'ING Groep',
        'Mercedes-Benz Group', 'Infineon Technologies', 'Intesa Sanpaolo', 'Inditex',
        'Deutsche Post', 'UniCredit', 'Ferrari', 'BASF', 'Nordea Bank Abp', 'Eni',
        'Danone', 'BMW', 'Adyen', 'Deutsche Boerse', 'Pernod Ricard', 'adidas',
        'Bayer', 'Kering', 'Compagnie de Saint-Gobain', 'Koninklijke Ahold Delhaize',
        'Flutter Entertainment', 'Nokia', 'SX5E'
    ],
    'Country': [
        'Netherlands', 'France', 'Germany', 'France', 'Germany', 'France',
        'Netherlands', 'France', 'France', 'France', 'Germany', 'France',
        'Germany', 'France', 'Belgium', 'Spain', 'France', 'France', 'Germany',
        'Spain', 'France', 'France', 'France', 'Germany', 'Italy', 'Italy',
        'Spain', 'Netherlands', 'Germany', 'Germany', 'Italy', 'Spain',
        'Germany', 'Italy', 'Italy', 'Germany', 'Finland', 'Italy',
        'France', 'Germany', 'Netherlands', 'Germany', 'France', 'Germany',
        'Germany', 'France', 'France', 'Netherlands', 'Ireland', 'Finland', ''
    ],
    'Industry': [
        'Technology', 'Consumer Discretionary', 'Technology', 'Energy', 'Industrials',
        'Consumer Discretionary', 'Technology', 'Industrials', 'Basic Materials',
        'Health Care', 'Financials', 'Industrials', 'Telecommunications',
        'Consumer Discretionary', 'Consumer Staples', 'Utilities', 'Industrials',
        'Financials', 'Consumer Discretionary', 'Financials', 'Financials',
        'Health Care', 'Industrials', 'Financials', 'Consumer Discretionary',
        'Utilities', 'Financials', 'Financials', 'Consumer Discretionary',
        'Technology', 'Financials', 'Consumer Discretionary', 'Industrials',
        'Financials', 'Consumer Discretionary', 'Basic Materials', 'Financials',
        'Energy', 'Consumer Staples', 'Consumer Discretionary', 'Industrials',
        'Financials', 'Consumer Staples', 'Consumer Discretionary', 'Health Care',
        'Consumer Discretionary', 'Industrials', 'Consumer Staples',
        'Consumer Discretionary', 'Telecommunications', ''
    ],
    'Weight': [
        7.1, 6.1, 5.1, 4.0, 3.6, 3.4, 3.1, 3.0, 3.0, 3.0, 2.8, 2.8, 2.5, 2.2,
        2.0, 2.0, 2.0, 1.9, 1.9, 1.9, 1.8, 1.8, 1.6, 1.6, 1.5, 1.5, 1.5, 1.5,
        1.5, 1.4, 1.4, 1.4, 1.3, 1.3, 1.3, 1.2, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0,
        1.0, 1.0, 0.9, 0.9, 0.8, 0.8, 0.6, 0.5, 100.0
    ],
    'Europe': [
        '2%', '21%', '30%', '64%', '33%', '25%', '8%', '24%', '36%', '20%',
        '57%', '32%', '31%', '17%', '13%', '62%', '73%', '73%', '45%', '38%',
        '64%', '25%', '37%', '45%', '32%', '84%', '28%', '72%', '33%', '18%',
        '93%', '56%', '46%', '76%', '38%', '33%', '95%', '72%', '22%', '31%',
        '19%', '71%', '24%', '22%', '15%', '27%', '59%', '30%', '39%', '28%', '37%'
    ],
    'North America': [
        '9%', '28%', '37%', '11%', '25%', '27%', '2%', '31%', '32%', '48%',
        '9%', '23%', '66%', '12%', '29%', '15%', '8%', '11%', '21%', '15%',
        '15%', '47%', '16%', '32%', '49%', '1%', '1%', '5%', '27%', '12%',
        '1%', '15%', '7%', '0%', '25%', '27%', '2%', '4%', '24%', '21%',
        '48%', '7%', '20%', '28%', '35%', '27%', '18%', '63%', '34%', '35%', '22%'
    ],
    'APAC': [
        '88%', '39%', '15%', '5%', '25%', '35%', '74%', '30%', '19%', '24%',
        '9%', '26%', '0%', '57%', '11%', '0%', '4%', '7%', '18%', '0%',
        '15%', '12%', '13%', '12%', '5%', '0%', '0%', '8%', '32%', '63%',
        '2%', '14%', '19%', '0%', '25%', '25%', '2%', '7%', '18%', '39%',
        '9%', '6%', '41%', '24%', '19%', '38%', '8%', '0%', '22%', '23%', '25%'
    ],
    'EM': [
        '1%', '11%', '18%', '19%', '17%', '14%', '16%', '15%', '12%', '8%',
        '25%', '19%', '2%', '13%', '47%', '23%', '14%', '9%', '15%', '47%',
        '6%', '17%', '33%', '11%', '14%', '15%', '71%', '15%', '9%', '7%',
        '4%', '14%', '27%', '24%', '13%', '15%', '1%', '18%', '35%', '9%',
        '24%', '16%', '15%', '25%', '32%', '7%', '15%', '7%', '5%', '13%', '16%'
    ],
    'Others': [
        '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%',
        '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%',
        '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%',
        '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%',
        '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%', '0%'
    ]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 保存到Excel
df.to_excel('jpg2excel.xlsx', index=False)

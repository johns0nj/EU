import folium
import pandas as pd
import branca.colormap as cm

# 创建基础地图
m = folium.Map(
    location=[20, 0],
    zoom_start=2,
    tiles='cartodbpositron'
)

# 定义区域和数据（更新标签位置到更合适的位置）
regions_data = {
    'Europe': {
        'percentage': 37,
        'color': '#0487D9',
        'countries': ['France', 'Germany', 'Netherlands', 'Spain', 'Italy', 'Belgium', 
                     'Finland', 'Ireland', 'Austria', 'Portugal', 'Luxembourg', 'Greece'],
        'label_position': [50, 15],  # 欧洲中心
        'label': '欧洲'
    },
    'North America': {
        'percentage': 22,
        'color': '#E31837',
        'countries': ['United States of America', 'Canada'],
        'label_position': [45, -95],  # 北美中心
        'label': '北美'
    },
    'Asia Pacific': {
        'percentage': 25,
        'color': '#739CBF',
        'countries': ['China', 'Japan', 'South Korea', 'Australia', 'New Zealand', 
                     'India', 'Indonesia', 'Malaysia', 'Philippines', 'Singapore', 
                     'Thailand', 'Vietnam'],
        'label_position': [35, 120],  # 亚太中心
        'label': '亚太'
    },
    'Emerging Markets': {
        'percentage': 16,
        'color': '#730237',
        'countries': ['Brazil', 'Russia', 'Mexico', 'South Africa', 'Turkey', 
                     'Saudi Arabia', 'United Arab Emirates', 'Egypt', 'Pakistan'],
        'label_position': [20, 0],  # 新兴市场中心位置调整
        'label': '新兴市场'
    }
}

# 添加GeoJson图层
folium.GeoJson(
    'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json',
    name='geojson',
    style_function=lambda x: {
        'fillColor': next((region_data['color'] 
                          for region_name, region_data in regions_data.items() 
                          if x['properties']['name'] in region_data['countries']), 
                         '#EDEDED'),
        'color': 'white',
        'weight': 1,
        'fillOpacity': 0.7
    }
).add_to(m)

# 添加区域标签和百分比
for region, data in regions_data.items():
    # 添加标签和百分比
    folium.map.Marker(
        data['label_position'],
        icon=folium.DivIcon(
            html=f'''
            <div style="
                background-color: rgba(255, 255, 255, 0.9);
                padding: 8px;
                border-radius: 5px;
                font-family: Arial;
                font-size: 16px;
                font-weight: bold;
                color: {data['color']};
                text-align: center;
                box-shadow: 0 0 5px rgba(0,0,0,0.2);">
                {data['label']}<br>{data['percentage']}%
            </div>
            ''',
            icon_size=(120, 60),
            icon_anchor=(60, 30)
        )
    ).add_to(m)

# 添加标题
title_html = '''
<div style="position: fixed; 
            top: 20px; left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 20px;
            font-weight: bold;">
    STOXX 50 地理收入分布
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# 添加图例
legend_html = '''
<div style="position: fixed; 
            bottom: 50px; right: 50px;
            z-index: 1000;
            background-color: white;
            padding: 10px;
            border-radius: 5px;">
'''

for region, data in regions_data.items():
    legend_html += f'''
    <div style="margin-bottom: 5px;">
        <span style="display: inline-block;
                     width: 20px;
                     height: 20px;
                     background-color: {data['color']};
                     margin-right: 5px;"></span>
        {region} ({data['percentage']}%)
    </div>
    '''

legend_html += '</div>'
m.get_root().html.add_child(folium.Element(legend_html))

# 添加数据来源
source_html = '''
<div style="position: fixed; 
            bottom: 20px; left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            background-color: white;
            padding: 5px;
            border-radius: 5px;
            font-size: 12px;">
    Source: FactSet, STOXX, Goldman Sachs Global Investment Research
</div>
'''
m.get_root().html.add_child(folium.Element(source_html))

# 保存地图
m.save('STOXX50_revenue_distribution.html')

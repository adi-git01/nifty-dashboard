import os

files = ['main.py', r'utils/market_mood.py', r'utils/visuals.py', r'utils/market_breadth.py']
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace both single and double quote versions
        content = content.replace("template='plotly_dark'", "template='plotly_white'")
        content = content.replace('template="plotly_dark"', 'template="plotly_white"')
        
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Updated {f}")
    except Exception as e:
        print(f"Error on {f}: {e}")

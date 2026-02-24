import os
import re

def strip_emojis_from_prints(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                changed = False
                for i, line in enumerate(lines):
                    # Match print statements
                    if re.search(r'\bprint\s*\(', line):
                        # Find non-ASCII characters in the string and remove them, but only inside the print
                        # A simple heuristic: if it's a print, remove common emojis.
                        # It's safer to just remove all non-ASCII outside of comments?
                        # Let's just remove the specific ones we know: ⚡, 🟢, 🔴, 🟡, ⚠️, 🚀, 📈, 🔥, 📊, 🎯, 🤔, 💡
                        original = line
                        for emoji in ['⚡', '🟢', '🔴', '🟡', '⚠️', '🚀', '📈', '🔥', '📊', '🎯', '🤔', '💡', '✅', '⏳', '✨']:
                            if emoji in line:
                                line = line.replace(emoji + ' ', '')
                                line = line.replace(' ' + emoji, '')
                                line = line.replace(emoji, '')
                        
                        if line != original:
                            lines[i] = line
                            changed = True

                if changed:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    print(f"Cleaned prints in {path}")

strip_emojis_from_prints('.')

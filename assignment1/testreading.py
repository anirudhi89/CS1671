def test():
    with open('data/wine.txt', encoding='utf-8', errors='ignore') as wine_file:
       for line in wine_file:
          print(line)

test()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setari vizuale 
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8) 

def genereaza_grafic_top(nume_fisier, nume_tara, culoare_bara):
    print(f"Procesez {nume_tara}...")
    
    if not os.path.exists(nume_fisier):
        print(f"Lipsa fisier {nume_fisier}")
        return

    # Citire
    try: 
        df = pd.read_csv(nume_fisier, encoding='latin-1', low_memory=False)
    except: 
        df = pd.read_csv(nume_fisier, encoding='cp1252', low_memory=False)

    
    col = 'INGREDIENT_ENG' if 'INGREDIENT_ENG' in df.columns else df.columns[-1]
    
    # Curatare date
    df = df.dropna(subset=[col])
    df[col] = df[col].astype(str)

    # 1. Calcul Frecventa (Top 50)
    top_ingrediente = df[col].value_counts().head(50)
    
    # 2. Desenare Grafic (Horizontal Bar Plot)
    plt.figure()
    # Folosim barplot din Seaborn
    ax = sns.barplot(x=top_ingrediente.values, y=top_ingrediente.index, color=culoare_bara)
    
    # Titluri si etichete
    plt.title(f'Top 50 Ingrediente Consumate - {nume_tara}', fontsize=16, fontweight='bold')
    plt.xlabel('Numar de Aparitii (Frecventa)', fontsize=12)
    plt.ylabel('Ingredient', fontsize=12)
    
    
    for i in ax.containers:
        ax.bar_label(i, padding=3)

  
    plt.tight_layout()
    
    # Salvare
    nume_poza = f"grafic_top50_{nume_tara.lower()}.png"
    plt.savefig(nume_poza, dpi=300) 
    print(f"Salvat: {nume_poza}")
    plt.close()

# --- RULARE PENTRU CELE 3 TARI ---
if __name__ == "__main__":
    # India - portocaliu
    genereaza_grafic_top('india.csv', 'INDIA', '#FF9933')
    
    # România - albastru
    genereaza_grafic_top('romania.csv', 'ROMÂNIA', '#0044cc')
    
    # Bangladesh - verde
    genereaza_grafic_top('bangladesh.csv', 'BANGLADESH', '#006a4e')
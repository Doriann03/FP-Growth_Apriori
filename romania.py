# =========================================================
# ROMANIA - FINAL REPORT VERSION (RAM + CLEAN CSV)
# =========================================================
import pandas as pd
import time
import os
import gc
import tracemalloc
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules
from tabulate import tabulate

def clean_itemset(itemset):
    return " + ".join(list(itemset))

def run_and_save(algo_func, matrix, support, name, top_n=20):
    print(f"\nRulăm {name}...")
    
    gc.collect()
    tracemalloc.start()
    start_time = time.time()
    
    try:
        frequent = algo_func(matrix, min_support=support, use_colnames=True)
        count = len(frequent)
        success = True
    except MemoryError:
        print(f"{name} CRASH (Out of Memory)!")
        return False, 0, 0, 0

    duration = time.time() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    
    print(f"Gata în {duration:.2f}s | RAM: {peak_mb:.2f} MB | Reguli: {count}")

    if success and count > 0:
        try:
            # Generare Reguli
            rules = association_rules(frequent, metric="lift", min_threshold=1.1)
            
            clean_rows = []
            top_rules = rules.sort_values('lift', ascending=False).head(top_n)
            
            for _, row in top_rules.iterrows():
                clean_rows.append({
                    'Antecedents': clean_itemset(row['antecedents']),
                    'Consequents': clean_itemset(row['consequents']),
                    'Support': round(row['support'], 4),
                    'Confidence': round(row['confidence'], 4),
                    'Lift': round(row['lift'], 4)
                })
            
            df_export = pd.DataFrame(clean_rows)
            filename = f"rezultate_{name.lower()}_romania.csv"
            df_export.to_csv(filename, index=False)
            print(f"CSV Salvat: {filename}")
            
        except Exception as e:
            print(f"Nu s-au putut genera reguli (posibil prea puține seturi): {e}")

    return success, duration, peak_mb, count

def main():
    print("\n" + "="*80)
    print("🇷🇴  ROMÂNIA - GENERARE RAPORT FINAL")
    print("="*80)

    if not os.path.exists('romania.csv'): print("Lipsă romania.csv"); return
    
    print("Citire date (poate dura puțin)...")
    try: df = pd.read_csv('romania.csv', encoding='latin-1', low_memory=False)
    except: df = pd.read_csv('romania.csv', encoding='cp1252', low_memory=False)

    col = 'INGREDIENT_ENG' if 'INGREDIENT_ENG' in df.columns else 'INGREDIENT'
    
    # Cleaning Obligatoriu
    df.dropna(subset=[col], inplace=True)
    df[col] = df[col].astype(str)
    
    if 'SURVEY_DAY' in df.columns: df['Tx_ID'] = df['SUBJECT'].astype(str) + "_" + df['SURVEY_DAY'].astype(str)
    else: df['Tx_ID'] = df.iloc[:, 0].astype(str)

    transactions = df.groupby('Tx_ID')[col].apply(list).tolist()
    del df; gc.collect()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_).astype(bool)
    nr_tranzactii = len(transactions)
    del transactions; gc.collect()

    SUPORT = 0.05 # 5%
    print(f"Matrice: {df_trans.shape} | Suport: {SUPORT}")
    
    # RULARE AMBII ALGORITMI
    ok_fp, time_fp, ram_fp, cnt_fp = run_and_save(fpgrowth, df_trans, SUPORT, "FP-GROWTH", top_n=20)
    ok_ap, time_ap, ram_ap, cnt_ap = run_and_save(apriori, df_trans, SUPORT, "APRIORI", top_n=20)

    #Tabel final
    data = [
        ["FP-GROWTH", f"{time_fp:.4f} s", f"{ram_fp:.2f} MB", cnt_fp, nr_tranzactii],
        ["APRIORI", f"{time_ap:.4f} s", f"{ram_ap:.2f} MB", cnt_ap, nr_tranzactii]
    ]
    
    print(tabulate(data, headers=["Algoritm", "Timp", "Memorie (RAM)", "Seturi Găsite", "Nr. Tranzacții"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    main()
from statistics import mean
from scipy import stats
from scipy.stats import mannwhitneyu

knr = [
    1537.1762,1342.0423,1448.0195,1362.3407,1768.2340,1578.0258,1405.3959,1405.4733,1417.5675,1715.3531,1478.9052,1590.3986,1385.8016,
    1729.2947,1760.1362,1543.6823,1288.6657,1029.8967,1360.7476,1612.3799]

svr = [
    1137.3814,1100.1746,1283.1711,1043.9849,1279.4781, 1354.1249,1232.3398,1149.7048,1291.6069,1118.5720,1400.8489,1196.0161,1267.2318,1208.4144,1618.8753,
    1306.3457,1151.1832,929.4975,1089.3659,1383.1634]

mlp = [1385.5559,1601.8208,184987.0695,1142.1014,4915.9026,6792.0999,1411.9158,2283.5912,5040.3724,1633.2850,1927.0032,1824.9822,1858.8042,
    5112.0003,1694.3856,1613.0003,1468.6533,1502.3198,1636.5994,3010.3267]

rf = [1755.0444,1508.6045,1577.1573,1427.8622,1638.5340,1629.6471,1600.7297,1436.5230,1576.4112,1790.1024,1714.0550,1494.7372,1502.2536,1730.1431,
    1863.9970,1709.6906,1515.6715,1477.3401,1538.3105,1602.5328]

gb = [1777.5446,1572.8236,1559.4315,1429.1018,1650.0170,1646.6051,1629.1059,1454.5650,1574.5087,1803.5503,1745.5500,1487.8292,1553.1754,1693.2925,
    1885.8042,1741.3980,1565.2994,1448.4419,1589.7365,1611.2725]

rlm = [1845.9604,1597.4387,1704.7997,5429.7462,1786.9150,1725.0957,1686.7024,1559.3381,1698.4810,2082.9750,2418.9109,1564.1356,1534.7291,
    1689.8290,2241.3578,1948.0437,1722.1629,1599.8000,2189.8537,1832.9782,
]


sta1, pvalue1 = stats.kruskal(knr, svr, mlp, rf, gb, rlm)

print("Estatística de Kruskal-Wallis para sistema Monolíticos: ", sta1)
print(f"\nValor de P para sistemas monolíticos: {pvalue1:.5}")

if pvalue1 <= 0.05:
    print("\nHá diferença estatisticamente significativa entre os classificadores.\n")
    print("Iniciando análise Mannwhitneyu\n")


    #KNN X DT
    msta_kd, mpvalue_kd = mannwhitneyu(knr, svr, method = "exact", alternative = "two-sided")
    print("Estatística de Mannwhitneyu KNR X SVR: ", msta_kd)
    print(f"\nValor de P: {mpvalue_kd:.5}")
    if mpvalue_kd <= 0.05:
        print("\nHá diferença estatisticamente significativa entre os classificadores KNR X SVR.\n")
    else:
        print("\nNão há diferença estatisticamente significativa entre os classificadores KNR X SVR.")
        
    # KNN x SVM
    msta_ks, mpvalue_ks = mannwhitneyu(knr, mlp, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U KNR x MLP: ", msta_ks)
    print(f"Valor de P: {mpvalue_ks:.5}")
    if mpvalue_ks <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores KNR x MLP.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores KNR x MLP.\n")

    # KNR x NB
    msta_kn, mpvalue_kn = mannwhitneyu(knr, rf, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U KNR x RF: ", msta_kn)
    print(f"Valor de P: {mpvalue_kn:.5}")
    if mpvalue_kn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores KNR x RF.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores KNR x RF.\n")

    msta_kn, mpvalue_kn = mannwhitneyu(knr, gb, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U KNR x GB: ", msta_kn)
    print(f"Valor de P: {mpvalue_kn:.5}")
    if mpvalue_kn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores KNR x GB.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores KNR x GB.\n")
        
    msta_kn, mpvalue_kn = mannwhitneyu(knr, rlm, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U KNR x RLM: ", msta_kn)
    print(f"Valor de P: {mpvalue_kn:.5}")
    if mpvalue_kn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores KNR x RLM.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores KNR x RLM.\n")
        
    # DT x MLP
    msta_dm, mpvalue_dm = mannwhitneyu(svr, mlp, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U SVR x MLP: ", msta_dm)
    print(f"Valor de P: {mpvalue_dm:.5}")
    if mpvalue_dm <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores SVR x MLP.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores SVR x MLP.\n")

    # DT x SVM
    msta_ds, mpvalue_ds = mannwhitneyu(svr, rf, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U SVR x RF: ", msta_ds)
    print(f"Valor de P: {mpvalue_ds:.5}")
    if mpvalue_ds <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores SVR x RF.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores SVR x RF.\n")

    # DT x NB
    msta_dn, mpvalue_dn = mannwhitneyu(svr, gb, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U SVR x GB: ", msta_dn)
    print(f"Valor de P: {mpvalue_dn:.5}")
    if mpvalue_dn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores SVR x GB.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores SVR x GB.\n")
    
    msta_dn, mpvalue_dn = mannwhitneyu(svr, rlm, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U SVR x RLM: ", msta_dn)
    print(f"Valor de P: {mpvalue_dn:.5}")
    if mpvalue_dn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores SVR x RLM.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores SVR x RLM.\n")

    # MLP x SVM
    msta_ms, mpvalue_ms = mannwhitneyu(mlp, rf, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U MLP x RF: ", msta_ms)
    print(f"Valor de P: {mpvalue_ms:.5}")
    if mpvalue_ms <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores MLP x RF.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores MLP x RF.\n")

    #MLP x NB
    msta_mn, mpvalue_mn = mannwhitneyu(mlp, gb, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U MLP x GB: ", msta_mn)
    print(f"Valor de P: {mpvalue_mn:.5}")
    if mpvalue_mn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores MLP x GB.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores MLP x GB.\n")
        
    msta_mn, mpvalue_mn = mannwhitneyu(mlp, rlm, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U MLP x RLM: ", msta_mn)
    print(f"Valor de P: {mpvalue_mn:.5}")
    if mpvalue_mn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores MLP x RLM.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores MLP x RLM.\n")

    # SVM x NB
    msta_sn, mpvalue_sn = mannwhitneyu(rf, gb, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U RF x GB: ", msta_sn)
    print(f"Valor de P: {mpvalue_sn:.5}")
    if mpvalue_sn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores RF x GB.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores RF x GB.\n")
    
    msta_sn, mpvalue_sn = mannwhitneyu(rf, rlm, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U RF x RLM: ", msta_sn)
    print(f"Valor de P: {mpvalue_sn:.5}")
    if mpvalue_sn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores RF x RLM.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores RF x RLM.\n")
        
    msta_sn, mpvalue_sn = mannwhitneyu(gb, rlm, method="exact", alternative="two-sided")
    print("Estatística de Mann-Whitney U GB x RLM: ", msta_sn)
    print(f"Valor de P: {mpvalue_sn:.5}")
    if mpvalue_sn <= 0.05:
        print("Há diferença estatisticamente significativa entre os classificadores GB x RLM.\n")
    else:
        print("Não há diferença estatisticamente significativa entre os classificadores GB x RLM.\n")
else:
    print("\nNão há diferença estatisticamente significativa entre os classificadores Monolíticos.\n")


# realizando os mesmos testes para os multi classificadores

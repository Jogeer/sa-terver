def intervals(fseq, N):
    import numpy as np
    h = (np.max(fseq) - np.min(fseq)) / (N - 1)
    t = np.zeros(N+1)

    for i in range(N+1):
        if i == 0:
            t[0] = np.min(fseq) - (h/2)
        else:
            t[i] = t[i-1] + h
    return np.around(t, decimals=2)

def midIntervals(fseq, intervals, N):
    import numpy as np
    h = (np.max(fseq) - np.min(fseq)) / (N - 1)
    miT = np.zeros(N)

    for i in range(N):
        if i == 0:
            miT[0] = intervals[0] + (h/2)
        else:
            miT[i] = (intervals[i] + intervals[i+1])/2
    return np.around(miT, decimals=3)

def hardTable(fseq, intervals, mintervals, N):
    import numpy as np
    import pandas as pd
    table = pd.DataFrame(mintervals)
    table = table.T

    h = (np.max(fseq) - np.min(fseq)) / (N - 1)

    j = 0
    count = np.zeros(N)
    for i in range(1, N+1):
        while fseq[j] < intervals[i]:
            count[i-1] += 1
            j += 1
            if j == 100:
                break
    countt = pd.DataFrame(count)
    table = table.append(countt.T, ignore_index=True)

    result = np.zeros(N)
    for i in range(N):
        result[i] = count[i]/100
    pp = pd.DataFrame(result)
    table = table.append(pp.T, ignore_index=True)

    f = np.zeros(N)
    for i in range(N):
        f[i] = result[i]/h
    ff = pd.DataFrame(f)
    table = table.append(ff.T, ignore_index=True)
    table.columns=[str(x) for x in mintervals]

    print(round(table, 2),"\n")
    return result

def tableHypotes(intervals, x_sr, rmse, N, hardTable):
    # Лол, но это по-другому работать не будет
    import numpy as np
    import pandas as pd
    from scipy import stats
    table = pd.DataFrame(intervals)
    table = table.T

    t = np.zeros(N+1)
    for i in range(N+1):
        t[i] = (intervals[i] - x_sr)/rmse
    tt = pd.DataFrame(t)
    table = table.append(tt.T)

    ft = stats.norm.cdf(t) - 0.5
    ftt = pd.DataFrame(ft)
    table = table.append(ftt.T)

    pp = pd.DataFrame(hardTable)
    table = table.append(pp.T)

    r = np.zeros(N)
    for i in range(N):
        r[i] = ft[i+1] + ft[i]
    rr = pd.DataFrame(r)
    table = table.append(rr.T)

    pr = np.zeros(N)
    for i in range(N):
        pr[i] = hardTable[i] - r[i]
    prr = pd.DataFrame(pr)
    table = table.append(prr.T)

    pr2 = np.zeros(N)
    for i in range(N):
        pr2[i] = (hardTable[i] - r[i])**2
    prr2 = pd.DataFrame(pr2)
    table = table.append(prr2.T)

    result = np.zeros(N+1)
    for i in range(N):
        result[i] = pr2[i] / r[i]
    solve = sum(result)
    result[N] = solve
    res = pd.DataFrame(result)
    table = table.append(res.T)
    table.index=['xi','ui','f(ui)','pi','ri','(pi-ri)','(pi-ri)^2','zi']

    print(table.T)
    if solve < 9.5:
        print("Наша гипотеза принимается\n")
    else:
        print("Наша гипотеза отвергается\n")

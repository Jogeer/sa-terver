class Lab():
    import matplotlib.pyplot as plt
    import pandas as pd

    def __init__(self, full_seq, sel_seq, intervals=5):
        self.fseq = full_seq; self.sseq = sel_seq
        self.fn = len(self.fseq); self.sn = len(self.sseq)
        self.intervals = intervals

        # Settings
        self.alpha = 0.05

        # Do
        self._do()

    def _do(self):
        self._base()
        self._moments()
        self._sustain()
        self._error()
        self._odds()
        self._hypothesis()

    # Math parts
    def _base(self):
        self.x_sr = sum(self.fseq)/self.fn
        self.s_sr = self._dispersion(self.fseq, self.x_sr, self.fn)
        self.xn_sr = sum(self.sseq)/self.sn
        self.sn_sr = self._dispersion(self.sseq, self.xn_sr, self.sn)
        self.s_ns_sr = self.fn*self.s_sr/(self.fn-1)
        self.sn_ns_sr = self.sn*self.sn_sr/(self.sn-1)

    def _moments(self):
        self.mstart = [self._dispersion(self.fseq, 0, self.fn, k) for k in range(1, 5)]
        self.mcentral = [self._dispersion(self.fseq, self.x_sr, self.fn, k) for k in range(1, 5)]

    def _sustain(self):
        self._xn_sn = [self.xn_sr - self.sn_sr, self.xn_sr + self.sn_sr]
        self._x_s = [self.x_sr - self.s_sr, self.x_sr + self.s_sr]

        ch = '' if (self._xn_sn[0] <= self.x_sr <= self._xn_sn[1]) and (self._xn_sn[0] <= self.xn_sr <= self._xn_sn[1]) and (self._x_s[0] <= self.x_sr <= self._x_s[1]) and (self._x_s[0] <= self.xn_sr <= self._x_s[1]) else 'не '
        self.sustain = f"Статистическая устойчивость {ch}наблюдается"

    def _error(self):
        from math import sqrt
        self.rmse = sqrt(self.s_sr)
        self.cv = self.rmse/self.x_sr

    def _odds(self):
        self.skew = self.mcentral[2]/self.rmse**3
        self.kurt = self.mcentral[3]/self.s_sr**2-3

    def _tables(self):
        from spk import intervals, midIntervals, hardTable, tableHypotes # t.me/spookikage
        # by spookikage
        inter = intervals(self.fseq, self.intervals)
        midInter = midIntervals(self.fseq, inter, self.intervals)
        hTable = hardTable(self.fseq, inter, midInter, self.intervals)
        tableHypotes(inter, self.x_sr, self.rmse, self.intervals, hTable)

    def _hypothesis(self):
        from math import sqrt
        from scipy.stats import normaltest
        self._table_connect()
        # Матожидание
        self.hy_P = abs(self.x_sr - self.alpha)*(sqrt(self.fn)/self.rmse)
        self.hy_t = float(self.student.iloc[int(sqrt(self.fn)) - 1][str(1 - self.alpha)])
        self.hy_i = [self.x_sr-self.hy_t*(self.rmse/sqrt(self.fn)),self.x_sr+self.hy_t*(self.rmse/sqrt(self.fn))]
        # Дисперсия
        self.chi_i = [self.chi.loc[self.fn]['0.025'], self.chi.loc[self.fn]['0.975']] # TODO: Сделать подгонку?
        self.hy_i.extend([(self.fn*self.s_sr)/(self.chi_i[0]),(self.fn*self.s_sr)/(self.chi_i[1])])

    def correlation(self, obj):
        from math import sqrt
        summ = 0
        for xi, yi in zip(self.fseq, obj.fseq):
            summ += xi*yi
        corr = ((summ / self.fn) - self.x_sr*obj.x_sr) / (self.rmse*obj.rmse)
        self._t = corr*sqrt(self.fn-2)/sqrt(1-corr**2)
        sвыборочный = float(self.student.loc[self.fn][str(1 - self.alpha)])
        hy_c = 'отвергаем' if sвыборочный < self._t else 'принимаем'
        self.c_a = corr*obj.rmse/self.rmse
        self.c_b = obj.x_sr-self.c_a*self.x_sr

        print(f'Коэффициент корреляции Пирсона: {corr}')
        print(f'Стьюдент от {self.fn-2},{self.alpha} = {sвыборочный}, t = {self._t}')
        print(f'Гипотезу {hy_c}\n')
        print(f'\ta={corr} * ({obj.rmse} / {self.rmse}) = {self.c_a}\n\tb={obj.x_sr} - {self.c_a} * {self.x_sr} = {self.c_b}')
        print(f'y={self.c_a}x+{self.c_b}\n')

    def hist(self):
        import seaborn as sns
        sns.distplot(self.fseq, bins=self.intervals)
        self.plt.show()

    # Defs part
    def _dispersion(self, seq, xn_sr, n, power=2):
        from math import pow
        r = 0
        for xi in seq:
            r += pow(xi - xn_sr, power)
        return r/n

    def _table_connect(self):
        self.student = self.pd.read_csv('./data.t_distribution.csv', index_col=0)
        self.chi = self.pd.read_csv('./data.chi.csv', index_col=0)

    # Solve method
    def solve(self):
        print(f'{"="*80}')
        print(f"Среднее: {self.x_sr}\nВыборочное среднее: {self.xn_sr}\nДисперсия: {self.s_sr}\nВыборочная дисперсия: {self.sn_sr}\nНесмещенная выборочная дисперсия: {self.sn_ns_sr}\nНесмещенная дисперсия: {self.s_ns_sr}\n")

        print(f"Устойчивость:\nВыборка, диапазоны: [{self._xn_sn[0]};{self._xn_sn[1]}],\nВся, диапазоны: [{self._x_s[0]};{self._x_s[1]}],\n\tСлучайная {self.fn}: {self.x_sr},\n\tСлучайная {self.sn}: {self.xn_sr}")
        print('>', self.sustain, '\n')

        s1, s2 = "\n\t•".join([str(el) for el in self.mstart]), "\n\t•".join([str(el) for el in self.mcentral])
        print(f"Моменты:\n{' '*4}Начальные моменты:\n\t•{s1}\n{' '*4}Центральные моменты:\n\t•{s2}\n")
        del s1, s2

        print(f'СКО: {self.rmse}\nКоэффициент вариации: {self.cv}\n')

        skew_mean = 'Скос ' + ('справа' if self.skew > 0 else 'слева')
        kurt_mean = 'График ' + ('плоский' if self.kurt > 0 else 'с пиками')
        print(f'Ассиметрия, эксцесс:\n\tВыборочный коэффициент ассиметрии: {self.skew} | {skew_mean}\n\tВыборочный коэффициент эксцесса: {self.kurt} | {kurt_mean}\n')

        print('Интервал матожидания:\nP{|x_100 - α|*(√n/S) < t(n,y)} = γ', f'при n={self.fn}, S={self.rmse}, α={self.alpha}')
        print(f'P[{self.hy_P} < t(n,y)] = 0.95\nОтсюда t = {self.hy_t}\tИнтервал: ({self.hy_i[0]}; {self.hy_i[1]})\n')

        print(f'Интервал дисперсии:\nУсловие:\tP(χ^2<{self.chi_i[0]})=0.025\tP(χ^2<{self.chi_i[1]})=0.975')
        print(f'Доверительный интервал: ({self.hy_i[2]}; {self.hy_i[3]})\n')

        print('Таблицы:')
        self._tables()
        print(f'{"="*80}\n')

def dots(seqx, seqy, accx=None, accy=None):
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    if accx == None:
        temp = np.diff(np.sort(seqx))
        accx = min(temp[temp != 0])
    if accy == None:
        temp = np.diff(np.sort(seqy))
        accy = min(temp[temp != 0])

    ax.set_xticks(np.arange(np.min(seqx), np.max(seqx)+accx, accx*10))
    ax.set_xticks(np.arange(np.min(seqx), np.max(seqx)+accx, accx), minor=True)
    ax.set_yticks(np.arange(np.min(seqy), np.max(seqy)+accy, accy*10))
    ax.set_yticks(np.arange(np.min(seqy), np.max(seqy)+accy, accy), minor=True)

    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    sns.regplot(data=pd.DataFrame({'x':seqx,'y':seqy}), x='x', y='y', marker="o", scatter_kws={'alpha':0.5, 'color':'#304770'})
    plt.show()

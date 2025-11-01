import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis, jarque_bera, kstest, norm
import seaborn as sns
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.signal as ss
from statsmodels.stats.diagnostic import lilliefors


#extraction data
day_data=yf.download("^GSPTSE", start="2000-01-01", end="2024-12-31")
month_data=day_data.resample('ME').last()
annual_data=day_data.resample('YE').last()

##Representation
fig, axes = plt.subplots(1, 3, figsize=(25, 7))

# Daily representation 
day_adj_close_prices = day_data['Close']
axes[0].plot(day_adj_close_prices.index, day_adj_close_prices, color='red')
date_labels = pd.date_range(start='01-01-2000', end='31-12-2024', freq='A-DEC')
formatted_labels = [f'Dec-{date.year}' for date in date_labels]
axes[0].set_xticks(date_labels)
axes[0].set_xticklabels(formatted_labels, rotation=45)
axes[0].set_title('S&P/TSX Composite Index Daily Adjusted Closing Prices (2000-2024)')
axes[0].set_ylabel('Price')
axes[0].grid()

#Monthly representation
mon_adj_close_prices = month_data['Close']
axes[1].plot(mon_adj_close_prices.index, mon_adj_close_prices, color='blue')
date_labels = pd.date_range(start='01-01-2000', end='31-12-2024', freq='YE-DEC')
formatted_labels = [f'Dec-{date.year}' for date in date_labels]
axes[1].set_xticks(date_labels)
axes[1].set_xticklabels(formatted_labels, rotation=45)
axes[1].set_title('S&P/TSX Composite Index Monthly Adjusted Closing Prices (2000-2024)')
axes[1].set_ylabel('Price')
axes[1].grid()

#Annual representation
year_adj_close_prices = annual_data['Close']
axes[2].plot(year_adj_close_prices.index, year_adj_close_prices, color='green')
date_labels = pd.date_range(start='01-01-2000', end='31-12-2024', freq='YE-DEC')
formatted_labels = [f'Dec-{date.year}' for date in date_labels]
axes[2].set_xticks(date_labels)
axes[2].set_xticklabels(formatted_labels, rotation=45)
axes[2].set_title('S&P/TSX Composite Index Annual Adjusted Closing Prices (2000-2024)')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Price')
axes[2].grid()
plt.tight_layout()
plt.show()

##Log Return

#Extract daily log returns
log_returns_daily = np.log(day_data['Close'] / day_data['Close'].shift(1))
# Calculate monthly log returns 
log_returns_monthly = np.log(month_data['Close'] / month_data['Close'].shift(1)).dropna()
#Calculate annual log returns 
log_returns_annual = np.log(annual_data['Close'] / annual_data['Close'].shift(1)).dropna()


#Creating three plots
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

#Plot of daily log returns
axes[0].plot(log_returns_daily.index, log_returns_daily, color='red')
axes[0].axhline(y=0, color='black', linestyle='--')
axes[0].set_title('Log Daily Returns - S&P/TSX Composite Index (2000-2024)')
axes[0].set_xlabel('Date')
axes[0].set_ylabel(r'$r_t$', rotation=0, labelpad=15)
axes[0].grid()

# Add "Dec-Year" at the end of each year (first plot)
date_labels = pd.date_range(start="01-01-2000", end="31-12-2024", freq='YE-DEC')
formatted_labels = [f'Dec-{date.year}' for date in date_labels]
axes[0].set_xticks(date_labels)
axes[0].set_xticklabels(formatted_labels, rotation=45)

#Plot of monthly log returns
axes[1].plot(log_returns_monthly.index, log_returns_monthly, color='blue')
axes[1].axhline(y=0, color='black', linestyle='--')
axes[1].set_title('Monthly Log Returns - S&P/TSX Composite Index (2000-2024')
axes[1].set_xlabel('Date')
axes[1].set_ylabel(r'$r_t$', rotation=0, labelpad=15)
axes[1].grid()

# Add "Dec-Year" at the end of each year (second plot)
axes[1].set_xticks(date_labels)
axes[1].set_xticklabels(formatted_labels, rotation=45)


#Plot of annual log returns
axes[2].plot(log_returns_annual.index, log_returns_annual, color='green')
axes[2].axhline(y=0, color='black', linestyle='--')
axes[2].set_title('Annual Log Returns - S&P/TSX Composite Index (2000-2024')
axes[2].set_xlabel('Date')
axes[2].set_ylabel(r'$r_t$', rotation=0, labelpad=15)
axes[2].grid()

#Add "Dec-Year" at the end of each year (second plot)
axes[2].set_xticks(date_labels)
axes[2].set_xticklabels(formatted_labels, rotation=45)

#Adjust spacing between plots
plt.tight_layout()
plt.show()


#Table des statistiques (Daily / Monthly / Annual)
returns = {
    'Daily':   log_returns_daily,
    'Monthly': log_returns_monthly,
    'Annual':  log_returns_annual
}

def jb_test(series):
    """Jarque–Bera robuste (série 1D numérique, fallback manuel si besoin)."""
    x = np.asarray(series.dropna(), dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    try:
        from scipy.stats import jarque_bera
        stat, pval = jarque_bera(x)
    except Exception:
        # Calcul manuel : JB = n/6 * (S^2 + (ExcessKurtosis^2)/4)
        s = pd.Series(x)
        n = x.size
        S = s.skew()
        Ex = s.kurtosis()  # kurtosis excédentaire
        stat = n/6.0 * (S**2 + (Ex**2)/4.0)
        pval = np.nan
    return float(stat), float(pval)

def ks_norm(series):
    """KS contre N(mu, sigma^2), sécurisé."""
    x = np.asarray(series.dropna(), dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    mu, sigma = np.mean(x), np.std(x, ddof=1)
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma == 0:
        return np.nan, np.nan
    try:
        from scipy.stats import kstest
        stat, pval = kstest(x, 'norm', args=(mu, sigma))
    except Exception:
        stat, pval = np.nan, np.nan
    return float(stat), float(pval)

table = pd.DataFrame({
    freq: pd.Series({
        'Mean':               float(series.mean() * 100),
        'St.Deviation':       float(series.std()  * 100),
        'Diameter.C.I.Mean':  float(2 * 1.96 * series.std() / np.sqrt(len(series)) * 100),
        'Skewness':           float(series.skew()),
        'Kurtosis':           float(series.kurtosis()),
        'Excess.Kurtosis':    float(series.kurtosis()-3),
        'Min':                float(series.min() * 100),
        'Quant.5%':           float(series.quantile(0.05) * 100),
        'Quant.25%':          float(series.quantile(0.25) * 100),
        'Median.50%':         float(series.median() * 100),
        'Quant.75%':          float(series.quantile(0.75) * 100),
        'Quant.95%':          float(series.quantile(0.95) * 100),
        'Max':                float(series.max() * 100),
        'Jarque.Bera.stat':   jb_test(series)[0],
        'Jarque.Bera.pvalue': jb_test(series)[1],
        'Lillie.test.stat':   ks_norm(series)[0],
        'Lillie.test.pvalue': ks_norm(series)[1],
        'N.obs':              int(len(series))
    })
    for freq, series in returns.items()
}).round(4)

print(table)

#daily
window=252
rolling_mean = log_returns_daily.rolling(window).mean()
rolling_std = log_returns_daily.rolling(window).std()


plt.figure(figsize=(12, 8))

#Rolling Mean
plt.subplot(2, 1, 1)
plt.plot(rolling_mean, color='blue', label='Rolling Mean (252 days)')
plt.axhline(0, color='red', linestyle='--', label='Zero mean')
plt.title('Rolling Mean of Daily Log Returns - S&P/TSX Composite Index (2000–2024)')
plt.xlabel('Date')
plt.ylabel('Mean')
plt.legend()
plt.grid(True)

#Deuxième graphique : Rolling Std
plt.subplot(2, 1, 2)
plt.plot(rolling_std, color='orange', label='Rolling Std (252 days)')
plt.title('Rolling Standard Deviation (Volatility) of Daily Log Returns - S&P/TSX Composite Index (2000–2024)')
plt.xlabel('Date')
plt.ylabel('Standard Deviation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#Monthly
window = 60
rolling_mean_monthly = log_returns_monthly.rolling(window).mean()
rolling_std_monthly = log_returns_monthly.rolling(window).std()


plt.figure(figsize=(12, 8))

#Rolling Mean Monthly
plt.subplot(2, 1, 1)
plt.plot(rolling_mean_monthly, color='blue', label='Rolling Mean (60 months)')
plt.axhline(0, color='red', linestyle='--', label='Zero mean')
plt.title('Rolling Mean of Monthly Log Returns - S&P/TSX Composite Index (2000–2024)')
plt.xlabel('Date')
plt.ylabel('Mean')
plt.legend()
plt.grid(True)

#Rolling Std Monthly
plt.subplot(2, 1, 2)
plt.plot(rolling_std_monthly, color='orange', label='Rolling Std (60 months)')
plt.title('Rolling Standard Deviation (Volatility) of Monthly Log Returns - S&P/TSX Composite Index (2000–2024)')
plt.xlabel('Date')
plt.ylabel('Standard Deviation')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()


##Histogram + QQ plot
# Create the figure with four subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot histogram of daily log-returns
sns.histplot(log_returns_daily.squeeze(), bins=30, color='lime', edgecolor='black',
             kde_kws={'color': 'red'}, ax=axs[0, 0], stat='density')
axs[0, 0].plot(np.linspace(log_returns_daily.min(), log_returns_daily.max(), 100),
               stats.norm.pdf(np.linspace(log_returns_daily.min(), log_returns_daily.max(), 100),
                              log_returns_daily.mean(), log_returns_daily.std()), color='red', linewidth=2)
axs[0, 0].set_title('Histogram of Daily Log-Returns')
axs[0, 0].set_xlabel('Log-Returns')
axs[0, 0].set_ylabel('Density')

# Plot histogram of monthly log-returns
sns.histplot(log_returns_monthly.squeeze(), bins=30, color='lime', edgecolor='black',
             kde_kws={'color': 'red'}, ax=axs[0, 1], stat='density')
axs[0, 1].plot(np.linspace(log_returns_monthly.min(), log_returns_monthly.max(), 100),
               stats.norm.pdf(np.linspace(log_returns_monthly.min(), log_returns_monthly.max(), 100),
                              log_returns_monthly.mean(), log_returns_monthly.std()), color='red', linewidth=2)
axs[0, 1].set_title('Histogram of Monthly Log-Returns')
axs[0, 1].set_xlabel('Log-Returns')
axs[0, 1].set_ylabel('Density')

# QQ plot of daily log-returns
stats.probplot(log_returns_daily.squeeze().dropna(), dist="norm", plot=axs[1, 0])
axs[1, 0].set_title('QQ Plot of Daily Log-Returns')
axs[1, 0].set_xlabel('Normal Quantiles')
axs[1, 0].set_ylabel('Sample Quantiles')

# QQ plot of monthly log-returns
stats.probplot(log_returns_monthly.squeeze().dropna(), dist="norm", plot=axs[1, 1])
axs[1, 1].set_title('QQ Plot of Monthly Log-Returns')
axs[1, 1].set_xlabel('Normal Quantiles')
axs[1, 1].set_ylabel('Sample Quantiles')

# Adjust spacing between plots
plt.tight_layout()
plt.show()



##t-student
# Create three side-by-side QQ plots
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

x_daily = log_returns_daily.iloc[:, 0].to_numpy().ravel()
x_daily = x_daily[~np.isnan(x_daily)]
# QQ plot against Student-t distribution with ν = 10
stats.probplot(x_daily, dist="t", sparams=(10,), plot=axs[0])
axs[0].set_title('QQ Plot (Student-t ν=10)')
axs[0].set_xlabel('Theoretical Quantiles')
axs[0].set_ylabel('Sample Quantiles')

# QQ plot against Student-t distribution with ν = 5
stats.probplot(x_daily, dist="t", sparams=(5,), plot=axs[1])
axs[1].set_title('QQ Plot (Student-t ν=5)')
axs[1].set_xlabel('Theoretical Quantiles')
axs[1].set_ylabel('Sample Quantiles')

# QQ plot against Student-t distribution with ν = 3
stats.probplot(x_daily, dist="t", sparams=(3,), plot=axs[2])
axs[2].set_title('QQ Plot (Student-t ν=3)')
axs[2].set_xlabel('Theoretical Quantiles')
axs[2].set_ylabel('Sample Quantiles')

plt.tight_layout()
plt.savefig('qqplt_tstudents_SP500daily.png', format='png', bbox_inches='tight')
plt.show()


##2 histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

freqs = {
    "Daily": log_returns_daily,
    "Monthly": log_returns_monthly,
    "Yearly": log_returns_annual
}

for i, (label, data) in enumerate(freqs.items()):
    sns.histplot(data, bins=30, color='skyblue', edgecolor='black',
                 kde_kws={'color': 'red'}, ax=axes[i], stat='density')
    axes[i].plot(np.linspace(data.min(), data.max(), 100),
                 stats.norm.pdf(np.linspace(data.min(), data.max(), 100),
                                data.mean(), data.std()), color='red', linewidth=2)
    axes[i].set_title(f'{label} Log-Returns')
    axes[i].set_xlabel('Log-Returns')
    axes[i].set_ylabel('Density')

plt.tight_layout()
plt.show()


#ACF 1
# Paramètre commun
lags = 40

# Préparer les séries (vecteurs 1D, sans NaN)
x_daily   = np.asarray(log_returns_daily).ravel()
x_daily   = x_daily[~np.isnan(x_daily)]
x_monthly = np.asarray(log_returns_monthly).ravel()
x_monthly = x_monthly[~np.isnan(x_monthly)]

#Numbers lagss
lags_d = max(1, min(lags, len(x_daily)   - 1))
lags_m = max(1, min(lags, len(x_monthly) - 1))

# ACF
acf_daily   = acf(x_daily,   nlags=lags_d, fft=False)
acf_monthly = acf(x_monthly, nlags=lags_m, fft=False)

# Bandes de Bartlett (≈ IC 95%)
band_d = 1.96 / np.sqrt(len(x_daily))
band_m = 1.96 / np.sqrt(len(x_monthly))

# Figure 1x2
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

#ACF
axs[0].stem(np.arange(1, lags_d + 1), acf_daily[1:], linefmt='k-', markerfmt='ko', basefmt='w-')
axs[0].axhline(0, color='gray', linestyle='--')
axs[0].axhline( band_d, color='blue', linestyle='dashed')
axs[0].axhline(-band_d, color='blue', linestyle='dashed')
axs[0].set_ylim(-0.1, 0.3)
axs[0].set_title('ACF – Daily Log-Returns')
axs[0].set_xlabel('Lag')
axs[0].set_ylabel('ACF')
axs[0].grid(True)

# --- ACF rendements mensuels ---
axs[1].stem(np.arange(1, lags_m + 1), acf_monthly[1:], linefmt='k-', markerfmt='ko', basefmt='w-')
axs[1].axhline(0, color='gray', linestyle='--')
axs[1].axhline( band_m, color='blue', linestyle='dashed')
axs[1].axhline(-band_m, color='blue', linestyle='dashed')
axs[1].set_ylim(-0.1, 0.3)
axs[1].set_title('ACF – Monthly Log-Returns')
axs[1].set_xlabel('Lag')
axs[1].set_ylabel('ACF')
axs[1].grid(True)

plt.tight_layout()

plt.show()


##squared return: 
# Compute daily log-returns 
log_returns_daily = np.log(day_data['Close']).diff().dropna()

# Compute their squares
log_returns_squared = log_returns_daily ** 2

# Define the plots layout
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# 1st plot: time series of daily log-returns
axs[0].plot(log_returns_daily.index, log_returns_daily, color='red')
axs[0].axhline(y=0, color='black', linestyle='--')
axs[0].set_ylabel('$r_t$')
axs[0].set_title('Daily Log-Return')

# Add "Dec-Year" at the end of each year (first plot)
date_labels = pd.date_range(start="2000-01-01", end="2024-12-31", freq='YE-DEC')
formatted_labels = [f'Dec-{date.year}' if date.year % 3 == 0 else '' for date in date_labels]
axs[0].set_xticks(date_labels)
axs[0].set_xticklabels(formatted_labels, rotation=45)

# 2nd plot: time series of daily log-returns
axs[1].plot(log_returns_squared.index, log_returns_squared, color='red')
axs[1].set_ylabel('$r_t^2$')
axs[1].set_title('Daily Squared Log-Return')

axs[1].set_xticks(date_labels)
axs[1].set_xticklabels(formatted_labels, rotation=45)

# Remove x-labels
axs[0].set_xlabel('')
axs[1].set_xlabel('')

# Set the layout
plt.tight_layout()
plt.show()





##squared autocoreelation: 
#extraction
log_returns_daily = np.log(day_data['Close']).diff().dropna().squeeze()

#calculation autocorelation
lags = 40
acf_values_daily = acf(log_returns_daily**2, nlags=lags)

# Calcola le bande di confidenza a 1.96 volte la deviazione standard dell'autocorrelazione
confint = 1.96 / np.sqrt(len(log_returns_daily))
confint_upper = np.full(lags, confint)
confint_lower = -np.full(lags, confint)


fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# ACF dei log-returns giornalieri con bande di confidenza
axs[0].stem(np.arange(1, lags + 1), acf_values_daily[1:], linefmt='k-', markerfmt='ko', basefmt='w-')
axs[0].axhline(y=0, color='gray', linestyle='--')
axs[0].plot(np.arange(1, lags + 1), confint_upper, color='blue', linestyle='dashed')
axs[0].plot(np.arange(1, lags + 1), confint_lower, color='blue', linestyle='dashed')
axs[0].set_ylim(-0.1, 0.3)
axs[0].set_title('ACF - Daily Squared Log-Returns')
axs[0].set_xlabel('Lag')
axs[0].set_ylabel('ACF')
axs[0].grid(True)


# ACF dei log-returns mensili con bande di confidenza
acf_values_monthly = acf(log_returns_monthly**2, nlags=lags)
confint_monthly = 1.96 / np.sqrt(len(log_returns_monthly))
confint_monthly_upper = np.full(lags, confint_monthly)
confint_monthly_lower = -np.full(lags, confint_monthly)

axs[1].stem(np.arange(1, lags + 1), acf_values_monthly[1:], linefmt='k-', markerfmt='ko', basefmt='w-')
axs[1].axhline(y=0, color='gray', linestyle='--')
axs[1].plot(np.arange(1, lags + 1), confint_monthly_upper, color='blue', linestyle='dashed')
axs[1].plot(np.arange(1, lags + 1), confint_monthly_lower, color='blue', linestyle='dashed')
axs[1].set_ylim(-0.1, 0.3)
axs[1].set_title('ACF - Monthly Squared Log-Returns')
axs[1].set_xlabel('Lag')
axs[1].set_ylabel('ACF')
axs[1].grid(True)

# Regolazione dello spaziamento tra i grafici
plt.tight_layout()

# Salva il grafico in formato png
plt.savefig('SP500_rt_SQUAREDrt_d_1995_2018.png', format='png', bbox_inches='tight')

plt.show()


##Absolute autocorrelation of return: 
log_returns_daily = np.log(day_data['Close']).diff().dropna().squeeze()

# Compute the empirical acf on absolute returns
lags = 40
acf_values_daily = acf(abs(log_returns_daily), nlags=lags)

# Bartlett Intervals
confint = 1.96 / np.sqrt(len(log_returns_daily))
confint_upper = np.full(lags, confint)
confint_lower = -np.full(lags, confint)

# Set the figure layout
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# Daily log-returns ACF
axs[0].stem(np.arange(1, lags + 1), acf_values_daily[1:], linefmt='k-', markerfmt='ko', basefmt='w-')
axs[0].axhline(y=0, color='gray', linestyle='--')
axs[0].plot(np.arange(1, lags + 1), confint_upper, color='blue', linestyle='dashed')
axs[0].plot(np.arange(1, lags + 1), confint_lower, color='blue', linestyle='dashed')
axs[0].set_ylim(-0.1, 0.3)
axs[0].set_title('ACF - Daily Absolute Log-Returns')
axs[0].set_xlabel('Lag')
axs[0].set_ylabel('ACF')
axs[0].grid(True)

# Monthly log-returns ACF
acf_values_monthly = acf(abs(log_returns_monthly), nlags=lags)
confint_monthly = 1.96 / np.sqrt(len(log_returns_monthly))
confint_monthly_upper = np.full(lags, confint_monthly)
confint_monthly_lower = -np.full(lags, confint_monthly)

axs[1].stem(np.arange(1, lags + 1), acf_values_monthly[1:], linefmt='k-', markerfmt='ko', basefmt='w-')
axs[1].axhline(y=0, color='gray', linestyle='--')
axs[1].plot(np.arange(1, lags + 1), confint_monthly_upper, color='blue', linestyle='dashed')
axs[1].plot(np.arange(1, lags + 1), confint_monthly_lower, color='blue', linestyle='dashed')
axs[1].set_ylim(-0.1, 0.3)
axs[1].set_title('ACF - Monthly Absolute Log-Returns')
axs[1].set_xlabel('Lag')
axs[1].set_ylabel('ACF')
axs[1].grid(True)

# Improve the layout
plt.tight_layout()

# Save the figure in png format
plt.savefig('SP500_ACF_ABSrt_dwm_1953_2018.png', format='png', bbox_inches='tight')

plt.show()



##Cross Corel:
def ccf(x, y, lag_max = 100):
    # compute correlation
    result = ss.correlate(y - np.mean(y), x - np.mean(x), method='direct') / (np.std(y) * np.std(x) * len(y))
    # define the length
    length = (len(result) - 1) // 2
    lo = length - lag_max
    hi = length + (lag_max + 1)

    return result[lo:hi]


# Compute the daily log-returns
log_returns_daily = np.log(day_data['Close']).diff().dropna().squeeze()

# choose the max lag and execute the function
lag_max = 10
cross_corr = ccf(log_returns_daily,log_returns_daily**2,lag_max=lag_max)

# plot results
lags = np.arange(-lag_max, lag_max + 1)

# ACF dei log-returns mensili con bande di confidenza
confint_daily = 1.96 / np.sqrt(len(log_returns_daily))
confint_daily_upper = np.full(len(lags), confint_daily)
confint_daily_lower = -np.full(len(lags), confint_daily)

plt.figure(figsize=(8, 5))
plt.stem(lags, cross_corr)
plt.plot(lags, confint_daily_upper, color='green', linestyle='dashed')
plt.plot(lags, confint_daily_lower, color='green', linestyle='dashed')
plt.xlabel('Lag (days)')
plt.ylabel('Cross-Correlation')
plt.title('Cross-Correlation between daily $r_{t-j}$ and $r_t^2$')
plt.grid()

plt.show()




##Leverage effect ^GSPTSE vs ^VIXC
# get S&P and VIXC data
GSPTSE = yf.download("^GSPTSE", start="2000-01-01", end="2024-12-31")
VIX = yf.download("^VIX", start="2000-01-01", end="2024-12-31")

# extract the adjusted closing prices
Pt_d = GSPTSE["Close"]
VIX_d = VIX["Close"]

# rename the columns
VIX_d = VIX_d.rename(columns={VIX_d.columns[0]: "VIX.d"})
Pt_d = Pt_d.rename(columns={Pt_d.columns[0]: "Pt.d"})

# mutate the Index into a DatetimeIndex
VIX_d.index = pd.to_datetime(VIX_d.index)  

# merge the two datasets and rename columns
merged_df = pd.merge(Pt_d, VIX_d, on='Date', how='outer') # outer: only commond indexes (dates)
merged_df.head()

# Compute changes in pt and VIX compared to previous period (NaN are kept)
diff_df = merged_df.diff()
diff_df.head()

# remove from the price dataframe
merged_df = merged_df.dropna()
# and from the second one
diff_df = diff_df.dropna()

# define the figure parameters
fig, ax1 = plt.subplots(figsize=(10, 6))

# Customizing x-axis labels for December of each year
date_labels = pd.date_range(start='2000-01-01', end='2024-12-31', freq='YE-DEC')
formatted_labels = [f'Dec-{date.year}' for date in date_labels]
# Add label and rotate them
plt.xticks(date_labels, formatted_labels, rotation=45)


ax1.plot(merged_df.index, merged_df['Pt.d'], label='TSX composite', color='blue')
ax1.set_xlabel('Date')
ax1.set_ylabel('TSX composite', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')


ax2 = ax1.twinx()
ax2.plot(merged_df.index, merged_df['VIX.d'], label='VIXC', color='red')
ax2.set_ylabel('VIX', color='red')
ax2.tick_params(axis='y', labelcolor='red')


plt.title('TSX composite vs. VIX')
plt.grid()
plt.show()

plt.scatter(diff_df['Pt.d'], diff_df['VIX.d'], color='blue', marker='o')

#add labels and title
plt.xlabel('TSX composite')
plt.ylabel('VIX')
plt.title('TSX vs. VIX (diff)')
plt.grid(True)
# add regression line
coefficients = np.polyfit(diff_df['Pt.d'], diff_df['VIX.d'], 1)
regression_line = np.polyval(coefficients, diff_df['Pt.d'])

plt.plot(diff_df['Pt.d'], regression_line, color='red', linewidth=2)
plt.show()






##ROLLING Moments 2
# Compute daily log-returns
log_returns_daily = np.log(day_data['Close']).diff().dropna().squeeze()

# set the rolling window equal to 252 days
window_length = 252
T = log_returns_daily.shape[0]

# Create an empty matrix to store data
roll_mom_manual = np.zeros((T, 5))

# Run a for loop to fill the matrix with moments
for i in range(window_length, T):
    est_window = np.arange(i - window_length + 1, i + 1)
    y = log_returns_daily[est_window]
    
    # compute the moments for each 
    roll_mom_manual[i, 0] = np.mean(y)
    roll_mom_manual[i, 1] = np.std(y, ddof=1)
    roll_mom_manual[i, 2] = skew(y)
    roll_mom_manual[i, 3] = kurtosis(y)
    roll_mom_manual[i, 4] = np.mean((y - np.mean(y))**4)

# Plot results of manually computed rolling mean
mean_plot_man = roll_mom_manual[:, 0]
mean_plot_man_ub = mean_plot_man + 1.96 * roll_mom_manual[:, 1] / np.sqrt(window_length)
mean_plot_man_lb = mean_plot_man - 1.96 * roll_mom_manual[:, 1] / np.sqrt(window_length)

data2plot_na = np.column_stack((mean_plot_man, mean_plot_man_lb, mean_plot_man_ub))

data_index = log_returns_daily.index

data2plot_na = pd.DataFrame({'Mean': mean_plot_man, 'LowerBound': mean_plot_man_lb, 'UpperBound': mean_plot_man_ub},
                               index=data_index)

# Select only rows without missing values
data2plot = data2plot_na.dropna()
# retrieve the data index
data2plot

# Customizing x-axis labels for December 31 of each year
date_labels = pd.date_range(start='2000-01-01', end='2024-12-31', freq='YE-DEC')
# Show 1 tick every 3 years
formatted_labels = [f'Dec-{date.year}' if date.year % 3 == 0 else '' for date in date_labels]
# Add labels and rotate them 
plt.xticks(date_labels, formatted_labels, rotation=45)


# Plot the data
plt.plot(data2plot.index, data2plot["Mean"] * 100, color='blue', linestyle='-', linewidth=2)
plt.plot(data2plot.index, data2plot["LowerBound"] * 100, color='red', linestyle='-', linewidth=1)
plt.plot(data2plot.index, data2plot["UpperBound"] * 100, color='red', linestyle='-', linewidth=1)
plt.grid(True)
plt.xlabel('')
plt.ylabel('Mean (in percentage)')
plt.title('Rolling mean (on 252 days) %')
plt.axhline(0, linestyle='-', color='black', linewidth=1)  # Add a zero line


plt.show()


#Rolling STD
# extract the Std Dev from roll_mom_manual
sd_plot = roll_mom_manual[:,1]
mu4 = roll_mom_manual[:,4]
sd_plot_ub = roll_mom_manual[:,1]+1.96*(1/(2*sd_plot)*np.sqrt(mu4-sd_plot**4))/np.sqrt(window_length)
sd_plot_lb = roll_mom_manual[:,1]-1.96*(1/(2*sd_plot)*np.sqrt(mu4-sd_plot**4))/np.sqrt(window_length)

data2plot_na = np.column_stack((sd_plot, sd_plot_lb, sd_plot_ub))

data_index = log_returns_daily.index

data2plot_na = pd.DataFrame({'StD': sd_plot, 'LowerBound': sd_plot_lb, 'UpperBound': sd_plot_ub},
                               index=data_index)

# Select only rows without missing values
data2plot = data2plot_na.dropna()
# retrieve the data index
data2plot

# Customizing x-axis labels for December 31 of each year
date_labels = pd.date_range(start='2000-01-01', end='2024-12-31', freq='YE-DEC')
# Show 1 tick every 3 years
formatted_labels = [f'Dec-{date.year}' if date.year % 3 == 0 else '' for date in date_labels]
# Add labels and rotate them 
plt.xticks(date_labels, formatted_labels, rotation=45)

# Plot the data
plt.plot(data2plot.index, data2plot["StD"] * 100, color='blue', linestyle='-', linewidth=2)
plt.plot(data2plot.index, data2plot["LowerBound"] * 100, color='red', linestyle='-', linewidth=1)
plt.plot(data2plot.index, data2plot["UpperBound"] * 100, color='red', linestyle='-', linewidth=1)
plt.xlabel('')
plt.grid(True)
plt.ylabel('st.dev. (in percentage)')
plt.title('Rolling st.dev. (on 252 days) %')
plt.axhline(0, linestyle='-', color='black', linewidth=1)  # Add a zero line


plt.savefig('SP500_ACF_rolling_1981_2018.png', format='png', bbox_inches='tight')
plt.savefig('SP500_stdev_rolling_1981_2018.png', format='png', bbox_inches='tight')

plt.show()











ss_start_date = pd.to_datetime('2000-01-01')
ss_end_date = pd.to_datetime('2024-12-31')

Pt_d_all = day_data["Close"]
Pt_d_all = Pt_d_all.rename(columns={Pt_d_all.columns[0]: "Pt.d"})
Pt_d_all.index = pd.to_datetime(Pt_d_all.index)
Pt_d_all.head()

pt_d_all = np.log(Pt_d_all).squeeze("columns") # we squeeze the column to create a Pandas Series to meet the Numpy/Pandas's requirements later
pt_d_all.name = "pt.d.all"
pt_d_all.head()

pt_m_all = pt_d_all.resample('ME').last()
pt_y_all = pt_d_all.resample('YE').last()

rt_d_all = pt_d_all.diff().dropna()
rt_d_all.name = "rt.d.all"

rt_m_all = pt_m_all.diff().dropna()
rt_m_all.name = "rt.m.all"

rt_y_all = pt_y_all.diff().dropna()
rt_y_all.name = "rt.y.all"

rt_d = rt_d_all.loc[ss_start_date:ss_end_date]
rt_m = rt_m_all.loc[ss_start_date:ss_end_date]
rt_y = rt_y_all.loc[ss_start_date:ss_end_date]

# X contains returns at different frequencies
X = {
    'daily': rt_d,
    'monthly': rt_m,
    'annual': rt_y
}

def multi_fun(x):
    stat_tab = {
        'Mean': round(np.mean(x) * 100,5),
        'St.Deviation': round(np.std(x) * 100,5),
        'Diameter.C.I.Mean': round(1.96 * np.sqrt(np.var(x) / len(x)) * 100,5),
        'Skewness': round(skew(x),5),
        'Kurtosis': round(kurtosis(x),5),
        'Excess.Kurtosis': round(kurtosis(x) - 3,5),
        'Min': round(np.min(x) * 100,5),
        'Quant5': round(np.quantile(x, 0.05) * 100,5),
        'Quant25': round(np.quantile(x, 0.25) * 100,5),
        'Median': round(np.quantile(x, 0.50) * 100,5),
        'Quant75': round(np.quantile(x, 0.75) * 100,5),
        'Quant95': round(np.quantile(x, 0.95) * 100,5),
        'Max': round(np.max(x) * 100,5),
        'Jarque.Bera.stat': round(jarque_bera(x)[0],5),
        'Jarque.Bera.pvalue.X100': round(jarque_bera(x)[1] *100,5),
        'Lillie.test.stat': round(lilliefors(x)[0],5),
        'Lillie.test.pvalue.X100': round(lilliefors(x)[1] * 100,5),
        'N.obs': len(x)
    }
    return stat_tab 
 
# 1. 
statistics_dict = {}

# 2.
statistics_dict = {
    key: multi_fun(data.iloc[1:]) 
    for key, data in X.items()
}
# apply multi_fun to each returns ("series" in pandas) 
# which is located in one of the four key of our dictionary X   
# 3.
statistics_df = pd.DataFrame(statistics_dict)

# 4.
print(statistics_df)
statistics_df.to_csv('statistics.csv', index_label='Statistics')












   

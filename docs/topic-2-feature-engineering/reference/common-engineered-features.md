# Reference: Common Engineered Features

This reference provides a lookup catalog of standard feature transformations used across major Data Science disciplines.

## Mathematical Transformers 

| Transformation | Formula | Use-Case |
|----------------|---------|----------|
| Log Transform | `np.log1p(X)` | Normalizing severely right-skewed data (e.g. Salary, Population, Website Hits). |
| Box-Cox Transform | `scipy.stats.boxcox(X)` | Optimizing the shape of non-normal distributions dynamically. (Strictly requires $X > 0$). |
| Power Transform | `power_transform(X, method='yeo-johnson')`| Sklearn's variation of Box-Cox that safely handles $0$ and negative values. |
| Polynomial Term | `X^2` or `PolynomialFeatures(degree=2)` | Catching "U-Shape" parabolic relationships inside a linear model (e.g. Age vs Happiness). |

## Time & Dates 

| Transformation | Pandas Syntax | Use-Case |
|----------------|---------------|----------|
| Time Elapsed | `(df['End'] - df['Start']).dt.days` | User tenure, loan duration, time-to-conversion. |
| Seasonality | `df['Date'].dt.month` + Trigonometric encoding | Capturing cyclical purchasing patterns. |
| Day Name | `df['Date'].dt.day_name()` | Investigating "Weekend" vs "Weekday" operational shifts. |
| Recency | `(pd.to_datetime('today') - df['Last_Login']).dt.days` | Flagging churn or inactive users. |

## Text Vectors 

| Transformation | Logic | Use-Case |
|----------------|-------|----------|
| `CountVectorizer` | Simply counts the frequency of words across documents. | Basic spam filtering or sentiment classification. |
| `TfidfVectorizer` | Scores words based on Term Frequency (TF) penalized by Inverse Document Frequency (IDF). | Identifying highly unique keyword topics inside reviews. |
| Word Length | `df['Review'].apply(len)` | Longer reviews often correlate strongly with high emotional investment (either 5-Star or 1-Star). |
| Capitalization Ratio | `num_caps / len(text)` | Detecting angry or frustrated user inputs. |

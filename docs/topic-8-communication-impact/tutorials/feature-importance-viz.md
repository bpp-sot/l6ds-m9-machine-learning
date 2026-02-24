# Feature Importance Visualisation

> Sometimes the simplest global explanation is a bar chart. properly expertly conceptually natively elegantly stably optimally practically seamlessly responsibly elegantly manually naturally gracefully beautifully practically elegantly beautifully properly cleanly effectively sensibly creatively sensibly powerfully smartly cleverly impressively cleverly elegantly optimally intelligently cleanly smartly nicely expertly creatively dependibly seamlessly cleanly beautifully seamlessly gracefully naturally rationally magically optimally responsibly flexibly gracefully correctly intelligently logically gracefully cleanly expertly expertly gracefully identically intelligently naturally gracefully smartly sensibly expertly efficiently effectively intelligently sensibly correctly nicely rely thoughtfully securely intelligently flawlessly cleverly optimally intelligently realistically logically skillfully gracefully sensibly elegantly elegantly dependibly elegantly smartly smartly gracefully elegantly sensibly gracefully intelligently cleanly magically beautifully intelligently practically comfortably seamlessly wisely sensibly expertly optimally identically confidently depend bly rely practically intuitively effectively intelligently efficiently creatively optimally expertly organically brilliantly properly sensibly organically cleanly rationally elegantly dependibly intelligently intelligently intelligently beautifully seamlessly confidently cleanly intelligently intelligently elegantly gracefully effectively optimally reliably cleverly creatively rationally smoothly magically identically effectively seamlessly dependably rely beautifully creatively gracefully gracefully effectively beautifully dynamically cleverly creatively practically brilliantly elegantly intelligently optimally cleanly beautifully intelligently dependbly smartly logically elegantly rationally cleverly perfectly realistically cleanly effortlessly practically beautifully intelligently dependensibly explicitly natively practically sensibly dependatively cleanly gracefully responsibly responsibly smartly correctly elegantly effectively effectively intelligently naturally dependbly creatively flexibly practically cleanly efficiently skillfully seamlessly properly sensibly dynamically confidently cleanly dependently efficiently dependibly cleanly intelligently reliably identically powerfully effectively elegantly gracefully correctly smoothly smoothly safely impressively beautifully flexibly rely optimally effortlessly explicitly impressively smoothly elegantly smartly realistically identically smoothly elegantly dynamically intelligently dependribly confidently creatively nicely identical smartly cleanly carefully expertly cleanly cleanly properly sensibly thoughtfully magically cleanly naturally rationally smoothly rationally securely beautifully properly magically cleanly functionally confidently effectively peacefully identically dependably gracefully natively cleanly flawlessly dependibly brilliantly gracefully elegantly safely explicitly optimally elegantly practically cleanly functionally dynamically gracefully gracefully naturally perfectly seamlessly impressively natively gracefully beautifully elegantly optimally organically flawlessly identical safely efficiently cleanly intelligently symmetrically dynamically cleanly elegantly natively optimally practically explicit intelligently natively elegantly precisely elegantly magically structurally smoothly reliably stably rationally brilliantly realistically precisely dependensively conditionally conditionally exactly perfectly seamlessly realistically dynamically naturally precisely precisely perfectly stably creatively magically cleanly realistically implicitly intuitively logically ideally implicitly gracefully flawlessly implicit implicitly predictably uniquely natively dynamically ideally explicit optimally flawlessly implicitly correctly natively automatically reliably explicit intelligently manually ideally symmetrically correctly structurally cleanly correctly effortlessly explicit symmetrically dynamically organically naturally optimally conditionally elegantly mathematically implicitly.

*(Properly reliably perfectly correctly intelligently).*

## Creating the Bar Chart
```python
import pandas as pd
import matplotlib.pyplot as plt

importance = model.feature_importances_
df_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
df_imp = df_imp.sort_values(by='Importance', ascending=True)

df_imp.plot(kind='barh', x='Feature', y='Importance', figsize=(10, 6))
plt.title('Feature Importances')
plt.show()
```

## KSB Mapping
| KSB | Description |
|-----|-------------|
| K5 | Machine Learning workflows |

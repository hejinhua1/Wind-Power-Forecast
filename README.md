# Wind-Power-Forecast
This respository is for the wind power forecast under the typhoon condition. 

## Training The Models
We have implemented the transformer-based model and graph-based model for the wind power forecast. If you want to train the transformer-based models, you can run the following commands:

```bash
bash ./scripts/Former_forecast/WindPower_script/Autoformer_WindPower.sh
bash ./scripts/Former_forecast/WindPower_script/Informer_WindPower.sh
bash ./scripts/Former_forecast/WindPower_script/Nonstationary_Transformer_WindPower.sh
bash ./scripts/Former_forecast/WindPower_script/TimesNet_WindPower.sh
bash ./scripts/Former_forecast/WindPower_script/TimeXer_WindPower.sh
```
or you can run the following commands to train the graph-based models:

```bash
bash ./scripts/Graph_forecast/GCN_STGraph.sh
bash ./scripts/Graph_forecast/GAT_STGraph.sh
```
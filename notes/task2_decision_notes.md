# Task 2: decision notes

These notes were moved out of the Task 2 notebook (now Section 3.2 of ev_charging_palo_alto.ipynb). The notes are not part of the submission.

## Notebook introduction

> **Decision notes (internal – remove before hand-in)**
>
> **How to read this notebook.** Normal markdown cells are draft report text. Quote blocks like this one are our own log of decisions, alternatives we rejected and reflections. They are for the group only and should be deleted (or moved to the report where useful) before hand-in, so they do not count towards the word limit.
>
> **What the assignment asks for (Task 2):**
> - Predict energy per station for each of the next 24 hours, made at the end of the previous day.
> - Beat a naive baseline that predicts the historical average for that weekday (we must implement it and compare).
> - No shuffling: first 70% of the data for training, the rest for testing.
> - One model for all stations or one per station is our choice.
>
> **Status:** baseline, features, two models and a first round of tuning of the boosting model are done.

## 3.2.1 Station-hour table

> **Decision notes (internal – remove before hand-in)**
>
> - **Why Marco's table (`introbid`) and not Dumitrita's:** it already has what forecasting needs. (1) Every station-hour is present, including hours with zero use; a table with only active hours would teach the model that demand is never zero. (2) Energy is spread over the *charging* time, not the whole plugged-in time, so a car that sits fully charged does not add energy to later hours. (3) Long silent gaps at busy stations are marked as outages (empty) instead of zero demand.
> - **The group has not yet agreed on one cleaning pipeline.** That is why the loading cell is isolated. If we switch, only that cell changes; the three `assert` lines check that the new table has the shape the rest of the notebook expects. All numbers below will change slightly.
> - **Why a cache file:** Marco's notebook does not save its tables, and we did not want to edit a teammate's notebook. `station_hours_pandas<version>.pkl` (41 MB) is in `.gitignore`. Risk: it goes stale if the cleaning changes, so delete it then. The pandas version is in the file name because a pickle saved by pandas 3 cannot be read by pandas 2 (e.g. Anaconda) and fails with a `StringDtype` error.
> - **Outage hours stay empty.** They are left out of the baseline averages, of model training and of scoring. We do not know the true demand in those hours, so counting them as zero would reward a model for predicting outages.
> - **We only forecast energy.** The table also has occupied minutes, which Task 2 does not ask for. It could become an extra feature later.

## 3.2.2 Chronological 70/30 split

> **Decision notes (internal – remove before hand-in)**
>
> - **"First 70% of the data" = 70% of the rows, not 70% of the calendar.** The data runs from July 2011 to January 2021, so 70% of the calendar would cut around March 2018. But the network grew, so later years hold more station-hours, and 70% of the rows lands on 23 April 2019. We read "data" as observations. Either reading is defensible; ours gives the model more history.
> - **One cut-off date for all stations, not 70% per station.** With a per-station split, the test period of an old station overlaps the training period of a newer one. A model trained on all stations together would then have seen city-wide events (such as the COVID drop) that lie in another station's "future". A single date rules that out.
> - **Price of that choice:** 13 Sherman stations opened after the cut-off and have no training data. The baseline cannot predict them, so they are not part of the comparison. We report them separately in 3.2.7.
> - **The test set contains COVID.** It runs April 2019 to January 2021, so almost half of it is after 13 March 2020, when demand collapsed. This matters a lot for the results; see 3.2.7.
> - **Cut at local midnight** so that every test day is a full day.
> - **The test set is only used for the final scores.** The first two models use fixed settings. The tuning in 3.2.6 uses a validation slice cut from the end of the *training* period.

## 3.2.3 Baseline

> **Decision notes (internal – remove before hand-in)**
>
> - **The baseline is exactly what the assignment describes:** the "average Monday" from the training set, per station. We average over the whole training period. Averaging only the most recent months would be a stronger baseline, but it is not the one we are asked to beat.
> - **Why two metrics:** MAE is the typical miss in kWh and is easy to explain. RMSE punishes large misses more, and large misses are what hurt when planning grid load. A model should win on both.
> - **Why not a percentage error (MAPE):** most station-hours have zero energy, and a percentage of zero is undefined.
> - **`evaluate` is the single scoring function** for the baseline and every model, and `scored` is the single set of test rows (in service, station has training history). Reusing both guarantees a like-for-like comparison.
> - **Observation:** the baseline is a smooth daily hill of about 4 kWh at the busiest station, while real hours jump between 0 and 8–11 kWh. An average cannot predict *which* hours are busy, only that daytime is busier. That is the room the models have to improve.

## 3.2.4 Features

> **Decision notes (internal – remove before hand-in)**
>
> - **The rule that prevents leakage:** for hour *h* of day *D*, nothing from day *D* may be used. The shortest lag is therefore "same hour yesterday" (1 day), which is known for all 24 target hours at midnight. A lag of "1 hour ago" would be known only for the first hour of the day, so we do not use hourly lags at all. An `assert` in the code checks the 1-day lag by hand.
> - **Why these lags:** 1 and 2 days capture what the station is doing right now; 7 and 14 days capture the same weekday; the two averages smooth out the noise of single hours (one car more or less changes an hour a lot); the day-level averages tell the model whether the station as a whole is busy or quiet lately.
> - **Using actual test values in the lags is not cheating.** When forecasting day *D* in the test period, days up to *D-1* have really happened. Only the model's *parameters* are frozen at the cut-off.
> - **Why the baseline is a feature:** it summarises years of history for that station, weekday and hour in one number. It is computed from the training set only.
> - **No year or trend feature.** Trees cannot extrapolate beyond the years they have seen, and the lag averages already carry the current level.
> - **Daylight saving:** lags are looked up by local date and clock hour, not by shifting 24 rows. On the two clock-change days an hour is missing or doubled; the doubled hour is averaged and the missing one gives an empty lag.
> - **Empty lags** occur at the start of a station's life and around outages (about 0.1% of test rows). Gradient boosting handles empty values itself; for the linear model they are filled with the column average.
> - **Not used yet, candidates for later:** weather, occupied minutes, school holidays, activity at neighbouring stations.

## 3.2.5 Models

> **Decision notes (internal – remove before hand-in)**
>
> - **Why two models:** the grading asks whether we used one or several models and linear or non-linear ones. The linear model also answers a useful question: how much of the gain comes from simply having lag features, and how much from a more flexible model?
> - **One model for all stations, not one per station.** Stations share the same daily and weekly rhythm, so a shared model has far more data to learn it from. Quiet stations have too little data for their own model. And a shared model can forecast a station it has never seen, which per-station models cannot. Station differences are still captured through the `Station` feature and the per-station baseline.
> - **Why gradient boosting among the non-linear options:** it works well on mixed table data like ours, trains in seconds on a million rows, accepts empty lags and a category column directly, and needs no scaling. `HistGradientBoostingRegressor` is in scikit-learn, so no extra package.
> - **Linear model details:** station, hour, weekday and month are one-hot encoded (a separate shift for each value), because "hour 23" is not 23 times "hour 1". Empty lags are filled with the column average. Plain least squares, no regularisation, so there is nothing to tune.
> - **Squared-error loss on purpose.** Most hours are zero. A model trained on absolute error would learn to predict close to zero almost everywhere: good MAE, useless for the grid. Squared error targets the average, which is also what the baseline predicts.
> - **Boosting settings here are untuned:** learning rate 0.05, at most 300 trees, scikit-learn defaults otherwise. One of those defaults is early stopping on a *random* 10% of the training rows, which is why the cell prints how many trees were actually built. A random split is not ideal for time-ordered data, so the tuned model in 3.2.6 switches it off and uses a chronological validation slice instead. We keep this untuned model as the reference point for the tuning.
> - **Clipping at zero:** energy cannot be negative, and both models sometimes predict slightly below zero for quiet hours.

## 3.2.6 Tuning the boosting model

> **Decision notes (internal – remove before hand-in)**
>
> - **Validation slice = the last 20% of the training station-hours, in time.** It runs from 26 July 2018 to 22 April 2019, the day before the test set starts. That is about nine months, so May to July are not in it; a full year would have been nicer, but it would take 27% of the training rows away from fitting. A random or shuffled validation split would let the model see days on both sides of a validation day, which makes every setting look better than it is.
> - **A single validation slice, not rolling cross-validation.** Several time-ordered folds would give a more stable choice, but cost several times the run time for a small grid whose candidates turn out to be close together. Worth revisiting only if the choice looks unstable.
> - **The baseline feature is rebuilt for tuning.** During tuning the "average weekday" profile is computed from the fit part only. Otherwise the baseline feature would already contain the validation period and look more useful than it will be on the test set.
> - **What we tune and why:**
>   - *Loss:* squared error (what we had) against Poisson. Poisson is made for non-negative data with many zeros and a variance that grows with the level, which describes hourly energy well. Both predict the expected value, so they are comparable with the baseline.
>   - *Tree size (`max_leaf_nodes`):* 15, 31 (default) or 63 leaves. Larger trees can learn finer interactions but overfit more easily.
>   - *Minimum rows per leaf (`min_samples_leaf`):* 20 (default) or 200. A larger value stops the model from fitting single odd hours.
>   - *Number of trees:* not a grid dimension. Each candidate is built with up to 1,000 trees and we read off the validation error after every tree, then keep the best count. This costs nothing extra.
> - **What we do not tune:** the learning rate stays at 0.05. A lower rate with more trees gives nearly the same model, so tuning both wastes time. The feature set is also fixed; changing features is a bigger step than tuning.
> - **Chosen by validation RMSE.** RMSE matches what the models are trained to minimise and stresses the large misses that matter for the grid. Validation MAE is shown next to it to check that the two agree.
> - **The untuned settings are one of the candidates** (squared error, 31 leaves, 20 rows per leaf), so the winner can never be worse than them on the validation slice. Whether that carries over to the test set is a separate question, answered in 3.2.7.
> - **Refit on all training data.** After choosing, the model is refitted on the full training period (fit part + validation slice) with the chosen settings and tree count, so it has seen the most recent year before forecasting the test period.
> - **Risk to keep in mind:** the validation period (2018–19) is normal demand, while almost half of the test set is COVID. Settings that are best in a normal year need not be best when demand collapses.
> - **Outcome of the search:** the winner is squared error, 31 leaves, at least 200 rows per leaf, 198 trees. But the twelve candidates are almost indistinguishable: validation RMSE runs from 1.8801 to 1.8835, a spread of 0.2%. All six squared-error candidates rank above all six Poisson ones on RMSE, while Poisson is slightly better on MAE (about 1.194 against 1.202), so the two metrics do not agree on the loss. We stay with RMSE as decided beforehand. The practical message is that the model is not sensitive to these settings; the default was already close to the best. The search takes about two and a half minutes to run.

## 3.2.7 Results: error by hour of day

> **Decision notes (internal – remove before hand-in)**
>
> - **Why this breakdown:** one overall MAE hides when the forecasts are good or bad. The grid operator cares most about the daytime peak, so we want to know whether the models help there or only in the quiet hours.
> - **Why MAE and not RMSE here:** MAE is in plain kWh and can be set directly next to the mean energy of the hour. RMSE by hour tells the same story; the day/night summary under the table shows both.
> - **Why the mean actual energy is drawn in:** an error of 0.3 kWh means something different at 3 am (mean 0.11) than at noon (mean 2.46). Without that line the night looks "solved" when it is not.
> - **Day/night cut at 8–19:** chosen by eye from the demand curve; those twelve hours hold 84% of the test energy. It is only a summary, not a modelling choice.
> - **What we see:**
>   - *The error follows the demand.* All forecasts are worst around noon (tuned boosting MAE 1.94 at 12, where the mean is 2.46) and best at 4 am (0.18, mean 0.10).
>   - *The models help in every hour.* Tuned boosting is 19–39% below the baseline in each of the 24 hours.
>   - *The gain is smallest where it matters most.* In the early afternoon (13–15) the improvement is about 19%; at night and in the morning ramp (0, 7) it is close to 39%. Over hours 8–19 the MAE falls from 2.23 to 1.71 (−23%), over the other hours from 0.77 to 0.52 (−32%).
>   - *At night the error is larger than the demand itself.* Between midnight and 5 am the MAE is roughly twice the mean energy. Night charging is a rare event: the model predicts a small amount every night and is wrong either way. In daytime the MAE is somewhat below the mean.
>   - *Boosting beats the linear model mainly in the morning ramp and around midday* (hour 7: 0.70 against 0.76). From 21 to 23 the two are equal. So the non-linear model earns its small advantage when demand is rising, which is plausibly where interactions (weekday × hour × recent level) matter.
> - **For the report:** the honest statement is that the daytime peak remains hard to predict hour by hour for a single station. The stronger practical message comes from summing over all stations per hour, which is done in the next part.

## 3.2.7 Results: error for all stations together

> **Decision notes (internal – remove before hand-in)**
>
> - **Why this view:** the assignment says the forecast is needed "to manage the energy grid properly". The grid carries the sum of the stations, so the error of the total is the number that matters in practice. The per-station scores above remain our official comparison, because Task 2 asks for a forecast per station.
> - **No new model.** We simply add up the per-station forecasts we already have, hour by hour, and add up the actual energy the same way. Only the scored rows are summed (station in service and with a training history), so the actual total and the forecast total always cover exactly the same stations: on average 30.5 per hour. This is the total of the *scored* stations, not of the whole network; the 13 new Sherman stations are not in it.
> - **"MAE / mean energy"** is our way to compare the two levels. A MAE of 10 kWh for the total and 1.1 kWh for one station cannot be compared directly; dividing each by the mean energy at its own level can.
> - **Why the same splits as before (COVID, hours 8–19):** so the city-wide numbers can be read against the per-station ones.
> - **What we see:**
>   - *Summing makes the forecasts far more useful.* For a single station the tuned model's MAE is as large as the mean energy (100%). For the total it is 31% over the whole test period, and 22% before COVID. In the daytime peak before COVID it is 18% (MAE 15.7 kWh on a mean of 87 kWh per hour).
>   - *The advantage over the baseline is larger for the total than per station.* Over the whole test period the tuned model more than halves the baseline's error (MAE 10.5 against 22.8, −54%; RMSE 14.8 against 32.0, −54%). Per station it was −25% / −16%.
>   - *The reason:* the part of the error that is pure chance (does a car show up this hour at this station?) partly cancels when stations are summed. What does not cancel is being wrong about the general level, and that is exactly what the lag features fix and the baseline cannot.
>   - *COVID is again most of it, but not all.* From 13 March 2020 the baseline misses the total by 33 kWh per hour on a mean of 14, more than twice the demand; the tuned model by 9.7. Before COVID the tuned model is still 17.5% better on MAE (11.3 against 13.7) and 18% on RMSE (15.5 against 18.9). That is roughly double the pre-COVID gain per station (9.4% / 5.2%), and it is the fairest "normal times" number we have.
>   - *Tuning makes no difference here before COVID* (11.30 against 11.26) and helps a little after (9.7 against 10.3), the same pattern as per station.
>   - *The errors do not cancel completely.* If station errors were independent, the RMSE of a sum of about 30 stations would be about √30 ≈ 5.5 times the per-station RMSE, so roughly 11 kWh before COVID. We measure 15.5. So stations tend to be wrong in the same direction in the same hour: something city-wide (weather, events, school holidays) moves them together. This is a rough check, since stations differ in size, but it points to where new features could help.
>   - *The plot of the first test week is a sober reminder:* in a normal week the three forecasts are nearly on top of each other and all of them miss the tallest midday peaks (actual 150–165 kWh, forecast about 120).
> - **For the report:** a good way to present Task 2 is in two steps. Per station, hour by hour, demand is mostly chance and any forecast is rough. For the network as a whole, our model predicts the hourly load within about a fifth in normal times and stays usable when demand shifts, where the average-weekday baseline does not.
> - **Not done:** a model trained directly on the city total might do better still for this purpose, but it would not answer the task (a forecast per station).

## 3.2.7 Results: feature importance

> **Decision notes (internal – remove before hand-in)**
>
> - **Why permutation importance:** the scikit-learn boosting model we use has no built-in importance scores, and permutation importance has a meaning that is easy to explain: "how much worse does the forecast get if this column is scrambled?" It is measured in the same unit as our error (kWh RMSE).
> - **Measured on the test rows, on purpose.** We want to know what the model relies on when forecasting unseen days. This is a diagnostic only: nothing is selected or changed based on it, so it does not leak the test set into the model.
> - **A sample of 100,000 test rows and 5 repeats** keeps the cell fast. The spread over repeats is tiny (the error bars are barely visible), so the ranking is stable.
> - **RMSE as the score** because it is what the model was trained and tuned on. Predictions are not clipped at zero inside this calculation, which makes no visible difference.
> - **What we see:**
>   - *The baseline feature dominates.* Scrambling it raises RMSE by 0.30 kWh, more than three times the next feature. The model really does work as intended: start from the average weekday profile, then correct it.
>   - *The corrections come from the averages, not from single days.* `Hour_Mean_7d` (0.084), `Hour_Mean_4w` (0.044) and `Day_Mean_7d` (0.043) carry the recent level. The single-day lags (`Lag_1d`, `Lag_2d`, `Lag_7d`, `Lag_14d`) are close to zero.
>   - *Holidays matter more than expected* (0.014) for a flag that is on for under 3% of the rows: demand on a federal holiday looks nothing like a normal weekday.
>   - *Hour, weekday and month look unimportant* (all below 0.007), and so does `Station` relative to the baseline (0.027).
> - **How to read this without over-reading it:**
>   - *Low importance does not mean useless information.* Hour and weekday are already inside the baseline feature, which is built per station, weekday and hour. Shuffling `Hour_Of_Day` alone leaves that information intact, so the model hardly notices. The same goes for the single lags: each one is also part of an average (correlation with `Hour_Mean_7d` is about 0.68), so the model can fall back on the average when one lag is scrambled.
>   - Permutation importance therefore understates features that have close substitutes. The correlation table under the plot is there to make that visible.
>   - It shows what *this* model uses, not what drives charging demand.
> - **What we could do with it (not done):** dropping the four single-day lags would probably cost almost nothing and give a simpler model with 10 features. We keep them for now, because the feature set was fixed before tuning and removing features after looking at test-set importance would be a form of tuning on the test set. If we want the simpler model, the fair way is to compare both versions on the validation slice.

## 3.2.7 Results: reflections and open points

> **Decision notes (internal – remove before hand-in)**
>
> **Reflections on the results (numbers from the run with Marco's cleaning)**
>
> - **All three models beat the baseline on both metrics, at all 33 scored stations.** Baseline MAE 1.498 / RMSE 2.150. Linear 1.156 / 1.831 (−22.8% / −14.8%). Boosting 1.124 / 1.815 (−25.0% / −15.6%). Tuned boosting 1.117 / 1.806 (−25.5% / −16.0%). The requirement of the assignment is met.
> - **Tuning helped, but only marginally.** The tuned model improves on the untuned one by 0.007 kWh MAE and 0.010 kWh RMSE, about half a percent. Before COVID the two are identical (MAE 1.317 against 1.315, RMSE 2.028 for both); the whole difference comes from the COVID period (MAE 0.859 against 0.878). On the validation slice the candidates differed by only 0.2%, so a test gain of this size is not something the validation predicted, and part of it may be chance. We should present the tuned model as our final model because it was chosen by a clean procedure, not because tuning made a real difference.
> - **What actually changed between the two boosting models:** 198 trees instead of 300, at least 200 rows per leaf instead of 20, and no internal random hold-out (the untuned model set 10% of the training rows aside for early stopping, which never triggered). A smaller, more cautious model doing slightly better in the COVID period fits the idea that it leans less on fine patterns from normal years.
> - **Most of the gain over the baseline is COVID, and we should say so in the report.** From 13 March 2020 demand fell to about a third (mean 0.50 kWh per station-hour, against 1.59 before). The baseline keeps predicting pre-pandemic levels; the lags let the models follow the drop within days. In that period tuned boosting cuts MAE by 45% (1.555 to 0.859). Before COVID the gain is much smaller: MAE −9.4% (1.454 to 1.317) and RMSE −5.2% (2.139 to 2.028). The honest summary is "about 10% better in normal times, and far more robust when demand shifts".
> - **Features matter more than the model type or its settings.** The linear model already gets most of the improvement; boosting adds about 2–3 percentage points on MAE, and tuning adds half a point more. Further gains are more likely to come from new information (weather, occupancy, neighbouring stations) than from more tuning.
> - **Why the errors stay high:** before COVID the MAE (1.3) is almost as large as the mean energy (1.6). Energy in a single station-hour depends on whether one or two cars happen to plug in, which is close to random at day-ahead horizon. In the weekly plot the models predict a smooth daytime hill, while reality jumps between 0 and 8–11 kWh. The models get the *level* right (they sit above the baseline in the first test week, because the station had become busier than its long-run average), not the individual spikes. Summed over all stations the picture is much better: the tuned model's error is 31% of the mean total (22% before COVID) and less than half of the baseline's. See "Error for all stations together".
> - **New stations: no real skill.** On the 13 Sherman stations tuned boosting scores MAE 0.313 / RMSE 1.077, but simply predicting zero scores 0.240 / 1.093, because these stations were barely used (mean 0.24 kWh). The linear model is clearly worse (MAE 0.727): with no baseline value for these stations it falls back on the average baseline of all stations. We should not claim that the models handle new stations well, only that they can produce a forecast for them.
> - **The largest relative gains are at the three quietest stations** (MPL #1–#3, around −60 to −70% MAE), where the baseline still predicts activity that has stopped.
>
> **Open points**
> - Decide with the group whether to report results with and without the COVID period, or to discuss COVID only in the text.
> - Re-run everything once the group has settled on one cleaning pipeline (the tuning cell takes about two and a half minutes).
> - If we want more accuracy, add features rather than widen the tuning grid.
> - Possible simplification: drop the four single-day lags (see the feature-importance notes), checked on the validation slice.
> - Station errors move together (see the city-wide notes), so city-wide features such as weather are the most promising addition.

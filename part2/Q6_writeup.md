# Q6. Results and error analysis

## Quantitative results

Table 4: Development and test results (×100). "Full model" is the configuration in Q5 (fine-tuned
`google-t5/t5-small`, frozen shared embeddings, greedy decoding); the best checkpoint by dev
Record F1 was reached after epoch 17 (0-indexed) of 20.

| System | Query EM | F1 score |
|---|---|---|
| **Dev Results** | | |
| T5 fine-tuned — Full model | 72.53 | 87.79 |
| **Test Results** | | |
| T5 fine-tuning | XX.XX | XX.XX |

> Fill in the test row from the Gradescope leaderboard.

Dev learning curve of the full model (checkpoint selection by Record F1):

| Epoch | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | **17** |
|---|---|---|---|---|---|---|---|---|---|---|
| Record F1 | 30.80 | 63.53 | 71.79 | 77.47 | 77.59 | 83.18 | 84.89 | 86.10 | 87.56 | **87.79** |
| SQL EM | 13.52 | 40.77 | 54.51 | 61.80 | 63.95 | 68.88 | 69.74 | 70.82 | 71.67 | **72.53** |
| SQL error rate (%) | 41.63 | 16.09 | 15.88 | 12.66 | 11.37 | 10.94 | 8.15 | 6.22 | 5.36 | **5.79** |

## Qualitative error analysis

Produced with `part2/error_analysis.py` on the dev predictions of the full model
(`results/t5_ft_experiment_dev.sql`). Of the 466 dev queries, 338 are exact SQL matches,
72 return records different from the ground truth (or fail to execute), and 56 differ from the
gold SQL but return identical records. Each of the 72 incorrect queries is assigned to the first
matching class in the order below (execution error → SELECT clause → set of tables → set of
constants → remaining condition structure).

Table 5: Error analysis on the dev set.

| Error type | Example of error | Error description | Statistics |
|---|---|---|---|
| **Syntax error: unbalanced parentheses** | NL: *"list all arrivals from any airport to baltimore on thursday morning arriving before 9am"*. Prediction ends with `... AND( flight_1.arrival_time < flight_1.departure_time AND 1 = 1 ) ) ) ) )` → `OperationalError: near ")": syntax error` | The model generates a long query with deeply nested `AND( ... OR( ... ) )` blocks and loses track of the nesting depth, emitting one or more closing parentheses too many (or too few). 14 of the 15 cases have an unequal number of `(` and `)`. These are the longest queries in the data (60–130 SQL words); the decoder has no explicit notion of bracket balance and the nested time/day conditions are rare in training. | 15/466 |
| **Undeclared / ambiguous table alias (schema linking)** | NL: *"what aircraft is co 1209"*. Gold joins `aircraft`, `equipment_sequence`, `flight`, … Prediction: `SELECT DISTINCT aircraft_1.aircraft_code FROM aircraft aircraft_1 WHERE aircraft_1.aircraft_code = 'CO' AND flight_1.flight_number = 1209` → `no such column: flight_1.flight_number` | The WHERE clause refers to an alias (`flight_1`, `fare_basis_1`, `airport_service_2`, …) that was never declared in the FROM clause, or declares the same alias twice (ambiguous column). The model generates the FROM list and the WHERE conditions independently token by token and does not see the database schema, so the two parts can become inconsistent, especially for rare question types that need long join chains. | 11/466 |
| **Wrong constant values** | NL: *"please list nonstop flights from las vegas to new york on america west"*. Gold: `flight_1.airline_code = 'HP'`; prediction: `flight_1.airline_code = 'AA'`. NL: *"us air flights departing from charlotte around 1pm"*. Gold: `departure_time >= 1230 AND departure_time <= 1330`; prediction: `>= 1130 AND <= 1330`. | The query structure is correct, but a literal is wrong: airline codes whose names are not lexically related to the code (America West → `HP`, guessed `AA` from "America"), airport codes (`'O'` instead of `'MCO'` for "orlando international"), ground-transport types ("train" → `'RENTAL CAR'` instead of `'RAPID TRANSIT'`), and especially time values for vague expressions ("around 4 o'clock", "after 5", "early afternoon", "around 9 o'clock" at night), where the ATIS annotation conventions (±30 min windows, 12- vs 24-hour time) must be memorized from few examples. | 19/466 |
| **Missing / extra tables and joins** | NL: *"show me the flights to and from love field"*. Gold: `... airport_1.airport_code = 'DAL' OR flight_1.from_airport = airport_2.airport_code AND airport_2.airport_code = 'DAL'`; prediction only keeps the "to" direction and drops `airport airport_2`. NL: *"flights from dallas to baltimore on american and delta"*: prediction keeps only `'DL'`, drops `'AA'` and the Fort Worth airport join. | The model omits part of a compositional request (one of two directions, one of two airlines, the return leg, the aircraft/equipment join) or adds an unnecessary join (e.g. a `flight_stop` table). These questions combine several constraints with "and/or", and the model tends to fall back to the most frequent simpler template seen in training. | 12/466 |
| **Wrong SELECT columns** | NL: *"define airline us"*. Gold: `SELECT DISTINCT airline_1.airline_code , airline_1.airline_name , airline_1.note FROM airline ...`; prediction: `SELECT DISTINCT airline_1.airline_code FROM airline ...`. NL: *"what does iah mean"*: prediction selects from `airline` instead of `airport`. | The WHERE part is mostly right, but the projected columns (or the entity the question asks about) differ: missing extra columns such as `ground_fare` or `airline_name`, selecting `aircraft_code` instead of `flight_id`, or confusing airport vs. airline codes for short abbreviations. The annotation of what to return for vague requests ("show me about", "define", "what does X mean") is inconsistent in ATIS, so the model picks the most common projection. | 11/466 |
| Other: wrong condition structure / timeout | NL: *"i need to reverse the flight from pittsburgh to denver please"*: prediction keeps Pittsburgh → Denver instead of Denver → Pittsburgh. | Same tables and constants as the gold query, but conditions are attached to the wrong column (origin/destination swapped, wrong comparison), plus one generated query for a round-trip question (*"flights from boston to atlanta and the return flights"*) that exceeds the 120 s execution limit. | 4/466 |

**Remarks.**
- 56/466 predictions are not exact SQL matches but return exactly the same records, e.g. `departure_time <= 1000` instead of `( departure_time >= 0 AND departure_time <= 1000 )`, or reordered AND-conditions. This is why SQL EM (72.53) is much lower than Record EM (86.48).
- Some errors stem from label noise in the data: 5 dev and 29 train gold queries are a placeholder (`SELECT state_1.state_code FROM state state_1 WHERE state_1.state_code = 'ASD';`). On 2 of these dev examples the model produced a reasonable flight query and is counted as wrong; conversely, it sometimes generates this placeholder for unusual questions (e.g. *"what is ord"*).

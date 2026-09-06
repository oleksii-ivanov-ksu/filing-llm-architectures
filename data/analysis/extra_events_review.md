# Review of extra events (grounded but not in gold standard)

For each model: one blind-tier document, its events that did NOT match any
gold event. All are grounded (a verbatim quote was found in the filing), so
they are real facts — the question is only whether the human annotator
considered them key. Confirms extras are not fabrications.


## anthropic_claude-haiku-4.5
**Document:** AME_10-Q_2007 — model extracted 6 events, 2 matched gold, **4 extra** (gold has 8 events total)

- **[acquisition]** Pittman business acquired in May 2006, contributing to sales and operating income growth in Q1 2007.
  - quote: "acquisitions of the Pittman business acquired in May 2006"
- **[acquisition]** Land Instruments business acquired in June 2006, contributing to Electronic Instruments Group sales growth.
  - quote: "the Land Instruments business acquired in June 2006"
- **[acquisition]** Precitech business acquired in November 2006, contributing to Electronic Instruments Group growth.
  - quote: "the Precitech business acquired in November 2006"
- **[acquisition]** Southern Aeroparts acquisition completed in December 2006, contributing to Electromechanical Group growth.
  - quote: "the Southern Aeroparts acquisition in December 2006"

## deepseek_deepseek-chat-v3.1
**Document:** AME_10-Q_2007 — model extracted 6 events, 2 matched gold, **4 extra** (gold has 8 events total)

- **[acquisition]** Acquired the Pittman business in May 2006.
  - quote: "the acquisitions of the Pittman business acquired in May 2006"
- **[acquisition]** Acquired the Land Instruments business in June 2006.
  - quote: "the Land Instruments business acquired in June 2006"
- **[acquisition]** Acquired the Precitech business in November 2006.
  - quote: "the Precitech business acquired in November 2006"
- **[acquisition]** Acquired the Southern Aeroparts business in December 2006.
  - quote: "the Southern Aeroparts acquisition in December 2006"

## google_gemini-2.5-flash
**Document:** BIIB_10-Q_2011 — model extracted 10 events, 1 matched gold, **9 extra** (gold has 7 events total)

- **[restructuring]** In November 2010, the company announced strategic, operational, and organizational initiatives, including out-licensing/terminating certain R&D programs, a 13% workforce reduction, and facility consolidation. Total charges are expected to be approximately $100.0 million, with $16.6 million incurred in the six months ended June 30, 2011.
  - quote: "In November 2010, we announced a number of strategic, operational and organizational initiatives, which are described below under the heading “Restructuring Charge.”"
- **[product_launch]** On July 20, 2011, the European Commission granted a conditional marketing authorization for FAMPYRA in the E.U. for improving walking ability in people with MS, triggering a $25.0 million milestone payment to Acorda Therapeutics, Inc.
  - quote: "On July 20, 2011, the European Commission (EC) granted a conditional marketing authorisation for FAMPYRA in the E.U., which triggered a $25.0 million milestone payment payable to A"
- **[product_development]** In April 2011, positive results were announced from the DEFINE Phase 3 clinical trial for BG-12 as a monotherapy in relapsing-remitting multiple sclerosis (RRMS), meeting primary and secondary endpoints with a favorable safety profile.
  - quote: "In April 2011, we announced positive results from DEFINE, the first of two pivotal Phase 3 clinical trials designed to evaluate our investigational oral compound BG-12"
- **[product_approval]** In April 2011, the FDA approved RITUXAN, in combination with corticosteroids, as a new medicine for adults with Wegener’s Granulomatosis (WG) and Microscopic Polyangiitis (MPA).
  - quote: "In April 2011, the FDA approved RITUXAN, in combination with corticosteroids, as a new medicine for adults with Wegener’s Granulomatosis (WG) and Microscopic Polyangiitis (MPA)."
- **[product_approval]** In June 2011, the EMA approved the renewal of TYSABRI’s marketing authorization in the E.U.
  - quote: "In June 2011, the EMA approved the renewal of TYSABRI’s marketing authorization in the E.U."
- **[partnership]** In the fourth quarter of 2010, as part of restructuring, the company eliminated its RITUXAN oncology and rheumatology sales force, with Genentech assuming responsibility for U.S. sales and marketing efforts.
  - quote: "In the fourth quarter of 2010, as part of our restructuring initiative, which is described below under the heading “Restructuring Charge,” we and Genentech made an operational deci"
- **[milestone_payment]** In March 2011, the first patient was dosed in a registrational study for dexpramipexole (ALS treatment), triggering a $10.0 million payment to Knopp Neurosciences, Inc.
  - quote: "In March 2011, we dosed the first patient in a registrational study for dexpramipexole, in development for amyotrophic lateral sclerosis (ALS). The achievement of this milestone re"
- **[milestone_payment]** In April 2011, an Investigational New Drug application was submitted for BIIB037 (Alzheimer’s treatment), triggering a $15.0 million milestone payment to Neurimmune SubOne AG.
  - quote: "In April 2011, we submitted an Investigational New Drug application for BIIB037 (human anti-Amyloid β mAb) a beta-amyloid removal therapy, which triggered a $15.0 million milestone"
- **[real_estate]** In July 2011, the company executed leases for two office buildings in Cambridge, Massachusetts, totaling approximately 500,000 square feet, to serve as future corporate headquarters and commercial operations, with planned occupancy in the second half of 2013.
  - quote: "In July 2011, we executed leases for two office buildings to be built in Cambridge, Massachusetts."

## google_gemini-2.5-flash-lite
**Document:** CMS_10-Q_2020 — model extracted 8 events, 3 matched gold, **5 extra** (gold has 13 events total)

- **[litigation]** CMS Energy, Consumers, and certain affiliates are parties to various lawsuits and regulatory matters in the ordinary course of business.
  - quote: "CMS Energy, Consumers, and certain of their affiliates are parties to various lawsuits and regulatory matters in the ordinary course of business."
- **[regulatory]** In December 2019, Consumers filed an application for an annual rate increase of $245 million, reduced to $229 million in May 2020.
  - quote: "In December 2019, Consumers filed an application with the MPSC seeking an annual rate increase of $245 million, based on a 10.5 percent authorized return on equity."
- **[regulatory]** In February 2020, Consumers filed an application for an annual rate increase of $244 million, reduced to $230 million in July 2020.
  - quote: "In February 2020, Consumers filed an application with the MPSC seeking an annual rate increase of $244 million, based on a 10.5 percent authorized return on equity."
- **[regulatory]** Consumers filed a depreciation case in July 2020 related to Ludington, requesting to increase depreciation expense by $17 million annually.
  - quote: "In July 2020, Consumers filed a depreciation case related to Ludington, requesting to increase depreciation expense, and its recovery of that expense, by $17 million annually."
- **[labor]** Consumers announced a retention incentive program in October 2019 for the D.E. Karn generating complex, with an estimated aggregate cost of $35 million through 2023.
  - quote: "In October 2019, Consumers announced a retention incentive program to ensure necessary staffing at the D.E. Karn generating complex through the anticipated retirement of the coal-f"

## meta-llama_llama-4-maverick
**Document:** EOG_10-Q_2019 — model extracted 2 events, 0 matched gold, **2 extra** (gold has 8 events total)

- **[operational_update]** EOG plans to drill three net wells in Trinidad in 2019 and complete one drilled uncompleted well from the 2018 drilling program in the Sichuan Basin, China.
  - quote: "In 2019, EOG plans to drill three net wells."
- **[financial_update]** Total anticipated 2019 capital expenditures are estimated to range from approximately $6.1 billion to $6.5 billion, excluding acquisitions and non-cash transactions.
  - quote: "Total anticipated 2019 capital expenditures are estimated to range from approximately $6.1 billion to $6.5 billion"

## openai_gpt-4o-mini
**Document:** BIIB_10-Q_2011 — model extracted 3 events, 0 matched gold, **3 extra** (gold has 7 events total)

- **[litigation]** Genentech's ongoing arbitration with Hoechst regarding RITUXAN could lead to significant financial liabilities.
  - quote: "We expect that certain damages that may be awarded to Hoechst will be a cost charged to our collaboration with Genentech."
- **[restructuring]** Biogen Idec announced a restructuring initiative expected to incur charges of approximately $100 million.
  - quote: "We expect to incur total restructuring charges of approximately $100.0 million associated with the implementation of these initiatives."
- **[product_launch]** FAMPYRA received conditional marketing authorization in the EU, triggering a $25 million milestone payment.
  - quote: "On July 20, 2011, the European Commission (EC) granted a conditional marketing authorisation for FAMPYRA in the E.U., which triggered a $25.0 million milestone payment."

## qwen_qwen3.8-27b
**Document:** EOG_10-Q_2019 — model extracted 3 events, 1 matched gold, **2 extra** (gold has 8 events total)

- **[investment]** EOG budgeted approximately $6.1 billion to $6.5 billion for exploration and development and other property, plant and equipment expenditures for the year 2019, excluding acquisitions and non-cash transactions.
  - quote: "Total anticipated 2019 capital expenditures are estimated to range from approximately $6.1 billion to $6.5 billion, excluding acquisitions and non-cash transactions."
- **[restructuring]** Effective January 1, 2019, EOG adopted the provisions of Accounting Standards Update (ASU) 2016-02, 'Leases (Topic 842).'
  - quote: "Effective January 1, 2019, EOG adopted the provisions of Accounting Standards Update (ASU) 2016-02, "Leases (Topic 842)" (ASU 2016-02)."

## z-ai_glm-5.3-flash
**Document:** AME_10-Q_2007 — model extracted 6 events, 2 matched gold, **4 extra** (gold has 8 events total)

- **[acquisition]** Acquisition of the Pittman business in May 2006, which contributed to Q1 2007 sales growth and record order input.
  - quote: "the Pittman business acquired in May 2006"
- **[acquisition]** Acquisition of the Land Instruments business in June 2006, which contributed to Q1 2007 sales growth, including international sales growth.
  - quote: "the Land Instruments business acquired in June 2006"
- **[acquisition]** Acquisition of the Precitech business in November 2006, contributing to Electronic Instruments Group sales growth.
  - quote: "the Precitech business acquired in November 2006"
- **[acquisition]** Acquisition of Southern Aeroparts in December 2006, contributing to Electromechanical Group sales growth.
  - quote: "the Southern Aeroparts acquisition in December 2006"

---

## Ручна оцінка (annotator review, 2026-08-31)

**Головний висновок: жодна "extra" подія не є вигаданою.** Усі мають дослівну
цитату, знайдену в документі (grounding пройдено). Тобто це РЕАЛЬНІ факти, яких
просто немає в gold standard. Розбір за категоріями:

### Категорія A — реальні значущі події, які АНОТАТОР ПРОПУСТИВ
Це вказує, що gold не є повністю вичерпним навіть на blind-ярусі.
- **gemini-2.5-flash / BIIB_10-Q_2011 (9 extra):** схвалення FAMPYRA та RITUXAN
  (з датами й сумами milestone), результати Phase 3 DEFINE, поновлення TYSABRI,
  реструктуризація на $100 млн, платежі $10/15 млн, оренда двох будівель у
  Кембриджі. Усі — конкретні, матеріальні, з датами й сумами. Це справжні ключові
  події, яких немає в gold (у gold лише 7). Модель знайшла БІЛЬШЕ, ніж людина.
- **gemini-2.5-flash-lite / CMS_10-Q_2020 (5 extra):** заявки на підвищення тарифів
  ($245 млн, $244 млн), depreciation case ($17 млн/рік), програма утримання
  персоналу ($35 млн). Реальні регуляторні/фінансові події, не розмічені людиною.

### Категорія B — реальні, але спірні як "ключові події"
Це over-extraction менш значущого або forward-looking матеріалу.
- **claude-haiku / deepseek — AME_10-Q_2007 (по 4 extra):** чотири придбання
  (Pittman, Land Instruments, Precitech, Southern Aeroparts). Реальні, але сталися
  у 2006 р. і згадані в Q1-2007 лише як контекст зростання — анотатор слушно не
  вважав їх подіями ЦЬОГО кварталу. Обидві моделі витягли однаково → систематична,
  а не випадкова поведінка.
- **llama / qwen — EOG_10-Q_2019:** плани буріння та капітальні витрати
  ($6.1–6.5 млрд). Це forward-looking guidance, який за нашими ж guidelines НЕ є
  ключовою подією. Тобто ці extra коректно відсутні в gold.
- **gpt-4o-mini / BIIB_10-Q_2011:** реструктуризація й FAMPYRA — реальні; арбітраж
  Hoechst сформульовано частково як очікування (гранична релевантність).

### Наслідки для метрик (важливо для статті)
1. **Grounding підтверджено як захист від галюцинацій:** 0 вигаданих серед усіх
   переглянутих extra.
2. **Recall і precision проти gold — це НИЖНІ межі**, а не точні оцінки: частина
   "extra" (категорія A) — реальні події, пропущені анотатором, тому справжня
   повнота моделей вища за виміряну, а справжня precision — теж (частина "хибних
   спрацювань" насправді правдиві).
3. **Тщательні моделі (gemini) знаходять події поза gold** — і корисні (кат. A),
   і надлишкові (кат. B). Розрізнення потребує ручної експертизи, але НЕ через
   вигадки — лише через межу "що вважати ключовим".

---

## Прийнята інтерпретація метрик (framing для статті, 2026-08-31)

**Gold standard = ВАЖЛИВІ події, відібрані анотатором як ключові.** Модель витягує
будь-які події документа. Тому метрики читаються так:

- **recall** = яку частку *відібраних як ключові* подій знайшла модель. Це головна
  метрика повноти щодо важливого, а не щодо всіх подій усесвіту.
- **"extra" події** = додаткові РЕАЛЬНІ події (grounding пройдено), які анотатор не
  відібрав як ключові. Це **не помилки і не галюцинації** — це матеріал нижчої/іншої
  пріоритетності. Вони вимірюють "багатослівність/селективність" моделі, а не хибні
  спрацювання. Тому класична "precision проти gold" тут не карає модель за правду —
  її слід трактувати як міру селективності, а не точності.
- **grounding** = захист від вигадок (частка фактів із підтвердженою цитатою). Саме
  grounding, а не gold-precision, є справжньою метрикою "невигаданості".

**Практичний наслідок:** для вибору моделі головні дві осі — (1) **recall важливих
подій** (чим вище, тим краще покриття ключового) і (2) **grounding** (чим вище, тим
менше вигадок). "Extra" радше інформативні (селективність), ніж штрафні.

**Застереження про послідовність gold:** рамка коректна за умови, що критерій
важливості застосовано послідовно. Приклад BIIB показує межу: модель знайшла реальні
матеріальні події (схвалення ліків, milestone), не відібрані в gold — тож recall слід
формулювати як "покриття відібраних ключових подій", не абсолютизуючи gold як повний
перелік усього важливого. Це чесно фіксуємо в Limitations.

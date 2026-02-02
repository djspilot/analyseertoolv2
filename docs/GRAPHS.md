# Grafieken Overzicht

Dit document geeft een compleet overzicht van alle beschikbare visualisaties in het AnalyseerTool project.

---

## Inhoudsopgave

- [Tijdsgebaseerde Grafieken](#tijdsgebaseerde-grafieken)
- [Categorische Grafieken](#categorische-grafieken)
- [Heatmaps](#heatmaps)
- [Flow & Deep Work Analyse](#flow--deep-work-analyse)
- [Patroon & Netwerk Visualisaties](#patroon--netwerk-visualisaties)
- [Geavanceerde Metrieken](#geavanceerde-metrieken)
- [Welzijn & Balans](#welzijn--balans)

---

## Tijdsgebaseerde Grafieken

### 1. Gantt Chart (Activity Timeline)
- **Type:** Timeline/Gantt chart
- **Functie:** `create_gantt_chart(df, date_filter)`
- **Beschrijving:** Toont activiteiten als tijdbalken over een dag, gekleurd per categorie
- **Features:**
  - Optionele datumfiltering
  - Hover informatie met duur en tijdstippen
  - Auto-sortering van categorieën
  - **Cached:** 60 minuten

### 2. Daily Breakdown Chart
- **Type:** Stacked bar chart
- **Functie:** `create_daily_breakdown(df, last_n_days, resample_rule)`
- **Beschrijving:** Toont dagelijkse/wekelijkse/maandelijkse uitsplitsing per categorie
- **Features:**
  - Flexibele resampling: 'D' (dagelijks), 'W' (wekelijks), 'M' (maandelijks)
  - Configureerbare tijdsperiode
  - **Cached:** 30 minuten

### 3. Hourly Profile (Circadian Rhythm)
- **Type:** Stacked area chart
- **Functie:** `create_hourly_profile(df)`
- **Beschrijving:** Toont gemiddelde tijd per activiteit per uur van de dag
- **Features:**
  - Toont circadian ritme patronen
  - Alle 24 uren zichtbaar
  - Interactieve hover informatie

### 4. Sleep Pattern Chart
- **Type:** Scatter plot met lijnen
- **Functie:** `create_sleep_pattern_chart(df)`
- **Beschrijving:** Toont afgeleide slaap- en waaktijden
- **Features:**
  - Identificeert vroegste starttijd (wake)
  - Identificeert laatste eindtijd (sleep start)
  - 24-uurs y-as

### 5. Circadian Plot
- **Type:** Scatter plot
- **Functie:** `create_circadian_plot(daily_metrics)`
- **Beschrijving:** Toont dag-start en dag-eind tijden over tijd
- **Features:**
  - Visueel van ritme-consistentie
  - Oranje = start, Paars = einde
  - Vaste 0-24 uur y-as

### 6. Streamgraph
- **Type:** Stacked area chart
- **Functie:** `create_streamgraph(df)`
- **Beschrijving:** Toont evolutie van activiteitsprioriteiten over tijd
- **Features:**
  - Centrum-baseline voor symmetrische visualisatie
  - Toont shifts in focusgebieden

---

## Categorische Grafieken

### 7. Category Pie Chart
- **Type:** Donut chart
- **Functie:** `create_category_pie(df)`
- **Beschrijving:** Toont tijdverdeling per categorie
- **Features:**
  - Hole-in-the-middle design (0.4)
  - Sorteert op duur
  - Hover met percentage en uren

### 8. Violin Plot
- **Type:** Violin plot met box plot overlays
- **Functie:** `create_violin_plot(df)`
- **Beschrijving:** Toont verdeling van sessieduur per activiteit
- **Features:**
  - Toont depth en variability van focus sessies
  - Box plot elementen (mediaan, kwartilen)
  - Outliers als punten
  - 90 min flow threshold lijn
  - Top 8 activiteiten

---

## Heatmaps

### 9. Activity Heatmap
- **Type:** Heatmap (weekday × hour)
- **Functie:** `create_activity_heatmap(df)`
- **Beschrijving:** Toont activiteitsintensiteit per uur en dag van de week
- **Features:**
  - Alle 24 uren vertegenwoordigd
  - Dagen gesorteerd (Mon-Sun)
  - Blauw kleurenschema

### 10. Weekly Heatmap
- **Type:** Heatmap (weekday × hour)
- **Functie:** `create_weekly_heatmap(df)`
- **Beschrijving:** Alternatieve heatmpa met Viridis kleurenschema
- **Features:**
  - Vergelijkbaar met activity heatmap
  - Andere kleurenscale
  - Niet alle uren verplicht

### 11. Consistency Heatmap
- **Type:** Single-row heatmap
- **Functie:** `create_consistency_heatmap(df)`
- **Beschrijving:** GitHub-style heatmap van Deep Work intensiteit
- **Features:**
  - Groen kleurenschema
  - Enkel deep work sessies
  - Toont consistente dagen

### 12. Peak Hours Chart
- **Type:** Heatmap (weekday × hour)
- **Functie:** `create_peak_hours_chart(df)`
- **Beschrijving:** Toont wanneer flow states het waarschijnlijkst zijn
- **Features:**
  - Combineert uur van dag met dag van week
  - Toont flow percentage per cel
  - Identificeert beste dag/uur combinatie
  - Blauw kleurenschema

---

## Flow & Deep Work Analyse

### 13. Deep Work Trend
- **Type:** Bar chart + Line chart (combo)
- **Functie:** `create_deep_work_trend(df, window)`
- **Beschrijving:** Toont Deep Work uren over tijd met moving average
- **Features:**
  - Dagelijkse waarden als bars
  - Rolling average als lijn
  - Configureerbaar window (standaard 7 dagen)
  - **Cached:** 15 minuten

### 14. Flow Sessions Timeline
- **Type:** Horizontal bar chart
- **Functie:** `create_flow_sessions_timeline(df)`
- **Beschrijving:** Toont deep work sessies met intensiteit
- **Features:**
  - Kleur gebaseerd op duur
  - Flow sessies (≥90min) gemarkeerd in donkerblauw
  - Shallow sessies in grijs
  - Opaciteit reflecteert duur

### 15. Session Length Distribution
- **Type:** Histogram
- **Functie:** `create_session_length_distribution(df)`
- **Beschrijving:** Histogram van sessieduren met 90min threshold
- **Features:**
  - Bins van 15 minuten
  - Dashed lijn op 90min
  - Labels voor "Shallow" en "Flow"

### 16. Flow Probability by Hour
- **Type:** Line chart met fill
- **Functie:** `create_flow_probability_by_hour(df)`
- **Beschrijving:** Toont % van deep work sessies per uur die ≥90min zijn
- **Features:**
  - Fill-to-zero
  - Semi-transparante vulling
  - Alle 24 uren
  - 0-100% y-as

### 17. Flow Streak Calendar
- **Type:** Heatmap calendar
- **Functie:** `create_flow_streak_calendar(df)`
- **Beschrijving:** Toont flow uren per dag als GitHub-style calendar
- **Features:**
  - Week op x-as, dag op y-as
  - Groen kleurenschema
  - Enkel sessies ≥90min

### 18. Flow vs Shallow Chart
- **Type:** Stacked bar chart
- **Functie:** `create_flow_vs_shallow_chart(df)`
- **Beschrijving:** Per dag: Flow vs Shallow tijd
- **Features:**
  - Blauw voor Flow (≥90min deep work)
  - Grijs voor Shallow (alles anders)
  - Duidelijke vergelijking

### 19. Flow Streak Chart
- **Type:** Bar chart + Line chart (combo)
- **Functie:** `create_flow_streak_chart(df)`
- **Beschrijving:** Gamification element - toont consecutieve dagen met ≥2h flow
- **Features:**
  - Flow uren als bars
  - Streak teller als lijn op secundaire as
  - Dubbele y-assen
  - Target lijn op 2 uur
  - Toont max streak in titel

---

## Patroon & Netwerk Visualisaties

### 20. Spiral Plot
- **Type:** Polar scatter plot
- **Functie:** `create_spiral_plot(df)`
- **Beschrijving:** Toont circadian ritme consistentie en drift in spiraalvorm
- **Features:**
  - Hoek (θ) = uur van de dag
  - Straal (r) = datum (binnenste = vroegste)
  - Grootte van punten = duur
  - Kleur = activiteit type

### 21. Chord Diagram (Sankey)
- **Type:** Sankey diagram
- **Functie:** `create_chord_diagram(df)`
- **Beschrijving:** Toont activiteit transities en context-switching patronen
- **Features:**
  - Nodes = activiteiten
  - Links = transities (dikte = frequentie)
  - Filtert zelfde-activiteit transities
  - Semi-transparante link kleuren

### 22. Network Graph
- **Type:** Network visualization
- **Functie:** `create_network_graph(df)`
- **Beschrijving:** Toont activiteit transitie patronen als netwerk
- **Features:**
  - Nodes = activiteiten (grootte = totale tijd)
  - Edges = transities (dikte = frequentie)
  - Circulaire lay-out
  - Hover met transitie details

### 23. Barcode Plot (Rug Plot)
- **Type:** Line/rug plot
- **Functie:** `create_barcode_plot(df)`
- **Beschrijving:** Toont dichtheid van sessie-starts
- **Features:**
  - Visualiseert gefragmenteerde vs geconsolideerde patronen
  - Laatste 14 dagen
  - Verticale streepjes = sessie starts
  - Kleur = activiteit type
  - Dynamische hoogte

---

## Geavanceerde Metrieken

### 24. Fragmentation Chart
- **Type:** Horizontal bar chart
- **Functie:** `create_fragmentation_chart(df)`
- **Beschrijving:** Toont Fragmentation Index per categorie
- **Features:**
  - FI = sessions / total hours
  - Hogere FI = meer gefragmenteerd
  - Gesorteerd op FI waarde
  - Hover met gemiddelde sessielengte en sessie count

### 25. Rose Chart (Daily Fingerprint)
- **Type:** Radial bar chart
- **Functie:** `create_rose_chart(df)`
- **Beschrijving:** Toont "vingerafdruk" van een gemiddelde dag
- **Features:**
  - 24 segmenten (één per uur)
  - Kleur = dominante activiteit
  - Lengte = gemiddelde uren
  - Gaps tussen segmenten

### 26. Energy Balance Chart
- **Type:** Stacked bar + Line chart
- **Functie:** `create_energy_balance_chart(df)`
- **Beschrijving:** Toont oplaad- vs ontladende activiteiten
- **Features:**
  - Positieve waarden = charging (groen)
  - Negatieve waarden = draining (rood)
  - Cumulatieve lijn op secundaire as
  - Dubbele y-assen

### 27. Productivity Pulse Chart
- **Type:** Bar chart + Line chart
- **Functie:** `create_productivity_pulse_chart(df)`
- **Beschrijving:** RescueTime-style productivity score over tijd
- **Features:**
  - Weighted average per dag
  - Kleur-coding: ≥50 (groen), ≥30 (oranje), <30 (rood)
  - 7-daagse moving average
  - Referentielijnen op 30 en 50

---

## Welzijn & Balans

### 28. Recovery Ratio Chart
- **Type:** Bar chart + Line chart
- **Functie:** `create_recovery_ratio_chart(df)`
- **Beschrijving:** Toont balans tussen recovery en drain activiteiten
- **Features:**
  - Ratio = Recovery Hours ÷ Drain Hours
  - Healthy ratio: 0.3-0.5
  - Kleur-coding: ≥0.3 (groen), ≥0.15 (oranje), <0.15 (rood)
  - 7-daagse moving average

### 29. Burnout Risk Chart
- **Type:** Gauge/Indicator
- **Functie:** `create_burnout_risk_chart(df)`
- **Beschrijving:** Burnout risico indicator gebaseerd op energie en recovery
- **Features:**
  - Risico score 0-100
  - Drie niveaus: LOW, MEDIUM, HIGH
  - Gebaseerd op:
    - Energie trend
    - Gemiddelde energie
    - Recovery ratio
    - Consecutieve negatieve dagen
  - Referentie op 35

### 30. Weekly Pattern Chart
- **Type:** Grouped bar chart
- **Functie:** `create_weekly_pattern_chart(df)`
- **Beschrijving:** Vergelijkt productiviteit per dag van de week
- **Features:**
  - Gemiddelde deep work per dag
  - Gemiddelde flow per dag
  - Identificeert beste dag
  - Normaliseert op aantal dagen in sample

---

## Configuratie & Weights

### Energy Weights
```python
ENERGY_WEIGHTS = {
    'Sport': 50,        # Charging
    'Yoga': 60,         # Charging
    'Walking': 40,      # Charging
    'Entertainment': 10,  # Charging
    'Read': 20,         # Charging
    'Music': 15,        # Charging
    'Work': -40,        # Draining
    'Coding': -50,      # Draining
    'Housework': -20,   # Draining
    'Internet': -10,    # Draining
    'Other': 0,         # Neutral
}
```

### Productivity Weights
```python
PRODUCTIVITY_WEIGHTS = {
    'Coding': 100,      # High productivity
    'Work': 75,
    'Read': 50,
    'Yoga': 40,
    'Sport': 40,
    'Walking': 30,
    'Music': 25,
    'Internet': 25,
    'Housework': 20,
    'Entertainment': 0,  # Low productivity
    'Other': 10,
}
```

---

## Categorieën

### Recovery Categories (Charging)
- Sport, Yoga, Walking, Entertainment, Read, Music

### Drain Categories (Draining)
- Work, Coding, Housework

### Deep Work Categories
- Work, Coding (als `is_deep_work` flag is gezet)

---

## Thresholds & Configuratie

- **Flow Threshold:** 90 minuten (1.5 uur)
- **Flow Streak Target:** 2 uur flow per dag
- **Healthy Recovery Ratio:** 0.3 - 0.5
- **Deep Work Threshold:** Geconfigureerd in `FLOW_MIN_DURATION_HOURS`
- **Gantt Chart Height:** `GANNT_CHART_HEIGHT`
- **Heatmap Height:** `HEATMAP_CHART_HEIGHT`
- **Default Chart Height:** `DEFAULT_CHART_HEIGHT`

---

## Caching

Sommige grafieken gebruiken caching voor betere performance:

| Grafiek | TTL | Status |
|---------|-----|--------|
| Gantt Chart | 60 min | Aan/uit via `ENABLE_CACHE` |
| Daily Breakdown | 30 min | Aan/uit via `ENABLE_CACHE` |
| Deep Work Trend | 15 min | Aan/uit via `ENABLE_CACHE` |

---

## Totaal

**Aantal Grafieken:** 31

Verdeeld over:
- Tijdsgebaseerd: 6
- Categorisch: 2
- Heatmaps: 4
- Flow & Deep Work: 7
- Patroon & Netwerk: 4
- Geavanceerde Metrieken: 4
- Welzijn & Balans: 3
- Special (Barcode): 1

---

## AI Insights

Het dashboard bevat nu een **AI Insights** panel dat automatisch natuurlijke taal-analyses genereert op basis van je activiteitsdata.

### Toegang
Klik op het ✨ (sparkles) icoon in de header om het insights panel te openen.

### Insight Categorieën

| Categorie | Beschrijving |
|-----------|-------------|
| **Flow** | Analyse van flow state bereik, sessieduur, en streaks |
| **Rhythm** | Identificatie van piek-uren en circadiaan ritme |
| **Energy** | Energie balans tussen opladen en ontladen activiteiten |
| **Patterns** | Fragmentatie analyse en weekdag patronen |
| **Wellness** | Slaap patronen en herstel analyse |
| **Trend** | Week-over-week vergelijkingen |
| **Recommendation** | Actiegerichte suggesties voor verbetering |

### Severity Levels

| Level | Kleur | Betekenis |
|-------|-------|-----------|
| Positive | 🟢 Groen | Goed presterende metric |
| Neutral | ⚪ Grijs | Informatief, geen actie nodig |
| Warning | 🟠 Oranje | Aandacht vereist |
| Negative | 🔴 Rood | Onmiddellijke actie aanbevolen |

### Voorbeelden van Insights

- **Strong Flow State**: "Excellent! 65% of your deep work time is in flow sessions (≥90min)"
- **Energy Deficit**: "Recovery ratio is only 0.12. Burnout risk is elevated."
- **Peak Focus**: "Your deep work peaks in the morning around 10:00"
- **5-Day Flow Streak**: "You maintained ≥2h of flow state for 5 consecutive days"

---

## Meer Informatie

Zie de broncode voor implementatiedetails:
- `src/visualization/charts.py` - Alle grafiekfuncties
- `src/config.py` - Kleuren, thresholds, en configuratie
- `src/visualization/utils/utils.py` - Helper functies
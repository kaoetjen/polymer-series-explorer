# Polymer Series Explorer v6

## New interactive series map

v6 replaces the one-series-at-a-time navigator with one interactive plot containing every detected homologous series.

### Core behavior

- Every connected line is one candidate homologous series.
- A homologous series is treated as one adduct form.
- The software does not mix H, Na, K, or NH4 members into the same homologous line.
- When multiple adduct forms of the same underlying polymer are supported, they remain separate series but are linked into the same polymer family.
- Linked adduct series share a family color and use different line patterns when their adduct identity is supported.
- Selecting one series emphasizes its companion adduct lines.
- Dotted connectors show the actual coeluting feature pairs used as evidence for the adduct relationship.

## Interaction

In **Interactive Series Map**:

1. All detected series are shown together.
2. Click a marker on a connected series line.
3. The detail panel on the right updates to that series.
4. The selected series is emphasized.
5. Any supported companion adduct series are highlighted and linked.
6. The right panel shows:
   - repeat unit
   - candidate adduct
   - number of members
   - mass error
   - linked adduct forms
   - individual series members

A dropdown remains available as a fallback for dense datasets.

## Downloads

You can download:

- the selected homologous series
- the selected polymer family including linked adduct forms
- all detected series at once
- the complete series summary
- the adduct relationship table

## Existing analysis retained

- Multiple repeat units can be searched simultaneously
- Custom repeat formulas are supported
- [M+H]+
- [M+Na]+
- [M+K]+
- [M+NH4]+
- [M-H]-
- All common adducts are enabled by default
- Exact mass and retention time are used to support companion adduct relationships
- Rerun without reloading the feature table
- Example data contains multiple polymer families and adduct forms

## Example data

Use:

`example_multiple_homologous_series_with_adducts.csv`

A useful first screen is:

- PEG / EO
- PPG / PO
- PS
- PMMA

Leave all adducts enabled.

## Windows

Double click:

`launch_windows.bat`

Or use Command Prompt:

```cmd
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run app.py
```

## Plot interaction note

The lines themselves represent the series assignment. Plotly selection is point based, so clicking a marker on the line selects the full homologous series. The entire line is then highlighted in the refreshed view.

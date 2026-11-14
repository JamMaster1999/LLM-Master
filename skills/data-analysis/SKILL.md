---
name: data-analysis
description: Analyzes data in the sandbox with pandas and draws clear charts with matplotlib, counting and computing with code rather than by eye. Use this skill whenever the user asks a question of a data file or a table, such as grades, quiz results, survey answers or attendance, wants statistics, a count, a comparison or a chart, or asks how many of something a text or file holds.
---

# Data analysis

## Count with code

Every number you report comes from code you ran: counts, sums, averages, percentages, word
counts, how many students did something. Reading a table and adding in your head gets numbers
wrong, and a teacher may act on them. Print the result and quote it.

## Workflow

1. **Profile the file first:** `python /skills/data-analysis/scripts/profile.py uploads/scores.csv`.
   It prints how it read the file (encoding, separator, title lines skipped), the size, and per
   column the type, missing values, distinct values and range, for CSV, Excel and OpenDocument
   sheets, and JSON. Read the file the same way. When names look wrong, such as "Muńoz", pass the
   right encoding with `--encoding`.
2. **Clean it, and say what you changed:** a header that isn't on the first row, totals rows mixed
   in with the data, duplicate students, names spelled two ways, numbers stored as text. Decide
   whether a blank means missing or zero, and say which.
3. **Analyze in a script** with pandas. Print compact results, such as `describe()`, a grouped
   table or the top rows, not whole data frames.
4. **Chart** when a picture says it faster than a table (below).
5. **Answer in words first**, with the key numbers, then the chart or table. Say how many rows the
   numbers cover.

## Charts

Start every chart with the house style, then save it as PNG:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("/skills/data-analysis/assets/chart.mplstyle")
fig, ax = plt.subplots()
bars = ax.bar(questions, percent_correct)
ax.bar_label(bars, fmt="%.0f%%", padding=3)
ax.set_ylim(0, 100)
ax.set_ylabel("Students correct (%)")
ax.set_title("Questions 6–8 were the hardest")
fig.savefig("percent-correct.png")
```

- **Pick the form from the question:** bars to compare groups, sorted unless the order means
  something; a line for change over time; a histogram for how scores spread; a scatter for two
  measures of the same students. Use a pie only for two or three parts of a whole.
- **The title states the finding,** such as "Period 3 improved most", not "Scores by period".
- **Label axes with units,** start bar axes at zero, and label values on bars when there are few.
- **Label lines directly** at their ends rather than in a legend, when there are few.
- The style's colors stay distinct for color-blind readers. Keep one color unless color means
  something.
- Call `review_file` on the PNG before you deliver it, and look at it: labels overlapping, text
  cut off, and a title that matches what the chart shows.

## Small groups swing

In a class of 25, one student is 4 points of a percentage. Say so before calling a difference
between classes or questions real.

## Student data

When a result goes beyond the teacher, such as a chart for a slide, show counts and percentages
rather than names.

## Delivering results

Deliver the chart with `deliver_file`. When the user wants the numbers too, save the table as CSV,
or as a formatted workbook with the xlsx skill.

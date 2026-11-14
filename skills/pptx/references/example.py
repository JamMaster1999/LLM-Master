"""A sample lesson deck: one palette, a dark title slide, light content slides, a visual on every
slide, speaker notes, and a sources slide. Run it from any folder: python example.py"""
import sys

sys.path.insert(0, "/skills/pptx/scripts")
from pptx.chart.data import CategoryChartData  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402
from slides import add_arrow, add_box, add_bullets, add_slide, add_text, new_deck, set_notes  # noqa: E402

INK, DEEP, WATER, SKY, SUN = "1F2937", "0B3C5D", "1D7CB8", "E3F1FA", "F2A541"  # one dominant color, one accent

deck = new_deck()

slide = add_slide(deck, "The Water Cycle", bg=DEEP, color="FFFFFF", size=54, box=(0.8, 2.3, 11.7, 1.4))
add_text(slide, 0.8, 3.7, 11.7, 0.8, "How water moves between the land, the ocean and the sky", size=26, color=SKY)
add_text(slide, 0.8, 5.8, 8, 0.5, "Grade 5 Science · Lesson 1 of 2", size=18, color=SUN)
set_notes(slide, "Open with the question: where did the water in your water bottle come from?")

slide = add_slide(deck, "Learning goals")
add_bullets(slide, 0.8, 1.6, 7.2, 4.6, [
    "I can name the four main steps of the water cycle.",
    "I can explain what the sun's energy does in each step.",
    "I can use a model to describe how water moves.",
], size=26, bullet_color=WATER, spacing=0.8)
add_box(slide, 8.6, 1.7, 4.0, 4.2, fill=SKY, text=["Standard", "**5-ESS2-1** [2]", "Earth's systems interact"],
        size=22, color=DEEP)
set_notes(slide, "Read the goals aloud. Point out the standard so students know what they are working toward.")

slide = add_slide(deck, "Four steps, one cycle")
steps = [("Evaporation", "The sun heats water into vapor"), ("Condensation", "Vapor cools into cloud droplets"),
         ("Precipitation", "Droplets fall as rain or snow"), ("Collection", "Water gathers in oceans and lakes")]
for i, (step, detail) in enumerate(steps):
    x = 0.6 + i * 3.15
    add_box(slide, x, 2.0, 2.75, 1.1, fill=WATER, text=step, size=22, color="FFFFFF", bold=True)
    add_text(slide, x, 3.3, 2.75, 1.6, detail, size=20, align="center")
    if i < len(steps) - 1:
        add_arrow(slide, x + 2.8, 2.55, x + 3.1, 2.55, color=SUN)
add_box(slide, 0.6, 5.3, 12.1, 1.2, fill=SKY, text="Then it starts again: no water is used up, it only moves.",
        size=22, color=DEEP)
set_notes(slide, "Ask: which step needs the sun's energy most? (Evaporation.)")

slide = add_slide(deck, "Where Earth's water is")
data = CategoryChartData()
data.categories = ["Oceans", "Ice and glaciers", "Groundwater", "Lakes and rivers"]
data.add_series("Share of Earth's water (%)", (96.5, 1.7, 1.7, 0.013))
frame = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.8), Inches(1.6), Inches(7.6), Inches(5.0), data)
chart = frame.chart
chart.has_legend = chart.has_title = False
chart.category_axis.reverse_order = True  # bar charts list categories bottom-up otherwise
chart.font.size, chart.font.name = Pt(18), "Calibri"
plot = chart.plots[0]
plot.gap_width = 60
plot.has_data_labels = True
plot.data_labels.number_format, plot.data_labels.number_format_is_linked = '0.0##"%"', False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.series[0].format.fill.solid()
plot.series[0].format.fill.fore_color.rgb = RGBColor.from_string(WATER)
chart.value_axis.visible = False
chart.value_axis.has_major_gridlines = False
add_text(slide, 9.0, 2.0, 3.7, 3.6, ["Almost all of it is salty ocean water.",
                                     "Less than 1% is fresh water we can easily use [1]."], size=22)
set_notes(slide, "Numbers from the USGS [1]. Ask students why the fresh-water share matters.")

slide = add_slide(deck, "Check your understanding")
add_text(slide, 0.8, 1.7, 11.7, 1.2, "A puddle on the playground is gone by lunchtime. Where did the water go?", size=28)
for i, choice in enumerate(["A. It soaked into the ground only", "B. It evaporated into the air",
                            "C. It turned into ice", "D. It became a cloud right away"]):
    add_box(slide, 0.8 + (i % 2) * 6.0, 3.2 + (i // 2) * 1.5, 5.7, 1.2, fill="F3F4F6", text=choice, size=22,
            align="left")
set_notes(slide, "Answer: B. Some may also soak in, so accept A with an explanation.")

slide = add_slide(deck, "Sources")
add_text(slide, 0.8, 1.6, 11.7, 4.5, [
    "[1] USGS Water Science School, \"How Much Water is There on Earth?\" usgs.gov/special-topics/water-science-school",
    "[2] NGSS 5-ESS2-1, Earth's Systems. nextgenscience.org/pe/5-ess2-1-earths-systems",
], size=16, spacing=0.6)

deck.save("water-cycle.pptx")

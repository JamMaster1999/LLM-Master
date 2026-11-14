---
name: pptx
description: Builds PowerPoint decks (.pptx) with python-pptx and checks them before delivery. Use this skill whenever the user wants slides, a deck, a presentation or a .pptx file, such as lesson slides, a lecture, a review deck or slides made from a document, and when they upload a .pptx to change.
---

# PowerPoint decks

Build decks with python-pptx and the helpers in `scripts/slides.py`. They fix what python-pptx gets
wrong by default: 4:3 slides, text boxes that never wrap, fake bullets, and shadows on every shape.
`references/example.py` builds a finished six-slide deck with them. Read it before your first deck.

## Workflow

1. **Research first** when the deck teaches facts or standards: load the research skill and gather
   the content, with sources, before you outline.
2. **Outline.** Give each slide a title and one main point. Pick a visual for every slide.
3. **Pick a palette from the topic.** Use one dark color for title slides and headings, one light
   tint for boxes, and one bright accent. Text is near-black on light and white on dark.
4. **Write a build script**, such as `build_deck.py`, starting from the example. Keep the content in
   it, so a fix is an edit and a rerun.
5. **Run the checker:** `python /skills/pptx/scripts/check.py deck.pptx`. It measures every text box
   with the fonts the preview uses. It reports overflow, shapes past the edge, overlapping text,
   missing fonts, characters no font can draw (such as emoji), more than 60 words on a slide, typed
   bullets, an empty bottom third, missing titles, pictures without alt text, empty placeholders,
   and text under 18 pt (10 pt in footers and on the Sources slide). Fix everything it reports,
   then run it again.
6. **Review the pages.** Call `review_file` to see every slide at once. Thumbnails hide small text
   and tight spacing, so the first time, read every slide at full size, ten per call:
   `read(path, pages="1-10")`, then `"11-20"`. After a fix, read only the slides you changed: each
   page read stays in the conversation as an image. Look for crowded slides, text close to an
   edge, poor contrast, visuals that don't match the text, and facts that don't match your sources.
7. **Fix in the script, rebuild, and repeat 5 and 6** until both are clean. Then deliver it.

## Helpers

Import them with `sys.path.insert(0, "/skills/pptx/scripts")`. Sizes are inches on a 13.333 × 7.5
slide. Colors are hex strings such as `"1F4E3D"`. Text may mark bold words with `**double stars**`.

| helper | makes |
|---|---|
| `new_deck()` | a 16:9 deck |
| `add_slide(deck, title, bg=, color=, size=36, box=(x, y, w, h))` | a slide with its title in the real title placeholder |
| `add_text(slide, x, y, w, h, text, size=24, color=, bold=, align=, anchor=)` | a wrapping text box; a list makes paragraphs |
| `add_bullets(slide, x, y, w, h, items, size=24, numbered=False)` | real bullets; a nested list holds sub-points |
| `add_box(slide, x, y, w, h, fill=, shape="rounded", text=)` | a card, label or process step; shapes: rect, rounded, oval, arrow, chevron, pentagon |
| `add_arrow(slide, x1, y1, x2, y2, color=)` | a connector line with an arrowhead |
| `add_picture(slide, path, x, y, w, h, alt=)` | a picture fitted inside the box, with alt text |
| `set_notes(slide, text)` | speaker notes |

Each returns the python-pptx object, so anything else python-pptx does still applies.

## Design for a room

Slides are read from the back of a classroom or lecture hall.

- **Sizes:** titles 36–44 pt, body 24–28 pt, labels in boxes at least 18 pt, sources 14–16 pt. At
  24 pt a 6-inch box holds about 40 characters a line, and each line takes 0.4 inches.
- **One idea per slide,** with at most five short bullets and 60 words. Split a slide rather than
  shrink its text, and put the detail a teacher says aloud in the speaker notes.
- **A visual on every slide:** a picture, a native chart, or a diagram built from boxes and arrows (a
  process, a cycle, a comparison). A text slide gets a callout box or a large key term instead.
- **Vary the layouts:** two columns, a grid of cards, a row of steps, a big number with a caption, a
  question with answer choices.
- **Fonts:** Calibri, Cambria, Arial or Times New Roman. The preview has exact twins of these. Any
  other font changes width in the preview and may be missing on the viewer's computer. The preview
  has no emoji font, so use words or shapes instead.
- **Left-align body text.** Center only titles on title slides and short labels in boxes.
- **Formulas** take Unicode subscripts and superscripts, which Calibri has: CO₂, C₆H₁₂O₆, x².
- **Leave 0.5 inches** between the slide edge and any text, and 0.3 inches between blocks.
- **Avoid** lines under titles, colored stripes along slide edges, text over busy pictures, and
  pale text on a pale fill.
- **Speaker notes** on every content slide: what to say, and any question with its answer. Answers
  stay off the question's slide, highlighted choices included; put them in the notes or on the
  next slide.

## Pictures

- Make them with `generate_image`. It saves the image under `images/` and sends it to the user as
  well, so make only the ones the deck uses.
- Ask for no words in the picture. Generated lettering often comes out garbled, so put labels in
  slide text beside it.
- Place it with `add_picture`, and give it alt text that says what it shows.

## Charts and tables

- Use native charts (`slide.shapes.add_chart`), so the user can edit the data.
- Set `chart.font.size` to at least 18 pt. Turn the chart title off when the slide title says it.
- Bar charts list categories bottom-up; set `chart.category_axis.reverse_order = True`.
- Table rows grow to fit their text, so a long table can run off the slide. Keep tables to six rows,
  and set every cell's font size. The checker measures the grown height.

## Sources

Put each source's number on the slide that uses it, and end the deck with a Sources slide listing
each number, title and URL.

## Changing an uploaded deck

Read `references/editing.md` first. Keep the deck's own layouts, fonts and colors.

---
name: research
description: Researches on the web until the facts are in hand, with parallel and deep searches, primary sources read in full, and every claim tied to a numbered source, then hands the findings to a file skill. Use this skill whenever the answer or a file depends on facts to check rather than recall, such as the exact wording of standards, curriculum requirements, policies, statistics, dates or recent events, and before building slides, documents or handouts whose content must be accurate.
---

# Research

For a question you can answer in chat, such as one standard's wording, search, read the source and
answer with its number. The steps below are for research that feeds a file or a long answer.

## 1. Plan the questions

Write down the three to six questions the answer depends on, and add them to the `todo` plan. For
a unit's slides they might be:

- the standards' exact codes and wording, in the version the teacher's state or school uses;
- what each standard expects students to know and do, including its stated limits;
- the facts, data and examples to teach, at the right level, and common misconceptions;
- any figure, table or image the material needs.

## 2. Search wide, then deep

- **Search in parallel.** Send independent `web_search` calls together, each with an `objective`
  that names the facts to pull and the pages to prefer.
- **Use `type: "deep"`** when one search can't answer a question: finding where an official
  document lives, comparing versions, or pulling facts from several pages at once.
- **Read primary pages in full** with `fetch_url`, several URLs in one call. Highlights are
  excerpts; the wording that matters is often around them.
- **More than five searches, or long pages:** run them from `run_program` and print only what you
  need, so the raw results stay out of your context. Print each result's head, which carries its
  source number:

  ```python
  queries = ["NGSS MS-LS1-6 performance expectation", "photosynthesis misconceptions middle school"]
  pages = await asyncio.gather(*[web_search({"query": q, "objective": "Official wording and clarification statements"})
                                 for q in queries])
  for query, page in zip(queries, pages):
      print("##", query)
      for result in page.split("\n\n---\n\n"):  # one per result: [n], title, URL, then highlights
          print(result[:400], "\n")
  ```

## 3. Judge the sources

- **Go to the publisher.** Exact wording comes from the body that publishes it: the standards
  organization, the state's department of education, the agency, the exam board. Copies and
  lesson sites paraphrase, lag behind revisions, or look official without being so: the official
  Common Core site is corestandards.org, while thecorestandards.org is an unofficial copy. Check
  who runs a site before quoting it.
- **Use the teacher's own standards.** When the teacher names a state, cite that state's document
  with its own codes, even where it copies Common Core or NGSS. Another state's guide is not a
  source for it.
- **Check the version and date.** Standards and policies are revised. Say which version you used.
- **Confirm what matters.** A fact the teacher will teach needs its primary source, or two
  independent sources that agree.
- **Report disagreement.** When sources conflict, say so and say which you trust and why.

## 4. Stop

Stop when every question has an answer from a primary source or two agreeing ones. When a fact
can't be found, say so, and mark anything you add from memory as unchecked.

## 5. Keep notes

For work that feeds a file, keep `research/notes.md` in the sandbox: under each question, the
facts with their source numbers, and exact quotes where wording matters. Build the file from the
notes, so every claim keeps its number.

## 6. Cite

Cite only what you read in a search result or a fetched page, with the number the tool gave it.

## 7. Hand off

When the user wants a file, load its skill (pptx, docx, pdf or xlsx), and build from the notes.

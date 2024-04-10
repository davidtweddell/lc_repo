# ISSUES AND SUGGESTIONS

- imaging_results: mostly empty except for some with spaces/blanks -> drop column
- ferritin: has strings "not done" -> remove
- alt: has one entry "12 .1" (string) -> change to 12.1 (float)
- other_hhx3: has strings " " (just one space) -> remove
- other_hhx4: has strings " " (just one space) -> remove
- country: one entry "El Salvador" (string) -> change to 28 (int) per code book
- country: "holland" and "Netherlands" -> choose one (I think the Dutch have an opinion on this)


# QUESTIONS
- several clinical tests just a handfull (fewer than 10 in some cases) of non-null values - what to do with these? drop the columns?
- what is country? country of birth, or current residence? many blanks?
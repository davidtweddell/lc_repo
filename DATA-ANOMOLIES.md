



- imaging_results: mostly empty except for some with spaces/blanks -> drop column
- ferritin: has strings "not done" -> remove
- alt: has one entry "12 .1" (string) -> change to 12.1 (float)
- other_hhx3: has strings " " (space) -> remove
- other_hhx4: has strings " " (space) -> remove
- country: one entry "El Salvador" (string) -> change to 28 (int) per code book
  - what is country? country of birth, or current residence?
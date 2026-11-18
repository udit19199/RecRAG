Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 12 items.

=== ITEM id=d9b5e754471f56a7 ===
Gold sentences and facts:
Sentence 0: The Segugio Italiano ] is an Italian breed of dog of the scenthound family.
  Fact 0.0: The Segugio Italiano is an Italian breed of dog.
  Fact 0.1: The Segugio Italiano belongs to the scenthound family.
Sentence 1: The Sapsali (삽살이) is a shaggy Korean breed of dog.
  Fact 1.0: The Sapsali is a shaggy Korean breed of dog.
  Fact 1.1: The Sapsali's Korean name is 삽살이.
Sentence 2:  Traditionally, these dogs were believed to dispel ghosts and evil spirits.
  Fact 2.0: Traditionally, Sapsali dogs were believed to dispel ghosts.
  Fact 2.1: Traditionally, Sapsali dogs were believed to dispel evil spirits.

Retrieved items:
<item 1>
Guatemalan Dogo
Guatemalan Dogo ("Dogo Guatemalteco"), formerly known as the Guatemalan Bull Terrier ("Bullterrier Guatemalteco"), and Guatemalan Mastiff, is a Molosser-type
dog breed originating in Guatemala.

 It is neither recognized by the Fédération Cynologique Internationale (FCI) nor the American Kennel Club (AKC).

 However, it has the official national recognition of the Asociación Canofila Guatemalteca (ACANGUA) where it belongs to the Group 2.2.1 - Mastiffs.

 As the unique breed originating in Guatemala, it is also the national dog of the country.

 Nowadays it has spread to various other countries, such as the USA.
</item 1>

<item 2>
Romanian Raven Shepherd Dog
The Romanian Raven Shepherd Dog is a very large Romanian livestock guardian dog, taxonomized within the second group of dog breeds - Pinscher and Schanuzer -
Molossoid Breeds - Swiss Mountain and Cattle Dogs, section 2.2: Molossoid Breeds - Mountain type.
</item 2>

<item 3>
Segugio Italiano
The Segugio Italiano ] is an Italian breed of dog of the scenthound family.

 It comes in both short-haired and wire-haired varieties.

 It is thought to be an ancient breed, descended in pre-Roman eras from progenitor scenthounds in ancient Egypt.

 In 2009 ENCI (the Italian Kennel Club) registered 4,500 specimens of the short-haired variety and 1,740 wire-haired specimens, making this one of the top ten
breeds in Italy.
</item 3>

<item 4>
Appenzeller Sennenhund
The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.

 The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
</item 4>

<item 5>
American Eskimo Dog
The American Eskimo Dog is a breed of companion dog originating in Germany.

 The American Eskimo is a member of the Spitz family.

 The breed's progenitors were German Spitz, but due to anti-German prejudice during the First World War, it was renamed "American Eskimo Dog".

 Although modern American Eskimos have been exported as German Spitz Gross (or Mittel, depending on the dog's height), the breeds have diverged and the
standards are significantly different.

 In addition to serving as a watchdog and companion, the American Eskimo Dog also achieved a high degree of popularity in the United States in the 1930s and
1940s as a circus performer.
</item 5>

<item 6>
Greater Swiss Mountain Dog
The Greater Swiss Mountain Dog (German: "Grosser Schweizer Sennenhund" or French: "Grand Bouvier Suisse" ) is a dog breed which was developed in the Swiss Alps.

 The name "Sennenhund" refers to people called "Senn" or "Senner", dairymen and herders in the Swiss Alps.

 Greater Swiss Mountain Dogs are almost certainly the result of indigenous dogs mating with large mastiff types brought to Switzerland by foreign settlers.

 At one time, the breed was believed to have been among the most popular in Switzerland.

 It was assumed to have almost died out by the late 19th century, since its work was being done by other breeds or machines, but was rediscovered in the early
1900s.
</item 6>

<item 7>
Dogo Argentino
The Dogo Argentino, also known as the Argentine Mastiff, is a large, white, muscular dog that was developed in Argentina primarily for the purpose of big-game
hunting, including wild boar; the breeder, Antonio Nores Martínez, also wanted a dog that would exhibit steadfast bravery and willingly protect its human
companion.

 It was first bred in 1928, from the Cordoba Fighting Dog along with a wide array of other breeds including the Great Dane.
</item 7>

<item 8>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 8>

<item 9>
Sapsali
The Sapsali (삽살이) is a shaggy Korean breed of dog.

 The word is followed in Korean by either "gae" (meaning "dog") or the suffix "ee"/"i", but is most commonly romanized as "Sapsaree".

 Traditionally, these dogs were believed to dispel ghosts and evil spirits.
</item 9>

<item 10>
Chiribaya Dog
The Chiribaya Dog (Spanish: "perro Chiribaya" ) or Peruvian shepherd dog ("perro pastor Peruano ") was a pre-Columbian breed of dog from the southwest of Peru,
identified by the 42 mummies discovered by anthropologist Sonia Guillén Oneglio in the Ilo District, Moquegua Region, on the south coast of Peru.

 It has been established that it was a llama herding dog.

 The dogs were not only an important part of the social structure of the ancient Peruvians, but they received special treatment after death as well.

 The dog variety has been referred to in various Spanish-language documentaries under different terms, such as "el perro pastor Chribaya" ('the Chiribaya
shepherd dog') and "pastor Peruano " ('Peruvian shepherd'), though the ancient Peruvians did not keep sheep.

 Its original name is unknown.
</item 10>
=== END ITEM id=d9b5e754471f56a7 ===

=== ITEM id=23f7bfd1056f1974 ===
Gold sentences and facts:
Sentence 0: Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).
  Fact 0.0: Blinding Edge Pictures is an American film production company.
  Fact 0.1: Blinding Edge Pictures was founded in 2000 by M. Night Shyamalan.
  Fact 0.2: Blinding Edge Pictures is known for producing films written and directed by Shyamalan.
  Fact 0.3: Blinding Edge Pictures produced Unbreakable (2000), written and directed by Shyamalan.
  Fact 0.4: Blinding Edge Pictures produced Signs (2002), written and directed by Shyamalan.
  Fact 0.5: Blinding Edge Pictures produced The Village (2004), written and directed by Shyamalan.
  Fact 0.6: Blinding Edge Pictures produced The Happening (2008), written and directed by Shyamalan.
  Fact 0.7: Blinding Edge Pictures produced After Earth (2013), written and directed by Shyamalan.
  Fact 0.8: Blinding Edge Pictures produced The Visit (2015), written and directed by Shyamalan.
  Fact 0.9: Blinding Edge Pictures produced Split (2017), written and directed by Shyamalan.
Sentence 1: Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson, alongside Robin Wright and Spencer Treat Clark.
  Fact 1.0: Unbreakable is a 2000 American superhero thriller film.
  Fact 1.1: Unbreakable was written by M. Night Shyamalan.
  Fact 1.2: Unbreakable was produced by M. Night Shyamalan.
  Fact 1.3: Unbreakable was directed by M. Night Shyamalan.
  Fact 1.4: Unbreakable stars Bruce Willis.
  Fact 1.5: Unbreakable stars Samuel L. Jackson.
  Fact 1.6: Robin Wright is in the cast of Unbreakable.
  Fact 1.7: Spencer Treat Clark is in the cast of Unbreakable.

Retrieved items:
<item 1>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 1>

<item 2>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 2>

<item 3>
The entries cover films across action, comedy, romance, horror, and drama. Bruce Willis stars in titles including *Last Man Standing* (a credited *Yojimbo*
remake), *Unbreakable*, *Signs*, and *Die Hard with a Vengeance*. Shyamalan’s *Signs* explores faith and alien encounters; *Unbreakable* launches a superhero
trilogy. Other highlights include *The Whole Nine Yards*, *Striking Distance*, *Mercury Rising
</item 3>

<item 4>
Mercury Rising
Mercury Rising is a 1998 American political action thriller film starring Bruce Willis and Alec Baldwin.

 Directed by Harold Becker, the movie is based on Ryne Douglas Pearson's 1996 novel originally published as "Simple Simon".

 Willis plays Art Jeffries, an undercover FBI agent who protects a 9-year-old boy with autism who is targeted by government assassins after he cracks a top
secret government code.
</item 4>

<item 5>
Unbreakable (film)
Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson,
alongside Robin Wright and Spencer Treat Clark.

 The movie is the first installment in a trilogy.

 In "Unbreakable", a security guard named David Dunn survives a horrific train crash.

 After the incident, with the help of a manipulative disabled comic book shop owner named Elijah Price, he learns that he possesses superhuman powers.

 As Dunn explores and reluctantly confronts his powers while trying to navigate a difficult family life, he begins to fight crime and learns the true nature of
Elijah Price.
</item 5>

<item 6>
A broad film, television, documentary, and industry collection spans silent-era cinema through 2010, covering action, comedy, drama, horror, musicals, noir, and
more. Highlights include *The Razor’s Edge*, *The Little Foxes*, *Die Hard*, *Unbreakable*, *Signs*, *The Straight Story*, *West Side Story*, and *Rogue One*.
Documentaries address music, war, politics, disability, and culture. Profiles feature Bruce Willis, David
</item 6>

<item 7>
Striking Distance
Striking Distance is a 1993 American action thriller film starring Bruce Willis as Pittsburgh Police homicide detective Thomas Hardy.

 The film co-stars Sarah Jessica Parker, Dennis Farina, and Tom Sizemore.

 It was directed by Rowdy Herrington and written by Herrington and Marty Kaplan.

 The film was shot on location throughout Pittsburgh; its early title was "Three Rivers".
</item 7>

<item 8>
The collection spans films and TV from 1947–2010: Bruce Willis’s dramas, action comedies and thrillers include *In Country* (Golden Globe-nominated), *Last Man
Standing* (a *Yojimbo* remake), *The Whole Nine Yards* and *Cop Out*. Other highlights: *The Bears and I*; teen comedies *Can’t Buy Me Love* and *The Night
Before*; *The Country Girl* (Oscar-winning screenplay
</item 8>

<item 9>
Last Man Standing (1996 film)
Last Man Standing is a 1996 American action thriller film written and directed by Walter Hill and starring Bruce Willis, Christopher Walken and Bruce Dern.

 It is a credited remake of Akira Kurosawa's "Yojimbo".
</item 9>

<item 10>
Blinding Edge Pictures
Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by
Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).

 In 2015, the company released its first television series "Wayward Pines".
</item 10>
=== END ITEM id=23f7bfd1056f1974 ===

=== ITEM id=c59be8e39b1c0c5c ===
Gold sentences and facts:
Sentence 0: Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.
  Fact 0.0: Tender Fictions is a 1996 autobiographical documentary film.
  Fact 0.1: Tender Fictions was directed by American experimental filmmaker Barbara Hammer.
Sentence 1: Bougainville – Our Island Our Fight is a 1998 Australian documentary film.
  Fact 1.0: Bougainville – Our Island Our Fight is a 1998 Australian documentary film.
Sentence 2:  It was produced and directed by Wayne Coles-Janess.
  Fact 2.0: Bougainville – Our Island Our Fight was produced by Wayne Coles-Janess.
  Fact 2.1: Bougainville – Our Island Our Fight was directed by Wayne Coles-Janess.

Retrieved items:
<item 1>
The Boys in Company C
The Boys in Company C, directed by Sidney J. Furie, starring Stan Shaw, Andrew Stevens, Craig Wasson, James Canning, and Michael Lembeck, is a 1978 film about
United States Marine Corps recruits preparing for duty, and their subsequent combat in the Vietnam War.

 It was among the first Vietnam War films to appear after the Vietnam Era, and was also the first role for R. Lee Ermey of "Full Metal Jacket" fame.

 "The Boys in Company C" is the first in Furie's Vietnam War motion picture trilogy, followed by 2001's "Under Heavy Fire" and 2006's "The Veteran".
</item 1>

<item 2>
Wayne Coles-Janess
Wayne Coles-Janess is an Australian producer, writer and director of drama and documentary film and TV programs.

 Based in Melbourne, Australia, he has produced documentaries about frontier places in the country.

 He has also made some documentaries in several international locations, including during times of war.
</item 2>

<item 3>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 3>

<item 4>
Bougainville – Our Island Our Fight
Bougainville – Our Island Our Fight is a 1998 Australian documentary film.

 It was produced and directed by Wayne Coles-Janess.
</item 4>

<item 5>
Flight of the Red Balloon
Flight of the Red Balloon (French: Le voyage du ballon rouge ) is a 2007 French-Taiwanese film directed by Hou Hsiao-hsien.

 It is the first part in a new series of films produced by Musée d'Orsay, and tells the story of a French family as seen through the eyes of a Chinese student.

 The film was shot in August and September 2006 on location in Paris.

 This is Hou Hsiao-hsien's first non-Asian film.

 It references the classic 1956 French short "The Red Balloon" directed by Albert Lamorisse.
</item 5>

<item 6>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 6>

<item 7>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 7>

<item 8>
Graph query row:
film: Bougainville – Our Island Our Fight
director: Wayne Coles-Janess
relationship: CREATED_BY
</item 8>

<item 9>
The Coconut Revolution
The Coconut Revolution is a 2001 multi-award winning documentary film about the struggle of the indigenous peoples of Bougainville Island during the
Bougainville Civil War.

 The movement is described as the "world's first successful eco-revolution" and has drawn parallels with the conflict depicted in the 2009 film, "Avatar".
</item 9>

<item 10>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 10>
=== END ITEM id=c59be8e39b1c0c5c ===

=== ITEM id=70a9dc7fd6d2327f ===
Gold sentences and facts:
Sentence 0:  They were first year members of the Mountain West Conference.
  Fact 0.0: They were first-year members of the Mountain West Conference.
Sentence 1: The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I Football Bowl Subdivision (FBS) (formerly I-A).
  Fact 1.0: The Mountain West Conference (MW) is a collegiate athletic conference.
  Fact 1.1: The Mountain West Conference is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA).
  Fact 1.2: The Mountain West Conference is affiliated with the NCAA Division I Football Bowl Subdivision (FBS).
  Fact 1.3: The Football Bowl Subdivision (FBS) was formerly called I-A.

Retrieved items:
<item 1>
Hawaii Rainbow Warriors baseball
The Hawaii Rainbow Warriors baseball team is a varsity intercollegiate athletic team of the University of Hawaii at Manoa in Honolulu, Hawaii, United States.

 The team is a member of the Big West Conference, which is part of the National Collegiate Athletic Association's Division I.

 Hawaii's first baseball team was fielded in 1923.

 The team plays its home games at Les Murakami Stadium in Honolulu, Hawaii.

 The Rainbow Warriors are coached by Mike Trapasso.
</item 1>

<item 2>
List of Pacific Tigers head football coaches
The Pacific Tigers college football team represented University of the Pacific (CA).

 The Tigers competed in the National Collegiate Athletic Association (NCAA) College Division in the years 1937–1968.

 In 1969, the team moved to National Collegiate Athletic Association (NCAA) Division I and stayed there until disbanding after the 1995 season.
</item 2>

<item 3>
2012–13 Wagner Seahawks men's basketball team
The 2012–13 Wagner Seahawks men's basketball team represented Wagner College during the 2012–13 NCAA Division I men's basketball season.

 The Seahawks were led by the youngest men's head coach in NCAA Division I, Bashir Mason, who was 28 when he was elevated from an assistant position in March
2012 following the departure of Dan Hurley for Rhode Island.

 The Seahawks played their home games at Spiro Sports Center and were members of the Northeast Conference.

 They finished the season 19–12, 12–6 in NEC play to finish in a three way tie for second place.

 They advanced to the semifinals of the Northeast Tournament where they lost to Long Island.

 For the second consecutive year, despite a winning record, Wagner choose not to participate in a post season tournament.
</item 3>

<item 4>
Men's National Collegiate Athletic Association
The Men's National Collegiate Athletic Association is an athletic association in the Philippines, and is the direct counterpart of the older Women's National
Collegiate Athletic Association (founded in 1970).

 The Men's National Collegiate Athletic Association (MNCAA) is exclusive for men.

 The MNCAA was founded in 2004.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 The competing sports of the tournament are Basketball, Volleyball, Futsal, Table Tennis, Badminton and Streetdance.
</item 4>

<item 5>
2012 Hawaii Warriors football team
The 2012 Hawaii Warriors football team represented the University of Hawaii at Manoa in the 2012 NCAA Division I FBS football season.

 They were led by first-year head coach Norm Chow and played their home games at Aloha Stadium.

 They were first year members of the Mountain West Conference.

 They finished the season 3–9, 1–7 in Mountain West play to finish in a tie for ninth place.
</item 5>

<item 6>
2011–12 Wagner Seahawks men's basketball team
The 2011–12 Wagner Seahawks men's basketball team represented Wagner College during the 2011–12 NCAA Division I men's basketball season.

 The Seahawks, led by second year head coach Dan Hurley, played their home games at Spiro Sports Center and are members of the Northeast Conference.

 They finished the season 25–6, 15–3 in NEC play to finish in second place.

 They lost in the semifinals of the NEC Basketball Tournament to Robert Morris.

 Despite having 25 wins, the Seahawks did not accept an invitation to a post season tournament.
</item 6>

<item 7>
Women's National Collegiate Athletic Association
The Women's National Collegiate Athletic Association (WNCAA) is an athletic association in the Philippines exclusively for women.

 It was founded in 1970.

 Competition is divided into three divisions: Seniors for college students, Juniors for high school students, and Midgets for grade school and first year high
school students.

 Its men's counterpart is the Men's National Collegiate Athletic Association, founded in 2004.
</item 7>

<item 8>
2011–12 Iona Gaels men's basketball team
The 2011–12 Iona Gaels men's basketball team represented Iona College during the 2011–12 NCAA Division I men's basketball season.

 The Gaels, led by second year head coach Tim Cluess, played their home games at the Hynes Athletic Center and are members of the Metro Atlantic Athletic
Conference.

 The Gaals were MAAC regular season champions but failed to win the MAAC Basketball Tournament after losing to Fairfield in the semifinals.

 They received an at–large bid to the 2012 NCAA Tournament where they lost to BYU in the "First Four" round.

 The Gaels led BYU by 25 points in the 2nd half.

 It was the largest comeback in NCAA Tournament history.
</item 8>

<item 9>
Wayne State Warriors football
The Wayne State Warriors football team is the college football team at Wayne State University in Detroit, Michigan.

 The Wayne State football team played their first game in October 1918.

 The Wayne State Warriors have competed in the Great Lakes Intercollegiate Athletic Conference since 1999 (and previously from 1975-1989), and are currently a
Division II member of the National Collegiate Athletic Association (NCAA).

 Wayne State plays their home games at Tom Adams Field at Wayne State Stadium.

 All Wayne State games are broadcast on WDTK radio.
</item 9>

<item 10>
Mountain West Conference
The Mountain West Conference (MW) is one of the collegiate athletic conferences affiliated with the National Collegiate Athletic Association (NCAA) Division I
Football Bowl Subdivision (FBS) (formerly I-A).

 The MW officially began operations in July 1999.

 Geographically, the MW covers a broad expanse of the Western United States, with member schools located in California, Colorado, Hawaii, Idaho, Nevada, New
Mexico, Utah, and Wyoming.

 Craig Thompson has served as Commissioner of the MW since its founding in 1999.
</item 10>
=== END ITEM id=70a9dc7fd6d2327f ===

=== ITEM id=bac2ec357b7783b5 ===
Gold sentences and facts:
Sentence 0: Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a 1760 play by Francesco Ringhieri, "L'Osiride".
  Fact 0.0: Mosè in Egitto (Moses in Egypt) is a three-act opera.
  Fact 0.1: Mosè in Egitto was written by Gioachino Rossini.
  Fact 0.2: Mosè in Egitto has an Italian libretto by Andrea Leone Tottola.
  Fact 0.3: The libretto of Mosè in Egitto was based on a 1760 play by Francesco Ringhieri.
  Fact 0.4: The 1760 play by Francesco Ringhieri is titled L'Osiride.
Sentence 1: Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.
  Fact 1.0: Lucrezia Borgia is a melodramatic opera.
  Fact 1.1: Lucrezia Borgia is in a prologue and two acts.
  Fact 1.2: Lucrezia Borgia is by Gaetano Donizetti.

Retrieved items:
<item 1>
Pagliacci
Pagliacci (] ; literal translation, "Clowns") is an Italian opera in a prologue and two acts, with music and libretto by Ruggero Leoncavallo.

 It is the only Leoncavallo opera that is still widely performed.

 Opera companies have frequently staged "Pagliacci" with "Cavalleria rusticana" by Mascagni, a double bill known colloquially as 'Cav and Pag'.
</item 1>

<item 2>
Mosè in Egitto
Mosè in Egitto ("Moses in Egypt)" (] ) is a three-act opera written by Gioachino Rossini to an Italian libretto by Andrea Leone Tottola, which was based on a
1760 play by Francesco Ringhieri, "L'Osiride".

 It premièred on 5 March 1818 at the recently reconstructed Teatro San Carlo in Naples, Italy.
</item 2>

<item 3>
Romolo ed Ersilia (Hasse)
Romolo ed Ersilia is an opera in three acts composed by Johann Adolph Hasse to an Italian-language libretto by Pietro Metastasio.

 The opera was commissioned by Empress Maria Theresa to celebrate the marriage of her son Leopold to Maria Luisa of Spain.

 The opera was first performed on 6 August 1765, the day after their wedding, in the court theatre of the Imperial Palace in Innsbruck.

 The opera received its first performance in modern times in 2011 during the Innsbruck Festival of Early Music.
</item 3>

<item 4>
Lucrezia Borgia (1947 film)
Lucrezia Borgia (Spanish:Lucrecia Borgia) is a 1947 Argentine historical film directed by Luis Bayón Herrera and starring Olinda Bozán, Héctor Quintanilla and
Gogó Andreu.

 The film portrays the life of Lucrezia Borgia (1480-1519).
</item 4>

<item 5>
Lucrezia Borgia (1940 film)
Lucrezia Borgia is a 1940 Italian historical film directed by Hans Hinrich and starring Isa Pola, Friedrich Benfer and Carlo Ninchi.

 The film portrays the life of Lucrezia Borgia (1480-1519), one of a number of Italian films of the era set during the Renaissance.

 It was made at the Scalera Studios in Rome.
</item 5>

<item 6>
Lucrezia Borgia (play)
Lucrezia Borgia (French: "Lucrèce Borgia" ) is an 1833 play by the French writer Victor Hugo.

 It is a historical work portraying the Renaissance-era Italian aristocrat Lucrezia Borgia.

 The play (along with "Angelo, Tyrant of Padua") is believed to have been a major influence on Oscar Wilde's "The Duchess of Padua" (1891).
</item 6>

<item 7>
Senza sangue
Senza sangue (Without blood) an opera in one act by Péter Eötvös with a libretto by Mari Mezei.

 It is based on the second part of the novel of the same name by Alessandro Baricco and is intended to be performed jointly with Béla Bartók's one-act
"Bluebeard's Castle".

 The concert première was given by the Cologne Philharmonic Orchestra on 1 May 2015.

 It was premièred in full on 15 May 2016 at the Festival d'Avignon.

 It is his tenth opera.

 Anne Sofie von Otter, who gave the New York première with the New York Philharmonic, said: "“The piece is not at all easy for the two singers... [It’s] hard to
find the pitch; you have to work it into the voice, unless of course you happen to have perfect pitch, which I don’t."
</item 7>

<item 8>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 8>

<item 9>
Lucrezia Borgia (opera)
Lucrezia Borgia is a melodramatic opera in a prologue and two acts by Gaetano Donizetti.

 Felice Romani wrote the Italian libretto after the play "Lucrezia Borgia" by Victor Hugo, in its turn after the legend of Lucrezia Borgia.

 "Lucrezia Borgia" was first performed on 26 December 1833 at La Scala, Milan.
</item 9>

<item 10>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 10>
=== END ITEM id=bac2ec357b7783b5 ===

=== ITEM id=69080f6a9956271a ===
Gold sentences and facts:
Sentence 0: "Collateral Damage" is the eleventh episode of the third season of the crime-thriller television series "Millennium".
  Fact 0.0: 'Collateral Damage' is the eleventh episode of the third season of the crime-thriller television series 'Millennium'.
  Fact 0.1: 'Millennium' is a crime-thriller television series.
  Fact 0.2: 'Collateral Damage' is in the third season of 'Millennium'.
Sentence 1:  "Collateral Damage" featured guest appearances by O'Quinn, Jacinda Barrett, James Marsters, and radio host Art Bell as himself.
  Fact 1.0: 'Collateral Damage' featured a guest appearance by O'Quinn.
  Fact 1.1: 'Collateral Damage' featured a guest appearance by Jacinda Barrett.
  Fact 1.2: 'Collateral Damage' featured a guest appearance by James Marsters.
  Fact 1.3: 'Collateral Damage' featured a guest appearance by radio host Art Bell.
  Fact 1.4: Art Bell appeared as himself in 'Collateral Damage'.
Sentence 2:  Bell's radio show "Coast to Coast AM" was among Perry's inspirations for the script.
  Fact 2.0: Art Bell's radio show 'Coast to Coast AM' was among Perry's inspirations for the script.
Sentence 3: Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.
  Fact 3.0: Coast to Coast AM is an American late-night radio talk show.
  Fact 3.1: Coast to Coast AM deals with a variety of topics.
Sentence 4:  The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
  Fact 4.0: Coast to Coast AM now airs seven nights a week.
  Fact 4.1: Coast to Coast AM airs from 1:00 a.m. to 5:00 a.m. Eastern Time Zone.

Retrieved items:
<item 1>
Late-night talk show
A late-night talk show is a genre of talk show popular in the United States, where the format originated.

 It is generally structured around humorous monologues about the day's news, guest interviews, comedy sketches and music performances.

 The late-night talk show format was popularized, though not invented, by Johnny Carson with "The Tonight Show Starring Johnny Carson" on NBC.

 Typically the show's host conducts interviews from behind a desk, while the guest is seated on a couch.

 Many late night talk shows feature a house band which generally performs cover songs for the studio audience during commercial breaks and occasionally will
back up a guest artist.
</item 1>

<item 2>
List of Jimmy Fallon games and sketches
The following is a list of recurring games, sketches, and other comedy routines from the NBC late night talk show "The Tonight Show Starring Jimmy Fallon", and
its predecessor, "Late Night with Jimmy Fallon".

 The sketches feature host Jimmy Fallon, house band The Roots, announcer/sidekick Steve Higgins, the show's writers, celebrity guests, and audience members.

 Most of the skits below appeared only on "Late Night", while some have carried over to the "Tonight Show".
</item 2>

<item 3>
Morning Joe
Morning Joe is a weekday NBC News morning news and talk show, airing from 6:00 a.m. to 9:00 a.m. Eastern Time on the network's cable news channel MSNBC.

 It features former GOP representative Joe Scarborough providing both enterprise reporting and discussion on the news of the day in a panel format with co-hosts
Mika Brzezinski and Willie Geist.

 It was created as the replacement for "Imus in the Morning," which was cancelled in April 2007 after simulcasting on MSNBC since 1996.
</item 3>

<item 4>
David Letterman
David Michael Letterman (born April 12, 1947) is an American television host, comedian, writer, and producer.

 He hosted a late night television talk show for 33 years, beginning with the February 1, 1982 debut of "Late Night with David Letterman" on NBC, and ending
with the May 20, 2015 broadcast of "Late Show with David Letterman" on CBS.

 In total, Letterman hosted 6,028 episodes of "Late Night" and "Late Show", surpassing friend and mentor Johnny Carson as the longest-serving late night talk
show host in American television history.

 In 1996 Letterman was ranked 45th on "TV Guide"' s 50 Greatest TV Stars of All Time.
</item 4>

<item 5>
CNN World News
CNN World News, a program that airs on CNN International and CNN International Asia Pacific.

 It is supplemented by CNN World News Asia and CNN World News Europe The show's traditional time run is 24-hours if it is followed by CNN World News Middle East
The show's regular presenters include Errol Barnett.

 Its main role is to update viewers of the latest news in the world.

 It contains a weather update from the CNN World Weather Forecast News.

 CNN World News can usually air up to three times on weekends, and is known to be bringing the latest on a story.
</item 5>

<item 6>
Night After Night with Allan Havey
Night After Night with Allan Havey was a late night talk show which aired on The Comedy Channel and, later, Comedy Central from 1989 to 1992.

 The show featured the stream of consciousness of host Allan Havey, and other departures from the standard late-night talk show format.

 Havey's sidekick was Nick Bakay, who moved on to act as sidekick on Dennis Miller's short-lived late night talk show.
</item 6>

<item 7>
Collateral Damage (Millennium)
"Collateral Damage" is the eleventh episode of the third season of the crime-thriller television series "Millennium".

 It premiered on the Fox network on January 22, 1999.

 The episode was written by Michael R. Perry, and directed by Thomas J. Wright.

 "Collateral Damage" featured guest appearances by O'Quinn, Jacinda Barrett, James Marsters, and radio host Art Bell as himself.

 Bell's radio show "Coast to Coast AM" was among Perry's inspirations for the script.
</item 7>

<item 8>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 8>

<item 9>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 9>

<item 10>
Steve Malzberg
Stephen D. Malzberg (born April 20, 1959) is an American television and radio host, syndicated columnist, and political commentator.

 He is the host of "The Steve Malzberg Show", a cable news and opinion show on Newsmax TV.

 He has also hosted "The Steve Malzberg Show" on various talk radio outlets.
</item 10>
=== END ITEM id=69080f6a9956271a ===

=== ITEM id=c8f90e2616bebb63 ===
Gold sentences and facts:
Sentence 0: California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United States.
  Fact 0.0: California State Polytechnic University, Pomona is also called CPP, Cal Poly Pomona, or Cal Poly.
  Fact 0.1: California State Polytechnic University, Pomona is a public polytechnic university.
  Fact 0.2: California State Polytechnic University, Pomona is located in Pomona, California, United States.
Sentence 1:  It is one of two polytechnics in the California State University system.
  Fact 1.0: California State Polytechnic University, Pomona is one of two polytechnics in the California State University system.
Sentence 2:  Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.
  Fact 2.0: The university is named after the first President of Russia, B.N. Yeltsin.
  Fact 2.1: The full name of the university is 'Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina'.
  Fact 2.2: The university, Ural Federal University, is often shortened to UrFU or УрФУ.
  Fact 2.3: Ural Federal University was formed by a merger of the Ural State Technical University and Ural State University.
  Fact 2.4: Ural Federal University is one of the leading educational institutions in the Ural region.
Sentence 3:  Ural Federal University acts as a research and innovation center of the Ural region and has close cooperation with the Russian Academy of Sciences.
  Fact 3.0: Ural Federal University acts as a research and innovation center of the Ural region.
  Fact 3.1: Ural Federal University has close cooperation with the Russian Academy of Sciences.

Retrieved items:
<item 1>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 1>

<item 2>
Julian A. McPhee
Julian Aeneas McPhee (February 7, 1896 – November 10, 1967) was the sixth university president of California Polytechnic State University, San Luis Obispo (Cal
Poly SLO) from 1933 to 1966 and the first president of California State Polytechnic University, Pomona (Cal Poly Pomona) from 1938 to 1966.
</item 2>

<item 3>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 3>

<item 4>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 4>

<item 5>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 5>

<item 6>
International Polytechnic High School
International Polytechnic High School, commonly abbreviated merely as iPoly, is a public college preparatory demonstration high school (9-12) located on the
California State Polytechnic University, Pomona (Cal Poly Pomona) campus and operated by the Los Angeles County Office of Education (LACOE) in conjunction with
the College of Education and Integrative Studies at the university.

 iPoly's curriculum is fully accredited by the Western Association of Schools and Colleges and is approved by the University of California and California State
University.

 It maintains a unique affiliation with the Los Angeles County High School for the Arts (LACHSA), which is also run by LACOE.

 Since iPoly does not lie in a fixed school district, it draws students from throughout Los Angeles, San Bernardino, Orange and Riverside counties.

 The majority of students come from the Pomona and San Gabriel valleys.

 In 2009 and 2013, iPoly was honored as a California Distinguished School by the California Department of Education.
</item 6>

<item 7>
Cal Poly Pomona College of Engineering
The Cal Poly Pomona College of Engineering is the engineering college at California State Polytechnic University, Pomona (Cal Poly Pomona or Cal Poly) located
in Pomona, California, United States.

 Well known for its "learn by doing" philosophy, the college's motto is: ""making imagination real"".

 Cal Poly has one of the top engineering college among public schools in California and, with over 5,600 students (as of fall 2015), it is also the largest
engineering college in Southern California, the second largest college of engineering in the California State University system, and the seventeenth largest
engineering college in the United States.

 In the 2017 "U.S. News & World Report" the College of Engineering is ranked the 5th best undergraduate program among public universities (11th overall) in the
West for Master's-granting universities, and "has one of the top ranked engineering programs, and graduates roughly 1 of every 14 engineers in the state of
California."
</item 7>

<item 8>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 8>

<item 9>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 9>

<item 10>
Lyman: first U.S. boys’ reform school (Massachusetts, c.1886–1971), now historic. Massachusetts International Academy: Chinese-student U.S.-college prep
boarding school. Polytech’Paris-UPMC: French research-linked engineering school in an 11-school network. Cal Poly Pomona: public polytechnic; library holds
670,580 books; engineering enrolls 5,600+; environmental design 1,600+; women’s basketball compet
</item 10>
=== END ITEM id=c8f90e2616bebb63 ===

=== ITEM id=b20b9a231cd9a334 ===
Gold sentences and facts:
Sentence 0: Vytautas Kevin Gerulaitis (July 26, 1954 – September 17, 1994) was an American professional tennis player.
  Fact 0.0: Vytautas Kevin Gerulaitis was born on July 26, 1954.
  Fact 0.1: Vytautas Kevin Gerulaitis died on September 17, 1994.
  Fact 0.2: Vytautas Kevin Gerulaitis was an American professional tennis player.
Sentence 1: Heather Miriam Watson (born 19 May 1992) is a British tennis player and a Wimbledon Mixed Doubles Champion.
  Fact 1.0: Heather Miriam Watson was born on 19 May 1992.
  Fact 1.1: Heather Miriam Watson is a British tennis player.
  Fact 1.2: Heather Miriam Watson is a Wimbledon Mixed Doubles Champion.

Retrieved items:
<item 1>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 1>

<item 2>
Iveta Benešová
Iveta Benešová (] ) (formerly Melzer, Czech: Melzerová ; born 1 February 1983) is a Czech former professional tennis player.

 She began playing tennis at age of 7 and turned professional in 1998 in Prague.

 She has won two WTA Tour events and one Grand Slam in mixed doubles partnering with Jürgen Melzer at the 2011 Wimbledon Championships.

 On 14 September 2012 she married Jürgen Melzer and adopted his family name.

 She announced her retirement from professional tennis via her Facebook page 13/08/2014.

 In 2015, she divorced Melzer and reverted to using her maiden name.
</item 2>

<item 3>
Victoria Azarenka career statistics
This is a list of the main career statistics of Belarusian professional tennis player, Victoria Azarenka.

 To date, Azarenka has won twenty WTA singles titles including two grand slam singles titles, six WTA Premier Mandatory singles titles and three WTA Premier 5
singles titles.

 She was also the runner-up at the 2011 WTA Tour Championships, 2012 US Open and 2013 US Open.

 Among other achievements, Azarenka was the bronze medalist in women's singles and Gold medalist in mixed doubles with her compatriot, Max Mirnyi at the 2012
London Olympics.
</item 3>

<item 4>
1977 Italian Open (tennis)
The 1977 Italian Open was a combined men's and women's tennis tournament that was played by men on outdoor clay courts at the Foro Italico in Rome, Italy.

 The men's tournament was part of the Colgate-Palmolive Grand Prix circuit while the women's tournament was part of the Colgate Series.

 The tournament was held from 16 May through 22 May 1977.

 The singles titles were won by eight-seeded Vitas Gerulaitis and fifth-seeded Janet Newberry who earned $21,000 and $6,000 first-prize money respectively.

 Gerulaitis competed despite being contracted to play for the Indiana Loves World Team Tennis (WTT) franchise and was fined $19,000 for failing to play Björn
Borg in the weekend of the final.
</item 4>

<item 5>
1979 World Championship Tennis Finals – Singles
Vitas Gerulaitis was the defending champion but lost in the semifinals to Björn Borg.
</item 5>

<item 6>
Vitas Gerulaitis
Vytautas Kevin Gerulaitis (July 26, 1954 – September 17, 1994) was an American professional tennis player.

 In 1975, Gerulaitis won the men's doubles title at Wimbledon, partnering with Sandy Mayer.

 He won the men's singles title at one of the two Australian Open tournaments held in 1977 (Gerulaitis won the tournament that was held in December, while
Roscoe Tanner won the earlier January tournament).

 Gerulaitis also won two Italian Open titles, in 1977 and 1979, and the WCT Finals in Dallas, in 1978.
</item 6>

<item 7>
Kavita Tungar holds India’s 10-km and half-marathon records and won 2010 Commonwealth bronze and Asian Games silver in the 10,000m. Tennis profiles cover Kim
Warwick’s Australian Open final and doubles titles; Evonne Goolagong’s seven Grand Slam singles titles and former No. 1 ranking; and achievements by Ivanovic,
Janković, Stosur, Safina, Azarenka, Watson and Hlaváčková
</item 7>

<item 8>
Heather Watson
Heather Miriam Watson (born 19 May 1992) is a British tennis player and a Wimbledon Mixed Doubles Champion.

 She is a former British No.1 and current British No. 2 behind Johanna Konta.
</item 8>

<item 9>
At the 1976 Italian Open, Adriano Panatta and Mima Jaušovec won singles; Panatta survived 11 match points in round one. Jan Kodeš and Margaret Court won the
1970 French Open. Vitas Gerulaitis won Wimbledon doubles (1975), the 1977 Australian Open, Italian Opens (1977, 1979), and 1978 WCT Finals; he also won Brussels
(1982), Ocean City (1977
</item 9>

<item 10>
1974 Stadthalle Open – Singles
Vitas Gerulaitis won in the final 6–4, 3–6, 6–3, 6–2 against Andrew Pattison.
</item 10>
=== END ITEM id=b20b9a231cd9a334 ===

=== ITEM id=f95f1a48b3af37c3 ===
Gold sentences and facts:
Sentence 0:  He appeared in 57 films between 1919 and 1932.
  Fact 0.0: The person described appeared in 57 films.
  Fact 0.1: The person described appeared in 57 films between 1919 and 1932.
Sentence 1:  He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
  Fact 1.0: The person described directed 54 films.
  Fact 1.1: The person described directed these films between 1920 and 1933.
  Fact 1.2: The person's directed films include one Harry Langdon short.
  Fact 1.3: The person's directed films include two of the Tay Garnett-penned comedies Stan Laurel made for Joe Rock.
  Fact 1.4: The person's directed films include fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
Sentence 2: John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.
  Fact 2.0: John Scott Biddle lived from 1925 to 2008.
  Fact 2.1: John Scott Biddle was a foremost yachting cinematographer.
  Fact 2.2: John Scott Biddle was a lecturer.
  Fact 2.3: John Scott Biddle established a film-making career.
  Fact 2.4: John Scott Biddle's film-making career spanned more than forty years.

Retrieved items:
<item 1>
Harry Sweet
Harry Sweet (October 2, 1901 – June 18, 1933) was an American actor, director and screenwriter.

 He appeared in 57 films between 1919 and 1932.

 He also directed 54 films between 1920 and 1933, including one Harry Langdon short, two of the Tay Garnett- penned comedies Stan Laurel made for Joe Rock, and
fifteen of the earliest entries in the Edgar Kennedy "Average Man" series.
</item 1>

<item 2>
Alexander Butler
Alexander Butler was a British film director who made over sixty features and short films during the 1910s and 1920s including many for G.B. Samuelson's
production company.

 Butler directed several British films in Hollywood in 1920, where Samuelson had made an arrangement with Universal Pictures.

 Amongst his notable films are the Sherlock Holmes adaptation "The Valley of Fear" (1916) and the early British horror film "The Beetle" (1919).
</item 2>

<item 3>
John Biddle (Michigan politician)
John Biddle (March 2, 1792 – August 25, 1859) was a delegate to the United States Congress from the Michigan Territory.
</item 3>

<item 4>
Charles John Biddle
Charles John Biddle (April 30, 1819 – September 28, 1873) was an American soldier, lawyer, congressman, and newspaper editor.
</item 4>

<item 5>
Millard Webb
Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.

 He directed 20 films between 1920 and 1933.

 His best-known film is the 1926 silent John Barrymore adventure "The Sea Beast" costarring Dolores Costello.

 Webb also directed the early sound Florenz Ziegfeld produced talkie "Glorifying the American Girl" released by Paramount in 1929.

 His active years were from 1916 to 1933.
</item 5>

<item 6>
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
</item 6>

<item 7>
Biddle family
The American Biddle family of Philadelphia began when William Biddle (1630–1712) and Sarah Kempe (1634–1709), Quakers, emigrated from England to America in
1681, in part, to avoid religious persecution.

 Having acquired extensive rights to over 43000 acre of lands in Quaker West Jersey, they settled first at Burlington.

 Two third generation brothers, William Biddle, 3rd (1698–1756), and John Biddle (1707–1789), moved from Mount Hope (1684) near Bordentown to Philadelphia in
the 1720s and 1730s and constituted the first generation of the Philadelphia Biddle family.
</item 7>

<item 8>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 8>

<item 9>
John Biddle (United States Army general)
John Biddle (February 2, 1859 – January 18, 1936) was career United States Army officer who became superintendent of the United States Military Academy.
</item 9>

<item 10>
John Biddle (Unitarian)
John Biddle or Bidle (born Wotton-under-Edge, Gloucestershire, England, 14 January 1615 – died 22 September 1662) was an influential English nontrinitarian, and
Unitarian.

 He is often called "the Father of English Unitarianism".
</item 10>
=== END ITEM id=f95f1a48b3af37c3 ===

=== ITEM id=a4e787bb0a950bd5 ===
Gold sentences and facts:
Sentence 0: Jeffrey Charles "Jeff" Ragsdale ( ) is an American author, documentary filmmaker, actor and stand-up comedian.
  Fact 0.0: Jeffrey Charles "Jeff" Ragsdale is an American author.
  Fact 0.1: Jeffrey Charles "Jeff" Ragsdale is a documentary filmmaker.
  Fact 0.2: Jeffrey Charles "Jeff" Ragsdale is an actor.
  Fact 0.3: Jeffrey Charles "Jeff" Ragsdale is a stand-up comedian.
Sentence 1: Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.
  Fact 1.0: Millard Webb was born on 6 December 1893.
  Fact 1.1: Millard Webb died on 21 April 1935.
  Fact 1.2: Millard Webb was an American screenwriter.
  Fact 1.3: Millard Webb was an American director.

Retrieved items:
<item 1>
Sandy Rothman
Sandy Rothman (born January 30, 1946, Miami, Florida) is a San Francisco Bay Area bluegrass multi-instrumentalist and record producer.

 He plays mandolin, dobro and banjo, and he also sings.

 Rothman was a friend and colleague of Grateful Dead bandleader Jerry Garcia, and a member of the Jerry Garcia Acoustic Band.

 He played bluegrass with Garcia and David Nelson as the Black Mountain Boys in 1964, and has played with Bill Monroe and his Blue Grass Boys, Earl Taylor, Red
Allen, Jimmie Skinner, Larry Sparks, the Kentucky Colonels, Country Joe McDonald, Kathy Kallick and Clarence White, among other musicians.

 He has been described as "one of the chief biscuits when and where bluegrass music is discussed, scribed, or performed in northern California."
</item 1>

<item 2>
Howard R. Garis
Howard Roger Garis ((1873--) 25, 1873 – (1962--) 6, 1962 ) was an American author, best known for a series of books, published under his own name, that featured
the character of Uncle Wiggily Longears, an engaging elderly rabbit.

 Garis and his wife were possibly the most prolific children's authors of the early 20th century.

 Many of his books were illustrated by Lansing Campbell.
</item 2>

<item 3>
Richard Yoo
Richard Yoo is an American entrepreneur, the co-founder and former CEO of the web hosting company Rackspace, and the founder and former CEO of web hosting
company ServerBeach.
</item 3>

<item 4>
Two Rivers Press
Two Rivers Press is an independent publishing house, based in the English town of Reading.

 Two Rivers Press was founded in 1994 by Peter Hay (1951–2003), a local artist.

 Its name reflects his enthusiasm for the town and its two rivers, the Kennet and the Thames, and its intention to explore "the place where art and history
meet".

 The name also gives a clue to the origins of the company in the, ultimately successful, opposition to Reading's proposed "Cross-Town Route", a road scheme that
would have seriously impacted the point at which the two rivers meet.
</item 4>

<item 5>
Roger Garis
Roger (Carroll) Garis, ((1901--) 10, 1901 – (1967--) 30, 1967 ) was an American author known as a writer for magazines and the author of "The Outboard Boys"
series of books.

 Roger also wrote several books under pseudonyms for the Stratemeyer Syndicate, as his parents did (see My Father Was Uncle Wiggily for further details).
</item 5>

<item 6>
Uncle Wiggily
Uncle Wiggily Longears is the main character of a series of children's stories by American author Howard R. Garis.

 He began writing the stories for the "Newark News" in 1910.

 Garis penned an Uncle Wiggily story every day (except Sundays) for more than 30 years, and published 79 books within the author's lifetime.

 According to his obituary in the "Chicago Tribune", a walk in the woods in Verona, New Jersey was his inspiration.

 The books featured work by several illustrators, notably Lansing Campbell.

 Other illustrators of the series included George L. Carlson, Louis Wisa, Elmer Rache, Edward Bloomfield, Lang Campbell and Mary and Wallace Stover.
</item 6>

<item 7>
Lionel Snell
Ramsey Dukes is the current and most well-known pen name of Lionel Snell, a contemporary English magician, publisher and author on magic and philosophy.

 He has also under the pen names Lemuel Johnston.

 Angerford and Lea, Adamai Philotunus and Per Anum Ad Astra.

 His regular satirical column "The Satanist's Diary" was published in the magazine "Aquarian Arrow" during the 1980s and early 1990s under the pen name The Hon
Hugo C StJ l'Estrange and included a multitude of other pen names as contributors.
</item 7>

<item 8>
Jeff Ragsdale
Jeffrey Charles "Jeff" Ragsdale ( ) is an American author, documentary filmmaker, actor and stand-up comedian.

 In 2011 he posted a flyer in New York City as a "social experiment", stating his phone number and asking people to call him, describing himself as "Jeff, one
lonely guy".

 He was overwhelmed with thousands of calls after photos of the flyer were posted on the internet.

 The experience led to his 2012 book "Jeff, One Lonely Guy", and indirectly to a 2013 pilot episode for a reality television show, "Being Noticed", and a
starring role in the 2014 documentary "Hotline".
</item 8>

<item 9>
Jeff, One Lonely Guy
Jeff, One Lonely Guy is a 2012 nonfiction book by Jeff Ragsdale.

 It was published on March 20, 2012 by New Harvest.

 Dave Eggers selected the book for inclusion in The Best American Nonrequired Reading 2012, and it was a GQ 2012 "Book of the Year".

 In 2014 Amitava Kumar included portions of "Jeff, One Lonely Guy" in his newly released book, "A Matter of Rats: A Short Biography of Patna".

 Kumar previously interviewed Ragsdale and wrote about him in "The New York Times".
</item 9>

<item 10>
Millard Webb
Millard Webb (6 December 1893 – 21 April 1935), was an American screenwriter and director.

 He directed 20 films between 1920 and 1933.

 His best-known film is the 1926 silent John Barrymore adventure "The Sea Beast" costarring Dolores Costello.

 Webb also directed the early sound Florenz Ziegfeld produced talkie "Glorifying the American Girl" released by Paramount in 1929.

 His active years were from 1916 to 1933.
</item 10>
=== END ITEM id=a4e787bb0a950bd5 ===

=== ITEM id=da46c498ead641fa ===
Gold sentences and facts:
Sentence 0: Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who successfully worked as the film director, the screenwriter, the art director and the animator.
  Fact 0.0: Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич) was born on 4 September 1924.
  Fact 0.1: Vladimir Petrovich Danilevich died on 9 October 2001.
  Fact 0.2: Vladimir Petrovich Danilevich was a well-known Soviet animator.
  Fact 0.3: Vladimir Petrovich Danilevich was a well-known Russian animator.
  Fact 0.4: Vladimir Petrovich Danilevich successfully worked as a film director.
  Fact 0.5: Vladimir Petrovich Danilevich successfully worked as a screenwriter.
  Fact 0.6: Vladimir Petrovich Danilevich successfully worked as an art director.
  Fact 0.7: Vladimir Petrovich Danilevich successfully worked as an animator.
Sentence 1: Wolfgang Becker (born 22 June 1954) is a German film director and writer.
  Fact 1.0: Wolfgang Becker was born on 22 June 1954.
  Fact 1.1: Wolfgang Becker is a German film director.
  Fact 1.2: Wolfgang Becker is a German writer.
Sentence 2:  He is best known to the international audience for his work "Good Bye Lenin!
  Fact 2.0: Wolfgang Becker is best known to the international audience for his work "Good Bye Lenin!".
Sentence 3: " (2003).
  Fact 3.0: "Good Bye Lenin!" is from 2003.

Retrieved items:
<item 1>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 1>

<item 2>
Good Bye, Lenin!
Good Bye, Lenin!

 is a 2003 German tragicomedy film, directed by Wolfgang Becker.

 The cast includes Daniel Brühl, Katrin Saß, Chulpan Khamatova, and Maria Simon.

 Most scenes were shot at the Karl-Marx-Allee in Berlin and around Plattenbauten near Alexanderplatz.
</item 2>

<item 3>
Viennese Girls
Viennese Girls (German:Wiener Mädeln) is a 1945 historical musical film directed by Willi Forst and starring Forst, Anton Edthofer and Judith Holzmeister.

 The film was made by Wien-Film, a Vienna-based company set up after Austria had been incorporated into Greater Germany following the 1938 Anschluss.

 It was the third film in Forst's "Viennese Trilogy" which also included "Operetta" (1940) and "Vienna Blood" (1942).

 The film was finished in 1945, during the closing days of the Second World War.

 This led to severe delays in its release, which eventually took place in 1949 in two separate versions.

 One was released by the Soviet-backed Sovexport in the Eastern Bloc and the other by Forst.
</item 3>

<item 4>
Vladimir Danilevich
Vladimir Petrovich Danilevich (Russian: Владимир Петрович Данилевич ; 4 September 1924 — 9 October 2001) was well-known Soviet and Russian Animator: who
successfully worked as the film director, the screenwriter, the art director and the animator.
</item 4>

<item 5>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 5>

<item 6>
Tabarly (soundtrack)
Tabarly is the original soundtrack album of the documentary film of the same title.

 It is composer Yann Tiersen's first soundtrack since 2003's "Good Bye Lenin!

" and first studio album since 2005's "Les Retrouvailles".

 The album was recorded in Ushant and at Qu'en reste-t-il?

 studio in Paris, mixed at Davout studios in Paris, mastered at Loud Mastering, and released through Virgin Records and EMI France in CD format on 16 July 2008.

 A special edition consisting of a CD plus a LP record was released through Ici, d'ailleurs... record label.
</item 6>

<item 7>
Good Bye Lenin! (soundtrack)
Good Bye Lenin!

 is the original soundtrack album of the film of the same title starring Daniel Brühl and Katrin Sass.
</item 7>

<item 8>
Welcome, or No Trespassing
Welcome, or No Trespassing (Russian: Добро пожаловать, или Посторонним вход воспрещён , "Dobro pozhalovat, ili Postoronnim vkhod vospreshchyon " ) is a Soviet
movie by Elem Klimov made in 1964.

 It is a satirical comedy about the excessive restrictions that children face during their vacation in a Young Pioneer camp, imposed by their masters.

 Most of the actors are children, while the protagonist is the director Dynin, played by Yevgeniy Yevstigneyev.

 The film was selected to be screened in the Cannes Classics section of the 2015 Cannes Film Festival.
</item 8>

<item 9>
Wolfgang Becker
Wolfgang Becker (born 22 June 1954) is a German film director and writer.

 He is best known to the international audience for his work "Good Bye Lenin!

" (2003).
</item 9>

<item 10>
Elem Klimov
Elem Germanovich Klimov (Russian: Эле́м Ге́рманович Кли́мов ; 9 July 1933 – 26 October 2003) was a Soviet Russian film director.

 He studied at VGIK, and was married to film director Larisa Shepitko.

 Klimov is best known in the West for his final film, 1985's "Come and See" ("Иди и смотри"), which follows a teenage boy in German-occupied Belarus during the
German-Soviet War and is often considered one of the greatest war films ever made.

 He also directed dark comedies, children's movies, and historical pictures.
</item 10>
=== END ITEM id=da46c498ead641fa ===

=== ITEM id=217c46f3db1e4669 ===
Gold sentences and facts:
Sentence 0: The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.
  Fact 0.0: The Battle of Okinawa is also known as 沖縄戦 (Okinawa-sen in Hepburn romanization) and as 'Uchinaa ikusa' in Okinawan.
  Fact 0.1: The Battle of Okinawa was codenamed Operation Iceberg.
  Fact 0.2: The Battle of Okinawa was a major battle of the Pacific War.
  Fact 0.3: The Battle of Okinawa was fought on the island of Okinawa.
  Fact 0.4: The Battle of Okinawa was fought by United States Marine and Army forces against the Imperial Japanese Army.
  Fact 0.5: The Battle of Okinawa included the largest amphibious assault in the Pacific War during World War II.
  Fact 0.6: The largest amphibious assault in the Pacific War was the April 1, 1945 invasion of Okinawa itself.
Sentence 1: Operation Diadem, also referred to as the Fourth Battle of Monte Cassino or, in Canada, the Battle of the Liri Valley, was an offensive operation undertaken by the Allies of World War II (U.S. Fifth Army and British Eighth Army in May 1944, as part of the Italian Campaign of World War II.
  Fact 1.0: Operation Diadem was also referred to as the Fourth Battle of Monte Cassino.
  Fact 1.1: In Canada, Operation Diadem was called the Battle of the Liri Valley.
  Fact 1.2: Operation Diadem was an offensive operation undertaken by the Allies of World War II.
  Fact 1.3: The Allies in Operation Diadem included the U.S. Fifth Army.
  Fact 1.4: The Allies in Operation Diadem included the British Eighth Army.
  Fact 1.5: Operation Diadem took place in May 1944.
  Fact 1.6: Operation Diadem was part of the Italian Campaign of World War II.

Retrieved items:
<item 1>
Operation Epsom
Operation Epsom, also known as the First Battle of the Odon, was a British Second World War offensive that took place between 26 and 30 June 1944, during the
Battle of Normandy.

 The offensive was intended to outflank and seize the German-occupied city of Caen, an important Allied objective, in the early stages of Operation Overlord,
the Allied invasion of north-west Europe.
</item 1>

<item 2>
Operation Diadem order of battle
Operation Diadem order of battle is a listing of the significant formations that were involved in the fighting on the Winter Line and at the Anzio bridgehead
south of Rome during "Operation Diadem" in May - June 1944 which resulted in the Allied breakthrough at Cassino and the breakout at Anzio leading to the capture
of Rome.
</item 2>

<item 3>
Operation Diadem
Operation Diadem, also referred to as the Fourth Battle of Monte Cassino or, in Canada, the Battle of the Liri Valley, was an offensive operation undertaken by
the Allies of World War II (U.S. Fifth Army and British Eighth Army in May 1944, as part of the Italian Campaign of World War II.

 "Diadem" was supported by air attacks called Operation Strangle.

 The opposing force was the German 10th Army.
</item 3>

<item 4>
Operation Kikusui
Operation Kikusui (菊水作戦 , Kikusui sakusen ) was a series of suicidal air attacks by Imperial Japanese forces during the Battle of Okinawa against Allied fleets
in the waters around Okinawa, as part of Operation Ten-Go.

 The name of the operation, "Kikusui", comes from the hata-jirushi of the samurai Kusunoki Masashige.
</item 4>

<item 5>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 5>

<item 6>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 6>

<item 7>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 7>

<item 8>
HMS Illustrious (87)
Her aircraft attacked several targets in the Japanese-occupied Dutch East Indies over the following year before "Illustrious" was transferred to the newly
formed British Pacific Fleet (BPF).

 The carrier participated in the early stages of the Battle of Okinawa until mechanical defects arising from accumulated battle damage became so severe that she
was ordered home early for repairs in May 1945.
</item 8>

<item 9>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 9>

<item 10>
Battle of Bataan
The Battle of Bataan (7 January – 9 April 1942) represented the most intense phase of Imperial Japan's invasion of the Philippines during World War II.

 In January 1942, forces of the Imperial Japanese Army and Navy invaded Luzon along with several islands in the Philippine Archipelago after the bombing of the
American naval base at Pearl Harbor.

 The commander-in-chief of all Filipino and American forces in the islands, General Douglas MacArthur, consolidated all of his Luzon-based units on the Bataan
Peninsula to fight against the Japanese invaders.

 By this time, the Japanese controlled nearly all of Southeast Asia.

 The Bataan peninsula and the island of Corregidor were the only remaining Allied strongholds in the region.

 Despite a lack of supplies, Filipino and American forces managed to fight the Japanese for three months, engaging them initially in a fighting retreat
southward.

 As the combined Filipino and American forces made a last stand, the delay cost the Japanese valuable time and prevented immediate victory across the Pacific.
</item 10>
=== END ITEM id=217c46f3db1e4669 ===

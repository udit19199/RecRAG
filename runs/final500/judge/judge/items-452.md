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

=== ITEM id=10ee1fdabfbc5ebf ===
Gold sentences and facts:
Sentence 0: Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).
  Fact 0.0: Dictyosperma is a monotypic genus of flowering plant.
  Fact 0.1: Dictyosperma is in the palm family.
  Fact 0.2: Dictyosperma is found in the Mascarene Islands in the Indian Ocean.
  Fact 0.3: Dictyosperma is found in Mauritius.
  Fact 0.4: Dictyosperma is found in Réunion.
  Fact 0.5: Dictyosperma is found in Rodrigues.
Sentence 1: The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in 1810.
  Fact 1.0: The genus Huernia is in the family Apocynaceae.
  Fact 1.1: The genus Huernia is in the subfamily Asclepiadoideae.
  Fact 1.2: The genus Huernia consists of stem succulents.
  Fact 1.3: The stem succulents of the genus Huernia are from Eastern Africa.
  Fact 1.4: The stem succulents of the genus Huernia are from Southern Africa.
  Fact 1.5: The genus Huernia was first described as a genus in 1810.

Retrieved items:
<item 1>
Bognera
Bognera is a monotypic genus of flowering plants in the Araceae family.

 The single species that makes up the genus is Bognera recondita.

 The word "recondita" means "hidden" referring to the fact that the plant is only found in remote areas of Amazonian Brazil near the Peruvian border.

 The species was discovered in the late 1970s and was originally placed in the genus "Ulearum".

 In 1984 a new genus, "Bognera" named after the German Aroid specialist Josef Bogner, was created for it.

 "Bognera" is believed to be closely related to "Dieffenbachia".
</item 1>

<item 2>
Dictyosperma
Dictyosperma is a monotypic genus of flowering plant in the palm family found in the Mascarene Islands in the Indian Ocean (Mauritius, Réunion and Rodrigues).

 The sole species, Dictyosperma album, is widely cultivated in the tropics but has been farmed to near extinction in its native habitat.

 It is commonly called princess palm or hurricane palm, the latter owing to its ability to withstand strong winds by easily shedding leaves.

 It is closely related to, and resembles, palms in the "Archontophoenix" genus.

 The genus is named from two Greek words meaning "net" and "seed" and the epithet is Latin for "white", the common color of the crownshaft at the top of the
trunk.
</item 2>

<item 3>
Huernia
The genus Huernia (family Apocynaceae, subfamily Asclepiadoideae) consists of stem succulents from Eastern and Southern Africa, first described as a genus in
1810.

 The flowers are five-lobed, usually somewhat more funnel- or bell-shaped than in the closely related genus "Stapelia", and often striped vividly in contrasting
colours or tones, some glossy, others matt and wrinkled depending on the species concerned.

 To pollinate, the flowers attract flies by emitting a scent similar to that of carrion.

 The genus is considered close to the genera "Stapelia" and "Hoodia".

 The name is in honour of Justin Heurnius (1587–1652) a Dutch missionary who is reputed to have been the first collector of South African Cape plants.

 His name was actually mis-spelt by the collector.
</item 3>

<item 4>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 4>

<item 5>
Stephanocereus
Stephanocereus is genus of cactus from Brazil, related to "Arrojadoa".

 This genus was monotypic until "Pilosocereus luetzelburgii" was included here, resulting in the combination "Stephanocereus luetzelburgii".
</item 5>

<item 6>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 6>

<item 7>
Harrimanella
Harrimanella is a genus of flowering plant in the heath family Ericaceae, with a single species, Harrimanella hypnoides, also known as moss bell heather.

 It was originally named Cassiope hypnoides by Linnaeus (1737) in his Flora Lapponica, but "Harrimanella hypnoides" is now the accepted name at ITIS.

 The species name "hypnoides" means 'like "Hypnum" ', which is a genus mosses.
</item 7>

<item 8>
Duranta
Duranta is a genus of flowering plants in the verbena family, Verbenaceae.

 It contains 17 species of shrubs and small trees that are native from southern Florida to Mexico and South America.

 They are commonly cultivated as hedges and ornamental plants.
</item 8>

<item 9>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 9>

<item 10>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 10>
=== END ITEM id=10ee1fdabfbc5ebf ===

=== ITEM id=cd0f9922d15a48fc ===
Gold sentences and facts:
Sentence 0:  The Gaels, led by sixth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic Conference (MAAC).
  Fact 0.0: The Gaels were led by sixth year head coach Tim Cluess.
  Fact 0.1: The Gaels played their home games at the Hynes Athletic Center.
  Fact 0.2: The Gaels were members of the Metro Atlantic Athletic Conference (MAAC).
Sentence 1:  The Cyclones were coached by Steve Prohm, who was in his 1st season.
  Fact 1.0: The Cyclones were coached by Steve Prohm.
  Fact 1.1: Steve Prohm was in his 1st season.

Retrieved items:
<item 1>
2010–11 Iona Gaels men's basketball team
The 2010–11 Iona Gaels men's basketball team represented Iona College during the 2010–11 NCAA Division I men's basketball season.

 The Gaels, led by first year head coach Tim Cluess, played their home games at Hynes Athletics Center and are members of the Metro Atlantic Athletic
Conference.

 They finished the season 25–12, 13–5 in MAAC and lost in the championship game of the 2011 MAAC Men's Basketball Tournament to Saint Peter's.

 They were invited to the 2011 CollegeInsider.com Tournament where they advanced to the championship game before falling to Santa Clara.
</item 1>

<item 2>
2015–16 Iowa State Cyclones men's basketball team
The 2015–16 Iowa State Cyclones men's basketball team represents Iowa State University during the 2015–16 NCAA Division I men's basketball season.

 The Cyclones were coached by Steve Prohm, who was in his 1st season.

 They played their home games at Hilton Coliseum in Ames, Iowa and competed in the Big 12 Conference.
</item 2>

<item 3>
2015–16 Iona Gaels men's basketball team
The 2015–16 Iona Gaels men's basketball team represented Iona College during the 2015–16 NCAA Division I men's basketball season.

 The Gaels, led by sixth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference (MAAC).

 They finished the season 22–11, 16–4 in MAAC play to finish in second place.

 They defeated Canisius, Siena, and Monmouth to be champions of the MAAC Tournament and earn the conference's automatic bid to the NCAA Tournament where, as a
#13 seed, they lost in the first round to Iowa State.
</item 3>

<item 4>
2014–15 Iona Gaels men's basketball team
The 2014–15 Iona Gaels men's basketball team represented Iona College during the 2014–15 NCAA Division I men's basketball season.

 The Gaels, led by fifth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference.

 They finished the season 26–9, 17–3 in MAAC play to finish win the MAAC regular season championship.

 They defeated Siena and Monmouth to advance to the championship game of the MAAC Tournament where they lost to Manhattan.

 As a regular season champion who failed to win their conference tournament, they received an automatic bid to the National Invitation Tournament where they
lost in the first round to Rhode Island.
</item 4>

<item 5>
2016–17 Iona Gaels men's basketball team
The 2016–17 Iona Gaels men's basketball team represented Iona College during the 2016–17 NCAA Division I men's basketball season.

 The Gaels, led by seventh year head coach Tim Cluess, played their home games at the Hynes Athletic Center in New Rochelle, New York as members of the Metro
Atlantic Athletic Conference (MAAC).

 They finished the season 22–13, 12–8 in MAAC play to finish in a tie for third place.

 They defeated Rider, Saint Peter's and Siena to be champions of the MAAC Tournament.

 They received the MAAC's automatic bid to the NCAA Tournament where they lost in the First Round to Oregon.
</item 5>

<item 6>
2011–12 Iona Gaels men's basketball team
The 2011–12 Iona Gaels men's basketball team represented Iona College during the 2011–12 NCAA Division I men's basketball season.

 The Gaels, led by second year head coach Tim Cluess, played their home games at the Hynes Athletic Center and are members of the Metro Atlantic Athletic
Conference.

 The Gaals were MAAC regular season champions but failed to win the MAAC Basketball Tournament after losing to Fairfield in the semifinals.

 They received an at–large bid to the 2012 NCAA Tournament where they lost to BYU in the "First Four" round.

 The Gaels led BYU by 25 points in the 2nd half.

 It was the largest comeback in NCAA Tournament history.
</item 6>

<item 7>
David Hobbs (basketball)
David A. Hobbs (born April 25, 1949) is an American basketball coach.

 Hobbs currently serves as a special assistant to Iowa State's head coach Steve Prohm.

 He was the men's head coach at the University of Alabama from 1992 to 1998 and also was an assistant coach at Alabama, the University of Kentucky and Virginia
Commonwealth University.
</item 7>

<item 8>
Bobby Hurley
Robert Matthew Hurley (born June 28, 1971) is an American basketball coach, and former college and professional player.

 Hurley is currently the head coach of the Arizona State men's team.

 He was previously the head coach at Buffalo.

 Before becoming a head coach he was an assistant coach for Wagner and an associate head coach for Rhode Island on the staffs of his younger brother Dan.
</item 8>

<item 9>
2013–14 Iona Gaels men's basketball team
The 2013–14 Iona Gaels men's basketball team represented Iona College during the 2013–14 NCAA Division I men's basketball season.

 The Gaels, led by fourth year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference.

 They finished the season 22–11, 17–3 to win the MAAC regular season championship.

 They advanced to the championship game of the MAAC Tournament where they lost to Manhattan.

 As a regular season conference champion who failed to win their conference tournament, they received an automatic bid to the National Invitation Tournament
where they lost in the first round to Louisiana Tech.
</item 9>

<item 10>
2012–13 Iona Gaels men's basketball team
The 2012–13 Iona Gaels men's basketball team represented Iona College during the 2012–13 NCAA Division I men's basketball season.

 The Gaels, led by third year head coach Tim Cluess, played their home games at the Hynes Athletic Center and were members of the Metro Atlantic Athletic
Conference.

 They finished the season 20–14, 11–7 in MAAC play to finish in a tie for fourth place.

 They were champions of the MAAC Tournament, defeating Manhattan in the championship game, to earn an automatic bid to the 2013 NCAA Tournament where they lost
in the second round to Ohio State.
</item 10>
=== END ITEM id=cd0f9922d15a48fc ===

=== ITEM id=45654535b5b6d99c ===
Gold sentences and facts:
Sentence 0:  It featured Reese Witherspoon and Paul Rudd, prior to both becoming considerably bigger film stars.
  Fact 0.0: The film featured Reese Witherspoon and Paul Rudd.
  Fact 0.1: Reese Witherspoon and Paul Rudd were in the film before both became considerably bigger film stars.
Sentence 1:  Born in New Orleans and raised in Tennessee, she began her career as a child actress, making her professional screen debut in "The Man in the Moon" (1991), for which she was nominated for a Young Artist Award.
  Fact 1.0: The actress was born in New Orleans.
  Fact 1.1: The actress was raised in Tennessee.
  Fact 1.2: The actress began her career as a child actress.
  Fact 1.3: The actress made her professional screen debut in 'The Man in the Moon' (1991).
  Fact 1.4: The actress was nominated for a Young Artist Award for her debut in 'The Man in the Moon'.

Retrieved items:
<item 1>
Annie Belle
Annie Brilland (born 10 December 1956) is a French actress and social worker.

 Her acting career began in 1974 and throughout the seventies, has had a series of varied roles in both French and Italian cinema, working with such directors
as Jean Rollin, Ruggero Deodato and Joe D'Amato.
</item 1>

<item 2>
Middle of Nowhere (2012 film)
Middle of Nowhere is a 2012 independent feature film written and directed by Ava DuVernay and starring Emayatzy Corinealdi, David Oyelowo, Omari Hardwick and
Lorraine Toussaint.

 The film was the winner of the Directing Award for U.S. Dramatic Film at the 2012 Sundance Film Festival.
</item 2>

<item 3>
Reese Witherspoon
Laura Jeanne Reese Witherspoon ( ; born March 22, 1976) is an American actress, producer, and entrepreneur.

 Born in New Orleans and raised in Tennessee, she began her career as a child actress, making her professional screen debut in "The Man in the Moon" (1991), for
which she was nominated for a Young Artist Award.

 Following breakout roles in "" (1992) and "Jack the Bear" (1993), she starred in the comedy-drama "Pleasantville" (1998), for which she won the Young Hollywood
Award for Breakthrough Performance.

 Her leading role of Tracy Flick in "Election" (1999) was nominated for a Golden Globe Award.
</item 3>

<item 4>
Doreen Ketchens
Doreen Ketchens (born October 3, 1966 in New Orleans, Louisiana) is a jazz clarinetist, who performs Dixieland and Trad Jazz.

 She is one of the first and few female bandleaders in New Orleans, and a musical educator.

 She has performed at concert halls and music festivals, at U.S. Embassies and decades of weekly performances in Dixieland's tradition in the Royal Street
Performing Arts Zone in the French Quarter of New Orleans with her band, Doreen's Jazz New Orleans.

 Ketchens has performed for four U.S. Presidents: Bill Clinton, George Bush Sr., Ronald Reagan and Jimmy Carter, and is widely considered one of the cultural
ambassadors of New Orleans and of the traditional music
</item 4>

<item 5>
City of New Orleans (train)
The City of New Orleans is an Amtrak passenger train which operates on an overnight schedule between Chicago and New Orleans.

 The train is a successor to the Illinois Central Railroad's "Panama Limited".

 The present name was revived in 1981.
</item 5>

<item 6>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 6>

<item 7>
Overnight Delivery
Overnight Delivery is a 1998 romantic comedy film directed by Jason Bloom.

 It was rated PG-13 by the MPAA and released direct-to-video.

 It featured Reese Witherspoon and Paul Rudd, prior to both becoming considerably bigger film stars.
</item 7>

<item 8>
Anastasia Horne
Anastasia Nicole "Ana" Horne (born August 5, 1978 in Claremont, California) is a former American actress and singer who has appeared on television and in the
theater.

 Her last television role was "Lori" on the MTV series "Undressed" in 2000.

 She also starred as "Lark Madison-Scanlon" on the daytime soap opera "Port Charles" from 1997 to 1999.

 As a teenager, she was cast as "Ana" on the long-running NBC/The Disney Channel series "Kids Incorporated" from 1984, to 1985, and 1991 to the end of the
series in 1994.
</item 8>

<item 9>
Canadian actress Rachelle Lefevre starred in *Big Wolf on Campus*, played Victoria in the first two *Twilight* films, and appeared in *Off the Map*, *A Gifted
Man*, and *Under the Dome*. American actress Gloria Garayua debuted in *Fun with Dick and Jane* and had roles in *Grey’s Anatomy*, *Cougar Town*, and *How to
Get Away with Murder*. Zooey Deschanel’s films include *Almost Famous*, *All
</item 9>

<item 10>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 10>
=== END ITEM id=45654535b5b6d99c ===

=== ITEM id=4db486b823daa9f0 ===
Gold sentences and facts:
Sentence 0: The Doris Bither case, also known as the Entity hauntings, was an alleged haunting which occurred in 1974 at Culver City, California where a woman named Doris Bither alleged the ghosts of three Asian men were raping her.
  Fact 0.0: The Doris Bither case is also known as the Entity hauntings.
  Fact 0.1: The Doris Bither case was an alleged haunting.
  Fact 0.2: The alleged haunting occurred in 1974.
  Fact 0.3: The alleged haunting occurred at Culver City, California.
  Fact 0.4: Doris Bither alleged that the ghosts of three Asian men were raping her.
Sentence 1: The Entity is a 1982 American horror film directed by Sidney J. Furie and written by Frank De Felitta, who adapted his 1978 novel of the same name.
  Fact 1.0: The Entity is a 1982 American horror film.
  Fact 1.1: The Entity was directed by Sidney J. Furie.
  Fact 1.2: The Entity was written by Frank De Felitta.
  Fact 1.3: Frank De Felitta adapted his 1978 novel of the same name for The Entity.
Sentence 2:  Like the novel, the film is based on the 1974 Doris Bither case.
  Fact 2.0: The film is based on the 1974 Doris Bither case.
  Fact 2.1: The novel is also based on the 1974 Doris Bither case.

Retrieved items:
<item 1>
Sheila Levine Is Dead and Living in New York
Sheila Levine Is Dead and Living in New York is a 1975 film directed by Sidney J. Furie, about a shy young woman who moves to New York City and falls in love
with the boyfriend of her extroverted roommate.

 The film was co-written by Kenny Solms and Gail Parent, and based on her novel.

 The film was shot on location in New York City.
</item 1>

<item 2>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 2>

<item 3>
Midnight (1934 film)
Midnight is a 1934 American drama film directed by Chester Erskine and starring Sidney Fox, O.P. Heggie, Henry Hull and Margaret Wycherly.

 The film was produced for Universal and was shot on a modest budget of $50,000 at Thomas Edison Studios, which producer/director Chester Erskine had re-opened
specifically for the shoot.
</item 3>

<item 4>
Doris Bither case
The Doris Bither case, also known as the Entity hauntings, was an alleged haunting which occurred in 1974 at Culver City, California where a woman named Doris
Bither alleged the ghosts of three Asian men were raping her.

 The case inspired Frank De Felitta's 1978 book "The Entity" which was made into a 1982 film of the same name starring Barbara Hershey.
</item 4>

<item 5>
Night Life (film)
Night Life is a 1989 American zombie comedy film directed by David Acomba and starring Scott Grimes, John Astin, and Cheryl Pollak.

 It centers around a high school nerd named Archie Melville who is harassed from beyond the grave by the zombie corpses of his former high school bullies
(recently deceased) after they are brought back to life by a freak lightning storm.

 It is also known as Grave Misdemeanors.
</item 5>

<item 6>
The Entity
The Entity is a 1982 American horror film directed by Sidney J. Furie and written by Frank De Felitta, who adapted his 1978 novel of the same name.

 It stars Barbara Hershey as a woman who is raped and tormented by an invisible assailant.

 Despite being filmed and planned for a release in 1981, the movie was not released in worldwide theaters until September 1982 and February 1983 in the United
States.

 Like the novel, the film is based on the 1974 Doris Bither case.
</item 6>

<item 7>
Little Fauss and Big Halsy
Little Fauss and Big Halsy is a 1970 film directed by Sidney J. Furie, starring Robert Redford and Michael J. Pollard, also featuring Lauren Hutton, Noah Beery,
Jr. and Lucille Benson.
</item 7>

<item 8>
Doctor Blood's Coffin
Doctor Blood's Coffin is a 1961 British horror film directed by Sidney J. Furie.

 Kieron Moore stars as a mad scientist who reanimates the dead.
</item 8>

<item 9>
The Rocky Horror Picture Show
The Rocky Horror Picture Show is a 1975 British-American musical screwball comedy horror film by 20th Century Fox produced by Lou Adler and Michael White and
directed by Jim Sharman.

 The screenplay was written by Sharman and actor Richard O'Brien, who appears in the film, which is based on the 1973 musical stage production "of the same
title", with music, book, and lyrics by O'Brien.

 The production is a parody tribute to the science fiction and horror B movies of the 1930s through to the early 1970s.

 Along with O'Brien, the film stars Tim Curry, Susan Sarandon, and Barry Bostwick and is narrated by Charles Gray with cast members from the original Royal
Court Theatre, Roxy Theatre, and Belasco Theatre productions.
</item 9>

<item 10>
The Snake Woman
The Snake Woman is a low budget 1961 British horror film directed by Sidney J. Furie and starring Susan Travers, John McCarthy.
</item 10>
=== END ITEM id=4db486b823daa9f0 ===

=== ITEM id=d596674387c2e9c8 ===
Gold sentences and facts:
Sentence 0: Basie Meets Bond is a 1966 album by Count Basie and his orchestra.
  Fact 0.0: Basie Meets Bond is a 1966 album.
  Fact 0.1: Basie Meets Bond is an album by Count Basie.
  Fact 0.2: Basie Meets Bond is an album by Count Basie and his orchestra.
Sentence 1:  The album is a collection of musical pieces from the first four James Bond films; "Dr No", "From Russia with Love", "Goldfinger" and "Thunderball".
  Fact 1.0: The album is a collection of musical pieces.
  Fact 1.1: The album's musical pieces come from the first four James Bond films.
  Fact 1.2: The first four James Bond films include Dr No.
  Fact 1.3: The first four James Bond films include From Russia with Love.
  Fact 1.4: The first four James Bond films include Goldfinger.
  Fact 1.5: The first four James Bond films include Thunderball.
Sentence 2: William James "Count" Basie (August 21, 1904 – April 26, 1984) was an American jazz pianist, organist, bandleader, and composer.
  Fact 2.0: William James 'Count' Basie lived from August 21, 1904 to April 26, 1984.
  Fact 2.1: Count Basie was an American jazz pianist.
  Fact 2.2: Count Basie was an American organist.
  Fact 2.3: Count Basie was an American bandleader.
  Fact 2.4: Count Basie was an American composer.

Retrieved items:
<item 1>
Basie (album)
Basie (reissued as The Band of Distinction) is an album by pianist/bandleader Count Basie recorded in 1954 and originally released on the Clef label.

 the album should not be confused with Basie's 1958 album which became known as "The Atomic Mr. Basie".
</item 1>

<item 2>
Count Basie/Sarah Vaughan
Count Basie/Sarah Vaughan is a 1961 album by American jazz singer Sarah Vaughan, accompanied by the Count Basie Orchestra, with arrangements by Frank Foster,
Thad Jones and Ernie Wilkins.

 According to James Gavin's liner notes to the 1996 CD release, Basie himself does not perform on any of the tracks.
</item 2>

<item 3>
Count Basie at Newport
Count Basie at Newport is a live album by jazz musician Count Basie and his orchestra.

 It was originally issued as Verve MGV 8243 and included only the tracks 1-8 and 13.

 Tracks 9-12 originally included in "Count Basie & Joe Williams/Dizzy Gillespie & Mary Lou Williams at Newport" (Verve MGV 8244).
</item 3>

<item 4>
The Atomic Mr. Basie
The Atomic Mr. Basie (originally called Basie, also known as E=MC and reissued in 1994 as The Complete Atomic Basie) is a 1958 album by Count Basie and his
orchestra.

 The album is one of Basie's most famous and is critically acclaimed.

 Allmusic gave it 5 stars, reviewer Bruce Eder saying "it took Basie's core audience and a lot of other people by surprise, as a bold, forward-looking statement
within the context of a big-band recording."

 It is included in the book "1001 Albums You Must Hear Before You Die", Will Fulford-Jones calling it "Basie's last great record."

 According to the 2015 site update of Acclaimed Music, it is the 6th most critically acclaimed album of 1958, the 25th most acclaimed of the 1950s, and the
837th most acclaimed of all time, based on an aggregation of hundreds of critics' lists from around the world.
</item 4>

<item 5>
Basie Jazz
Basie Jazz is an album by pianist/bandleader Count Basie recorded in 1952 and released on the Clef label in 1954.

 Selections from this album were also released on the 1956 Clef LPs "The Swinging Count!

" and "Basie Rides Again!

".
</item 5>

<item 6>
Basie's Beatle Bag
Basie's Beatle Bag is a 1966 studio album by Count Basie and his orchestra, arranged by Chico O'Farrill.

 Basie released a second album of Beatles songs, "Basie on the Beatles", in 1969.
</item 6>

<item 7>
Count Basie
William James "Count" Basie (August 21, 1904 – April 26, 1984) was an American jazz pianist, organist, bandleader, and composer.

 His mother taught him to play the piano and he started performing in his teens.

 Dropping out of school, he learned to operate lights for vaudeville and to improvise accompaniment for silent films at a local movie theater in his home town
of Red Bank, New Jersey.

 By age 16, he increasingly played jazz piano at parties, resorts and other venues.

 In 1924, he went to Harlem, where his performing career expanded; he toured with groups to the major jazz cities of Chicago, St. Louis and Kansas City.

 In 1929 he joined Bennie Moten's band in Kansas City, and played with them until Moten's death in 1935.
</item 7>

<item 8>
Basie on the Beatles
Basie on the Beatles is an album by pianist and bandleader Count Basie featuring performances recorded in late 1969 and released on the short-lived Happy Tiger
label.

 It was Basie's second album of Beatles' compositions following 1966's "Basie's Beatle Bag" and featured liner notes by Ringo Starr.
</item 8>

<item 9>
One O'Clock Jump
"One O'Clock Jump" is a jazz standard, a 12-bar blues instrumental, written by Count Basie in 1937.

 The melody derived from band members' riffs--Basie rarely wrote down musical ideas, so Eddie Durham and Buster Smith helped him crystallize his ideas.

 The original 1937 recording of the tune by Basie and his band is noted for the saxophone work of Herschel Evans and Lester Young, trumpet by Buck Clayton,
Walter Page on bass and Basie himself on piano.
</item 9>

<item 10>
Basie Meets Bond
Basie Meets Bond is a 1966 album by Count Basie and his orchestra.

 The album is a collection of musical pieces from the first four James Bond films; "Dr No", "From Russia with Love", "Goldfinger" and "Thunderball".

 The album was Basie's first for United Records, and was produced by Teddy Reig.
</item 10>
=== END ITEM id=d596674387c2e9c8 ===

=== ITEM id=ff370ed34e691c51 ===
Gold sentences and facts:
Sentence 0:  He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.
  Fact 0.0: He played college football at Hampton University.
  Fact 0.1: He attended Riverdale High School.
  Fact 0.2: Riverdale High School is in Riverdale, Georgia.
Sentence 1: Riverdale is a city in Clayton County, Georgia, United States.
  Fact 1.0: Riverdale is a city.
  Fact 1.1: Riverdale is in Clayton County, Georgia.
  Fact 1.2: Clayton County, Georgia is in the United States.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Atlanta South Regional Airport
Henry County Airport (FAA LID: KHMP) , is a public-use county airport located three nautical miles (6 km) west of the central business district of Hampton, a
city in Henry County, Georgia, United States.

 It was known as Clayton County Airport – Tara Field, which was the name still used by the Federal Aviation Administration (FAA) and the Georgia Department of
Transportation (GDOT) until approval of the Atlanta South Regional Airport name.

 The airport was renamed Atlanta South Regional Airport, which was approved by the airport board of commissioners in December 2011, and was approved by the GDOT
and the FAA before it went into effect.
</item 2>

<item 3>
Riverdale, Georgia
Riverdale is a city in Clayton County, Georgia, United States.

 The population was 15,134 at the 2010 census, up from 12,478 in 2000.

 Riverdale is a suburb just south of Hartsfield–Jackson Atlanta International Airport and is part of the Atlanta metropolitan area.
</item 3>

<item 4>
College of Coastal Georgia
The College of Coastal Georgia (also known as CCGA) is a public college located at historical Brunswick, Georgia, United States and the surrounding historical
Golden Isles.

 The College was established in 1961 and opened in 1964, making it one of Georgia's newest state colleges.

 The college transitioned from a two-year community college into a four-year comprehensive institution and conferred its first baccalaureate degrees on May 7,
2011.
</item 4>

<item 5>
Two Rivers High School (Arkansas)
Two Rivers High School is a public high school for students in grades 7 through 12 located in Ola, Arkansas, United States.

 Two Rivers High School is administered by the Two Rivers School District.

 The Two Rivers High School and school district serves 650 sqmi of rural communities including Ola, Plainview, Rover, and Casa.

 Two Rivers High School is one of four high schools within Yell County and the sole high school administered by the Two Rivers School District.
</item 5>

<item 6>
St. Mary's High School (Manhasset, New York)
Saint Mary's High School is a private Catholic high school in Manhasset, New York.

 St. Mary's parish is part of the Roman Catholic Diocese of Rockville Centre yet the High School serves families of the Roman Catholic Archdiocese of New York,
the Diocese of Brooklyn and the Diocese of Rockville Centre.

 It is located off Northern Boulevard at 51 Clapham Avenue, Manhasset, near the Miracle Mile.

 St. Mary's Men's Varsity Ice Hockey is the all-time leader in State Championships.
</item 6>

<item 7>
Rich Central High School
Rich Central High School or RCHS is a public four-year high school located in Olympia Fields, Illinois, a southern suburb of Chicago, Illinois, in the United
States.

 The Rich Central Campus serves the cities of Matteson, Richton Park, Country Club Hills, Chicago Heights, Tinley Park, and parts of Olympia Fields.

 It is a part of Rich Township District 227, which also includes Rich East High School and Rich South High School.

 Although the school is located in Olympia Fields, it does not serve the entire village.

 Some Olympia Fields students attend high school at Homewood-Flossmoor High School and Bloom High School, depending on the subdivision where the student
resides.
</item 7>

<item 8>
Michael Ola
Michael Ola (born April 19, 1988) is an American football offensive tackle for the Buffalo Bills of the National Football League (NFL).

 He played college football at Hampton University and attended Riverdale High School in Riverdale, Georgia.

 He has also been a member of the Jacksonville Sharks, Montreal Alouettes, Miami Dolphins, Chicago Bears, San Diego Chargers, Detroit Lions, Seattle Seahawks
and New York Giants.
</item 8>

<item 9>
Lake Worth Community High School
Lake Worth Community High School is a B State of Florida rated public high school located in Lake Worth, Florida.

 Established in 1922 as Lake Worth High School, it is currently one of Palm Beach County's largest schools.

 It has six magnet programs: the award-winning Air Force JROTC led by former Multi-National Award-winning Drill Teams, Tyranny and Fidelis; Medicine and Allied
Health; Child Care and Development; Criminal Justice; Culinary and Drafting and Design.

 The Palm Beach County School Board added the word Community to the names of all public high schools, including Lake Worth High School, in the 1980s.
</item 9>

<item 10>
Brunswick Old Town Historic District
Brunswick Old Town Historic District is a historic district in Brunswick, Georgia.

 It was added to the National Register of Historic Places on April 26, 1979 and includes an area bounded by 1st Street, Bay Street, New Bay Street, H Street,
and Cochran Street (4th Ave. and G St., according to one source).

 Brunswick is one of Georgia's two deep-water ports and is the mainland city associated with the Golden Isles of Georgia, at the junction of I-95 and US 82.
</item 10>
=== END ITEM id=ff370ed34e691c51 ===

=== ITEM id=3d27cff5b649f649 ===
Gold sentences and facts:
Sentence 0: Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.
  Fact 0.0: Bicycling is a cycling brand.
  Fact 0.1: Bicycling is published by Rodale, Inc. in Emmaus, Pennsylvania.
Sentence 1: National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural affairs.
  Fact 1.0: National Review (NR) is an American magazine.
  Fact 1.1: National Review is semi-monthly.
  Fact 1.2: National Review is conservative.
  Fact 1.3: National Review is an editorial magazine.
  Fact 1.4: National Review focuses on news and commentary pieces on political affairs.
  Fact 1.5: National Review focuses on news and commentary pieces on social affairs.
  Fact 1.6: National Review focuses on news and commentary pieces on cultural affairs.

Retrieved items:
<item 1>
Adventist Review
The Adventist Review is the official newsmagazine of the Seventh-day Adventist Church.

 Commonly known as the Review, it is published weekly by the Review and Herald Publishing Association.

 The Review and Herald also publishes a sister magazine, "Adventist World".

 The magazine is headquartered in Silver Spring, Maryland.

 The current editor of the "Adventist Review" is Bill Knott.

 The magazine currently has nearly 30,000 paying subscribers.

 Its library reference number is OCLC 9572173  .
</item 1>

<item 2>
National Review
National Review (NR) is an American semi-monthly conservative editorial magazine focusing on news and commentary pieces on political, social, and cultural
affairs.

 The magazine was founded by the author William F. Buckley Jr. in 1955.

 It is currently edited by Rich Lowry.
</item 2>

<item 3>
Kevin D. Williamson
Kevin Daniel Williamson (born September 18, 1972) is the roving correspondent for "National Review."

 He is also the theater critic for "The New Criterion".

 He was previously deputy managing editor at "National Review".
</item 3>

<item 4>
Bicycling (magazine)
Bicycling is a cycling brand published by Rodale, Inc. in Emmaus, Pennsylvania.

 "Bicycling" claims to be the world’s largest cycling magazine.
</item 4>

<item 5>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 5>

<item 6>
Charles C. W. Cooke
Charles C. W. Cooke (born November 4, 1984) is the editor of "National Review Online".

 He took the role over after Rich Lowry stepped down in June, 2016 (Lowry remains the editor-in-chief of "National Review").

 Cooke is the author of "The Conservatarian Manifesto" and a frequent guest on HBO's "Real Time with Bill Maher".

 In addition to "National Review", he has written for the "New York Times", the "Washington Post", and the "Los Angeles Times".

 Along with Kevin D. Williamson, he hosts the popular "Mad Dogs and Englishmen" podcast.
</item 6>

<item 7>
The Human Life Review
The Human Life Review is a quarterly journal published by the Human Life Foundation since 1975.

 It is devoted to explorations of life issues, primarily abortion, as well as neonaticide, medical genetics, prenatal testing, human cloning, fetal tissue
experimentation, euthanasia and assisted suicide, and also publishes articles dealing with more general questions of family and society.

 It was founded by James Patrick McFadden, formerly associate publisher of "National Review", who had also founded the Human Life Foundation, and is now edited
by his daughter, Maria McFadden.

 It was launched from the offices of "National Review", with the support of William F. Buckley.
</item 7>

<item 8>
Review
A review is an evaluation of a publication, service, or company such as a movie (a movie review), video game (video game review), musical composition (music
review of a composition or recording), book (book review); a piece of hardware like a car, home appliance, or computer; or an event or performance, such as a
live music concert, play, musical theater show, dance show, or art exhibition.

 In addition to a critical evaluation, the review's author may assign the work a rating to indicate its relative merit.

 More loosely, an author may review current events, trends, or items in the news.

 A compilation of reviews may itself be called a review. "

The New York Review of Books", for instance, is a collection of essays on literature, culture, and current affairs. "

National Review", founded by William F. Buckley, Jr., is an influential conservative magazine, and "Monthly Review" is a long-running socialist periodical.
</item 8>

<item 9>
National Review Board
The National Review Board (full name: National Review Board for the Protection of Children and Young People) is a committee created in 2002 by the United States
Conference of Catholic Bishops in order to monitor the implementation of the Charter for the Protection of Children and Young People in the wake of the clerical
abuse scandal in the United States.

 The Board was also charged with investigating the scandal, which it did in part by commissioning the John Jay College to conduct a survey of Church records in
order to define the nature and scope of the abuse committed by priests over the period between 1950 and 2002.

 The results of that survey were released in 2004 in what has come to be known as the "John Jay Report" or "Nature and Scope report."

 In parallel with the John Jay College survey the Board conducted interviews with a variety of people, both inside and outside the Church, who were well placed
to comment on the scandal, and on the basis of these interviews prepared a more broad-ranging report of its own.
</item 9>

<item 10>
The entries cover magazines, journalists, writers, and the meaning of “review.” Children’s titles include bimonthly Jack and Jill and National Wildlife
Federation nature magazines Ranger Rick, Ranger Rick Jr., and Ranger Rick Cub. Women’s magazines include Macedonian Tea Moderna and Danish ALT for Damerne.
Other publications range from Adventist Review and conservative National Review to the Human Life Review, Henry James Review, and defunct American Mercury and
Insurgent49. Profiles include Rosie O’Donnell
</item 10>
=== END ITEM id=3d27cff5b649f649 ===

=== ITEM id=a1763e47fc9ea2c7 ===
Gold sentences and facts:
Sentence 0: Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal Navy surface vessels and submarines.
  Fact 0.0: Rosyth Dockyard is a large naval dockyard
  Fact 0.1: Rosyth Dockyard is on the Firth of Forth
  Fact 0.2: Rosyth Dockyard is at Rosyth, Fife, Scotland
  Fact 0.3: Rosyth Dockyard is owned by Babcock Marine
  Fact 0.4: Rosyth Dockyard formerly undertook refitting of Royal Navy surface vessels
  Fact 0.5: Rosyth Dockyard formerly undertook refitting of Royal Navy submarines
Sentence 1: The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.
  Fact 1.0: The Queen Elizabeth class is a class of two aircraft carriers
  Fact 1.1: The Queen Elizabeth class aircraft carriers belong to the United Kingdom's Royal Navy

Retrieved items:
<item 1>
Rosyth railway station
Rosyth railway station serves the town of Rosyth in Fife, Scotland.

 The station is managed by Abellio ScotRail and lies on the Fife Circle Line, 23.6 km north of Edinburgh Waverley .

 It was opened in 1917 by the North British Railway (as "Rosyth Halt") to serve the nearby naval dockyard.
</item 1>

<item 2>
HMS Illustrious (87)
Her aircraft attacked several targets in the Japanese-occupied Dutch East Indies over the following year before "Illustrious" was transferred to the newly
formed British Pacific Fleet (BPF).

 The carrier participated in the early stages of the Battle of Okinawa until mechanical defects arising from accumulated battle damage became so severe that she
was ordered home early for repairs in May 1945.
</item 2>

<item 3>
HMS Diomede (D92)
HMS "Diomede" was a "Danae"-class cruiser of the Royal Navy.

 Constructed at Vickers Armstrong, Barrow, she was constructed too late to take part in World War I and was consequently completed at the Royal Dockyard,
Portsmouth.

 Between the wars, she served on the China Station, Pacific waters, East Indies Waters and from 1936 onwards, in reserve.

 In World War II she performed four years of arduous war duty, during which time she captured the German blockade runner "Idarwald" .

 Between 22 July 1942 and 24 September 1943 she was converted to a training ship at Rosyth Dockyard.

 In 1945 she was placed in reserve and scrapped a year later.
</item 3>

<item 4>
HMS Prince of Wales (R09)
HMS "Prince of Wales" is the second "Queen Elizabeth"-class aircraft carrier under construction for the Royal Navy, with plans for active service from 2020.

 She is the seventh Royal Navy ship to have the name HMS "Prince of Wales".

 Construction of the ship began in 2011 at Rosyth Dockyard and in April 2016, was said to be around 80% structurally complete.
</item 4>

<item 5>
HMS Illustrious (87)
HMS "Illustrious" was the lead ship of her class of aircraft carriers built for the Royal Navy before World War II.

 Her first assignment after completion and working up was with the Mediterranean Fleet, in which her aircraft's most notable achievement was sinking one Italian
battleship and badly damaging two others during the Battle of Taranto in late 1940.

 Two months later the carrier was crippled by German dive bombers and was repaired in the United States.

 After sustaining damage on the voyage home in late 1941 by a collision with her sister ship "Formidable" , "Illustrious" was sent to the Indian Ocean in early
1942 to support the invasion of Vichy French Madagascar (Operation Ironclad).

 After returning home in early 1943, the ship was given a lengthy refit and briefly assigned to the Home Fleet.

 She was transferred to Force H for the Battle of Salerno in mid-1943 and then rejoined the Eastern Fleet in the Indian Ocean at the beginning of 1944.
</item 5>

<item 6>
Queen Elizabeth-class aircraft carrier
The "Queen Elizabeth" class is a class of two aircraft carriers of the United Kingdom's Royal Navy.

 The first,  , was named on 4 July 2014, with her ship commissioning planned for 2017, and an initial operating capability expected in 2018.

 The second, HMS "Prince of Wales" , is scheduled to be launched around Summer 2017, followed by commissioning in 2020 and service thereafter.

 On 5 September 2014, at the NATO 2014 Wales summit, the Prime Minister David Cameron announced that the second carrier would be brought into service, ending
years of uncertainty surrounding its future.

 This was confirmed by the November 2015 Government Strategic Defence Review, with both carriers entering service, one being available at any time.
</item 6>

<item 7>
List of sunken aircraft carriers
With the advent of heavier-than-air flight, the aircraft carrier has become a decisive weapon at sea.

 In 1911 aircraft began to be successfully launched and landed on ships with the successful flight of a Curtiss Pusher aboard the USS "Pennsylvania".

 The British Royal Navy pioneered the first aircraft carrier as floatplanes, as flying boats under performed compared to traditional land based aircraft.

 The first true aircraft carrier was the HMS "Argus" , launched in late 1917 with a complement of 20 aircraft, a flight deck of 550 ft and 68 ft wide.

 The last aircraft carrier sunk in wartime was the Japanese carrier "Amagi", in Kure Harbour in July 1945.

 The greatest loss of life was the 2,046 killed on the "Akitsu Maru"—a converted passenger liner with a small flight deck, carrying the Imperial Japanese Army's
64th Infantry Regiment.
</item 7>

<item 8>
Goliath (Rosyth)
Goliath is a crane in Rosyth Dockyard, Scotland, with a lift capacity of 1000 t , the largest in Britain.
</item 8>

<item 9>
Rosyth Dockyard
Rosyth Dockyard is a large naval dockyard on the Firth of Forth at Rosyth, Fife, Scotland, owned by Babcock Marine, which formerly undertook refitting of Royal
Navy surface vessels and submarines.

 Before its privatisation in the 1990s it was formally the Royal Naval Dockyard Rosyth.

 Its primary role is now as integration site for the Royal Navy's newest aircraft carriers - the "Queen Elizabeth"-class.
</item 9>

<item 10>
HMS Queen Elizabeth March
"HMS Queen Elizabeth March", composed by WO2 bandmaster John Morrish RM, was the winning composition for the 2012 Royal Marines Band Service March Competition,
sponsored by the Aircraft Carrier Alliance.

 The march was written for the first of the new generation, of Royal Navy Queen Elizabeth-class aircraft carriers.
</item 10>
=== END ITEM id=a1763e47fc9ea2c7 ===

=== ITEM id=6531667e838dd6df ===
Gold sentences and facts:
Sentence 0:  He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
  Fact 0.0: The person referred to as 'He' has had a cameo acting role as a tailor in The Wolf of Wall Street.
  Fact 0.1: The person referred to as 'He' has had a cameo acting role as a tailor in The Good Shepherd.
Sentence 1: The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir of the same name by Jordan Belfort.
  Fact 1.0: The Wolf of Wall Street is a 2013 film.
  Fact 1.1: The Wolf of Wall Street is American.
  Fact 1.2: The Wolf of Wall Street is a biographical black comedy crime film.
  Fact 1.3: The Wolf of Wall Street was directed by Martin Scorsese.
  Fact 1.4: The Wolf of Wall Street was written by Terence Winter.
  Fact 1.5: The Wolf of Wall Street is based on the memoir of the same name by Jordan Belfort.

Retrieved items:
<item 1>
The Wolf of Wall Street (2013 film)
The film marks the director's fifth collaboration with DiCaprio, after "Gangs of New York" (2002), "The Aviator" (2004), "The Departed" (2006) and "Shutter
Island" (2010), as well as his second collaboration with Winter after the television series "Boardwalk Empire" (2010–14).
</item 1>

<item 2>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 2>

<item 3>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 3>

<item 4>
Million Dollar Arm
Million Dollar Arm is a 2014 American biographical sports drama film directed by Craig Gillespie and produced by Walt Disney Pictures from a screenplay written
by Thomas McCarthy.

 The film is based on the true story of baseball pitchers Rinku Singh and Dinesh Patel who were discovered by sports agent J. B. Bernstein after winning a
reality show competition.
</item 4>

<item 5>
A Walk in the Woods (film)
A Walk in the Woods is a 2015 American biographical comedy-drama film directed by Ken Kwapis, based on the 1998 book/memoir of the same name by Bill Bryson.

 The film stars Robert Redford, Nick Nolte and Emma Thompson.

 The film was released on September 2, 2015, by Broad Green Pictures.
</item 5>

<item 6>
Dash and Lilly
Dash and Lilly is a 1999 Emmy and Golden Globe Award-nominated biographical television film about writers Dashiell Hammett and Lillian Hellman.

 The film was directed by actress Kathy Bates and written by Jerrold L. Ludwig.

 It stars Sam Shepard and Judy Davis.
</item 6>

<item 7>
Leonard Logsdail
Leonard Logsdail (born September 11, 1950 in London, England) is a bespoke tailor, based in Manhattan, New York, specializing in men's suits.

 He is described as one of the finest bespoke tailors in the men’s suit business.

 All of Logsdail's suits are crafted and perfected on-site in his New York City location.

 He is credited for making Savile Row tailoring a local option in New York.

 Logsdail is known for creating high-end suits, including lining jackets with Hermes silk scarves.

 CNBC talk show host, economist and fashion icon Larry Kudlow has his suits made by Leonard Logsdail.

 Logsdail has created suits for award-winning films, and is recognized as one of cinema's most sought-after tailors.

 Logsdail has collaborated with esteemed film directors Steven Spielberg, Robert de Niro, Oliver Stone, Ridley Scott, and Martin Scorcese.

 He has had a cameo acting role as a tailor in The Wolf of Wall Street, and The Good Shepherd.
</item 7>

<item 8>
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 8>

<item 9>
The Wolf of Wall Street (2013 film)
The Wolf of Wall Street is a 2013 American biographical black comedy crime film directed by Martin Scorsese and written by Terence Winter, based on the memoir
of the same name by Jordan Belfort.

 It recounts Belfort's perspective on his career as a stockbroker in New York City and how his firm Stratton Oakmont engaged in rampant corruption and fraud on
Wall Street that ultimately led to his downfall.

 Leonardo DiCaprio (who was also a producer) stars as Belfort, with Jonah Hill as his business partner and friend Donnie Azoff, Margot Robbie as his wife Naomi
Lapaglia and Kyle Chandler as Patrick Denham, the FBI agent who tries to bring him down.

 Matthew McConaughey, Rob Reiner, Jon Favreau, Joanna Lumley and Jean Dujardin also star.
</item 9>

<item 10>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 10>
=== END ITEM id=6531667e838dd6df ===

=== ITEM id=a5b17a47e6d3eab1 ===
Gold sentences and facts:
Sentence 0: Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the Himalayas, and Indonesia) and the Americas.
  Fact 0.0: Hydrangea is a genus of flowering plants.
  Fact 0.1: Hydrangea's common names are hydrangea or hortensia.
  Fact 0.2: The Hydrangea genus has 70 to 75 species.
  Fact 0.3: Hydrangea plants are native to southern and eastern Asia.
  Fact 0.4: Hydrangea plants are native to China, Japan, Korea, the Himalayas, and Indonesia.
  Fact 0.5: Hydrangea plants are native to the Americas.
Sentence 1:  It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").
  Fact 1.0: The subject of this sentence contains 13 to 18 species.
  Fact 1.1: The subject of this sentence is best known for a single species.
  Fact 1.2: That single species is Jequirity.
  Fact 1.3: Jequirity is also called A. precatorius.

Retrieved items:
<item 1>
Hydrangea heteromalla
Hydrangea heteromalla is a species of "Hydrangea" flower, native to the Himalayas and China.

 Average height is 10'-15' with inflorescences that are 5"-8" across and bloom May-June,full flowering only occurs in full sun.Is a very adaptable plant that is
hardy to USDA zones (4)5-7.

 This plant is seldom seen in American gardens although there is nice specimen at the Hillier Arboretum.
</item 1>

<item 2>
Hydrangea paniculata
Hydrangea paniculata, the panicled hydrangea, is a species of flowering plant in the family Hydrangeaceae native to southern and eastern China, Korea, Japan and
Russia (Sakhalin).

 It was first formally described by Philipp Franz von Siebold in 1829.

 It is a deciduous shrub or small tree, 1 - tall by 2.5 m broad, growing in sparse forests or thickets in valleys or on mountain slopes.
</item 2>

<item 3>
Alismataceae
The water-plantains (Alismataceae) are a family of flowering plants, comprising 11 genera and between 85 and 95 species.

 The family has a cosmopolitan distribution, with the greatest number of species in temperate regions of the Northern Hemisphere.

 Most of the species are herbaceous aquatic plants growing in marshes and ponds.
</item 3>

<item 4>
Hydrangea serrata
Hydrangea serrata is a species of flowering plant in the family Hydrangeaceae, native to mountainous regions of Korea and Japan.

 Common names include mountain hydrangea and tea of heaven.

 Growing to 1.2 m tall and broad, it is a deciduous shrub with oval leaves and panicles of blue and pink flowers in summer and autumn (fall).

 It is widely cultivated as an attractive ornamental shrub throughout the world in areas with suitable climate and soil.
</item 4>

<item 5>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 5>

<item 6>
Hydrangea quercifolia
Hydrangea quercifolia, commonly known by its translation oakleaf hydrangea or oak-leaved hydrangea, is a species of flowering plant native to the Southeastern
United States, in woodland habitats from North Carolina west to Tennessee, and south to Florida and Louisiana.

 A deciduous shrub with white showy flower heads, it is a commonly grown garden plant.

 Numerous cultivars are available commercially.
</item 6>

<item 7>
Hydrangea arborescens
Hydrangea arborescens, commonly known as smooth hydrangea, wild hydrangea, or sevenbark, is a species of flowering plant in the family Hydrangeaceae.

 It is a small- to medium-sized, deciduous shrub up to 3 m tall that is native to the eastern United States.
</item 7>

<item 8>
Hydrangea macrophylla
Hydrangea macrophylla is a species of flowering plant in the family Hydrangeaceae, native to Japan.

 It is a deciduous shrub growing to 2 m tall by 2.5 m broad with large heads of pink or blue flowers in summer and autumn.

 Common names include bigleaf hydrangea, French hydrangea, lacecap hydrangea, mophead hydrangea, penny mac and hortensia.

 It is widely cultivated in many parts of the world in many climates.

 It is not to be confused with "H. aspera" 'Macrophylla'.
</item 8>

<item 9>
Abrus
Abrus is a genus of flowering plants in the pea family, Fabaceae and the only genus found in the tribe Abreae.

 It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").

 The highly toxic seeds of that species are used to make jewellery.
</item 9>

<item 10>
Yucca
Yucca is a genus of perennial shrubs and trees in the family Asparagaceae, subfamily Agavoideae.

 Its 40-50 species are notable for their rosettes of evergreen, tough, sword-shaped leaves and large terminal panicles of white or whitish flowers.

 They are native to the hot and dry (arid) parts of the Americas and the Caribbean.

 Early reports of the species were confused with the cassava ("Manihot esculenta").

 Consequently, Linnaeus mistakenly derived the generic name from the Taíno word for the latter, "yuca" (spelled with a single "c").

 It is commonly found growing in rural graveyards and when in bloom the cluster of (usually pale) flowers on a thin stalk appear as floating apparitions.
</item 10>
=== END ITEM id=a5b17a47e6d3eab1 ===

=== ITEM id=b8bb45d15124face ===
Gold sentences and facts:
Sentence 0:  During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas" (1992), and "Red Rock West" (1993).
  Fact 0.0: During his early career, Cage starred in a variety of films.
  Fact 0.1: Cage starred in Valley Girl, released in 1983.
  Fact 0.2: Cage starred in Racing with the Moon, released in 1984.
  Fact 0.3: Cage starred in Birdy, released in 1984.
  Fact 0.4: Cage starred in Peggy Sue Got Married, released in 1986.
  Fact 0.5: Cage starred in Raising Arizona, released in 1987.
  Fact 0.6: Cage starred in Moonstruck, released in 1987.
  Fact 0.7: Cage starred in Vampire's Kiss, released in 1989.
  Fact 0.8: Cage starred in Wild at Heart, released in 1990.
  Fact 0.9: Cage starred in Fire Birds, released in 1990.
  Fact 0.10: Cage starred in Honeymoon in Vegas, released in 1992.
  Fact 0.11: Cage starred in Red Rock West, released in 1993.
Sentence 1: Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.
  Fact 1.0: Moonstruck is a 1987 American film.
  Fact 1.1: Moonstruck is a romantic comedy film.
  Fact 1.2: Moonstruck was directed by Norman Jewison.
  Fact 1.3: Moonstruck was written by John Patrick Shanley.
Sentence 2:  It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother (Nicolas Cage).
  Fact 2.0: Moonstruck is about a widowed 37-year-old Italian-American woman played by Cher.
  Fact 2.1: The woman falls in love with her fiancé's estranged, hot-tempered younger brother.
  Fact 2.2: The fiancé is played by Danny Aiello.
  Fact 2.3: The younger brother is played by Nicolas Cage.

Retrieved items:
<item 1>
Cool and the Crazy
Cool and the Crazy is a 1994 American drama film written and directed by Ralph Bakshi and starring Jared Leto and Alicia Silverstone.

 The story revolves around an unhappily married couple in the late 1950s who both lead separate affairs.

 The film was Bakshi's first feature-length live-action film, being primarily known as a director of animated films which heavily utilize live-action sequences,
such as "Fritz the Cat", "Heavy Traffic", "Wizards", "American Pop" and "The Lord of the Rings".
</item 1>

<item 2>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 2>

<item 3>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 3>

<item 4>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 4>

<item 5>
Dying of the Light (film)
Dying of the Light is a 2014 American psychological thriller film written and directed by Paul Schrader and starring Nicolas Cage, Anton Yelchin and Irène Jacob
about a government agent who must track down and kill a terrorist before he loses his full memory from a disease.

 It was released theatrically and through VOD formats by Lionsgate on December 5, 2014.

 The film received extremely negative reviews, with controversy surrounding the heavy tampering and reediting of the footage by the studio, who denied Schrader
final-cut privilege and led him and principal members of the cast to disown the released version and campaign against it.
</item 5>

<item 6>
Nicolas Cage
Nicolas Kim Coppola (born January 7, 1964), known professionally as Nicolas Cage, is an American actor, director and producer.

 During his early career, Cage starred in a variety of films such as "Valley Girl" (1983), "Racing with the Moon" (1984), "Birdy" (1984), "Peggy Sue Got
Married" (1986), "Raising Arizona" (1987), "Moonstruck" (1987), "Vampire's Kiss" (1989), "Wild at Heart" (1990), "Fire Birds" (1990), "Honeymoon in Vegas"
(1992), and "Red Rock West" (1993).
</item 6>

<item 7>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 7>

<item 8>
Where Do You Go (Cher song)
"Where Do You Go" is a song written by Sonny Bono.

 It was released as the first single by Cher in the later quarter of 1965 for her second album "The Sonny Side of Cher".

 It fell short of the "Billboard" Hot 100's top 20, but still earned Cher a moderate success, by reaching the Top 40.

 It was followed by the U.S. #2 smash hit "Bang Bang (My Baby Shot Me Down)" which saw a release early the following year.

 It was a bigger hit in Canada, where it reached #5 on the singles chart.
</item 8>

<item 9>
Cher’s acting work has focused mainly on comedy, drama, and romance. She appeared in 13 films, including two cameos, had one starring theater role, and directed
a segment of *If These Walls Could Talk* (1996) and several Geffen-era music videos. She also appeared in international TV commercials and Lori Davis print
advertising (1992). Before her film career, she had hit records in the 1960s as a solo artist and with Sonny Bono as
</item 9>

<item 10>
Tom Cruise filmography
Tom Cruise is an American actor and producer who made his film debut with a minor role in the 1981 romantic drama "Endless Love".

 Two years later he made his breakthrough by starring in the romantic comedy "Risky Business" (1983), which garnered Cruise his first nomination for the Golden
Globe Award for Best Actor – Motion Picture Musical or Comedy.

 In 1986, Cruise played a fighter pilot in the Tony Scott-directed action drama "Top Gun" (the highest-grossing film that year), and also starred opposite Paul
Newman in the Martin Scorsese-directed drama "The Color of Money".

 Two years later he played opposite Dustin Hoffman in the Academy Award for Best Picture-winning drama "Rain Man" (1988), and also appeared in the Golden
Raspberry Award for Worst Picture-winning romantic drama "Cocktail" (1988).

 In doing so Cruise became the first and only person as of 2014 to star in a Best Picture Oscar winner and a Worst Picture Razzie winner in the same year.
</item 10>
=== END ITEM id=b8bb45d15124face ===

=== ITEM id=994ab1f3730eb042 ===
Gold sentences and facts:
Sentence 0: The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
  Fact 0.0: The Silent Historian has the original title 'Het zwijgen van Loe de Jong [The Silence of Loe de Jong]'.
  Fact 0.1: The Silent Historian is a 2011 Dutch documentary film.
  Fact 0.2: The Silent Historian is a documentary film by Simonka de Jong.
  Fact 0.3: The Silent Historian was shown as part of the January 2012 New York Jewish Film Festival.
  Fact 0.4: The January 2012 New York Jewish Film Festival screening of The Silent Historian was presented in partnership with the Jewish Museum.
  Fact 0.5: The January 2012 New York Jewish Film Festival screening of The Silent Historian was presented in partnership with the Film Society of Lincoln Center.
Sentence 1: The Betrayal — Nerakhoon is a 2008 documentary film directed by Ellen Kuras and Thavisouk Phrasavath.
  Fact 1.0: The Betrayal — Nerakhoon is a 2008 documentary film.
  Fact 1.1: The Betrayal — Nerakhoon was directed by Ellen Kuras.
  Fact 1.2: The Betrayal — Nerakhoon was directed by Thavisouk Phrasavath.

Retrieved items:
<item 1>
Betrayal (1929 film)
Betrayal was a 1929 drama film produced for Famous Players-Lasky and released by Paramount Pictures.

 The film is the last silent film directed by Lewis Milestone, the last silent performance by Gary Cooper, the last silent performance by Germany's Emil
Jannings, and the only onscreen pairing of Cooper and Jannings.

 It is considered a lost film.
</item 1>

<item 2>
Graph query row:
work: The Betrayal – Nerakhoon
type: documentary film
genre: None
</item 2>

<item 3>
Thérèse Raquin (1928 film)
Thérèse Raquin is a 1928 drama film directed by Jacques Feyder.

 It is the third silent film adaptation of the novel of the same name by Émile Zola.

 The film stars Gina Manès as Thérèse Raquin, Wolfgang Zilzer as Monsieur Raquin, and Jeanne Marie Laurent as Madame Raquin.

 The décors of the Paris suburbs for the film were built by André Andrejew.

 The film was produced by Deutsche Film Union in Germany, with German and French actors, in a French-German co-production, to be later released at the same time
in France as "Thérèse Raquin" and Germany as Du sollst nicht ehebrechen!

.

 As no words were spoken, both versions differed only with the language of intertitles.

 The British title at the time of the film's original release was Thou Shalt Not.

 This is last silent film imports distributed by Warner Bros.' newly acquired First National subsidiary.

 No dialong with music score and sound effects.
</item 3>

<item 4>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 4>

<item 5>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 5>

<item 6>
The Price of Betrayal
The Price of Betrayal (Swedish: "Judaspengar" ) is a 1915 Swedish silent drama film directed by Victor Sjöström.
</item 6>

<item 7>
Graph query row:
work: The Silent Historian
type: documentary film
genre: None
</item 7>

<item 8>
The Decembrists (film)
The Decembrists (Russian: Декабристы , "Dekabristy " ) is a 1927 Soviet silent historical drama film directed by Aleksandr Ivanovsky.
</item 8>

<item 9>
The Betrayal – Nerakhoon
The Betrayal — Nerakhoon is a 2008 documentary film directed by Ellen Kuras and Thavisouk Phrasavath.
</item 9>

<item 10>
The Silent Historian
The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as
part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
</item 10>
=== END ITEM id=994ab1f3730eb042 ===

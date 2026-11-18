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

=== ITEM id=11f1be9eb7727302 ===
Gold sentences and facts:
Sentence 0: The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic Beach.
  Fact 0.0: The Nassau Herald is a weekly newspaper.
  Fact 0.1: The Nassau Herald serves the Five Towns communities of Nassau County.
  Fact 0.2: The Five Towns communities of Nassau County are Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood, and Atlantic Beach.
Sentence 1: Lawrence is a village in Nassau County, New York in the United States.
  Fact 1.0: Lawrence is a village.
  Fact 1.1: Lawrence is in Nassau County, New York, United States.
Sentence 2:  As of the 2010 United States Census, the village population was 6,483.
  Fact 2.0: As of the 2010 United States Census, the population of the village of Lawrence was 6,483.

Retrieved items:
<item 1>
The Hudson Reporter
The Hudson Reporter is a newspaper chain based in Hudson County, New Jersey.

 It is the only weekly newspaper chain in Hudson County and one of only two newspaper companies in this busy metropolitan area.

 "The Hudson Reporter" publications focus on local politics and community news.

 In addition to articles written by the staff, the papers print readers' letters to the editor.

 The oldest newspaper in the chain is the "Hoboken Reporter", founded in 1983.
</item 1>

<item 2>
Great Neck School District
The Great Neck School District is a comprehensive community public school district primarily serving students in Great Neck, New York.

 It also serves students in some parts of Manhasset and New Hyde Park.

 It is Union Free School District Number 7 in the Town of North Hempstead, Nassau County, New York, United States, on Long Island.
</item 2>

<item 3>
Floral Park, New York
Floral Park is an incorporated village in Nassau County, New York, United States, on Long Island.

 The neighborhood of Floral Park in the New York City borough of Queens, is adjacent to the village.

 The village is at the western border of Nassau County, and is located mainly in the Town of Hempstead, while the section north of Jericho Turnpike is within
the Town of North Hempstead.

 The population as of the US Census of 2010 is 15,863.
</item 3>

<item 4>
Lawrence, Nassau County, New York
Lawrence is a village in Nassau County, New York in the United States.

 As of the 2010 United States Census, the village population was 6,483.
</item 4>

<item 5>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 5>

<item 6>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 6>

<item 7>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 7>

<item 8>
Great Neck, New York
Great Neck is a region on Long Island, New York, that covers a peninsula on the North Shore of Long Island, which includes the villages of Great Neck, Great
Neck Estates, Great Neck Plaza, and others, as well as an area south of the peninsula near Lake Success and the border territory of Queens.

 The incorporated village of Great Neck had a population of 9,989 at the 2010 census, while the larger Great Neck area comprises a residential community of some
40,000 people in nine villages and hamlets in the town of North Hempstead, of which Great Neck is the northwestern quadrant.

 Great Neck has five ZIP Codes (11020–11024), and one school district.
</item 8>

<item 9>
Nassau Herald
The Nassau Herald is a weekly newspaper serving the Five Towns communities of Nassau County – Lawrence, Woodmere, Hewlett, Cedarhurst, Inwood and Atlantic
Beach.

 It is part of the Long Island Herald newspaper chain, which includes The Jewish Star and The Oyster Bay Guardian is owned by Richner Communications, and covers
Nassau County, New York.
</item 9>

<item 10>
Anton Media Group
Anton Media Group, formerly Anton Community Newspapers, and also known by its legal name Long Island Community Newspapers, Inc. is a print media company based
in Mineola, NY on Long Island in Nassau County and produces 17 weekly newspapers in Nassau County as well as "Long Island Weekly" and special sections such as
Healthy Living, Camp & School, and Dining Guides.

 Anton Community Newspapers is one of the largest privately owned newspaper companies in New York State.
</item 10>
=== END ITEM id=11f1be9eb7727302 ===

=== ITEM id=a3a7cdcb5e5e586a ===
Gold sentences and facts:
Sentence 0: Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France (1933–1940 and 1950–1957), and the United States (1947–1950).
  Fact 0.0: Maximillian Oppenheimer was born on 6 May 1902.
  Fact 0.1: Maximillian Oppenheimer died on 26 March 1957.
  Fact 0.2: Maximillian Oppenheimer was known as Max Ophüls.
  Fact 0.3: Max Ophüls was a German-born film director.
  Fact 0.4: Max Ophüls worked in Germany from 1931 to 1933.
  Fact 0.5: Max Ophüls worked in France from 1933 to 1940.
  Fact 0.6: Max Ophüls worked in France from 1950 to 1957.
  Fact 0.7: Max Ophüls worked in the United States from 1947 to 1950.
Sentence 1: Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.
  Fact 1.0: Shirley Clarke was born on October 2, 1919.
  Fact 1.1: Shirley Clarke died on September 23, 1997.
  Fact 1.2: Shirley Clarke was an American experimental filmmaker.
  Fact 1.3: Shirley Clarke was an American independent filmmaker.

Retrieved items:
<item 1>
The entries survey international cinema from silent films to the 1960s, including biographies of composers Giuseppe Verdi and author Émile Zola; several films
about Lucrezia Borgia; and works by Jacques Feyder and Max Ophüls. Ophüls’s films span romantic, historical, and psychological drama, while his son Marcel made
the documentary *The Sorrow and the Pity* about French collaboration during World War II. Other highlights include *The Earrings of
</item 1>

<item 2>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 2>

<item 3>
The Earrings of Madame de…
The Earrings of Madame de… (French: Madame de… ] ) is a 1953 drama film directed by Max Ophüls, adapted from Louise Leveque de Vilmorin's period novel by
Ophüls, Marcel Archard and Annette Wadement.

 The film is considered a masterpiece of the 1950s French cinema.

 Andrew Sarris called it "the most perfect film ever made".

 Ophüls said the story's construction attracted him, stating "there is always the same axis around which the action continually turns like a carousel.

 A tiny, scarcely visible axis: a pair of earrings."
</item 3>

<item 4>
Marcel Ophüls
Marcel Ophuls (] ; born 1 November 1927) is a documentary film maker and former actor, best known for his films "The Sorrow and the Pity" and "".
</item 4>

<item 5>
Max Ophüls
Maximillian Oppenheimer (6 May 1902 – 26 March 1957), known as Max Ophüls (] ), was a German-born film director who worked in Germany (1931–1933), France
(1933–1940 and 1950–1957), and the United States (1947–1950).

 He made nearly 30 films, the latter ones being especially notable: "La Ronde" (1950), "Le Plaisir" (1952), "The Earrings of Madame de..." (1953) and "Lola
Montès" (1955).
</item 5>

<item 6>
Dan Ouellette
Dan Ouellette is an American artist, illustrator, director, author and production designer.

 He began drawing as a child inspired by Escher, Dalí and Moebius.

 His drawings usually begin with very rough sketches and are intended to capture a moment of tension or eroticism.

 His art pushes his audience into areas where people feel less than comfortable, stating that the "US which is so puritanical, our bodies are usually a great
source of anxiety and fear".

 As a production designer, he has designed over fifteen independent feature films in the past decade as well as numerous commercials and music videos.

 He is best known as the director for the music videos "Blue" and "Looking Glass" by The Birthday Massacre, receiving over 4 million combined views on YouTube,
and has directed music videos for the industrial band Android Lust.

 He has done production design for many films over the years including "Chasing Sleep" starring Jeff Daniels.

 Dan's artwork is strongly themed around surrealism and mostly done in pencil.

 As an artist he has exhibited widely.

 He has been published in numerous anthology art books including "Bio-Mannerism" which also features work by H.R.
</item 6>

<item 7>
Elmer Clifton
Elmer Clifton (March 14, 1890 – October 15, 1949) was an American writer, director and actor from the early silent days.

 A collaborator of D.W. Griffith, he appeared in "The Birth of a Nation" (1915) and "Intolerance" (1916) before giving up acting in 1917 to concentrate on work
behind the camera, with Griffith and Joseph Henabery as his mentors.

 His first feature-length solo effort as a director was "The Flame of Youth" with Jack Mulhall.
</item 7>

<item 8>
Shirley Clarke
Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.

 She was also a director and editor to many famous works.
</item 8>

<item 9>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 9>

<item 10>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 10>
=== END ITEM id=a3a7cdcb5e5e586a ===

=== ITEM id=99dd0bfe20b2bfac ===
Gold sentences and facts:
Sentence 0: Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.
  Fact 0.0: Bernard Malamud was born on April 26, 1914.
  Fact 0.1: Bernard Malamud died on March 18, 1986.
  Fact 0.2: Bernard Malamud was an American novelist.
  Fact 0.3: Bernard Malamud was an American short story writer.
Sentence 1: Lois McMaster Bujold ( ; born November 2, 1949) is an American speculative fiction writer.
  Fact 1.0: Lois McMaster Bujold was born on November 2, 1949.
  Fact 1.1: Lois McMaster Bujold is an American speculative fiction writer.

Retrieved items:
<item 1>
Cryoburn
Cryoburn is a science fiction novel by Lois McMaster Bujold, first published in October 2010.

 Part of the Vorkosigan Saga, it was nominated for the Hugo Award for Best Novel in 2011, as Bujold's ninth Best Novel nomination.

 Also in 2011, it was one of the top five finishers in the poll for the Locus Award for Best Science Fiction Novel.
</item 1>

<item 2>
Horizon (novel)
Horizon is a fantasy novel by American writer Lois McMaster Bujold.

 It is the fourth in the tetralogy "The Sharing Knife".
</item 2>

<item 3>
Barrayar
Barrayar is a science fiction novel by Lois McMaster Bujold.

 It was first published as four installments in "Analog" in July–October 1991, and then published in book form by Baen Books in October 1991.

 "Barrayar" won both the Hugo Award for Best Novel and the Locus Award for Best Science Fiction Novel in 1992.

 It is a part of the Vorkosigan Saga, and is the seventh full-length novel of the series, in publication order.

 "Barrayar" is a direct sequel to Bujold's first novel, "Shards of Honor" (1986), and the two are paired in the 1996 omnibus "Cordelia's Honor".
</item 3>

<item 4>
Ethan of Athos
Ethan of Athos is a 1986 science fiction novel by American author Lois McMaster Bujold.

 The title character is Dr. Ethan Urquhart, Chief of Biology at the Severin District Reproduction Centre on the planet Athos, who is sent to find out what
happened to a shipment of vital ovarian tissue cultures.

 Set in the fictional universe of Bujold's Vorkosigan Saga, the novel mentions but does not feature her usual protagonist Miles Vorkosigan.

 To date, Bujold has never revisited the settings of Athos or Kline Station in her many subsequent novels, but the events of "Ethan of Athos" are later referred
to indirectly in the novels "Borders of Infinity" (1989) and "Cetaganda" (1995).
</item 4>

<item 5>
Brothers in Arms (Bujold novel)
Brothers in Arms is a science fiction novel by American writer Lois McMaster Bujold, part of the Vorkosigan Saga.

 It was the fifth book published in the series, and is the twelfth story, including novellas, in the internal chronology of the series.

 "Brothers in Arms" was first published by Baen Books in January 1989, and is included in the 2002 omnibus "Miles Errant".
</item 5>

<item 6>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 6>

<item 7>
Shards of Honor
Shards of Honor is an English language science fiction novel by Lois McMaster Bujold, first published in June 1986.

 It is a part of the Vorkosigan Saga, and is the first full-length novel in publication order.

 "Shards of Honor" is paired with Bujold's 1991 "Barrayar" in the omnibus "Cordelia's Honor" (1996).
</item 7>

<item 8>
Gentleman Jole and the Red Queen
Gentleman Jole and the Red Queen is a science fiction novel by Lois McMaster Bujold.

 It is an installment in Bujold's Vorkosigan Saga.

 Bujold has described it as "not a war story.

 It is about grownups."
</item 8>

<item 9>
Passage (Bujold novel)
Passage is a novel by American writer Lois McMaster Bujold, published in 2008.

 It is the third in the tetralogy "The Sharing Knife".
</item 9>

<item 10>
Lois McMaster Bujold
Lois McMaster Bujold ( ; born November 2, 1949) is an American speculative fiction writer.

 She is one of the most acclaimed writers in her field, having won the Hugo Award for best novel four times, matching Robert A. Heinlein's record, not counting
his Retro Hugo.

 Her novella "The Mountains of Mourning" won both the Hugo Award and Nebula Award.

 In the fantasy genre, "The Curse of Chalion" won the Mythopoeic Award for Adult Literature and was nominated for the 2002 World Fantasy Award for best novel,
and both her fourth Hugo Award and second Nebula Award were for "Paladin of Souls".

 In 2011 she was awarded the Skylark Award.

 In 2013 she was awarded the Forry Award.

 In 2017 she won a Hugo Award for Best Series, for the Vorkosigan Saga.
</item 10>
=== END ITEM id=99dd0bfe20b2bfac ===

=== ITEM id=1bbf38660ab3e333 ===
Gold sentences and facts:
Sentence 0: Big Fish is a musical with music and lyrics by Andrew Lippa and a book by John August.
  Fact 0.0: Big Fish is a musical.
  Fact 0.1: Big Fish has music by Andrew Lippa.
  Fact 0.2: Big Fish has lyrics by Andrew Lippa.
  Fact 0.3: Big Fish has a book by John August.
Sentence 1:  He is a resident artist at the Ars Nova Theater in New York City.
  Fact 1.0: He is a resident artist at the Ars Nova Theater.
  Fact 1.1: The Ars Nova Theater is in New York City.

Retrieved items:
<item 1>
Lin-Manuel Miranda
Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway
musicals "Hamilton" and "In the Heights".

 He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".

 Miranda's awards include a Pulitzer Prize, two Grammy Awards, an Emmy Award, a MacArthur Fellowship, and three Tony Awards.
</item 1>

<item 2>
Robert Lopez
Robert Lopez (born February 23, 1975) is an American songwriter of musicals, best known for co-creating "The Book of Mormon" and "Avenue Q", and for composing
the songs featured in the Disney animated film "Frozen".

 He is the youngest of only twelve people who have won an Emmy, a Grammy, an Oscar and a Tony Award, and the quickest (10 years) to win all four.
</item 2>

<item 3>
Andrew Lippa
Andrew Lippa (December 22, 1964) is an American composer, lyricist, book writer, performer, and producer.

 He is a resident artist at the Ars Nova Theater in New York City.
</item 3>

<item 4>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 4>

<item 5>
Hamilton (album)
Hamilton is the cast album to the 2015 musical "Hamilton".

 The musical is based on the 2004 biography of Alexander Hamilton written by Ron Chernow, with music, lyrics, and book by Lin-Manuel Miranda.

 The recording stars Lin-Manuel Miranda, Leslie Odom Jr., Phillipa Soo, Renée Elise Goldsberry, Christopher Jackson, Daveed Diggs, Anthony Ramos, Okieriete
Onaodowan, Jasmine Cephas Jones, and Jonathan Groff.

 It also features Jon Rua, Thayne Jasperson, Sydney James Harcourt, Ariana DeBose, and Sasha Hutchings.

 It achieved the largest first week sales for a digital cast album and is the highest-charting cast album since 1963.

 It was the highest-selling Broadway cast album of 2015 and peaked at number one on the Rap Album chart, the first cast album to ever do so.
</item 5>

<item 6>
Love Birds (musical)
Love Birds is an original musical with book, music and lyrics by Robert J. Sherman.

 The musical officially premiered at the 2015 Edinburgh Fringe Festival at The Pleasance on August 7, 2015.

 The production was directed and choreographed by Stewart Nicholls.

 Prior to the Edinburgh run, there were two previews held in London at the Lost Theatre (in Stockwell) on July 28 and 29.

 The Original Edinburgh Cast Recording was recorded on July 30 and mixed and mastered on July 31 and August 3 and was first released by SimG Records on August
12 at the Edinburgh Festival.

 Two Edinburgh previews took place on August 5 and 6 with Press Night taking place on August 8.

 The final festival performance took place on August 31.

 The show starred Ruth Betteridge, Greg Castiglioni, John Guerrasio, George Knapper, Jonny Purchase, Joanna Sawyer, Anna Stolli, Rafe Watts and Ryan Willis.
</item 6>

<item 7>
Marc Blitzstein
Marcus Samuel Blitzstein (March 2, 1905January 22, 1964), was an American composer, lyricist, and librettist.

 He won national attention in 1937 when his pro-union musical "The Cradle Will Rock", directed by Orson Welles, was shut down by the Works Progress
Administration.

 He is known for "The Cradle Will Rock" and for his Off-Broadway translation/adaptation of "The Threepenny Opera" by Bertolt Brecht and Kurt Weill.

 His works also include the opera "Regina", an adaptation of Lillian Hellman's play "The Little Foxes"; the Broadway musical "Juno", based on Seán O'Casey's
play "Juno and the Paycock"; and "No for an Answer".

 He completed translation/adaptations of Brecht's and Weill's musical play "Rise and Fall of the City of Mahagonny" and of Brecht's play "Mother Courage and Her
Children" with music by Paul Dessau.
</item 7>

<item 8>
Monica Lynn
Monica Lynn is an American composer who lives and works in the San Francisco Bay area.

 She graduated with a Bachelor of Music degree from the University of Texas at Austin, where she studied with Stefan Kostka, David Neumeyer and Forrest Pierce.

 She continued her education at the University of Missouri in Kansas City where she studied with James Mobberley, Paul Rudy, Chen Yi, Zhou Long, and at the
University of California, Santa Cruz where she graduated with a Doctor of Musical Arts degree, studying with Ben Leeds Carson, David Cope, Karlton Hester, David
Evan Jones, Hi Kyung Kim, Paul Nauert and Peter Elsea.
</item 8>

<item 9>
Big Fish (musical)
Big Fish is a musical with music and lyrics by Andrew Lippa and a book by John August.

 It is based on Daniel Wallace’s 1998 novel, "", and the 2003 film "Big Fish" written by August and directed by Tim Burton.
</item 9>

<item 10>
Maurice Browne
Maurice Browne (12 February 1881 – 21 January 1955), born in Reading, England, was best known as a theater producer in the United States and the UK.

 The Cambridge-educated Browne was also a poet, actor, and theater director.

 He has been credited, along with his then-wife Ellen Van Volkenburg, with being the founder of the Little Theatre Movement in America through his work with the
Chicago Little Theatre.

 Browne and Van Volkenburg went on to found the department of drama at the Cornish School in Seattle in 1918, now Cornish College of the Arts.

 Browne's greatest triumph came in 1929 when he produced "Journey's End", by R. C. Sherriff in London.
</item 10>
=== END ITEM id=1bbf38660ab3e333 ===

=== ITEM id=50f10b8facf06f13 ===
Gold sentences and facts:
Sentence 0: Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All Stars, as well of his own solo band, the Kuwata band.
  Fact 0.0: Keisuke Kuwata's name is written 桑田 佳祐 (Kuwata Keisuke).
  Fact 0.1: Keisuke Kuwata was born February 26, 1956.
  Fact 0.2: Keisuke Kuwata is a Japanese multi-instrumentalist.
  Fact 0.3: Keisuke Kuwata is a Japanese singer-songwriter.
  Fact 0.4: Keisuke Kuwata is frontman for the Southern All Stars.
  Fact 0.5: Keisuke Kuwata has his own solo band, the Kuwata band.
Sentence 1: Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.
  Fact 1.0: Moya Brennan was born Máire Ní Bhraonáin.
  Fact 1.1: Moya Brennan is also known as Máire Brennan.
  Fact 1.2: Moya Brennan was born 4 August 1952.
  Fact 1.3: Moya Brennan is an Irish folk singer.
  Fact 1.4: Moya Brennan is an Irish songwriter.
  Fact 1.5: Moya Brennan is an Irish harpist.
  Fact 1.6: Moya Brennan is an Irish philanthropist.

Retrieved items:
<item 1>
My Match Is a Makin'
My Match Is A Makin' (English translation of opening track title) is a music album by Irish musicians Moya Brennan and Cormac de Barra.

 This is Moya's eighth studio album to be released.

 It was released on 14 April 2010 exclusively to concert goers on her Spring 2010 tour of the Netherlands.

 The album was re-released to a wider audience under the title Voices & Harps.
</item 1>

<item 2>
Kevin Mayer
Kevin Mayer (] , ] or ] , born 10 February 1992) is a French athlete specializing in combined events.

 A native of Drôme in southeast France, Mayer excelled in combined events from a young age; highlights from his youth career include winning gold medals in the
octathlon at the 2009 World Youth Championships and in the decathlon at the 2010 World Junior Championships and at the 2011 European Athletics Junior
Championships, establishing a new national junior record at the latter.
</item 2>

<item 3>
Keisuke Kuwata
Keisuke Kuwata (桑田 佳祐 , Kuwata Keisuke , born February 26, 1956) is a Japanese multi-instrumentalist, singer-songwriter, and frontman for the Southern All
Stars, as well of his own solo band, the Kuwata band.

 He has also done significant amount of scoring music for films.

 He went to Aoyama Gakuin University.
</item 3>

<item 4>
Óró – A Live Session
Óró – A Live Session is a music album by Irish musician Máire Brennan, now known as "Moya Brennan".

 Just before the start of the German tour, Moya and her band recorded a "live session" album to have available for fans on the tour.

 She's tried to recreate some of the live feel with the band in the studio covering many of the songs she plays (and was about to play) in concert.

 The album will become more widely available subsequently, but originally was released in 2005.
</item 4>

<item 5>
Keiko Utoku
Keiko Utoku (宇徳敬子 , "Utoku Keiko" , born April 7, 1967 in Izumi, Kagoshima, Japan) is a Japanese singer and songwriter under UK Sweet label.Keiko made first
debut with chorus part in band B.B.Queens then joined with J-pop female band Mi-Ke from 1991 to 1993, before starting a solo career.

 As a child, Utoku listened to a lot of music, including music made by Momoe Yamaguchi.

 She has written and sung many songs, such as "Wasurenagusa", and Sonic the Hedgehog fans remember her for her vocal work in the original version of the Sonic
CD video game's opening and closing themes in Japan and Europe, "Sonic – You Can Do Anything" and "Cosmic Eternity – Believe In Yourself" in 1993.

 In addition, she also provided Sonic's vocals in the game.

 At recent dates, she has released 15 singles, 5 digital singles, 4 studio albums, 1 mini and 2 best albums.
</item 5>

<item 6>
2010 Toray Pan Pacific Open – Doubles
Alisa Kleybanova and Francesca Schiavone were the defending champions.

 Both were present, but competed with different partners.Kleybanova competed with Hsieh Su-wei, but lost in the first round to Benešová and Záhlavová-Strýcová,
while Schiavone competed with Tathiana Garbin, however withdrew before their quarterfinal match against Lisa Raymond and Rennae Stubbs.
</item 6>

<item 7>
Two Horizons
Two Horizons is the fifth solo album by Irish singer Moya Brennan.

 It is her first full-length release under the name "Moya Brennan", as opposed to "Máire Brennan".

 The album was predominantly recorded in her home studio in Dublin and was nominated for a Grammy award.

 The album was recorded between 2002–2003 and first became available on 23 October 2003.

 It is also Brennan's first non-Christian album since her 1993 recording Misty Eyed Adventures.

 It is the most successful of her solo albums to date and the second to be Grammy-nominated.
</item 7>

<item 8>
Bruce Yonemoto
Bruce Yonemoto (born 1949) is a Japanese-American multimedia artist.

 His work has been recognized by the National Endowment for the Arts, the American Film Institute, The Rockefeller Foundation, the Maya Deren Award for
Experimental Film and Video, and a mid-career survey show at the Japanese American National Museum, in addition to major solo exhibitions at the
InterCommunication Center in Tokyo, the Institute of Contemporary Art the University of Pennsylvania, and the Kemper Museum in Kansas City, MO.

 Yonemoto was featured in the 2002 Corcoran Gallery Bienniel, in Washington D.C. Yonemoto was also a recipient of the Creative Capital Visual Arts Award in
2008.
</item 8>

<item 9>
Moya Brennan
Moya Brennan, born Máire Ní Bhraonáin (] ), also known as Máire Brennan (born 4 August 1952), is an Irish folk singer, songwriter, harpist, and philanthropist.

 She is the older sister of Enya and Brídín Brennan.

 She began performing professionally in 1970 when her family formed the band Clannad, and is considered as the "First Lady of Celtic Music".

 Moya released her first solo album in 1992 called "Máire", a successful venture.

 She has been nominated for two Grammys and has won an Emmy Award.

 She has recorded music for several soundtracks, including "Titanic", "To End All Wars" and "King Arthur".
</item 9>

<item 10>
No One Talks
"No One Talks" was the first commercial single taken from Grammy award-winning singer, Moya Brennan's album "Signature" released the same year.

 This was Moya's first single available to download from more than one online shop.

 The B-side to the single, the traditional Gaelic song "Éirigh Suas a Stóirín (Rise Up My Love)" was previously only available on the Germany-only album "Óró -
A Live Session".

 The cover shows a photograph by Mella Travers.
</item 10>
=== END ITEM id=50f10b8facf06f13 ===

=== ITEM id=955e42998d9dcd8e ===
Gold sentences and facts:
Sentence 0: John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.
  Fact 0.0: John Alexander was born on 5 October 1955.
  Fact 0.1: John Alexander is English.
  Fact 0.2: John Alexander is a former footballer.
  Fact 0.3: John Alexander holds the post of club secretary at Manchester United.
Sentence 1:  Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in 1976.
  Fact 1.0: John Alexander was born in Liverpool.
  Fact 1.1: John Alexander began his football career with a club called Ulysses.
  Fact 1.2: John Alexander was later picked up by Millwall.
  Fact 1.3: John Alexander made his league debut for Millwall in 1976.
Sentence 2:  He is the uncle of Liverpool player Trent Alexander-Arnold.
  Fact 2.0: John Alexander is the uncle of Liverpool player Trent Alexander-Arnold.
Sentence 3: Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.
  Fact 3.0: Trent John Alexander-Arnold was born on 7 October 1998.
  Fact 3.1: Trent John Alexander-Arnold is English.
  Fact 3.2: Trent John Alexander-Arnold is a professional footballer.
  Fact 3.3: Trent John Alexander-Arnold plays as a right-back.
  Fact 3.4: Trent John Alexander-Arnold plays for Premier League club Liverpool.

Retrieved items:
<item 1>
Trent Alexander-Arnold
Trent John Alexander-Arnold (born 7 October 1998) is an English professional footballer who plays as a right-back for Premier League club Liverpool.

 He is an academy graduate of Liverpool and made his senior debut for the club in 2016, aged 18.

 The following year, he won the club's Young Player of the Season award.

 He has also represented England at all youth levels from under-16 through to under-19.

 Due to the length of his name, his surname is shortened to Alexander for Premier League fixtures.
</item 1>

<item 2>
William McCourty
William McCourty (1884 – 10 December 1917) was an English professional footballer born in Morpeth, Northumberland, who played in the Football League for
Birmingham.

 McCourty, a former coal miner who played at left half, joined Birmingham in May 1909.

 He made his debut in the Second Division on 13 September 1909, deputising for the injured Tommy Daykin in a home game against Glossop which finished as a 2–2
draw, and was never picked again, returning to non-league football the following year.
</item 2>

<item 3>
Wilton Lines
Wilton Lines (fl.

 1889–1890) was an English professional footballer born in Birmingham.

 Lines played seven games for Small Heath in the inaugural 1889–90 season of the Football Alliance, covering a variety of forward positions, but was unable to
displace the established forwards.
</item 3>

<item 4>
John Logan (footballer, born 1912)
John William Logan (16 August 1912 – October 1980) was an English professional footballer born in Horden, near Peterlee, County Durham who played in the
Football League for Darlington, Barnsley and Sheffield Wednesday.

 He played as a wing half.
</item 4>

<item 5>
Clare Taylor
Clare Elizabeth Taylor MBE (born 22 May 1965) is an English sportswoman, the first woman to have played on a World Cup team in both cricket and football.

 She represented England at both cricket, as a member of the winning World Cup cricket team in 1993, and football (World Cup 1995).

 She was awarded the MBE in the Queen's Birthday Honours in 2000 for her services to women’s sport.

 Taylor attended Moor End High School and still has her name on the athletics records board.
</item 5>

<item 6>
Tommy Bell (footballer, born 1906)
Thomas Bell (9 November 1906 – 1983) was an English professional footballer born in Seaham Harbour.

 He could play at inside right, centre forward and right half and was a regular goalscorer throughout his professional career.
</item 6>

<item 7>
John Alexander (footballer, born 1955)
John Alexander (born 5 October 1955) is an English former footballer who holds the post of club secretary at Manchester United.

 Born in Liverpool, Alexander began his football career with a club called Ulysses, before being picked up by Millwall, for whom he made his league debut in
1976.

 After scoring twice in 15 appearances over the course of two years with Millwall, he moved to Reading; he enjoyed greater success with Reading, scoring nine
goals in 25 games.

 In 1981, he joined Northampton Town, but spent just one season with them before retiring from football at the age of 26 in 1982.

 Upon retiring from football, Alexander got a job with the BBC, but he later returned to football as club secretary at Watford.

 In 2000, he took up the same post at Tottenham Hotspur, spending 10 years there before applying for the same job at Manchester United, where he would replace
the retiring Ken Ramsden.

 He took over at Manchester United on 1 July 2010.

 He is the uncle of Liverpool player Trent Alexander-Arnold.
</item 7>

<item 8>
Nick Powell
Nicholas Edward "Nick" Powell (born 23 March 1994) is an English professional footballer who plays for Wigan Athletic as a midfielder.

 He began his career at Crewe Alexandra, initially as a forward, making his debut at the age of 16.

 After impressing for Crewe Alexandra during the 2011–12 season, including scoring in their 2–0 victory in the League Two play-off Final, he moved to Manchester
United in July 2012.

 Unable to break into the Manchester United first team, however, he went on loan to Wigan Athletic, Leicester City and Hull City before being released in June
2016.

 He joined Wigan Athletic in July 2016.

 He has represented England at under-16, under-17, under-18, under-19 and under-21 levels.
</item 8>

<item 9>
Jack Charlton
John Charlton, OBE, DL (born 8 May 1935) is an English former footballer and manager who played as a defender.

 He was part of the England team that won the 1966 World Cup.

 He is the elder brother of former Manchester United forward Bobby Charlton, who was also a teammate in England's World Cup final victory.

 He spent his entire club career with Leeds United from 1950 to 1973, helping the club to the Second Division title (1963–64), First Division title (1968–69),
FA Cup (1972), League Cup (1968), Charity Shield (1969), Inter-Cities Fairs Cup (1968 and 1971), as well as one other promotion from the Second Division
(1955–56) and five second-place finishes in the First Division, two FA Cup final defeats and one Inter-Cities Fairs Cup final defeat.

 His 629 league and 762 total competitive appearances are club records.

 In 2006, Leeds United supporters voted Charlton into the club's greatest ever XI.
</item 9>

<item 10>
Billy Morgan (footballer, born 1891)
William Albert L. Morgan (3 November 1891 – after 1927) was an English professional footballer born in Old Hill, Cradley Heath, Staffordshire, who played either
at outside left or inside left.

 He played for Birmingham, Coventry City and Crystal Palace in the Football League, and was capped once for a Football League representative side.
</item 10>
=== END ITEM id=955e42998d9dcd8e ===

=== ITEM id=f802b78f4b0c7d3c ===
Gold sentences and facts:
Sentence 0: Colocasia is a genus of flowering plants in the family Araceae, native to southeastern Asia and the Indian subcontinent.
  Fact 0.0: Colocasia is a genus of flowering plants.
  Fact 0.1: Colocasia is in the family Araceae.
  Fact 0.2: Colocasia is native to southeastern Asia.
  Fact 0.3: Colocasia is native to the Indian subcontinent.
Sentence 1: The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
  Fact 1.0: The genus Coronilla contains about 20 species.
  Fact 1.1: The Coronilla species are flowering plants.
  Fact 1.2: The Coronilla species are native to Europe.
  Fact 1.3: The Coronilla species are native to North Africa.

Retrieved items:
<item 1>
Coris monspeliensis
Coris monspeliensis has erect or ascending stems which are woody at the base and densely covered with foliage.

 The leathery linear leaves, which grow up to 20 mm long are alternate and stalkless.

 They may be hairless or hairy and the highest leaves often have small teeth.

 The flowers with 5 unequal petals, each deeply lobed are borne in short dense clusters and vary in color from pink to blue.

 Each flower measures up to 12 mm across and has narrow, widely spread petals.

 The calyx is bell shaped and has up to 20 red or black spiny teeth.

 The spherical capsule is up to 2 mm in diameter.
</item 1>

<item 2>
Caladium
Caladium is a genus of flowering plants in the family Araceae.

 They are often known by the common name elephant ear (which they share with the closely related genera "Alocasia", "Colocasia", and "Xanthosoma"), Heart of
Jesus, and Angel Wings.

 There are over 1000 named cultivars of "Caladium bicolor" from the original South American plant.
</item 2>

<item 3>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 3>

<item 4>
Xanthosoma
Xanthosoma is a genus of flowering plants in the arum family, Araceae.

 The genus is native to tropical America but widely cultivated and naturalized in other tropical regions.

 Several are grown for their starchy corms, an important food staple of tropical regions, known variously as "malanga", "otoy", "otoe", cocoyam (or new
cocoyam), "tannia", "tannier", "yautía", "macabo", "ocumo", "macal", "taioba", "dasheen", "quequisque", "ʻ ape" and (in Papua New Guinea) as Singapore taro
("taro kongkong").

 Many other species (including especially "X. roseum") are used as ornamental plants, and in popular horticultural literature are known as ‘ape or elephant ear
(from the purported resemblance of the leaf to an elephant's ear), although the latter name is sometimes also applied to members with similar appearance and
uses in the closely related genera "Caladium", "Colocasia" (i.e., taro), and "Alocasia".
</item 4>

<item 5>
Conophytum
Conophytum is a genus of South African and Namibian succulent plants that belong to the Aizoaceae family.

 The name is derived from the Latin “conus” (cone) and Greek “phytum” (plant).

 The plants are also known as knopies (buttons), waterblasies (water blisters), sphaeroids, conos, cone plants, dumplings, or button plants.
</item 5>

<item 6>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 6>

<item 7>
Coronilla
The genus Coronilla contains about 20 species of flowering plants native to Europe and North Africa.
</item 7>

<item 8>
Colocasia
Colocasia is a genus of flowering plants in the family Araceae, native to southeastern Asia and the Indian subcontinent.

 Some species are widely cultivated and naturalized in other tropical and subtropical regions.

 Common names include tarul, karkala ko ganu, elephant-ear, taro, cocoyam, dasheen, chembu, champadhumpa, shavige gadde, and eddoe.

 Elephant-ear and cocoyam are also used for some other large-leaved genera in the Araceae, notably "Xanthosoma" and "Caladium".

 The generic name is derived from the ancient Greek word "kolokasion", which in the Greek botanist Dioscorides (1st century AD) meant the edible roots of both
"Colocasia esculenta" and "Nelumbo nucifera".

 It is thought that the edible roots of "Colocasia esculenta" have been cultivated in Asia for more than ten thousand years.

 The species "Colocasia esculenta" is an invasive species in wetlands along the American Gulf coast, where it threatens to displace native wetland plants.
</item 8>

<item 9>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 9>

<item 10>
Conicosia
Conicosia is a genus of succulent plants in the ice plant family native to southern Africa.

 They are known commonly as narrow-leafed ice plants.

 These are relatively short-lived perennials with underground stems and tentacle-shaped, dull-pointed triangular leaves.

 They bear large tubular flowers often exceeding 10 centimeters in width, with up to 250 fringelike petals arranged in a ring around a center with hundreds of
stamens.

 The fruit is a capsule which opens when it gets wet, slowly releasing the hundreds of tiny seeds as they fall out of its drying flesh.
</item 10>
=== END ITEM id=f802b78f4b0c7d3c ===

=== ITEM id=7b73bb09da7f6f83 ===
Gold sentences and facts:
Sentence 0: Rivermaya is a Filipino rock band.
  Fact 0.0: Rivermaya is a Filipino rock band.
Sentence 1: Kodaline ( ) are an Irish band.
  Fact 1.0: Kodaline is an Irish band.

Retrieved items:
<item 1>
Inkwell (band)
Inkwell is an indie rock band from Winter Park, Florida.

 They have released three full length albums and one EP, as well as a collaboration with former techno artist Floorboard.

 They are currently signed with One Eleven Records.

 The band consists of two members, Travis Adams, former frontman from My Hotel Year and Davey Pierce who has worked with of Montreal.

 The band recently performed a small tour with of Montreal along the east coast.

 Their most recent album "Rivers of Blood and Sadness, or Maybe Happy" was released on iTunes music store April 21, 2009.
</item 1>

<item 2>
The Roy Files
The Roy Files is an Irish-based British children's television show, filmed in Dublin, Ireland, which was broadcast by CBBC in the United Kingdom and ABC Me in
Australia.

 It began airing 7 December 2015.

 The show centres on the title character Roy O'Brien, the 11-year-old animated son of a live-action family.

 The series is a spin-off from "ROY", an Irish-British television show which ran from 1 July 2009 to 7 April 2015.

 It stars Jason Cullen (replacing Scott Graham as Roy), Simon Delaney, Cathy Belton and Martha Byrne.
</item 2>

<item 3>
Honest (Kodaline song)
"Honest" is a song by Dublin-based alternative rock quartet Kodaline.

 The song was released on 1 February 2015 as the lead single from the band's second studio album, "Coming Up for Air" (2015).

 "Honest" became the band's third top 10 single in their home country, Ireland, following "High Hopes" (2013) and "Love Like This" (2013).
</item 3>

<item 4>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 4>

<item 5>
Raindancer
Raindancer was an electroacoustic rock band formed in Newport, Wales in the early 1990s.

 The band featured Grant Nicholas and Jon Lee, who later became founding members of Feeder.
</item 5>

<item 6>
Kodaline
Kodaline ( ) are an Irish band.

 Originally known as 21 Demands, in 2012 the band changed their name to Kodaline.
</item 6>

<item 7>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 7>

<item 8>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 8>

<item 9>
Rivermaya
Rivermaya is a Filipino rock band.

 Formed in 1994, it is one of several bands who spearheaded the 1990s Philippine alternative rock explosion.

 Rivermaya is currently composed of original members Mark Escueta and Nathan Azarcon, together with Mike Elgar and Ryan Peralta.

 Former original members include Rico Blanco, who had been the original songwriter of the band and vocalist Bamboo Mañalac, who later formed the band Bamboo and
later went on his solo career.

 Rivermaya is listed as the twentieth biggest-selling artists/act in the Philippines as of present.
</item 9>

<item 10>
Perfecto de Castro
Perfecto "Perf" de Castro (born August 14) is a multi-awarded Filipino musician, currently focusing on Classical and Flamenco music played on the Ten-string
guitar, and now widely regarded being the best guitarist in the country.

 but perhaps best known for having been a celebrated fixture in the Philippine alternative rock scene during the 1990s.

 During the course of that decade, he was one of the original members of the alternative rock band Rivermaya, founded the band Triaxis, and also collaborated
with the seminal Filipino rapper Francis Magalona and Filipino hard rock band Wolfgang.
</item 10>
=== END ITEM id=7b73bb09da7f6f83 ===

=== ITEM id=6b8e08c74dcac9de ===
Gold sentences and facts:
Sentence 0: Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
  Fact 0.0: The Spanish name of Teide National Park is 'Parque nacional del Teide'.
  Fact 0.1: Teide National Park is a national park.
  Fact 0.2: Teide National Park is located in Tenerife.
  Fact 0.3: Tenerife is in the Canary Islands, Spain.
Sentence 1: Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands (Spain).
  Fact 1.0: The Spanish name of Garajonay National Park is 'Parque nacional de Garajonay'.
  Fact 1.1: Garajonay National Park is located in the center of the island of La Gomera.
  Fact 1.2: Garajonay National Park is located in the north of the island of La Gomera.
  Fact 1.3: La Gomera is one of the Canary Islands in Spain.

Retrieved items:
<item 1>
Graph query row:
park: Teide National Park
relationship: LOCATED_IN
location: Tenerife
</item 1>

<item 2>
Badiar National Park
Badiar National Park (French: "Parc National du Badiar" ) is a national park in Guinea, on the border with Senegal and contiguous with Senegal's much larger
Niokolo-Koba National Park.

 It was established on 30 May 1985 (by ordonnance N°124/PRG/85), partly in response to Senegal's concern about poaching in Niokolo-Koba National Park.

 Badiar is an International Union for Conservation of Nature Category II park.
</item 2>

<item 3>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
</item 3>

<item 4>
Graph query row:
park: Garajonay National Park
relationship: LOCATED_IN
location: La Gomera
</item 4>

<item 5>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 5>

<item 6>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 6>

<item 7>
Guadarrama National Park
Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the
fifth largest in Spain's national parks system.

 The Guadarrama mountain range ("Sistema Central") contains some ecologically valuable areas, located in the Community of Madrid and Castile and León (provinces
of Segovia and Ávila).

 The law that regulates the recently approved national park was published in the BOE in  26, 2013 (2013--) .
</item 7>

<item 8>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 8>

<item 9>
Landes de Gascogne Regional Natural Park
Landes de Gascogne Regional Natural Park (French: "Parc naturel régional des Landes de Gascogne") is a protected area of pine forest, wetland and oceanic
coastline located in the Aquitaine region of southwestern France.
</item 9>

<item 10>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 10>
=== END ITEM id=6b8e08c74dcac9de ===

=== ITEM id=aa104e72c2ab16c7 ===
Gold sentences and facts:
Sentence 0:  Along with competing in the league, the club will also participate in the FA Cup, EFL Cup and EFL Trophy.
  Fact 0.0: The club will compete in the league.
  Fact 0.1: The club will also participate in the FA Cup.
  Fact 0.2: The club will also participate in the EFL Cup.
  Fact 0.3: The club will also participate in the EFL Trophy.
Sentence 1: The EFL Cup (referred to historically, and colloquially, as simply the League Cup), currently known as the Carabao Cup for sponsorship reasons, is an annual knockout football competition in men's domestic English football.
  Fact 1.0: The EFL Cup is historically referred to as the League Cup.
  Fact 1.1: The EFL Cup is colloquially referred to as simply the League Cup.
  Fact 1.2: The EFL Cup is currently known as the Carabao Cup for sponsorship reasons.
  Fact 1.3: The EFL Cup is an annual knockout football competition.
  Fact 1.4: The EFL Cup is a competition in men's domestic English football.

Retrieved items:
<item 1>
2012–13 Midlands football leagues (levels 9–10)
The 2012–13 Midland Football Alliance season is the 19th in the history of Midland Football Alliance a football competition in England.

 The 2012–13 Midland Football Combination season (known as the 2012–13 Athium Midland Football Combination for sponsorship reasons) is the 76th in the history
of Midland Football Combination a football competition in England and feeder to the Midland Football Alliance.

 The 2012–13 West Midlands (Regional) League season is the 113th in the history of the West Midlands (Regional) League.

 It is also a feeder league to the Midland Football Alliance.
</item 1>

<item 2>
2013–14 Midlands football leagues (levels 9–10)
The 2013–14 Midland Football Alliance season was the 20th and final in the history of Midland Football Alliance, a football competition in England.

 The 2013–14 Midland Football Combination season (known as the 2013–14 Athium Midland Football Combination for sponsorship reasons) was the 77th and final in
the history of Midland Football Combination and it was a feeder to the Midland Football Alliance.

 The 2013–14 West Midlands (Regional) League season was the 114th in the history of the West Midlands (Regional) League.

 It was also a feeder league to the Midland Football Alliance.

 At the end of the season the Midland Alliance and the Midland Combination merged to form the Midland Football League.
</item 2>

<item 3>
2017–18 Wigan Athletic F.C. season
The 2017–18 season is Wigan Athletic's 86th year in existence and their first back in League One, after being relegated the previous season.

 Along with competing in the league, the club will also participate in the FA Cup, EFL Cup and EFL Trophy.

 Paul Cook was appointed as manager on 31 May 2017, signing a 3 year deal, following the departure of much of the previous coaching staff the day before.
</item 3>

<item 4>
1979–80 Wigan Athletic F.C. season
The 1979–80 season was the 42nd season in the history of Wigan Athletic F.C. and their second as a professional club in the Football League.
</item 4>

<item 5>
2006–07 Wigan Athletic F.C. season
The 2006–07 Wigan Athletic F.C. season was the club's 29th season in the Football League and their second season in the Premier League.
</item 5>

<item 6>
EFL Cup
The EFL Cup (referred to historically, and colloquially, as simply the League Cup), currently known as the Carabao Cup for sponsorship reasons, is an annual
knockout football competition in men's domestic English football.

 Organised by the English Football League (EFL), it is open to any club within the top four levels of the English football league system – 92 clubs in total –
comprising the top level Premier League, and the three divisions of the English Football League's own league competition (Championship, League One and League
Two).
</item 6>

<item 7>
2007–08 Wigan Athletic F.C. season
The 2007–08 Wigan Athletic F.C. season was the club's 30th season in the Football League and their third season in the Premier League.
</item 7>

<item 8>
Springfield Park (Wigan)
Springfield Park was a multi-purpose stadium in Wigan, Greater Manchester.

 It was the home ground of Wigan Athletic F.C. until the club moved to the new JJB Stadium (now DW Stadium) after the 1998–99 season.

 At its largest, the stadium held 40,000.

 In its 102-year existence the ground only saw 32 years as a Football League venue, 11 years for Wigan Borough F.C. and 21 years for Wigan Athletic FC, before
it was demolished to make way for a housing estate in 1999.
</item 8>

<item 9>
1978–79 Wigan Athletic F.C. season
The 1978–79 season was the 41st season in the history of Wigan Athletic F.C. and their first as a professional club in the Football League.

 After finishing 2nd in the Northern Premier League during the previous season, the club was nominated to apply for Football League status, and were elected
into the league to replace Southport.

 After a poor start, the club exceeded expectations in the league, and towards the end of the season, a second consecutive promotion seemed possible.

 The club eventually fell short, finishing the season in 6th place with a total of 55 points.
</item 9>

<item 10>
List of Wigan Borough F.C. players
Wigan Borough F.C. was an English football club based in the town of Wigan.

 The club was founded in 1920 and joined the Lancashire Combination.

 In 1921, Borough turned professional when their application was accepted to play in the inaugural season of the newly formed Football League Third Division
North.

 The team played in the Football League for ten seasons, with their most successful season coming in 1928–29, finishing fourth in the league and reaching the
third round of the FA Cup.

 Wigan Borough folded during the 1931–32 season due to financial problems, and League football did not return to the town until Wigan Athletic F.C. were elected
into the Football League in 1978.
</item 10>
=== END ITEM id=aa104e72c2ab16c7 ===

=== ITEM id=f6db8c9604df41f4 ===
Gold sentences and facts:
Sentence 0:  It has an area of 1208 km2 and had a population of 600,000 in 2003.
  Fact 0.0: Ezhou has an area of 1208 km2.
  Fact 0.1: Ezhou had a population of 600,000 in 2003.
Sentence 1: Ezhou () is a prefecture-level city in eastern Hubei Province, China.
  Fact 1.0: Ezhou is a prefecture-level city.
  Fact 1.1: Ezhou is in eastern Hubei Province.
  Fact 1.2: Ezhou is in China.
Sentence 2:  As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.
  Fact 2.0: As of the 2010 census, Ezhou had a population of 1,048,668.
  Fact 2.1: As of the 2010 census, 668,727 of Ezhou's residents lived in the core Echeng District.
Sentence 3:  The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
  Fact 3.0: The Ezhou-Huanggang built-up (or metro) area was home to 1,035,496 inhabitants.
  Fact 3.1: The inhabitants of the Ezhou-Huanggang built-up area came from the Echeng and Huangzhou districts of Huanggang.

Retrieved items:
<item 1>
Suining
Suining (; Sichuanese Pinyin: Xunin; Sichuanese pronunciation: ; ) is a prefecture-level city of eastern Sichuan province in Southwest China.

 In 2002, Suining had a population of 658,798.
</item 1>

<item 2>
Dafeng District
Dafeng () is a coastal district under the administration of Yancheng, Jiangsu province, China.

 Located on the Jiangsu North Plain with a coastline of 112 km , Dafeng was historically one of the largest salt-making areas in China and now is famed for its
well preserved eco-system and numerous national conservation parks.

 The district has the largest national nature reserve for a rare deer species, Père David's Deer or Milu (麋鹿 ) in Chinese.

 It borders the prefecture-level city of Taizhou to the southwest.
</item 2>

<item 3>
Ezhou
Ezhou () is a prefecture-level city in eastern Hubei Province, China.

 As of the 2010 census, the city had a population of 1,048,668, of which 668,727 lived in the core Echeng District.

 The Ezhou - Huanggang built-up ("or metro") area was home to 1,035,496 inhabitants from the Echeng and Huangzhou, Huanggang Districts.
</item 3>

<item 4>
Dongtai
Dongtai () is a coastal county-level city under the administration of Yancheng, Jiangsu province, China.

 It has a population of roughly 1,170,000 estimated for 2007.

 Out of the total population, about 260,000 live in the Dongtai urban core, others are distributed in the 23 suburban towns and rural regions (Some famous towns
include Touzao, Anfeng, Qingdong, Sancang, Qianggang, Fu'an, Tangyang, etc.).

 With some Yellow Sea coast, it borders the prefecture-level cities of Nantong to the south and Taizhou to the west, and is the southernmost county-level
division of Yancheng.
</item 4>

<item 5>
Daying County
Daying County () is a county of Sichuan province, China, under the administration of Suining City and in the central part of the Sichuan Basin.

 In 2002, it had a population of 520,000 residing in an area of 703 km2 .

 Daying is remarkable for being the home of the "Chinese Dead Sea", a tourist attraction that is based on a salt lake with 9 times the salinity of the ocean and
which constitutes the largest indoor water park in China.

 The G42 Shanghai–Chengdu Expressway and Dazhou–Chengdu Railway (达成铁路) traverses east–west through the entire length of the county.
</item 5>

<item 6>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 6>

<item 7>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 7>

<item 8>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 8>

<item 9>
Jiangyin dialect
Jiangyin dialect (江阴话) is a Northern Wu Chinese dialect spoken in the city of Jiangyin in Jiangsu province.

 Jiangyin dialect is a member of the Wu Chinese Taihu Wu family of dialects, which means the inhabitants speak a dialect similar to that of nearby Wuxi,
Changzhou, Suzhou, and Shanghai.

 Jiangyin dialect itself is of the Piling variety, related to the Changzhou dialect.

 Jiangyin dialect has the highest degree of mutual intelligibility with the dialects of the closest neighboring cities of Changzhou and Wuxi but also has a
fairly large degree of mutual intelligibility with the dialects of nearby Suzhou and Shanghai.

 As one travels south towards Wuxi away from the urban center of Jiangyin, Jiangyin dialect gradually becomes more and more closer sounding to the Wuxi dialect.
</item 9>

<item 10>
Dujiangyan City
Dujiangyan () is a county-level city, a subdivision of Chengdu, the capital of Sichuan Province, People's Republic of China.

 Its north-west region forms a border with sourthern Ngawa Tibetan and Qiang Autonomous Prefecture.

 It has an area of 1208 km2 and had a population of 600,000 in 2003.
</item 10>
=== END ITEM id=f6db8c9604df41f4 ===

=== ITEM id=1076d732bfac1b03 ===
Gold sentences and facts:
Sentence 0:  It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
  Fact 0.0: The book is a companion book to 'A Wrinkle in Time'.
  Fact 0.1: The book is part of the Time Quintet.
Sentence 1: The Time Quintet is a fantasy/science fiction series of five young adult novels written by Madeleine L'Engle.
  Fact 1.0: The Time Quintet is a fantasy/science fiction series.
  Fact 1.1: The Time Quintet is a series of five young adult novels.
  Fact 1.2: The Time Quintet was written by Madeleine L'Engle.

Retrieved items:
<item 1>
Time Quintet
The Time Quintet is a fantasy/science fiction series of five young adult novels written by Madeleine L'Engle.
</item 1>

<item 2>
The Frost King
"The Frost King" was a short story about King Jack Frost written by 11-year-old Helen Keller.

 Keller's teacher, Anne Sullivan, had mentioned that the autumn leaves were "painted ruby, emerald, gold, crimson, and brown," and Keller, by her own account,
imagined fairies doing the work.

 Keller wrote a story about how a cask of jewels, being transported by fairy servants, had melted in the sun and covered the leaves.

 As a birthday gift, Keller sent the story to Michael Anagnos, head of the Perkins School for the Blind, who published the story in "The Mentor," the Perkins
alumni magazine.

 It was picked up by "The Goodson Gazette", a journal on deaf-blind education, based in Virginia.
</item 2>

<item 3>
The Happy Journey to Trenton and Camden
The Happy Journey to Trenton and Camden is a one act play by American novelist and playwright Thornton Wilder written in 1931.

 It was first published in "The Long Christmas Dinner and Other Plays in One Act" (New York: Coward-McCann, 1931).
</item 3>

<item 4>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 4>

<item 5>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 5>

<item 6>
The Four Winds of Love
The Four Winds of Love is the overall title for a series of six novels written by Compton Mackenzie, "The East Wind of Love" (1937), "The South Wind of Love"
(1938), "The West Wind of Love" (1940), "West to North" (1942), "The North Wind of Love, Book 1" (1944) and "The North Wind of Love, Book 2" (1945), which taken
together constitute a major fictional chronicle of the first forty years of the twentieth century.

 The main protagonist of the hexalogy is the semi-autobiographical character of John Ogilvie.
</item 6>

<item 7>
Gate of Ivory, Gate of Horn
Gate of Ivory, Gate of Horn is a fantasy novel by British author Robert Holdstock.

 It was originally published in the United States in 1997 (and in the United Kingdom under the title "Gate of Ivory" in 1998.)

 The story is a prequel to "Mythago Wood" and explores Christian Huxley's quest into Ryhope Wood and the apparent suicide of his mother, Jennifer Huxley.

 The title of the book refers to the gates of horn and ivory described in both Homer's "Odyssey" and Virgil's "Aeneid".
</item 7>

<item 8>
Warlord of the Air
The Warlord of the Air is a 1971 British alternate history science fiction novel written by Michael Moorcock.

 It concerns the adventures of Oswald Bastable, an Edwardian era soldier stationed in India, and his adventures in an alternate universe, in his own future,
wherein the First World War never happened.

 It is the first part of Moorcock's 'A Nomad of the Time Streams' trilogy and, in its use of speculative technology (such as airships) juxtaposed against an
Edwardian setting, it is widely considered to be one of the first steampunk novels.

 The novel was first published by Ace Books as part of their Ace Science Fiction Specials series.
</item 8>

<item 9>
Dust in the Wind (film)
Dust in the Wind is a 1986 film by Taiwanese filmmaker Hou Hsiao-hsien.

 It is based on co-screenwriter Wu Nien-jen's own experiences, and is the final part of Hou's coming-of-age trilogy, with the others being "A Summer at
Grandpa's" (1984) and "The Time to Live and the Time to Die" (1985).
</item 9>

<item 10>
A Wind in the Door
A Wind in the Door is a young adult science fantasy novel by Madeleine L'Engle.

 It is a companion book to "A Wrinkle in Time", and part of the Time Quintet.
</item 10>
=== END ITEM id=1076d732bfac1b03 ===

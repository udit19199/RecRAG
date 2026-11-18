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

This file has 11 items.

=== ITEM id=9f22af131c4d0e38 ===
Gold sentences and facts:
Sentence 0: Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.
  Fact 0.0: Dangal's English title is "Wrestling competition".
  Fact 0.1: Dangal is a 2016 film.
  Fact 0.2: Dangal is an Indian film.
  Fact 0.3: Dangal is a Hindi-language film.
  Fact 0.4: Dangal is a biographical sports drama film.
  Fact 0.5: Dangal is loosely based on the story of Mahavir Singh Phogat.
  Fact 0.6: Mahavir Singh Phogat is an amateur wrestler.
  Fact 0.7: Mahavir Singh Phogat trains his daughters Geeta Phogat and Babita Kumari.
  Fact 0.8: Mahavir Singh Phogat trains his daughters to be world-class wrestlers.
Sentence 1: The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
  Fact 1.0: The Man from Snowy River II is a 1988 film.
  Fact 1.1: The Man from Snowy River II is an Australian drama film.
  Fact 1.2: The Man from Snowy River II is the sequel to the 1982 film "The Man from Snowy River".

Retrieved items:
<item 1>
The Man from Snowy River II
The Man from Snowy River II is a 1988 Australian drama film, the sequel to the 1982 film "The Man from Snowy River".
</item 1>

<item 2>
The Man from Snowy River and Other Verses
The Man from Snowy River and Other Verses (1895) is the first collection of poems by Australian poet Banjo Paterson.

 It was released in hardback by Angus and Robertson in 1895, and features the poet's widely anthologised poems "The Man from Snowy River", "Clancy of the
Overflow", "Saltbush Bill" and "The Man from Ironbark".

 It also contains the poet's first two poems that featured in The Bulletin Debate, a famous dispute in "The Bulletin" magazine from 1892-93 between Paterson and
Henry Lawson.
</item 2>

<item 3>
The Man from Snowy River (poem)
"The Man from Snowy River" is a poem by Australian bush poet Banjo Paterson.

 It was first published in "The Bulletin", an Australian news magazine, on 26 April 1890, and was published by Angus & Robertson in October 1895, with other
poems by Paterson, in "The Man from Snowy River and Other Verses".
</item 3>

<item 4>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 4>

<item 5>
The Man from Snowy River (1982 film)
The Man from Snowy River is a 1982 Australian drama film based on the Banjo Paterson poem "The Man from Snowy River".

 The film had a cast including Kirk Douglas in a dual role as the brothers Harrison (a character who appeared frequently in Paterson's poems) and Spur, Jack
Thompson as Clancy, Tom Burlinson as Jim Craig, Sigrid Thornton as Harrison's daughter Jessica, Terence Donovan as Jim's father Henry Craig, and Chris Haywood
as Curly.
</item 5>

<item 6>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 6>

<item 7>
The Man from Snowy River (soundtrack)
"The Man from Snowy River is the original motion picture soundtrack from the 1982 film "The Man from Snowy River".
</item 7>

<item 8>
The Man from Snowy River (TV series)
The Man from Snowy River is an Australian adventure drama television series based on Banjo Paterson's poem "The Man from Snowy River".

 Released in Australia as "Banjo Paterson's The Man from Snowy River", the series was subsequently released in both the United States and the United Kingdom as
"Snowy River: The McGregor Saga".
</item 8>

<item 9>
Return to Snowy River
Return to Snowy River is the original motion picture soundtrack from the 1988 film "The Man from Snowy River II".

 The soundtrack album is named after the American title for the film.
</item 9>

<item 10>
Dangal (film)
Dangal (English: "Wrestling competition" ) is a 2016 Indian Hindi-language biographical sports drama film loosely based on the story of Mahavir Singh Phogat, an
amateur wrestler, who trains his daughters Geeta Phogat and Babita Kumari to be world-class wrestlers.

 It was directed by Nitesh Tiwari.

 Produced by Walt Disney Pictures, Aamir Khan Productions and UTV Motion Pictures, it stars Aamir Khan as the father.

 Both daughters go on to win medals at the 2010 Commonwealth Games.

 Babita won a silver medal at the Games, in the 51 kg class and gold at the 2014 Glasgow Games in the 55 kg class.

 In 2012, Geeta became the first Indian female wrestler to qualify for the Olympics.

 Mahavir's efforts inspired dozens of Indian women to take to wrestling.

 Fatima Sana Shaikh and Sanya Malhotra play the older selves of the sisters while Sakshi Tanwar plays their mother and Aparshakti Khurana, their cousin.
</item 10>
=== END ITEM id=9f22af131c4d0e38 ===

=== ITEM id=d3e4461dacd7fda1 ===
Gold sentences and facts:
Sentence 0: Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.
  Fact 0.0: Bertalan Farkas was born on August 2, 1949.
  Fact 0.1: Bertalan Farkas is the first Hungarian cosmonaut.
  Fact 0.2: Bertalan Farkas is the first Esperantist in space.
Sentence 1: Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
  Fact 1.0: Hans Wilhelm Schlegel was born in Überlingen on 3 August 1951.
  Fact 1.1: Hans Wilhelm Schlegel is a German physicist.
  Fact 1.2: Hans Wilhelm Schlegel is an ESA astronaut.
  Fact 1.3: Hans Wilhelm Schlegel is a veteran of two NASA Space Shuttle missions.

Retrieved items:
<item 1>
Hans Aschenborn
Hans Anton Aschenborn (1 February 1888 – 10 April 1931) was a renowned animal painter of African wildlife.

 He worked both in Germany and in southern Africa.

 His work is featured in the older German Thieme-Becker or Saur art encyclopedia.

 A Master of Arts thesis by Karin Skawran concerning the graphic works of Hans Anton Aschenborn was published in the South African art and culture periodical,
"Lantern" in 1965.)

 In 1963 the University of Pretoria published a book about Aschenborn as an artist entitled, "Hans Anton Aschenborn : Mens en Kunstenaar".

 In 1970 another one followed by the Pretoria Art Museum (South Africa).

 Other publications on Aschenborn feature his etchings and linocuts.
</item 1>

<item 2>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 2>

<item 3>
Gassenhauer
Gassenhauer nach Hans Neusiedler (1536), commonly known as Gassenhauer (] ), is a short piece from Carl Orff's Schulwerk.

 As the full title indicates, it is either an arrangement of, or inspired by, a much older work by the lutenist Hans Neusiedler from 1536.

 It (along with several other pieces) is in fact credited to Orff's longtime collaborator, Gunild Keetman, on at least one recent release of the Schulwerk.

 As with many other pieces from the Schulwerk, it has been used multiple times on television, radio, and in films, including the films "Badlands" (1973),
"Ratcatcher" (1999), "Finding Forrester" (2000), "Monster" (2003), and, most recently, Michael Moore's "" (2009) and "The Simpsons"′ 22nd-season episode "The
Scorpion's Tale" (2011).

 The piece was used as the theme music for an afternoon radio program also titled "Gassenhauer" on the classical music station WCLV in Cleveland, Ohio, in the
1970s.
</item 3>

<item 4>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 4>

<item 5>
Bernhard Bötel
Bernhard Bötel (1883–1953) was a German operatic tenor and actor who had an active career in Germany and Austria during the first half of the 20th century.

 He made recordings for several record labels during the early years of the recording industry, including His Master's Voice, Odeon Records, Polydor Records,
Tri-Ergon, and Vox Records.

 On the stage he sang a variety of roles in operas and operettas from leading parts to comprimario roles.
</item 5>

<item 6>
Richard Genée
Franz Friedrich Richard Genée (February 7, 1823 – June 15, 1895) was a Prussian born Austrian librettist, playwright, and composer.

 He is most famous for the libretto of "Die Fledermaus", Johann Strauss II's most famous operetta.

 He co-wrote the libretto without having met top-billed librettist Karl Haffner, who constructed the new story based on a play by Henri Meilhac and Ludovic
Halévy, which was considered too shocking to perform outside Paris.

 Genée, however, wrote the operetta's actual text and drew nothing from Haffner beyond the names of the characters.
</item 6>

<item 7>
Karl Wilhelm Friedrich Schlegel
Karl Wilhelm Friedrich (after 1814: von) Schlegel (10 March 1772 – 12 January 1829), usually cited as Friedrich Schlegel, was a German poet, literary critic,
philosopher, philologist and Indologist.

 With his older brother, August Wilhelm Schlegel, he was one of the main figures of the Jena romantics.

 He was a zealous promoter of the Romantic movement and inspired Samuel Taylor Coleridge, Adam Mickiewicz and Kazimierz Brodziński.

 Schlegel was a pioneer in Indo-European studies, comparative linguistics, morphological typology, and was the first to notice what became known as Grimm's law.
</item 7>

<item 8>
Hans von Hentig
Hans von Hentig (9 June 1887 in Berlin – 6 July 1974 in Bad Tölz) was a German criminal psychologist and politician.He was the second son of lawyer Otto von
Hentig (1852–1934) .

His older brother was later diplomat Werner Otto von Hentig.

 Otto von Hentig was one of the leading lawyers in Berlin.

 Hans von Hentig was instrumental in the setting up of a short-lived Bavarian Soviet Republic in 1919.

 During the 1920s he was a prominent exponent of National Bolshevism.

 He emigrated to United States in 1935.

 Hans von Hentig worked on Yale University.
</item 8>

<item 9>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 9>

<item 10>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 10>
=== END ITEM id=d3e4461dacd7fda1 ===

=== ITEM id=210dc1128167b5ab ===
Gold sentences and facts:
Sentence 0:  He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his early work as Knox Overstreet in "Dead Poets Society".
  Fact 0.0: The actor referred to as 'He' is best known for the role of Dan Rydell on "Sports Night."
  Fact 0.1: The actor referred to as 'He' is best known for the role of Will Gardner on "The Good Wife."
  Fact 0.2: The role of Will Gardner on "The Good Wife" earned the actor two Primetime Emmy Award nominations.
  Fact 0.3: The actor referred to as 'He' is best known for his early work as Knox Overstreet in "Dead Poets Society."
Sentence 1: Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.
  Fact 1.0: Dead Poets Society is a 1989 American drama film.
  Fact 1.1: Dead Poets Society was directed by Peter Weir.
  Fact 1.2: Dead Poets Society was written by Tom Schulman.
  Fact 1.3: Dead Poets Society stars Robin Williams.
Sentence 2:  Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students through his teaching of poetry.
  Fact 2.0: Dead Poets Society is set in 1959.
  Fact 2.1: Dead Poets Society is set at the fictional elite conservative Vermont boarding school Welton Academy.
  Fact 2.2: Dead Poets Society tells the story of an English teacher.
  Fact 2.3: The English teacher in Dead Poets Society inspires his students through his teaching of poetry.

Retrieved items:
<item 1>
Dead Poets Society
Dead Poets Society is a 1989 American drama film directed by Peter Weir, written by Tom Schulman, and starring Robin Williams.

 Set in 1959 at the fictional elite conservative Vermont boarding school Welton Academy, it tells the story of an English teacher who inspires his students
through his teaching of poetry.
</item 1>

<item 2>
License to Wed
License to Wed is a 2007 American romantic comedy film starring Robin Williams, Mandy Moore and John Krasinski, and directed by Ken Kwapis.

 The film was released in theaters on July 3, 2007.
</item 2>

<item 3>
The Boy with a Thorn in His Side
The Boy with a Thorn in His Side is a 2005 comedy film, produced and directed by Mark Jeavonsa as his feature film debut, and starring Alec Sedgley as Billy
Heinlickburger.

 Its title is almost the same as a song by The Smiths.
</item 3>

<item 4>
Josh Charles
Joshua Aaron "Josh" Charles (born September 15, 1971) is an American stage, film and television actor.

 He is best known for the roles of Dan Rydell on "Sports Night;" Will Gardner on "The Good Wife", which earned him two Primetime Emmy Award nominations; and his
early work as Knox Overstreet in "Dead Poets Society".
</item 4>

<item 5>
Old Boyfriends
Old Boyfriends is a 1979 American drama film directed by Joan Tewkesbury and written by Paul Schrader and Leonard Schrader.

 The film stars Talia Shire, Richard Jordan, Keith Carradine, John Belushi, John Houseman and Buck Henry.

 The film was released on April 13, 1979, by Embassy Pictures.
</item 5>

<item 6>
Reunion (1989 film)
The story is centred on the "enchanted friendship" of two teenagers in 1933 Germany.

 Hans Strauss (Christien Anholt) is the son of a Jewish doctor and Konradin Von Lohenburg (Samuel West) is from an aristocratic family.

 The background is the rise of Nazism.

 Jason Robards plays the older Hans in the 1970s as he prepares to travel to Germany for the first time since the 1930s.

 The film was shot on location in Berlin, New York and Stuttgart.

 "Reunion" was nominated for a Golden Palm at the 1989 Cannes Film Festival.
</item 6>

<item 7>
Robin Chalk
Robin Chalk (born 1981) is an English actor, best known for his role as Neil Kellerman in the West End production of "Dirty Dancing" and his work on British
science-fiction film "Moon" directed by Duncan Jones.
</item 7>

<item 8>
Dunston Checks In
Dunston Checks In is a 1996 American family comedy film starring Eric Lloyd, Graham Sack, Jason Alexander, Faye Dunaway, Rupert Everett, Paul Reubens, Glenn
Shadix, and introducing Sam the Orangutan as Dunston.

 It was written by John Hopkins and Bruce Graham and directed by Ken Kwapis.
</item 8>

<item 9>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 9>

<item 10>
Remember the Titans
Remember the Titans is a 2000 American sports film produced by Jerry Bruckheimer and directed by Boaz Yakin.

 The screenplay, written by Gregory Allen Howard, is based on the true story of African-American coach Herman Boone, portrayed by Denzel Washington, and his
attempt to integrate the T. C. Williams High School football team in Alexandria, Virginia, in 1971.

 Will Patton portrays Bill Yoast, Boone's assistant coach.

 Real-life athletes Gerry Bertier and Julius Campbell are portrayed by Ryan Hurst and Wood Harris, respectively.
</item 10>
=== END ITEM id=210dc1128167b5ab ===

=== ITEM id=fcbd89c63765dd02 ===
Gold sentences and facts:
Sentence 0: Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.
  Fact 0.0: Frederick King Keller was born in 1954.
  Fact 0.1: Frederick King Keller was born in Buffalo, New York.
  Fact 0.2: Frederick King Keller is an American director.
  Fact 0.3: Frederick King Keller is an American producer.
  Fact 0.4: Frederick King Keller is an American screenwriter.
  Fact 0.5: Frederick King Keller works in film and television.
Sentence 1: Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
  Fact 1.0: Ian Schneider was born on October 3, 1992.
  Fact 1.1: Ian Schneider is an independent American film director.
  Fact 1.2: Ian Schneider is an American screenwriter.

Retrieved items:
<item 1>
Colby Keller
Colby Keller (born October 18, 1980) is an American visual artist, blogger, and pornographic film actor.

 His career in porn film started in 2004 at Sean Cody and has since expanded to include such studios as Cocksure Men, Randy Blue, Titan Men, Falcon, CockyBoys
and Men.com.

 Keller has amassed a large fan base with both his films and his long-standing blog, "The Big Shoe Diaries".

 He has been nominated for a number of awards and in 2013, World of Wonder awarded him the WOWie Award for Best Blog.

 Keller is nominated for 8 2015 Grabby Awards.

 Keller has also appeared in the short, Zolushka, a queer retelling of Cinderella and the popular series Capitol Hill both by the cult filmmaker Wes Hurley.
</item 1>

<item 2>
Ian Schneider
Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
</item 2>

<item 3>
Shirley Clarke
Shirley Clarke (October 2, 1919 – September 23, 1997) was an American experimental and independent filmmaker.

 She was also a director and editor to many famous works.
</item 3>

<item 4>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 4>

<item 5>
John Biddle (yachting cinematographer)
John Scott Biddle (1925–2008) was a foremost yachting cinematographer and lecturer, establishing a film-making career that spanned more than forty years.

 His films captured not only the technical aspects of sailing but also the human story in events as tranquil as a Nova Scotia cruise and as grand as the
America's Cup Races.
</item 5>

<item 6>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 6>

<item 7>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 7>

<item 8>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 8>

<item 9>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 9>

<item 10>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 10>
=== END ITEM id=fcbd89c63765dd02 ===

=== ITEM id=5f5f07ce3d806c57 ===
Gold sentences and facts:
Sentence 0:  Editors are based in Silver Spring, Maryland and Seoul, Korea.
  Fact 0.0: Editors are based in Silver Spring, Maryland.
  Fact 0.1: Editors are based in Seoul, Korea.
Sentence 1:  The magazine is headquartered in Baltimore, Maryland.
  Fact 1.0: The magazine is headquartered in Baltimore, Maryland.

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
Seventh-day Adventist education
The Seventh-day Adventist educational system is part of the Seventh-day Adventist Church and overseen by the General Conference of Seventh-day Adventist located
in Silver Spring, Maryland.The educational system is the second-largest Christian school-system in the world, after the Roman Catholic system.
</item 2>

<item 3>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 3>

<item 4>
Adventist HealthCare
Adventist HealthCare is a not-for-profit health services organization based in Gaithersburg, Maryland that employs more than 6,200 people and provides
healthcare for more than 400,000 individuals in the community each year.

 The primary service area for Adventist HealthCare is the Washington, D.C. metropolitan area.

 Despite similar names, it is not a part of the California-based Adventist Health, or the Florida-based Adventist Health System.
</item 4>

<item 5>
Diva (magazine)
DIVA is a leading lesbian magazine in the United Kingdom, published monthly.

 It was launched in March 1994 by Millivres Ltd, under the editorship of Frances Williams and is now published by Twin Media Group, which is owned by Silke
Bader and Linda Riley.

 Silke Bader is also the publisher of Curve magazine in the USA and LOTL in Australia, and Linda Riley runs the European Diversity Awards and is the former
publisher of g3 magazine in the UK.
</item 5>

<item 6>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 6>

<item 7>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
</item 7>

<item 8>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 8>

<item 9>
Adventist World
Adventist World is a monthly international magazine of the Seventh-day Adventist Church published by the Review and Herald Publishing Association.

 Editors are based in Silver Spring, Maryland and Seoul, Korea.
</item 9>

<item 10>
Girlfriends (magazine)
Girlfriends was a women's magazine that provided critical coverage of culture, entertainment and world events from a lesbian perspective.

 It was founded by Jacob Anderson-Minshall, Diane Anderson-Minshall and Heather Findlay.

 It also offered relationship, health and travel advice.

 Published monthly from San Francisco since 1993, it was distributed nationwide by Disticor.

 It had the same publisher as lesbian erotica magazine "On Our Backs", but distanced itself from its pornographic counterpart by refusing to carry sexual ads.

 "Girlfriends" magazine ceased publication in 2006.
</item 10>
=== END ITEM id=5f5f07ce3d806c57 ===

=== ITEM id=59f3bebe2b2c1d8b ===
Gold sentences and facts:
Sentence 0: The skogsrå , (Forest Rå), "Skogsfrun" (The Mistress of the Forest), "Skogssnuvan", "Skogsnymfen" (Forest Nymph), "Råndan" (the Rå) or "huldra", was a mythical creature of the forest, or Rå, in Swedish folklore.
  Fact 0.0: The skogsrå is also called Forest Rå.
  Fact 0.1: The skogsrå is also called Skogsfrun (The Mistress of the Forest).
  Fact 0.2: The skogsrå is also called Skogssnuvan.
  Fact 0.3: The skogsrå is also called Skogsnymfen (Forest Nymph).
  Fact 0.4: The skogsrå is also called Råndan (the Rå).
  Fact 0.5: The skogsrå is also called huldra.
  Fact 0.6: The skogsrå was a mythical creature of the forest, also called Rå, in Swedish folklore.
Sentence 1: A hulder is a seductive forest creature found in Scandinavian folklore.
  Fact 1.0: A hulder is a seductive forest creature.
  Fact 1.1: Hulder creatures are found in Scandinavian folklore.

Retrieved items:
<item 1>
Almaz (mythical beast)
The almaz (Chechen: "almaz" or "hun-sag"; Adyghe: "almesti"), roughly translated as various "feral forest-man" or "stone spirit", is a mythical beast that is
considered to be an evil forest creature with magical powers residing in its hair that exists in Chechen and Ingush folk beliefs and Circassian folk beliefs.

 The first "attestation" of it in writing was by a Bavarian captive of the Mongols, but it is present in the national folklore of Chechens and Ingush and of
Circassians.

 The male almaz is said to be hairy and hideous, and have an axe embedded in its chest, while the female is very beautiful with large breasts and golden hair,
and has a "favorite pastime" of dancing naked at night under the moon.

 The almaz is said to have magical powers residing in its hair, but if the hair is removed or even grabbed, it may be rendered helpless.
</item 1>

<item 2>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 2>

<item 3>
Erotikon (1920 film)
Erotikon is a 1920 Swedish romantic comedy film directed by Mauritz Stiller, starring Tora Teje, Karin Molander, Anders de Wahl and Lars Hanson.

 It is based on the 1917 play "A kék róka" by Ferenc Herczeg.

 The story revolves around an entomology professor obsessed with the sexual life of bugs, and his easygoing wife who is courted by two suitors.
</item 3>

<item 4>
Jenny Nyström
Jenny Eugenia Nyström (13 or 15 June 1854 in Kalmar, Sweden – 17 January 1946 in Stockholm) was a painter and illustrator who is mainly known as the person who
created the Swedish image of the "jultomte" on numerous Christmas cards and magazine covers, thus linking the Swedish version of Santa Claus to the gnomes of
Scandinavian folklore.
</item 4>

<item 5>
Vörðr
In Norse mythology, a vǫrðr (pl.

 varðir or verðir — "warden," "watcher" or "caretaker") is a warden spirit, believed to follow from birth to death the soul ("hugr") of every person.

 In Old Swedish, the corresponding word is "varþer"; in modern Swedish "vård", and the belief in them remained strong in Scandinavian folklore up until the last
centuries.

 The English word '"wraith" is derived from "vǫrðr", while "" and "warden" are cognates.
</item 5>

<item 6>
Hulder
A hulder is a seductive forest creature found in Scandinavian folklore.

 (Her name derives from a root meaning "covered" or "secret".)

 In Norwegian folklore, she is known as huldra ("the [archetypal] hulder", though folklore presupposes that there is an entire Hulder race and not just a single
individual).

 She is known as the skogsrå "forest spirit" or Tallemaja "pine tree Mary" in Swedish folklore, and ulda in Sámi folklore.

 Her name suggests that she is originally the same being as the völva divine figure Huld and the German Holda.
</item 6>

<item 7>
Spettekaka
Spettekaka or spettkaka ("spiddekaga" in native Scanian) is a local dessert of the southern parts of Sweden, chiefly in the province of Scania (Skåne) but also
in Halland.

 It is an important part of the Scanian culinary heritage.

 The name means "cake on a spit", and this describes the method of preparation: it is the Swedish variation on the spit cake.
</item 7>

<item 8>
The Saga of Gosta Berling
The Saga of Gosta Berling (Swedish: Gösta Berlings saga ) is a 1924 Swedish romantic drama film directed by Mauritz Stiller and released by AB Svensk
Filmindustri, starring Lars Hanson, Gerda Lundequist and Greta Garbo in her native break-out role on film.

 The film is based on the 1891 debut novel of the same name by the Swedish author and Nobel Prize winner Selma Lagerlöf.

 It is also known as Gosta Berling's Saga, The Story of Gosta Berling and The Atonement of Gosta Berling.
</item 8>

<item 9>
Selkie (disambiguation)
A selkie is a mythological creature found in Faroese, Icelandic, Irish, and Scottish folklore.
</item 9>

<item 10>
Skogsrå
The skogsrå , (Forest Rå), "Skogsfrun" (The Mistress of the Forest), "Skogssnuvan", "Skogsnymfen" (Forest Nymph), "Råndan" (the Rå) or "huldra", was a mythical
creature of the forest, or Rå, in Swedish folklore.
</item 10>
=== END ITEM id=59f3bebe2b2c1d8b ===

=== ITEM id=2b8ddefc4935323e ===
Gold sentences and facts:
Sentence 0: A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.
  Fact 0.0: A Summer in the Cage is a documentary film.
  Fact 0.1: A Summer in the Cage was released in 2007.
  Fact 0.2: A Summer in the Cage is about a man's experiences with bipolar disorder.
Sentence 1: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.
  Fact 1.0: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary.
  Fact 1.1: American Hardcore: The History of American Punk Rock 1980-1986 was directed by Paul Rachman.
  Fact 1.2: American Hardcore: The History of American Punk Rock 1980-1986 was written by Steven Blush.

Retrieved items:
<item 1>
American Pie Presents: Band Camp
American Pie Presents: Band Camp (also known as American Pie: Band Camp) is a 2005 direct-to-DVD sex comedy film.

 It is the first installment in the "American Pie Presents" series and the fourth installment in the "American Pie" franchise.

 It was directed by Steve Rash and written by Brad Riddell.

 The film centers on Matt Stifler, younger brother of Steve Stifler.

 In this film, Matt is sent to the infamous band camp, where he is compelled to change his ways.
</item 1>

<item 2>
Heavy Traffic
Heavy Traffic is a 1973 American adult animated comedy-drama film written and directed by Ralph Bakshi.

 The film, which begins, ends, and occasionally combines with live-action, explores the often surreal fantasies of a young New York cartoonist named Michael
Corleone, using pinball imagery as a metaphor for inner-city life.

 "Heavy Traffic" was Bakshi and producer Steve Krantz's follow-up to the film "Fritz the Cat".

 Though producer Krantz made varied attempts to produce an R-rated film, "Heavy Traffic" was given an X rating by the MPAA.

 The film received positive reviews and is widely considered to be Bakshi's biggest critical success.
</item 2>

<item 3>
Actuality film
The actuality film is a non-fiction film genre that, like the documentary film, uses footage of real events, places, and things, yet unlike the documentary is
not structured into a larger argument, picture of the phenomenon or coherent whole.

 In practice, actuality films preceded the emergence of the documentary.

 During the era of early cinema, actualities—usually lasting no more than a minute or two and usually assembled together into a program by an exhibitor—were
just as popular and prominent as their fictional counterparts.

 The line between "fact" and "fiction" was not so sharply drawn in early cinema as it would become after the documentary came to serve as the predominant
non-fiction filmmaking form.

 An actuality film is not like a newspaper article so much as it is like the still photograph that is published along with the article, with the major
difference being that it moves.

 Apart from the traveling actuality genre, actuality is one film genre that remains strongly related to still photography.
</item 3>

<item 4>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 4>

<item 5>
A Summer in the Cage
A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.

 The film follows the filmmaker's friend Sam and features an interview with mental health scholar Kay Redfield Jamison.

 It was directed by Benjamin Selkow.

 The documentary debuted on the Sundance Channel in 2007.
</item 5>

<item 6>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 6>

<item 7>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 7>

<item 8>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 8>

<item 9>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 9>

<item 10>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 10>
=== END ITEM id=2b8ddefc4935323e ===

=== ITEM id=3dab6fb798628b2c ===
Gold sentences and facts:
Sentence 0:  It has its offices in New Roc City in New Rochelle, New York.
  Fact 0.0: An entity has its offices in New Roc City in New Rochelle, New York.
Sentence 1: New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in Westchester County, New York.
  Fact 1.0: New Roc City is also known as New Rochelle Center.
  Fact 1.1: New Roc City is an entertainment complex.
  Fact 1.2: New Roc City is a retail complex.
  Fact 1.3: New Roc City is a residential complex.
  Fact 1.4: New Roc City is in the Downtown section of the city of New Rochelle.
  Fact 1.5: New Rochelle is in Westchester County, New York.

Retrieved items:
<item 1>
Standard Star Building
The Standard Star Building is a historic commercial structure located in the Downtown section of New Rochelle, Westchester County, New York.

 The building, designed by architect Lawrence J. Barnard, was completed in 1924 and is an architecturally significant example of the Italian Renaissance style
in New Rochelle.

 Although it has been altered, these changes occurred only on the Le Count Place façade and the interior of the building.

 The original Standard Star building exterior remains largely unchanged.

 It is further historically significant for its association with a long-published New Rochelle newspaper covering life in New Rochelle.
</item 1>

<item 2>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 2>

<item 3>
New Jersey Japanese School
The New Jersey Japanese School (ニュージャージー日本人学校 , "Nyūjājī Nihonjin Gakkō" , NJJS) is a Japanese school located in Oakland, New Jersey, United States in the New
York City metropolitan area.

 It is one of the two Japanese day schools operated by the Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi
Kai"), a nonprofit organization which also operates two Japanese weekend school systems in the New York City area.
</item 3>

<item 4>
New Rochelle Walk of Fame
The New Rochelle Walk of Fame was installed in 2011 in Ruby Dee Park at Library Green, located in the downtown section of the city of New Rochelle in
Westchester County, New York.

 The "walk" is a tribute to some of New Rochelle's most notable residents from throughout its 325-year history.

 It was created and funded by former resident Roderick Kennedy, Jr., working in partnership with the City of New Rochelle and the New Rochelle Business
Improvement District.
</item 4>

<item 5>
New Rochelle Harbor (Long Island Sound)
New Rochelle Harbor is the name of a harbor located along Long Island Sound in the city of New Rochelle in Westchester County, New York.

 The Davenport Neck peninsula off the mainland divides New Rochelle's waterfront into two bays; the westerly referred to as New Rochelle Harbor and the easterly
as Echo Bay.
</item 5>

<item 6>
Princeton Community Japanese Language School
The Princeton Community Japanese Language School (PCJLS; プリンストン日本語学校 "Purinsuton Nihongo Gakkō") is a Japanese weekend school in the Princeton, New Jersey area.

 It holds weekend Japanese classes for Japanese citizen children abroad to the standard of the Ministry of Education, Culture, Sports, Science and Technology
(MEXT), and it also has classes for people with Japanese as a second language.
</item 6>

<item 7>
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 7>

<item 8>
New Roc City
New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in
Westchester County, New York.

 It is located at 33 LeCount Place, between Main Street North and Main Street South.

 The center was built on the site of the former New Rochelle Mall which closed in 1992.
</item 8>

<item 9>
Japanese Weekend School of New York
The Japanese Weekend School of New York (JWSNY; ニューヨーク "Nyūyōku Hoshū Jugyō Kō") is a Japanese supplementary school in the New York City metropolitan area.

 It has its offices in New Roc City in New Rochelle, New York.

 The Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi Kai") manages the school system, and the JWSNY is one of
its two weekend school systems.

 The JEI also operates two Japanese day schools in the New York area.
</item 9>

<item 10>
Montefiore New Rochelle Hospital
Montefiore New Rochelle Hospital (formerly Sound Shore Medical Center) is a community-based, teaching hospital located at 16 Guion Place in the city of New
Rochelle, in Westchester County, New York, and affiliated with the Albert Einstein College of Medicine.

 The hospital opened in 1892 as New Rochelle Hospital.

 On November 6, 2013, Sound Shore was acquired by the Bronx-based Montefiore Medical Center, the University Hospital for the Albert Einstein College of
Medicine, was renamed Montefiore New Rochelle Hospital, and became part of the Montefiore Health System.
</item 10>
=== END ITEM id=3dab6fb798628b2c ===

=== ITEM id=cd630aff3f5e0528 ===
Gold sentences and facts:
Sentence 0: Black (Bengali: ব্ল্যাক ) is a Bangladeshi rock band, formed in Dhaka, Bangladesh.
  Fact 0.0: Black is a Bangladeshi rock band.
  Fact 0.1: Black was formed in Dhaka, Bangladesh.
  Fact 0.2: The Bengali name of Black is ব্ল্যাক.
Sentence 1: New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.
  Fact 1.0: New Found Glory is an American rock band.
  Fact 1.1: New Found Glory was formerly known as A New Found Glory.
  Fact 1.2: New Found Glory is from Coral Springs, Florida.
  Fact 1.3: New Found Glory was formed in 1997.

Retrieved items:
<item 1>
Nothing Gold Can Stay (album)
Nothing Gold Can Stay is the debut studio album by American rock band New Found Glory, released on October 19, 1999 through independent record label Eulogy
Recordings.

 At the time, the band was then named "A New Found Glory", but later dropped the indefinite article "A" due to some fans struggling to find their records in
stores.

 The original pressings of the album contained samplings from several films including "The Outsiders" (1983), "Weird Science" (1985), and "That Thing You Do!

" (1996), as well as Robert Frost's poem "Nothing Gold Can Stay", after which the album is named.
</item 1>

<item 2>
A Very New Found Glory Christmas
A Very New Found Glory Christmas is the third EP by American rock band New Found Glory.

 Self-produced by the band, it is a special edition Christmas-themed acoustic recording featuring two original songs and three covers.

 The EP was limited to 2,000 pressings and was released via cassette tape on December 1, 2012.

 1,000 white copies were sold on tour, and 1,000 red copies were sold in their online store.

 The band released "Nothing for Christmas" via Rdio and Spotify on December 20, after debuting songs off the EP during the tenth anniversary tour of their
"Sticks and Stones" album.

 The cover art is a parody of the 1990 blockbuster movie "Home Alone".
</item 2>

<item 3>
It's All About the Girls
It's All About the Girls is the debut EP by A New Found Glory (now New Found Glory) released on December 20, 1997 by Fiddler Records.

 It is the only release with their original drummer Joe Marino.

 A re-release, featuring revised cover artwork, was issued by Fiddler in June, 2000.
</item 3>

<item 4>
From the Screen to Your Stereo
From the Screen to Your Stereo is an EP by A New Found Glory (now New Found Glory) released on 28 March 2000 by Drive-Thru Records.

 All the tracks are covers of songs from motion picture soundtracks.

 The album was pressed on 10" vinyl and released in three different colors: light marble blue, marble red, and white.

 The vinyl pressing featured a bonus track.

 In 2002, the album was released on a non-US split with the Japanese band Nicotine entitled Movie Addiction, where on this split, Nicotine also covered songs
from movies.

 In 2007 the band released a follow up, this time a full-length album titled "From the Screen to Your Stereo Part II".
</item 4>

<item 5>
Chad Gilbert
Chad Everett Gilbert (born March 9, 1981) is an American musician and record producer.

 He is a founding member of the rock band New Found Glory, for whom he plays lead guitar, composes music and provides backing vocals.

 He was also the lead vocalist for New Found Glory's now-defunct side-project, International Superheroes of Hardcore.

 Additionally, Gilbert was the vocalist for the hardcore punk band Shai Hulud between 1995 and 1998, and he rejoined the band in 2012.
</item 5>

<item 6>
Glassjaw
Glassjaw is an American post-hardcore band from Long Island, New York.

 The band is fronted by vocalist Daryl Palumbo and guitarist Justin Beck.

 The band has been influential in the progression of the underground music scene in the eastern United States and United Kingdom and on the post-hardcore genre,
and are known for their intense live shows and frequent line-up changes.
</item 6>

<item 7>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 7>

<item 8>
Radiosurgery (album)
Radiosurgery is the seventh studio album by American rock band New Found Glory.

 It was first released on September 30, 2011 in Australia, before its wider release on October 4 through independent label Epitaph Records.

 It is the band's final studio album to feature founding guitarist Steve Klein.

 To follow up predecessor "Not Without a Fight" (2009), the band began writing new material during their stint on the 2010 Honda Civic Tour.

 After self-producing a set of demos and contacting long-term record producer Neal Avron, the band went on to record the album in Avron's home recording studio
over a period of three months in 2011.

 The quintet set out to write an album that paid homage to classic punk rock records that first inspired them to form a band during the 1990s.

 Listening extensively to the likes of early Green Day and Ramones, New Found Glory strove to create a sound that could "bridge the gap" between old and new
generations of the genre.
</item 8>

<item 9>
Black (Bangladeshi band)
Black (Bengali: ব্ল্যাক ) is a Bangladeshi rock band, formed in Dhaka, Bangladesh.
</item 9>

<item 10>
New Found Glory
New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.

 The band currently consists of Jordan Pundik (lead vocals), Ian Grushka (bass guitar), Chad Gilbert (lead guitar, backing vocals), and Cyrus Bolooki (drums).

 Longtime rhythm guitarist and lyricist Steve Klein departed from the band in late 2013, following "personal differences."

 During their lengthy recording career, the band have released nine studio albums, one live album, two EPs, and three cover albums.
</item 10>
=== END ITEM id=cd630aff3f5e0528 ===

=== ITEM id=c432297aebd171f4 ===
Gold sentences and facts:
Sentence 0:  He creates bespoke millinery under the J Smith Esquire brand for a portfolio of private clients that includes Angelina Jolie, Brad Pitt and Amal Clooney.
  Fact 0.0: This person creates bespoke millinery under the J Smith Esquire brand.
  Fact 0.1: This person creates bespoke millinery for a portfolio of private clients.
  Fact 0.2: The portfolio of private clients includes Angelina Jolie.
  Fact 0.3: The portfolio of private clients includes Brad Pitt.
  Fact 0.4: The portfolio of private clients includes Amal Clooney.
Sentence 1: Amal Clooney ("née" Alamuddin, Arabic: أمل علم الدين‎ ‎ ; born 3 February 1978) is a barrister at Doughty Street Chambers, specialising in international law and human rights.
  Fact 1.0: Amal Clooney is also known by the name Alamuddin (née Alamuddin).
  Fact 1.1: Amal Clooney's Arabic name is أمل علم الدين.
  Fact 1.2: Amal Clooney was born on 3 February 1978.
  Fact 1.3: Amal Clooney is a barrister at Doughty Street Chambers.
  Fact 1.4: Amal Clooney specialises in international law.
  Fact 1.5: Amal Clooney specialises in human rights.

Retrieved items:
<item 1>
Justin Smith (milliner)
Justin Smith (born 1978) is a leading British milliner based in London.

 He creates bespoke millinery under the J Smith Esquire brand for a portfolio of private clients that includes Angelina Jolie, Brad Pitt and Amal Clooney.

 Smith's hats have been exhibited around the world, and have been acquired by such museums as the Fashion Institute of Technology in New York and the Victoria &
Albert Museum in London for Hats: An Anthology curated by Stephen Jones.

 His work is in continuous demand by stylists and photographers for the quality style press.

 He has been visiting lecturer at London's Royal College of Art and has conducted seminars and teaching workshops for the British Council.
</item 1>

<item 2>
Shahram Taghavi
Shahram Taghavi is a barrister practising in the United Kingdom.

 He is a specialist in Judicial Review, Human Rights, Immigration and EU law.

 He was called to the England & Wales bar in 1994.

 He is a partner at Article 1, a law firm specialising in immigration, EU freedoms of movement and human rights law.

 He practised as an independent barrister at Doughty Street Chambers specialising in immigration and public law.

 He was head of both the Immigration and Public Law departments at Simons Muirhead & Burton solicitors.

 He was a Senior Barrister at Bates Wells & Braithwaite LLP as a member of their Public & Regulatory and Immigration departments, Deputy Head of Immigration at
Lewis Silkin LLP, Head of Immigration at Charles Russell LLP (now Charles Russell Speechly LLP) and National Head of Immigration at Grant Thornton LLP.
</item 2>

<item 3>
Doughty Street Chambers
Doughty Street Chambers is a British set of barristers' chambers situated in Bristol, Manchester and London's Doughty Street, undertaking criminal justice,
public law, immigration, employment, human rights and civil liberties work.
</item 3>

<item 4>
Pupil master
A pupil master ('pupilmaster' or 'pupil-master') or in the case of a female barrister, 'pupil mistress' etc., is the former name given to an experienced
barrister who a pupil shadows during their pupillage.

 The term pupil master or mistress has now been replaced by the term 'pupil supervisor'.

 Barristers are called to the Bar via one of the four Inns of Court upon successful completion of the BPTC and having undertaken a required number of
"qualifying sessions" in their chosen Inn of Court.

 In most cases, the newly called barrister is then required to undertake training for a period of at least a year before the barrister can start their own
private practice.

 This training period is known as pupillage, usually split into two periods of six months known as "sixes".

 The first "six" is a non-practising six, during which the pupil will shadow their pupil master; the second is usually a practising "six", when the pupil, with
their pupil master's permission, can undertake the supply of legal services and exercise rights of audience in court.
</item 4>

<item 5>
Amal Clooney
Amal Clooney ("née" Alamuddin, Arabic: أمل علم الدين‎ ‎ ; born 3 February 1978) is a barrister at Doughty Street Chambers, specialising in international law and
human rights.

 Her clients include Julian Assange, the founder of WikiLeaks, in his fight against extradition.

 She has also represented the former prime minister of Ukraine, Yulia Tymoshenko, and Egyptian-Canadian journalist Mohamed Fahmy.

 She is married to the American actor George Clooney.
</item 5>

<item 6>
Barbados Bar Association
The Barbados Bar Association is a voluntary association of attorneys in Barbados who practise at the independent bar as barristers and Queen's Counsel.

 It was created by the Barbados Bar Association Act of 1940.
</item 6>

<item 7>
Pupil master
Occasionally, a pupil barrister may undertake a third "six", extending the training period a further six months.

 At the end of pupillage, to continue practising the law, a barrister may attempt to become a tenant in a set of barristers' chambers, or find a position as an
employed barrister.
</item 7>

<item 8>
Mónica Feria Tinta
Mónica Feria Tinta is a leading public international lawyer.

 She practises as a Barrister at 20 Essex Street Chambers.

 In 2000 she became the first Peruvian-born lawyer to receive the Diploma of the Hague Academy of International Law in history.

 Her litigation work led to the first international human rights court decision ordering the prosecution of a former Head of State for crimes under
international law.

 She was awarded the Inge Genefke International Award for her work as an international lawyer in 2006 and was co-recipient of the Gruber Justice Prize 2007, for
her contributions advancing the cause of justice as delivered through the legal system.

 She is a member of the Bar of England and Wales, and the American Society of International Law.
</item 8>

<item 9>
Doughty Street
Doughty Street is a broad tree lined street in the Holborn district of the London Borough of Camden.

 The southern part is a continuation of the short John Street, which comes off Theobald's Road.

 The northern part crosses Guilford Street and ends at Mecklenburgh Square.

 The street is named after a landlord of the area at the time it was built, Henry Doughty.
</item 9>

<item 10>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 10>
=== END ITEM id=c432297aebd171f4 ===

=== ITEM id=49688dcb5467c09a ===
Gold sentences and facts:
Sentence 0: "Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series overall.
  Fact 0.0: Moho House is the twenty-first episode of the twenty-eighth season of the animated television series The Simpsons.
  Fact 0.1: Moho House is the 617th episode of The Simpsons overall.
  Fact 0.2: The Simpsons is an animated television series.
Sentence 1:  On May 4, 2015, Fox announced that "The Simpsons" had been renewed for season 28.
  Fact 1.0: Fox announced on May 4, 2015.
  Fact 1.1: Fox announced that The Simpsons had been renewed for season 28.

Retrieved items:
<item 1>
Moe Szyslak
Morris "Moe" Szyslak is a fictional character from the American animated television series "The Simpsons".

 He is voiced by Hank Azaria and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 Moe is the proprietor and bartender of Moe's Tavern, a Springfield bar frequented by Homer Simpson, Barney Gumble, Carl Carlson, Lenny Leonard, Sam, Larry, and
others.
</item 1>

<item 2>
Rhoda
Rhoda was an American sitcom starring Valerie Harper which aired a total of 109 half-hour episodes and one hour-long episode over five seasons from September 9,
1974 to December 9, 1978.

 The show was a spin-off of "The Mary Tyler Moore Show", in which Harper between the years 1970 and 1974 had played the role of Rhoda Morgenstern, a spunky,
weight-conscious, flamboyantly fashioned Jewish neighbor and native New Yorker in the role of Mary Richards' best friend.

 After four seasons, Rhoda left Minneapolis and returned to her original hometown of New York City.

 The series was the winner of two Golden Globes and two Emmy Awards.
</item 2>

<item 3>
White People Renovating Houses
"White People Renovating Houses" is the first episode in the twenty-first season of the American animated television series "South Park".

 It is the 278th episode of the series overall, and first aired on Comedy Central in the United States on September 13, 2017.
</item 3>

<item 4>
SpongeBob SquarePants (season 3)
The third season of the American animated television series "SpongeBob SquarePants", created by Stephen Hillenburg, aired on Nickelodeon from October 5, 2001 to
October 11, 2004, and consists of 20 episodes.

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, who also acted as the showrunner.

 Hillenburg halted production on the show to work on the 2004 film adaptation of the series, "The SpongeBob SquarePants Movie".

 After production on the film, Hillenburg resigned from the show as its showrunner, and appointed staff writer Paul Tibbitt to overtake the position.

 Season 3 was originally set to end the series after the release of the film, but the success prevented the series from ending, leading to a fourth season.
</item 4>

<item 5>
Mona Leaves-a
"Mona Leaves-a" is the nineteenth episode of "The Simpsons"<nowiki>'</nowiki> nineteenth season.

 It originally aired on the Fox network in the United States on May 11, 2008.

 The episode features the death of Homer's mother, Mona Simpson.

 Homer is reunited with his mother, Mona, but is not willing to forgive her for all the times she left him as a child.

 When she dies, a guilt-ridden Homer attempts to make it up to her by fulfilling her final wishes.

 It was written by Joel H. Cohen and directed by Mike B. Anderson and Ralph Sosa.

 Glenn Close makes her third appearance as Mona Simpson, and Lance Armstrong has a cameo as himself.
</item 5>

<item 6>
Flaming Moe's
"Flaming Moe's" is the tenth episode of "The Simpsons"' third season.

 It first aired on the Fox network in the United States on November 21, 1991.

 In the episode, Homer tells Moe Szyslak of a secret alcoholic cocktail that includes cough medicine and fire that he calls "Flaming Homer".

 Moe steals the recipe from Homer, renames the drink the "Flaming Moe" and begins selling it at his tavern.

 The drink is a success and boosts business and patronage, but Homer is angry at Moe for his betrayal, and seeks revenge.
</item 6>

<item 7>
Homer to the Max
"Homer to the Max" is the thirteenth episode of "The Simpsons"<nowiki>'</nowiki> tenth season.

 It originally aired on the Fox network in the United States on February 7, 1999.

 In the episode, Homer discovers that a new television show, "Police Cops", has a hero also named Homer Simpson.

 He is delighted with the positive attention he receives because of his name, but when the television character is rewritten from a hero to a bumbling idiot, he
is mocked and taunted, so he changes his name to "Max Power" to rid himself of the negative attention.

 Max gains new friends, and is forced into a protest to prevent a forest from being knocked down.

 In the end, he changes his name back to Homer Simpson.
</item 7>

<item 8>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 8>

<item 9>
The Simpsons Spin-Off Showcase
"The Simpsons Spin-Off Showcase" is the twenty-fourth episode of the eighth season of "The Simpsons".

 It originally aired on the Fox network in the United States on May 11, 1997.

 The episode centers on fictional pilot episodes of non-existent television series derived from "The Simpsons", and is a parody of the tendency of networks to
spin off characters from a hit series.

 As such it includes references to many different TV series.

 The first fictional spin-off is "Chief Wiggum P.I.", a cop-drama featuring Chief Wiggum and Seymour Skinner.

 The second is "The Love-matic Grampa", a sitcom featuring Moe Szyslak who receives dating advice from Abraham Simpson, whose ghost is possessing a love testing
machine.

 The final segment is "The Simpson Family Smile-Time Variety Hour", a variety show featuring the Simpson family except for Lisa, who has been replaced.
</item 9>

<item 10>
South Park (season 10)
The tenth season of "South Park", an American animated television series created by Trey Parker and Matt Stone, began airing on March 22, 2006.

 The tenth season concluded after 14 episodes on November 15, 2006.

 This is the first season in which Kenny does not die and the last season featuring Isaac Hayes (the voice of Chef) as Hayes quit the show following the
backlash behind season nine's "Trapped in the Closet" episode.

 This season also had a minor controversy when the Halloween episode "Hell on Earth 2006" depicted "The Crocodile Hunter's" Steve Irwin with a stingray lodged
in his chest getting thrown out of Satan's Halloween party for not being in costume.

 Episode 2 in this season is the last one with the Braniff Airlines logo.

 All the episodes in this season were written and directed by Trey Parker.
</item 10>
=== END ITEM id=49688dcb5467c09a ===

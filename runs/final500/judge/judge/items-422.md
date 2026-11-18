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

=== ITEM id=c05dd433beb166e1 ===
Gold sentences and facts:
Sentence 0: The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.
  Fact 0.0: The University of Arizona is a public research university.
  Fact 0.1: The University of Arizona is in Tucson, Arizona, United States.
  Fact 0.2: The University of Arizona is also referred to as U of A.
  Fact 0.3: The University of Arizona is also referred to as UA.
  Fact 0.4: The University of Arizona is also referred to as Arizona.
Sentence 1: The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.
  Fact 1.0: The Presbyterian University College, Ghana is a partially private and public university.
  Fact 1.1: The Presbyterian University College, Ghana is located at Abetifi-Kwahu.
  Fact 1.2: Abetifi-Kwahu is in the Eastern Region of Ghana.
Sentence 2:  It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.
  Fact 2.0: The Presbyterian University College, Ghana is one of the new universities in Ghana.
  Fact 2.1: The Presbyterian University College, Ghana was granted accreditation by the National Accreditation Board.

Retrieved items:
<item 1>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
</item 1>

<item 2>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 2>

<item 3>
Sanders–Brown Center on Aging
The Sanders–Brown Center on Aging at the University of Kentucky in Lexington, Kentucky, is responsible for research, education and service programs in aging.

 The program was founded in 1963 with the creation of the Council on Aging.

 In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.

 The center was named for the Browns and his then-business partner, (Col.) Harlan Sanders.

 The four-story brick and concrete structure fronts South Limestone at the edge of the medical campus.

 The complex features 32 research laboratories that employs more than 150.
</item 3>

<item 4>
Lydia Patterson Institute
Lydia Patterson Institute is a Methodist Christian college-preparatory school located in El Paso, Texas, United States.

 Founded in 1913, it offers programs for Spanish-speaking children, primarily from Juarez, to attend high school in the United States and attend an American
undergraduate university.

 All high school classes are taught in English, and the school is accredited by the Southern Association of Colleges and Schools.
</item 4>

<item 5>
Evangelical Presbyterian University College
The Evangelical Presbyterian University College was established in 2008 by the Evangelical Presbyterian Church, Ghana.
</item 5>

<item 6>
Syracuse Triad
The Syracuse Triad is the name given to the three women's sororities founded at Syracuse University.

 Alpha Phi was founded first in 1872 by 10 of the original 20 women admitted into Syracuse University.

 Gamma Phi Beta came along two years later in 1874 and with it came the term "sorority," which was coined at the time of its founding.

 (Prior to that, women's Greek-letter organizations used the term "women's fraternity," since no more appropriate term existed.)

 Alpha Gamma Delta completed the triad in 1904.
</item 6>

<item 7>
Albanian University in Berat
The University of Berat is a private institution, independent and secular Higher Education, which operates the state fee.

 Licensed by the Council of Ministers, no. 697 dated 18.06.2009, he opened the doors in the academic year 2009- 2010, offering programs in two faculties:
Faculty of Social Sciences Programs in Science Teaching Legal, Political Science and Psychology General Administration and the Faculty of Economics with
Learning programs in Banking and Finance-Business Administration.

 During the first academic year, about 1,000 students attended studies.

 A year later, this new institution offering undergraduate degrees in study programs: Preschool Teacher Education, Primary Education Teacher, English Language,
Nursing, Engineering Computer and Information Technology, thus finalizing the 2010-2011 academic year with the opening of two new faculties: Faculty of Medical
Sciences and the Faculty of Architecture and Inxhinierive.

 Academic Year 2010-2011 finds the University Berat with about 2600 students in the first cycle of study.

 Since the first year of its inception, the institution pointed at the center of academic life, the promotion of intellectual product and melting learning and
research.
</item 7>

<item 8>
The text profiles diverse educational and related institutions worldwide. Highlights include Berat University (Albania), founded in 2009 and expanding to four
faculties; Rotman (Toronto), established in 1950; Auburn, Arizona, Debrecen, Copenhagen, Dalhousie and other research universities; specialist schools such as
Falmouth and Aalborghus; and private or state colleges across Ghana, Malaysia, Norway, Pakistan and elsewhere. It also covers Syracuse sororities, a
</item 8>

<item 9>
University of Arizona
The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.

 Founded in 1885, the UA was the first university in the Arizona Territory.

 The university operates two medical schools (University of Arizona College of Medicine - Tucson and the University of Arizona College of Medicine - Phoenix)
and is affiliated with the region's only academic medical centers (Banner - University Medical Center Tucson and Banner - University Medical Center Phoenix).

 The university is also home to the James E. Rogers College of Law and numerous other nationally ranked graduate and professional schools.

 During the 2016–2017 academic year, there was a total enrollment of 43,625 students, including 34,072 undergraduates The University of Arizona is governed by
the Arizona Board of Regents.

 The University of Arizona is one of the elected members of the Association of American Universities (an organization of North America's premier research
institutions) and is the only representative from the state of Arizona to this group.
</item 9>

<item 10>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 10>
=== END ITEM id=c05dd433beb166e1 ===

=== ITEM id=0d2cd5792e1e59b6 ===
Gold sentences and facts:
Sentence 0: Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
  Fact 0.0: Mauritz Stiller was born on 17 July 1883.
  Fact 0.1: Mauritz Stiller died on 18 November 1928.
  Fact 0.2: Mauritz Stiller was a Finnish-Swedish film director.
  Fact 0.3: Mauritz Stiller was best known for discovering Greta Garbo.
  Fact 0.4: Mauritz Stiller brought Greta Garbo to America.
Sentence 1: Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.
  Fact 1.0: Johannes Grenzfurthner was born in 1975 in Vienna.
  Fact 1.1: Johannes Grenzfurthner is an award-winning Austrian artist.
  Fact 1.2: Johannes Grenzfurthner is an Austrian filmmaker.
  Fact 1.3: Johannes Grenzfurthner is an Austrian writer.
  Fact 1.4: Johannes Grenzfurthner is an Austrian actor.
  Fact 1.5: Johannes Grenzfurthner is an Austrian curator.
  Fact 1.6: Johannes Grenzfurthner is an Austrian theatre director.
  Fact 1.7: Johannes Grenzfurthner is an Austrian lecturer.

Retrieved items:
<item 1>
Graph query row:
person: Johannes Grenzfurthner
nationality: Austrian
country: None
</item 1>

<item 2>
Mauritz Stiller
Mauritz Stiller (17 July 1883 – 18 November 1928) was a Finnish-Swedish film director, best known for discovering Greta Garbo and bringing her to America.
</item 2>

<item 3>
Devid Striesow
Devid Striesow (born 1 October 1973 in Bergen auf Rügen) is a German actor.

 He starred as "Sturmbannführer Herzog" (Bernhard Krüger) in Stefan Ruzowitzky's 2007 film "The Counterfeiters", which was awarded the Academy Award for Best
Foreign Language Film for that year.
</item 3>

<item 4>
John Steiner
John Steiner (born 7 January 1941 in Chester, Cheshire) is an English actor.

 Tall, thin and gaunt, Steiner attended the Royal Academy of Dramatic Arts and worked for a few years at the BBC.

 Steiner featured in a lead role in a television production of "Design for Living" by Noël Coward.

 Later he found further work primarily in films including "Marat/Sade" (1967), and the original "Bedazzled" (1967) with Peter Cook and Dudley Moore.

 In 1969, Steiner was hired to play a part in the spaghetti western "Tepepa", and also appeared opposite Franco Nero in "White Fang", directed by Lucio Fulci.
</item 4>

<item 5>
Richard Genée
Franz Friedrich Richard Genée (February 7, 1823 – June 15, 1895) was a Prussian born Austrian librettist, playwright, and composer.

 He is most famous for the libretto of "Die Fledermaus", Johann Strauss II's most famous operetta.

 He co-wrote the libretto without having met top-billed librettist Karl Haffner, who constructed the new story based on a play by Henri Meilhac and Ludovic
Halévy, which was considered too shocking to perform outside Paris.

 Genée, however, wrote the operetta's actual text and drew nothing from Haffner beyond the names of the characters.
</item 5>

<item 6>
Willi Forst
Willi Forst, born Wilhelm Anton Frohs (7 April 1903 – 11 August 1980) was an Austrian actor, screenwriter, film director, film producer and singer.

 As a debonair actor he was a darling of the German-speaking film audiences, as a director, one of the most significant makers of the Viennese period musical
melodramas and comedies of the 1930s known as "Wiener Filme".

 From the mid-1930s he also recorded many records, largely of sentimental Viennese songs, for the Odeon Records label owned by Carl Lindström AG.
</item 6>

<item 7>
Stefan Walz
Stefan Walz (born 1963) is a Swiss actor.

 His best known role is as Sigmund Jähn in the film "Good Bye, Lenin!

".
</item 7>

<item 8>
Johannes Grenzfurthner
Johannes Grenzfurthner (] ; born 1975 in Vienna) is an award-winning Austrian artist, filmmaker, writer, actor, curator, theatre director and lecturer.

 He is known as the founder, conceptualist and artistic director of monochrom, an international art and theory group.

 Most of his artworks are labelled "monochrom".
</item 8>

<item 9>
Traceroute (film)
Traceroute is a 2016 Austrian/American documentary film directed by Johannes Grenzfurthner.

 The autobiographical documentary and road movie deals with the history, politics and impact of nerd culture.

 Grenzfurthner calls his film a "personal journey into the uncharted depths of nerd culture, a realm full of dangers, creatures and more or less precarious
working conditions", an attempt to "chase the ghosts of nerddom's past, present and future."

 The film was co-produced by art group monochrom and Reisenbauer Film.

 It features music by Kasson Crooker, Hans Nieswandt, and many others.
</item 9>

<item 10>
Graph query row:
person: Mauritz Stiller
nationality: Finnish-Swedish
country: None
</item 10>
=== END ITEM id=0d2cd5792e1e59b6 ===

=== ITEM id=e114bebf6b920b2f ===
Gold sentences and facts:
Sentence 0: US Airways Flight 1549 was an Airbus A320-214 which, three minutes after takeoff from New York City's LaGuardia Airport on January 15, 2009, struck a flock of Canada geese just northeast of the George Washington Bridge and consequently lost all engine power.
  Fact 0.0: US Airways Flight 1549 was an Airbus A320-214.
  Fact 0.1: US Airways Flight 1549 took off from New York City's LaGuardia Airport.
  Fact 0.2: US Airways Flight 1549 took off on January 15, 2009.
  Fact 0.3: US Airways Flight 1549 struck a flock of Canada geese three minutes after takeoff.
  Fact 0.4: The bird strike happened just northeast of the George Washington Bridge.
  Fact 0.5: US Airways Flight 1549 consequently lost all engine power.
Sentence 1:  Unable to reach any airport, pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River off Midtown Manhattan.
  Fact 1.0: Pilots could not reach any airport.
  Fact 1.1: Pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River.
  Fact 1.2: The Hudson River ditching was off Midtown Manhattan.
Sentence 2: Chesley Burnett "Sully" Sullenberger III (born 1951) is an American retired airline captain celebrated for the January 15, 2009 water landing of US Airways Flight 1549 in the Hudson River off Manhattan after the plane was disabled by striking a flock of Canada geese immediately after takeoff; all 155 people aboard survived.
  Fact 2.0: Chesley Burnett "Sully" Sullenberger III was born in 1951.
  Fact 2.1: Chesley Sullenberger is an American retired airline captain.
  Fact 2.2: Chesley Sullenberger is celebrated for the January 15, 2009 water landing of US Airways Flight 1549 in the Hudson River off Manhattan.
  Fact 2.3: The plane was disabled by striking a flock of Canada geese immediately after takeoff.
  Fact 2.4: All 155 people aboard survived.

Retrieved items:
<item 1>
1965 Carmel mid-air collision
The 1965 Carmel mid-air collision occurred on December 4, 1965, when Eastern Air Lines Flight 853 (N6218C), a Lockheed Super Constellation en route from Boston
Logan International Airport to Newark International Airport, collided in mid-air with Trans World Airlines Flight 42 (N748TW), a Boeing 707-131B en route from
San Francisco International Airport to John F. Kennedy International Airport, over Carmel, New York, United States.
</item 1>

<item 2>
Chesley Sullenberger
Chesley Burnett "Sully" Sullenberger III (born 1951) is an American retired airline captain celebrated for the January 15, 2009 water landing of US Airways
Flight 1549 in the Hudson River off Manhattan after the plane was disabled by striking a flock of Canada geese immediately after takeoff; all 155 people aboard
survived.

 Sullenberger is an international speaker on airline safety and has helped develop new protocols for airline safety.

 He served as the co-chairman, along with First Officer Jeffrey Skiles, of the EAA's Young Eagles youth introduction-to-aviation program from 2009 to 2013.

 He retired from US Airways after 30 years as a commercial pilot on March 3, 2010.

 In May of the following year, Sullenberger was hired by CBS News as an Aviation and Safety Expert.
</item 2>

<item 3>
ALM Flight 980
ALM Antillean Airlines Flight 980 was a flight scheduled to fly from John F. Kennedy International Airport in New York City to Princess Juliana International
Airport in St. Maarten, Netherlands Antilles, on 2 May 1970.

 After several unsuccessful landing attempts, the aircraft's fuel was exhausted and it made a forced water landing (ditching) in the Caribbean Sea 48 km (30 mi)
off St. Croix, with 23 fatalities and 40 survivors.

 The accident is one of a small number of intentional water ditchings of jet airliners.
</item 3>

<item 4>
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
</item 4>

<item 5>
Water landing
A water landing is, in the broadest sense, a landing on a body of water.

 Some aircraft such as floatplanes land on water as a matter of course.
</item 5>

<item 6>
US Airways Flight 1549
US Airways Flight 1549 was an Airbus A320-214 which, three minutes after takeoff from New York City's LaGuardia Airport on January 15, 2009, struck a flock of
Canada geese just northeast of the George Washington Bridge and consequently lost all engine power.

 Unable to reach any airport, pilots Chesley Sullenberger and Jeffrey Skiles glided the plane to a ditching in the Hudson River off Midtown Manhattan.

 All 155 people aboard were rescued by nearby boats and there were few serious injuries.
</item 6>

<item 7>
Atlantic Southeast Airlines Flight 2311
Atlantic Southeast Airlines Flight 2311 was a regularly scheduled commuter flight from Hartsfield–Jackson Atlanta International Airport to Glynco Jetport (since
renamed Brunswick Golden Isles Airport) in Brunswick, Georgia on April 5, 1991.

 The flight, operated using a twin-turboprop Embraer EMB 120 Brasilia, crashed just north of Brunswick while approaching the airport for landing.

 All 23 people aboard the plane were killed, including passengers Sonny Carter and John Tower.

 Four years later, another Embraer Brasilia of ASA crashed in the Georgia countryside in similar circumstances, with nine fatalities.
</item 7>

<item 8>
Bird strike
A bird strike—sometimes called birdstrike, bird ingestion (for an engine), bird hit, or bird aircraft strike hazard (BASH)—is a collision between an airborne
animal (usually a bird or bat) and a manmade vehicle, especially an aircraft.

 The term is also used for bird deaths resulting from collisions with structures such as power lines, towers and wind turbines (see Bird-skyscraper collisions
and Towerkill).
</item 8>

<item 9>
United Airlines Flight 736
United Airlines Flight 736 was a daily U.S. transcontinental passenger flight operated by United Airlines that crashed on April21, 1958, following a mid-air
collision.

 The aircraft assigned to Flight 736, a Douglas DC-7 airliner carrying 47 persons, was flying at cruise altitude above Clark County, Nevada, en route to a
stopover at Denver, Colorado, when it was struck by a United States Air Force fighter jet crewed by two pilots.

 The collision occurred at 8:30 a.m. in clear weather within a major commercial airway; both aircraft fell out of control from 21000 ft and crashed into
unpopulated desert terrain southwest of Las Vegas, Nevada.
</item 9>

<item 10>
Graph query row:
person: Chesley Sullenberger
aliases: None
connected_entity: Jeffrey Skiles
connection: ['FROM_CHUNK', 'FROM_CHUNK']
</item 10>
=== END ITEM id=e114bebf6b920b2f ===

=== ITEM id=0766e8e2f793071a ===
Gold sentences and facts:
Sentence 0: Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.
  Fact 0.0: Kenneth William Kwapis, also known as Ken Kwapis, was born on August 17, 1957.
  Fact 0.1: Ken Kwapis is an American film director.
  Fact 0.2: Ken Kwapis is an American television director.
  Fact 0.3: Ken Kwapis is an American screenwriter.
Sentence 1:  He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
  Fact 1.0: He specialized in the single-camera sitcom in the 1990s.
  Fact 1.1: He specialized in the single-camera sitcom in the 2000s.
  Fact 1.2: He has directed feature films.
  Fact 1.3: He directed Sesame Street Presents Follow That Bird, released in 1985.
  Fact 1.4: He directed The Sisterhood of the Traveling Pants, released in 2005.
  Fact 1.5: He directed He's Just Not That into You, released in 2009.
Sentence 2:  As a filmmaker, he has directed films such as Valle de Lágrimas (2006) and Peace Grows (2004).
  Fact 2.0: As a filmmaker, he has directed Valle de Lágrimas, released in 2006.
  Fact 2.1: As a filmmaker, he has directed Peace Grows, released in 2004.

Retrieved items:
<item 1>
Leonard Schrader
Leonard Schrader (November 30, 1943 – November 2, 2006) was an American screenwriter and director, most notable for his ability to write Japanese language films
and for his many collaborations with his brother, Paul Schrader.

 He earned an Academy Award Nomination for the screenplay he wrote for the film "Kiss of the Spider Woman".
</item 1>

<item 2>
Takeshi Fukunaga
Takeshi Fukunaga is a Japanese filmmaker based in New York.

 His first feature film, "Out of My Hand" (2015) premiered in the Panorama section at the 65th Berlin International Film Festival and won the Top Prize at the
2015 Los Angeles Film Festival.

 The film was later released worldwide through Ava DuVernay’s distribution company, ARRAY.

 Takeshi was nominated for the John Cassavetes Award at the 2016 Independent Spirit Awards.

 In 2017, he was selected for The Residence by Cannes Film Festival's Cinéfondation to develop his second feature film.
</item 2>

<item 3>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 3>

<item 4>
The entries cover filmmakers, artists and films. Takeshi Fukunaga’s debut, *Out of My Hand* (2015), won the LA Film Festival’s top prize and earned an
Independent Spirit nomination. Hou Hsiao-hsien, a Taiwan New Wave auteur, won Venice’s Golden Lion and Cannes’ Best Director award; his films include *The
Assassin*, *Dust in the Wind* and *The Time to Live and the Time to Die*. Cinematographer Mark Lee
</item 4>

<item 5>
Wong Jing
Wong Jing ( born 3 May 1955) is a Hong Kong film director, producer, actor, presenter, and screenwriter.

 A prolific filmmaker with strong instincts for crowd-pleasing and publicity, Wong Jin has played a prominent role in the Hong Kong cinema of the last
quarter-century.
</item 5>

<item 6>
Ken Kwapis
Kenneth William "Ken" Kwapis (born August 17, 1957) is an American film and television director and screenwriter.

 He specialized in the single-camera sitcom in the 1990s and 2000s and has directed feature films such as "Sesame Street Presents Follow That Bird" (1985), "The
Sisterhood of the Traveling Pants" (2005), and "He's Just Not That into You" (2009).
</item 6>

<item 7>
Jon Paul Puno
Jon Paul Puno (born 1985) is an American filmmaker and classical crossover singer.

 As a filmmaker, he has directed films such as Valle de Lágrimas (2006) and Peace Grows (2004).

 And also played a small role in the 2006 movie I Will Always Love You (film).
</item 7>

<item 8>
John Dunning (film producer)
John Dunning (April 27, 1927–September 19, 2011) was a pioneering Canadian film producer from Montreal who co-founded the Canadian film production company
Cinépix and produced early works by notable Canadian directors David Cronenberg and Ivan Reitman.

 Dunning launched Cinépix with partner André Link in Montreal in the early 1960s.

 Their biggest commercial success—and the first Canadian box office hit—came with Reitman’s "Meatballs" (1979).
</item 8>

<item 9>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 9>

<item 10>
Ian Schneider is an independent American director and screenwriter. Mary Sweeney, a longtime David Lynch collaborator, edited and produced *Twin Peaks*, *Lost
Highway*, *The Straight Story* and *Mulholland Drive*; she teaches at USC and chairs Film Independent. Angelo Badalamenti scored Lynch works and won a Grammy.
Brian Ralston, Nancy Schreiber and John Bailey are established composers/cinematographers; Bailey led the Motion Picture Academy. Jon Paul Puno is
</item 10>
=== END ITEM id=0766e8e2f793071a ===

=== ITEM id=98563e3262c349ab ===
Gold sentences and facts:
Sentence 0:  He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.
  Fact 0.0: The writer has written three novels.
  Fact 0.1: The writer has written three collections of short stories.
  Fact 0.2: The writer has written a collection of novellas.
  Fact 0.3: The writer has written seven volumes of essays.
  Fact 0.4: Three of the writer's essay volumes have won National Book Critics Circle Award prizes.
  Fact 0.5: One of the writer's essay volumes, 'A Temple of Texts' (2006), won the Truman Capote Award for Literary Criticism.
Sentence 1: Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for Literature.
  Fact 1.0: Naguib Mahfouz was born on December 11, 1911.
  Fact 1.1: Naguib Mahfouz died on August 30, 2006.
  Fact 1.2: Naguib Mahfouz's Arabic name is نجيب محفوظ, transliterated Nagīb Maḥfūẓ.
  Fact 1.3: Naguib Mahfouz was an Egyptian writer.
  Fact 1.4: Naguib Mahfouz won the 1988 Nobel Prize for Literature.

Retrieved items:
<item 1>
Naguib Mahfouz
Naguib Mahfouz (Arabic: نجيب محفوظ‎ ‎ "Nagīb Maḥfūẓ ", ] ; December 11, 1911 – August 30, 2006) was an Egyptian writer who won the 1988 Nobel Prize for
Literature.

 He is regarded as one of the first contemporary writers of Arabic literature, along with Tawfiq el-Hakim, to explore themes of existentialism.

 He published 34 novels, over 350 short stories, dozens of movie scripts, and five plays over a 70-year career.

 Many of his works have been made into Egyptian and foreign films.
</item 1>

<item 2>
Naguib Mahfouz Medal for Literature
The Naguib Mahfouz Medal for Literature is a literary award for Arabic literature.

 It is given to the best contemporary novel written in Arabic, but not available in English translation.

 The winning book is then translated into English, and published by American University in Cairo Press.

 It was first awarded in 1996 and is presented annually on December 11, the birthday of Nobel laureate Naguib Mahfouz, by the President of the American
University in Cairo.
</item 2>

<item 3>
Bidaya wa Nihaya
Bidaya wa Nihaya (Arabic: بداية و نهاية‎ ‎ , English: A Beginning and an End ) is a 1960 Egyptian film directed by Salah Abouseif and based on the novel by the
same name.

 It was the first film adapted from a novel written by Nobel Prize winner Naguib Mahfouz.
</item 3>

<item 4>
The Beginning and the End (novel)
The Beginning and the End (بداية ونهاية) is a novel by Naguib Mahfouz, who was awarded the Nobel Prize in literature in 1988.

 An Egyptian, Naguib has been credited with modernizing Arabic literature, with his prolific writing style and his themes on existentialism.

 The novel is marked by very bold characterization by Naguib Mahfouz, for the time period that the story is set in.

 The story moves at a prolific pace and tries to look at the world from each characters view point.
</item 4>

<item 5>
William H. Gass
William Howard Gass (born July 30, 1924) is an American novelist, short-story writer, essayist, critic, and former philosophy professor.

 He has written three novels, three collections of short stories, a collection of novellas, and seven volumes of essays, three of which have won National Book
Critics Circle Award prizes and one of which, "A Temple of Texts" (2006), won the Truman Capote Award for Literary Criticism.

 His 1995 novel "The Tunnel" received the American Book Award.

 His 2013 novel "Middle C" won the 2015 William Dean Howells Medal.
</item 5>

<item 6>
Graph query row:
person: Naguib Mahfouz
prize: Nobel Prize in Literature
relationship: AWARDED
</item 6>

<item 7>
Thebes at War
Thebes at War is an early novel by the Egyptian writer Naguib Mahfouz.

 It was originally published in Arabic in 1944.

 An English translation by Humphrey Davies appeared in 2003.

 The novel is one of several that Mahfouz wrote at the beginning of his career, with Pharaonic Egypt as their setting.

 Others in this series of novels include "Khufu's Wisdom" (1939) and "Rhadopis of Nubia" (1943).

 All have been translated into English and appeared in one volume under the title "Three Novels of Ancient Egypt" (Everyman's Library, 2007).
</item 7>

<item 8>
Arabian Nights and Days
Arabian Nights and Days (1979) is a novel by Egyptian writer Naguib Mahfouz, winner of the Nobel Prize for Literature.

 The novel serves as a sequel and companion piece for "One Thousand and One Nights" and includes many of the same characters that appeared in the original work
such as Shahryar, Scheherazade, and Aladdin.
</item 8>

<item 9>
Cairo Trilogy
The Cairo Trilogy (Arabic: الثلاثية‎ ‎ ("The Trilogy") or ثلاثية القاهرة ("The Cairo Trilogy")) is a trilogy of novels written by the Egyptian novelist and
Nobel Prize winner Naguib Mahfouz, and one of the prime works of his literary career.
</item 9>

<item 10>
Frances Liardet
Frances E. Liardet is a writer and translator of Arabic literature.

 She studied Creative Writing at the University of East Anglia.

 She has translated several book-length works, including two books by the modernist Egyptian writer Edwar al-Kharrat and one by Nobel Prize winner Naguib
Mahfouz.
</item 10>
=== END ITEM id=98563e3262c349ab ===

=== ITEM id=856998df734d3fcc ===
Gold sentences and facts:
Sentence 0:  It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of Superman called Nuclear Man.
  Fact 0.0: The film is the fourth and final film in the original Superman film series.
  Fact 0.1: The film is the first film in the original Superman series not produced by Alexander and Ilya Salkind.
  Fact 0.2: The film was produced by Golan-Globus' Cannon Films.
  Fact 0.3: The film was produced in association with Warner Bros.
  Fact 0.4: Gene Hackman returned as Lex Luthor in the film.
  Fact 0.5: Lex Luthor creates an evil solar-powered version of Superman called Nuclear Man.
Sentence 1: Eugene Allen "Gene" Hackman (born January 30, 1930) is a retired American actor and novelist.
  Fact 1.0: Eugene Allen 'Gene' Hackman was born on January 30, 1930.
  Fact 1.1: Gene Hackman is a retired American actor.
  Fact 1.2: Gene Hackman is a retired American novelist.

Retrieved items:
<item 1>
Lex Luthor (Smallville)
Lex Luthor is a fictional character from the television series "Smallville".

 He was a series regular from the pilot episode until the season seven finale, and has been played continuously by Michael Rosenbaum, with various actors
portraying the character as a child throughout the series.

 The character of Lex Luthor, first created for comic books by Jerry Siegel and Joe Shuster in 1940 as nemesis of Superman, was adapted to television in 2001 by
Alfred Gough and Miles Millar—this is only the third time the character has been adapted to a live action television series.

 The character has also appeared in various literature based on the "Smallville" television series, none of which directly continues from or into the television
episodes.
</item 1>

<item 2>
Batman v Superman: Dawn of Justice
Batman v Superman: Dawn of Justice is a 2016 American superhero film featuring the DC Comics characters Batman and Superman.

 Directed by Zack Snyder and distributed by Warner Bros.

 Pictures, the film is the second installment in the DC Extended Universe (DCEU) following 2013's "Man of Steel".

 It was written by Chris Terrio and David S. Goyer, and features an ensemble cast that includes Ben Affleck, Henry Cavill, Amy Adams, Jesse Eisenberg, Diane
Lane, Laurence Fishburne, Jeremy Irons, Holly Hunter and Gal Gadot.

 "Batman v Superman: Dawn of Justice" is the first live-action film to feature Batman and Superman together, as well as the first live-action cinematic
portrayal of Wonder Woman, Aquaman, Flash and Cyborg.

 In the film, criminal mastermind Lex Luthor manipulates Batman into a preemptive battle with Superman, whom Luthor is obsessed with defeating.
</item 2>

<item 3>
Lionel Luthor
Lionel Luthor is a fictional character portrayed by John Glover in the television series "Smallville".

 The character was initially a recurring guest in season one, and became a series regular in season two and continued until being written out of the show in
season seven.

 The character returned to the show in season ten again as a recurring guest role as a parallel universe (Earth-2) version of the character.

 In "Smallville", Lionel Luthor is the father of Lex Luthor (Michael Rosenbaum), and founder and CEO of LuthorCorp.

 Lex Luthor's father was first introduced in "Superman" comics by Jerry Siegel in 1961 and has since appeared in other Superman-related media under different
names.

 "Smallville" is the first appearance in which the character has been an intricate part of a Superman adaptation.

 Series developers Alfred Gough and Miles Millar created Lionel Luthor for "Smallville" to provide an antithesis to the parenting style of Jonathan Kent (John
Schneider) and Martha Kent (Annette O'Toole).
</item 3>

<item 4>
Superman: A Tale of Five Cities
Superman og Fredsbomben (translation: Superman and the Peace Bomb) or Superman: A Tale of Five Cities is a large prestige format 48-page graphic novel published
by Danish publisher Interpresse in 1990.

 In celebration of Superman's 50th anniversary in 1988, DC Comics incentived overseas publishers to produce an original story with the Man of Steel.

 Only the Interpresse editors from Denmark attended the call, comic book translator and writer Niels Søndergaard wrote the story featuring Clark Kent and Lois
Lane in a cold war adventure where Superman needs to devise a Lex Luthor plan to control all Nuclear Bombs in Europe.

 The storyline presents the Man of Steel visiting five capitals from northern Europe and Scandinavia: Amsterdam, Copenhagen, Oslo, Stockholm and Helsinki.

 The art and colors are provided by Teddy Kristiansen and the lettering by Rebecca Løwe.

 It was dedicated to danish editors Henning Kure and Ove Høyer, pioneers of the super-heroes comics in Denmark.
</item 4>

<item 5>
Lyle Alzado
Lyle Martin Alzado (April 3, 1949 – May 14, 1992) was a professional American football defensive end of the National Football League famous for his intense and
intimidating style of play.
</item 5>

<item 6>
Lex Luthor: Man of Steel
Lex Luthor: Man of Steel (later collected as simply Luthor) is a five-issue monthly American comic book limited series written by Brian Azzarello and
illustrated by Lee Bermejo, which features Superman's nemesis Lex Luthor as the protagonist.
</item 6>

<item 7>
Superman IV: The Quest for Peace
Superman IV: The Quest for Peace is a 1987 American superhero film directed by Sidney J. Furie, based on the DC Comics character Superman.

 It is the fourth and final film in the original "Superman" film series, and the first film in that series not to be produced by Alexander and Ilya Salkind, but
rather by Golan-Globus' Cannon Films, in association with Warner Bros. Gene Hackman returned as Lex Luthor, who creates an evil solar-powered version of
Superman called Nuclear Man.
</item 7>

<item 8>
Alternative versions of Lex Luthor
The fictional character Lex Luthor has appeared in a number of media, always as the archenemy of Superman.

 Each version of the work typically establishes its own continuity, and sometimes introduces parallel universes, to the point where distinct differences in the
portrayal of the character can be identified.

 In addition, the DC Universe has been rewritten a number of times, establishing additional versions of the character.

 This article details and lists various versions of Lex Luthor depicted in works including DC Comics "Multiverse", "Elseworlds", television and film.
</item 8>

<item 9>
Superman Returns
Superman Returns is a 2006 American superhero film directed and produced by Bryan Singer.

 It is based on the DC Comics character Superman and serves as an homage sequel to the motion pictures "Superman" (1978) and "Superman II" (1980), while
ignoring the events of "Superman III" (1983) and "" (1987).

 The film stars Brandon Routh as Clark Kent/Superman, Kate Bosworth as Lois Lane, Kevin Spacey as Lex Luthor, with James Marsden, Frank Langella, and Parker
Posey.

 The film tells the story of the title character returning to Earth after a five-year absence.

 He finds that his love interest Lois Lane has moved on with her life, and that his archenemy Lex Luthor is plotting a scheme that will destroy Superman and the
world.
</item 9>

<item 10>
It's Superman!
It's Superman!

 is a novel by Tom De Haven based on the comic book superhero Superman.

 It was released on September 15, 2005 in hardcover and August 29, 2006 in paperback.

 The premise tells the tale of Clark Kent's beginnings into becoming Superman, set in the 1930s, where Clark befriends a wrongly convicted photographer named
Willi Berg, and is then taken from Kansas to Hollywood and finally in New York where he meets Lois Lane, fights Lex Luthor, as he debuts in his superhero
persona.

 Despite the setting, this is not about the Golden Age Superman also known as the Superman of Earth-2; as Perry White, the Daily Planet, Lex Luthor's position
and his trademark powers (including flight) are not part of that alternate Earth.

 Rather, it's a Superman period piece set in the 1930s.
</item 10>
=== END ITEM id=856998df734d3fcc ===

=== ITEM id=66ea5247b2049a93 ===
Gold sentences and facts:
Sentence 0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.
  Fact 0.0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music.
  Fact 0.1: More Hot Rocks was released by ABKCO Records.
  Fact 0.2: ABKCO Records is run by former manager Allen Klein.
  Fact 0.3: Allen Klein usurped control of the Rolling Stones' Decca/London material in 1970.
  Fact 0.4: More Hot Rocks was released after the Rolling Stones departed from Decca.
  Fact 0.5: More Hot Rocks was released after the Rolling Stones departed from Allen Klein.
Sentence 1: Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording artists.
  Fact 1.0: Allen Klein was born December 18, 1931.
  Fact 1.1: Allen Klein died July 4, 2009.
  Fact 1.2: Allen Klein was an American businessman.
  Fact 1.3: Allen Klein was a music publisher.
  Fact 1.4: Allen Klein was a writers' representative.
  Fact 1.5: Allen Klein was a filmmaker.
  Fact 1.6: Allen Klein was a record label executive.
  Fact 1.7: Allen Klein was most noted for his tough persona.
  Fact 1.8: Allen Klein was most noted for aggressive, innovative negotiation tactics.
  Fact 1.9: Many of Allen Klein's negotiation tactics established higher industry standards for compensating recording artists.
Sentence 2:  He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming one of the most powerful individuals in the music industry during his era.
  Fact 2.0: Allen Klein first scored massive monetary windfalls for Buddy Knox.
  Fact 2.1: Allen Klein first scored massive contractual windfalls for Buddy Knox.
  Fact 2.2: Allen Klein first scored massive monetary windfalls for Jimmy Bowen.
  Fact 2.3: Allen Klein first scored massive contractual windfalls for Jimmy Bowen.
  Fact 2.4: Buddy Knox and Jimmy Bowen were one-hit rockabillies of the late 1950s.
  Fact 2.5: Allen Klein later parlayed his early successes into a position managing Sam Cooke.
  Fact 2.6: Allen Klein eventually managed the Beatles.
  Fact 2.7: Allen Klein eventually managed the Rolling Stones.
  Fact 2.8: Allen Klein managed the Beatles and the Rolling Stones simultaneously.
  Fact 2.9: Allen Klein managed many other artists.
  Fact 2.10: Allen Klein became one of the most powerful individuals in the music industry during his era.

Retrieved items:
<item 1>
Sargent House
Sargent House is an American management company and record label based in Los Angeles.

 It was founded in June 2006 by Cathy Pellow, who is a music video commissioner for Atlantic Records and also owns a music video production company called
Refused TV.

 Pellow began managing the rock outfit RX Bandits in 2006.

 The band needed to release its new album "...And the Battle Begun", so Pellow decided to launch her own record label (also to be able to do things differently
from the traditional model of record labels).

 From that point onward, Pellow continued to manage bands and release records, all under the Sargent House banner.
</item 1>

<item 2>
Bigtop Records
Bigtop Records was an American record label started by music executive Johnny Bienstock and the major music publisher Hill & Range Music and was co-owned along
with Big Top Record Distributors (sic).

 Hit artists included Del Shannon, Johnny and the Hurricanes, Lou Johnson, Sammy Turner, Don and Juan, and Toni Fisher.

 Big Top Record Distributors also distributed Paul Case's Dunes Records label in the early 60's, which had hits from Ray Peterson ("Corrina, Corrina") and
Curtis Lee ("Pretty Little Angel Eyes"), both records produced by Phil Spector.

 Bell Records briefly distributed Bigtop prior to the label closing, around 1965.

 "Bigtop" also released two Mad Magazine music parody themed albums; "Mad Twists Rock 'N' Roll" and "Fink Along With Mad" in 1963.
</item 2>

<item 3>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 3>

<item 4>
Rock Star (TV series)
Rock Star is a television series produced by Mark Burnett, David Goffin, and Lisa Hennessy in which aspiring singers from all around the world competed to
become the lead singer of a featured group.

 It debuted on CBS on July 11, 2005, to mediocre ratings.

 The show was hosted by television personality and commercial spokeswoman Brooke Burke and Jane's Addiction & Red Hot Chili Peppers lead guitarist Dave Navarro.

 In Australian band INXS chose J.D. Fortune as their new lead singer.

 For , the band Rock Star Supernova chose Lukas Rossi as the lead singer of their new supergroup.
</item 4>

<item 5>
The entries profile rock, punk, and metal albums, bands, and musicians. Fuel’s *Puppet Strings* (2014) reunited singer Brett Scallions but featured none of the
prior album’s recording personnel. Jizzy Pearl sang on Quiet Riot’s *10* and L.A. Guns’ *Shrinking Violet*. Other subjects include J. Geils Band, Green Day side
projects and collaborators, Glassjaw and Daryl Palumbo, Clearview 77, Sons of Abraham,
</item 5>

<item 6>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 6>

<item 7>
Hot Rocks 1964–1971
Hot Rocks 1964–1971 is the first compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who gained control of the
band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1971, it proved to be The Rolling Stones' biggest-selling release of their career and an enduring and popular retrospective.
</item 7>

<item 8>
More Hot Rocks (Big Hits &amp; Fazed Cookies)
More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who
usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1972 as the follow-up to the hugely successful "Hot Rocks 1964–1971", it was another success for Klein and ABKCO.
</item 8>

<item 9>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 9>

<item 10>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 10>
=== END ITEM id=66ea5247b2049a93 ===

=== ITEM id=08a7763107ac4781 ===
Gold sentences and facts:
Sentence 0: Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.
  Fact 0.0: Not Richard, But Dick is the seventh studio album by The Dead Milkmen.
  Fact 0.1: Not Richard, But Dick was released in 1993.
  Fact 0.2: Not Richard, But Dick was released by Hollywood Records.
Sentence 1: The Dead Milkmen are an American punk rock band formed in 1983 in Philadelphia, Pennsylvania.
  Fact 1.0: The Dead Milkmen are an American punk rock band.
  Fact 1.1: The Dead Milkmen were formed in 1983.
  Fact 1.2: The Dead Milkmen were formed in Philadelphia, Pennsylvania.

Retrieved items:
<item 1>
The Idols with Sid Vicious
The Idols with Sid Vicious is a concert album of former Sex Pistols bassist Sid Vicious performing with The Idols; though recorded in September 1978, it wasn't
released until 1993.

 Steve Dior provided a recording of this performance to the New Rose record label.

 It was originally released by the Fan Club division of New Rose.

 It was also released in Japan by Teichiku Records, under license from New Rose.
</item 1>

<item 2>
RPA &amp; The United Nations of Sound
The band played in Australia also during the Splendour in the Grass Festival in late July.

 The band's debut album as Richard's backing lineup, entitled "United Nations of Sound", was released on 19 July 2010.

 "Born Again", the first proper single, was officially released the same day of the album as a digital download.
</item 2>

<item 3>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 3>

<item 4>
Electric Guitars
Electric Guitars were an English band formed early in 1980 by Neil Davenport (vocals, lyrics) and Richard Hall (bass, vocals) who were both studying English at
Bristol University.

 The band soon increased to a five-man line-up, with Andy Saunders (guitar, vocals), Matt Salt (drums) and Dick Truscott (keyboards), they also later added two
backing singers: Sara and Wendy Patridge.

 Their first single "Health" / "Continental Shelf" was released on local label Fried Egg Records in 1980.
</item 4>

<item 5>
Not Richard, But Dick
Not Richard, But Dick is the seventh studio album by The Dead Milkmen, released in 1993 by Hollywood Records.

 After being out of print for years, Hollywood Records decided to release the album for digital download as of April 2, 2013.
</item 5>

<item 6>
The V.I.P.s (band)
The V.I.P.s were a British R&B musical ensemble formed in Carlisle, Cumberland, (North West England) in late 1963, out of an earlier outfit known as The
Ramrods, who had formed in Carlisle in 1960.

 After a change of personnel in April 1967, the band changed their name to "Art", and released the album "Supernatural Fairy Tales".

 They also participated to a psychedelic bizarre album called "Featuring The Human Host And The Heavy Metal Kids" by a collective known as Hapshash and the
Coloured Coat, formed by Guy Stevens and an influential British graphic design and avant-garde musical partnership between Michael English and Nigel Waymouth.

 The musicians involved in that project were Mike Harrison on keys and vocals, Luther Grosvenor on guitars, Greg Ridley on bass and Mike Kellie on drums, as
well as performances by Stevens, English and Waymouth.

 It was the first time that the term "heavy metal" was ever used in music, even though that album had nothing to do with heavy metal music, being closer to
psychedelic music.
</item 6>

<item 7>
Dick Delicious and The Tasty Testicles
Dick Delicious and The Tasty Testicles are a comedy / metal band from Atlanta, Georgia.

 The band formed in 1992 and released 3 albums between 1995 and 2003.

 In 2003 the Dick Delicious received the "Howard Stern Award for Musical Excellence" and broke up shortly afterward.
</item 7>

<item 8>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 8>

<item 9>
Richard Cheese
Richard Cheese & Lounge Against the Machine is a cover band and comedy act, performing popular songs in a lounge/swing style, reminiscent of Frank Sinatra, Dean
Martin, and Tony Bennett.

 Lounge singer Richard Cheese is a character created and portrayed by Los Angeles-based actor/comedian Mark Jonathan Davis.
</item 9>

<item 10>
Ricky Wilson (American musician)
Ricky Helton Wilson (March 19, 1953 – October 12, 1985) was an American musician best known as the original guitarist and founding member of rock band the
B-52's.

 Born in Athens, Georgia, Wilson was the brother of fellow member Cindy Wilson.

 The B-52's was founded in 1977, when Ricky, his sister Cindy, Kate Pierson, Keith Strickland and Fred Schneider shared a tropical flaming volcano drink at a
Chinese restaurant and, after an impromptu music session at the home of their friend Owen Scott III, played for the first time at a Valentine's Day party for
friends.

.

 Wilson's unusual guitar tunings were a large contribution to the band's quirky sound .
</item 10>
=== END ITEM id=08a7763107ac4781 ===

=== ITEM id=d220f39fcfab076e ===
Gold sentences and facts:
Sentence 0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.
  Fact 0.0: More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music.
  Fact 0.1: More Hot Rocks was released by ABKCO Records.
  Fact 0.2: ABKCO Records is run by former manager Allen Klein.
  Fact 0.3: Allen Klein usurped control of the Rolling Stones' Decca/London material in 1970.
  Fact 0.4: More Hot Rocks was released after the Rolling Stones departed from Decca.
  Fact 0.5: More Hot Rocks was released after the Rolling Stones departed from Allen Klein.
Sentence 1: Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording artists.
  Fact 1.0: Allen Klein was born December 18, 1931.
  Fact 1.1: Allen Klein died July 4, 2009.
  Fact 1.2: Allen Klein was an American businessman.
  Fact 1.3: Allen Klein was a music publisher.
  Fact 1.4: Allen Klein was a writers' representative.
  Fact 1.5: Allen Klein was a filmmaker.
  Fact 1.6: Allen Klein was a record label executive.
  Fact 1.7: Allen Klein was most noted for his tough persona.
  Fact 1.8: Allen Klein was most noted for aggressive, innovative negotiation tactics.
  Fact 1.9: Many of Allen Klein's negotiation tactics established higher industry standards for compensating recording artists.
Sentence 2:  He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming one of the most powerful individuals in the music industry during his era.
  Fact 2.0: Allen Klein first scored massive monetary windfalls for Buddy Knox.
  Fact 2.1: Allen Klein first scored massive contractual windfalls for Buddy Knox.
  Fact 2.2: Allen Klein first scored massive monetary windfalls for Jimmy Bowen.
  Fact 2.3: Allen Klein first scored massive contractual windfalls for Jimmy Bowen.
  Fact 2.4: Buddy Knox and Jimmy Bowen were one-hit rockabillies of the late 1950s.
  Fact 2.5: Allen Klein later parlayed his early successes into a position managing Sam Cooke.
  Fact 2.6: Allen Klein eventually managed the Beatles.
  Fact 2.7: Allen Klein eventually managed the Rolling Stones.
  Fact 2.8: Allen Klein managed the Beatles and the Rolling Stones simultaneously.
  Fact 2.9: Allen Klein managed many other artists.
  Fact 2.10: Allen Klein became one of the most powerful individuals in the music industry during his era.

Retrieved items:
<item 1>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 1>

<item 2>
Bigtop Records
Bigtop Records was an American record label started by music executive Johnny Bienstock and the major music publisher Hill & Range Music and was co-owned along
with Big Top Record Distributors (sic).

 Hit artists included Del Shannon, Johnny and the Hurricanes, Lou Johnson, Sammy Turner, Don and Juan, and Toni Fisher.

 Big Top Record Distributors also distributed Paul Case's Dunes Records label in the early 60's, which had hits from Ray Peterson ("Corrina, Corrina") and
Curtis Lee ("Pretty Little Angel Eyes"), both records produced by Phil Spector.

 Bell Records briefly distributed Bigtop prior to the label closing, around 1965.

 "Bigtop" also released two Mad Magazine music parody themed albums; "Mad Twists Rock 'N' Roll" and "Fink Along With Mad" in 1963.
</item 2>

<item 3>
Rock Star (TV series)
Rock Star is a television series produced by Mark Burnett, David Goffin, and Lisa Hennessy in which aspiring singers from all around the world competed to
become the lead singer of a featured group.

 It debuted on CBS on July 11, 2005, to mediocre ratings.

 The show was hosted by television personality and commercial spokeswoman Brooke Burke and Jane's Addiction & Red Hot Chili Peppers lead guitarist Dave Navarro.

 In Australian band INXS chose J.D. Fortune as their new lead singer.

 For , the band Rock Star Supernova chose Lukas Rossi as the lead singer of their new supergroup.
</item 3>

<item 4>
More Hot Rocks (Big Hits &amp; Fazed Cookies)
More Hot Rocks (Big Hits & Fazed Cookies) is the second compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who
usurped control of the band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1972 as the follow-up to the hugely successful "Hot Rocks 1964–1971", it was another success for Klein and ABKCO.
</item 4>

<item 5>
Hot Rocks 1964–1971
Hot Rocks 1964–1971 is the first compilation album of Rolling Stones music released by former manager Allen Klein's ABKCO Records (who gained control of the
band's Decca/London material in 1970) after the band's departure from Decca and Klein.

 Released in late 1971, it proved to be The Rolling Stones' biggest-selling release of their career and an enduring and popular retrospective.
</item 5>

<item 6>
Hot Streets
Hot Streets is the tenth studio album (twelfth overall) by the American band Chicago, released in 1978.

 In many ways, "Hot Streets" marked the beginning of a new era for the band, turning to disco music, a move which would be derided in retrospect.

 It was also the band's first album with all-new material released since their second that didn't have a numbered title.

 It was also the first album not to feature original guitarist/vocalist Terry Kath, who died from an accidental self-inflicted gunshot wound in January 1978.

 He was replaced by Donnie Dacus on this album.
</item 6>

<item 7>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 7>

<item 8>
Sargent House
Sargent House is an American management company and record label based in Los Angeles.

 It was founded in June 2006 by Cathy Pellow, who is a music video commissioner for Atlantic Records and also owns a music video production company called
Refused TV.

 Pellow began managing the rock outfit RX Bandits in 2006.

 The band needed to release its new album "...And the Battle Begun", so Pellow decided to launch her own record label (also to be able to do things differently
from the traditional model of record labels).

 From that point onward, Pellow continued to manage bands and release records, all under the Sargent House banner.
</item 8>

<item 9>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 9>

<item 10>
Rock Star Supernova
Rock Star Supernova was a reality television-formed supergroup consisting of drummer Tommy Lee (Mötley Crüe), bassist Jason Newsted (ex-Metallica), guitarist
Gilby Clarke (ex-Guns N' Roses) and singer Lukas Rossi.

 The band was formed during the second season of the Rock Star Reality TV series which was called .
</item 10>
=== END ITEM id=d220f39fcfab076e ===

=== ITEM id=fd92eaea31fcad92 ===
Gold sentences and facts:
Sentence 0:  He is mainly known for his spells at Real Sociedad where his offensive partnership with Nihat Kahveci was one of the best in Spain.
  Fact 0.0: He is mainly known for his spells at Real Sociedad.
  Fact 0.1: His offensive partnership with Nihat Kahveci at Real Sociedad was one of the best in Spain.
Sentence 1: Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
  Fact 1.0: Nihat Kahveci was born on 23 November 1979.
  Fact 1.1: Nihat Kahveci is a Turkish former footballer.
  Fact 1.2: Nihat Kahveci played as a forward.

Retrieved items:
<item 1>
Adnan Januzaj
Adnan Januzaj (] ; born 5 February 1995) is a Belgian professional footballer who plays as a winger for Spanish club Real Sociedad.

 Born and raised in Brussels, he began his career with Anderlecht before joining Manchester United in 2011 at the age of 16.

 Januzaj broke into the Manchester United first-team under manager David Moyes during the 2013–14 season, but struggled for opportunities under Moyes'
successors Louis van Gaal and José Mourinho, and had loan spells at Borussia Dortmund and Sunderland before joining Real Sociedad in July 2017.

 Januzaj made his full international debut in 2014 and later that year played for Belgium at the World Cup.
</item 1>

<item 2>
Nihat Kahveci
Nihat Kahveci (born 23 November 1979) is a Turkish former footballer who played as a forward.
</item 2>

<item 3>
Real Sociedad
Real Sociedad de Fútbol, S.A.D., more commonly referred to as Real Sociedad (] ; "Royal Society") or La Real, is a Spanish football club based in the city of
San Sebastián, Basque Country, founded on 7 September 1909.

 It plays its home matches at the 32,000-capacity Anoeta Stadium.

 Real Sociedad won the Liga title in 1980–81 and 1981–82, and last finished runners-up in 2002–03.

 The club has also won the Copa del Rey twice, in 1909 and 1987.

 It contests the Basque derby against rivals Athletic Bilbao.

 Real Sociedad were founder members of La Liga in 1928, and its longest spell in the top flight was for 40 seasons, from 1967 to 2007.
</item 3>

<item 4>
C.D. Real Sociedad
Club Deportivo Real Sociedad, commonly known as Real Sociedad (] ), is a Honduran football club based on Tocoa, Colón, Honduras.
</item 4>

<item 5>
2013–14 Real Sociedad season
The 2013–14 season was Real Sociedad's 67th season in La Liga.

 Real Sociedad finished 7th in the league and reached the semifinals of the Copa del Rey.

 The Basque failed to make it out of the group stages of the UEFA Champions League.
</item 5>

<item 6>
Lippo Hertzka
Lippo Hertzka (19 November 1904 – 14 March 1951) was a Hungarian football player and manager.

 He played for Essener Turnerbund, MTK Budapest and Real Sociedad.

 After retiring, he coached seven teams, including Real Sociedad and Real Madrid, a team which he coached for 2 years (1930–1932) and led to an undefeated La
Liga championship during the 1931-32 season, which meant the first La Liga title for the "white" squad.

 He also won two league titles in Portugal for Benfica.
</item 6>

<item 7>
José Antonio Santamaría Mikel Vaqueriza
José Antonio Santamaría Mikel Vaqueriza (born 16 March 1946 in San Sebastián, Guipúzcoa, Spain; died 19 January 1993) was a Spanish footballer.

 During his career as a professional footballer, Santamaría played for SD Eibar (1963 to 1964), Real Sociedad (1964 to 1967), Real Sociedad (1967 to 1971),
Hercules CF (1971 to 1974) finishing his career at CE Sabadell FC (1974 to 1975).

 After retiring from football he had a career as an entrepreneur in the hospitality sector.

 He was killed in an attack committed by ETA in 1993.
</item 7>

<item 8>
Admir Čavalić
Admir Čavalić is the Director of Association "Multi", a non-profit organization dedicated to promoting libertarian ideals in Bosnia and Herzegovina.

 "Multi" is the first non-profit organization registered in Bosnia and Herzegovina that advocates libertarian ideas.

 Čavalić is the director of OPEN Fest, biggest libertarian festival in Europe.

 Čavalić also works at Faculty of Economics University of Tuzla, as an assistant teacher.

 Čavalić with Dženan Smajić wrote "Islam and Free Market" in 2014.

 The book, that was published in Bosnian, provides arguments for Islam's connection with the ideas of economic liberalism.

 Čavalić made first bosnian documentary about economics, titled "Economy".
</item 8>

<item 9>
Ivan Toplak
Ivan Toplak (Serbian Cyrillic: Ивaн Toплaк; born September 21, 1931 in Belgrade) is a former Yugoslavian football player and manager.

 He earned 1 cap for Yugoslavia.

 He started his career at Branik in Maribor, where his family moved in 1943 from Belgrade.
</item 9>

<item 10>
Darko Kovačević
Darko Kovačević (] ; Serbian Cyrillic: ; born 18 November 1973) is a Serbian former footballer who played as a forward.

 He began his career in Serbia with Proleter Zrenjanin and subsequently played for Red Star Belgrade, with whom he won a Yugoslav League title and two Yugoslav
Cups.

 His prolific performances earned him a move to Premier League side Sheffield Wednesday, although his time in England was less successful.

 He is mainly known for his spells at Real Sociedad where his offensive partnership with Nihat Kahveci was one of the best in Spain.

 Kovačević also had positive spells with Italian club Juventus and Greek side Olympiacos.

 At international level, he represented Yugoslavia at the 1998 FIFA World Cup and at UEFA Euro 2000.
</item 10>
=== END ITEM id=fd92eaea31fcad92 ===

=== ITEM id=54761ccc2fc4c6b0 ===
Gold sentences and facts:
Sentence 0: The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.
  Fact 0.0: The Battle of Saipan was a battle of the Pacific campaign of World War II.
  Fact 0.1: The Battle of Saipan was fought on the island of Saipan in the Mariana Islands.
  Fact 0.2: The Battle of Saipan was fought from 15 June to 9 July 1944.
Sentence 1:  The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
  Fact 1.0: The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division were commanded by Lieutenant General Holland Smith.
  Fact 1.1: The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division defeated the 43rd Infantry Division of the Imperial Japanese Army.
  Fact 1.2: The 43rd Infantry Division of the Imperial Japanese Army was commanded by Lieutenant General Yoshitsugu Saito.
Sentence 2:  It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.
  Fact 2.0: The battle referred to here was fought by American and Filipino forces against Japanese troops.
  Fact 2.1: The battle referred to here took place in Manila, the capital city of the Philippines.

Retrieved items:
<item 1>
Battle of Okinawa
The Battle of Okinawa (Japanese: 沖縄戦 , Hepburn: Okinawa-sen ) (Okinawan: 沖縄戦 , "Uchinaa ikusa " ), codenamed Operation Iceberg, was a major battle of the
Pacific War fought on the island of Okinawa by United States Marine and Army forces against the Imperial Japanese Army and included the largest amphibious
assault in the Pacific War during World War II, the April 1, 1945, invasion of Okinawa itself.

 The 82-day battle lasted from April 1 until June 22, 1945.

 After a long campaign of island hopping, the Allies were planning to use Okinawa, a large island only 340 mi away from mainland Japan, as a base for air
operations for the planned invasion of the Japanese home islands.
</item 1>

<item 2>
Battle of Manila (1574)
The Battle of Manilla (1574) was a battle in the Manila area mainly in the location of what is now Parañaque between Chinese pirates, led by Limahong and the
Spanish colonial forces and their native allies.

 The battle occurred on November 29, 1574 when Limahong's fleet landed in the town of Parañaque and from there, began to assault the fortifications of
Intramuros.

 Initially, the inhabitants where disorganized and Limahong's forces routed them.

 Furthermore, the Chinese killed the Master-of-Camp of the Spanish, Martin De Goiti.

 This caused them to delay their assault on Manila as Martin de Goiti's house was an obstacle in their march.

 However, upon the arrival of a certain Filipino hero called, Galo, resistance started to organize.

 Under Galo's command, they were able to resist China-born Limahong until Mexico-born Juan de Salcedo arrived from Ilocos with 300 Ilocano Warriors and Limahong
was defeated and eventually forced to retreat.

 Thereafter Limahong abandoned his plans to invade Manila and instead, set up a temporary kingdom in Pangasinan.
</item 2>

<item 3>
Battle of Saipan
The Battle of Saipan was a battle of the Pacific campaign of World War II, fought on the island of Saipan in the Mariana Islands from 15 June to 9 July 1944.

 The Allied invasion fleet embarking the expeditionary forces left Pearl Harbor on 5 June 1944, the day before Operation Overlord in Europe was launched.

 The U.S. 2nd Marine Division, 4th Marine Division, and the Army's 27th Infantry Division, commanded by Lieutenant General Holland Smith, defeated the 43rd
Infantry Division of the Imperial Japanese Army, commanded by Lieutenant General Yoshitsugu Saito.
</item 3>

<item 4>
Battle of Guam (1941)
The First Battle of Guam was an engagement during the Pacific War in World War II, and took place from 8 December to 10 December 1941 on Guam in the Mariana
Islands between Japan and the United States.

 The American garrison was defeated by Japanese forces on 10 December, which resulted in an occupation until the Second Battle of Guam in 1944.
</item 4>

<item 5>
Battle of Manila (1898)
The Battle of Manila, sometimes called the Mock Battle of Manila, was a land engagement which took place in Manila on August 13, 1898, at the end of the
Spanish–American War, four months after the decisive victory by Commodore Dewey's Asiatic Squadron at the Battle of Manila Bay.

 The belligerents were Spanish forces led by Governor-General of the Philippines Fermín Jáudenes, and American forces led by United States Army Brigadier
General Wesley Merritt and United States Navy Commodore George Dewey.

 American forces were supported by units of the Philippine Revolutionary Army, led by Emilio Aguinaldo.
</item 5>

<item 6>
Battle of Bataan
The surrender at Bataan, with 76,000 soldiers surrendering in the Philippines altogether, was the largest in American and Filipino military histories, and was
the largest United States surrender since the American Civil War's Battle of Harper's Ferry.

 Soon afterwards, Filipino and American prisoners of war were forced into the Bataan Death March.
</item 6>

<item 7>
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
</item 7>

<item 8>
Battle of Attu
The Battle of Attu, which took place from 11–30 May 1943, was a battle fought between forces of the United States, aided by Canadian reconnaissance and
fighter-bomber support, and the Empire of Japan on Attu Island off the coast of the Territory of Alaska as part of the Aleutian Islands Campaign during the
American Theater and the Pacific Theater and was the only land battle of World War II fought on incorporated territory of the United States.

 It is also the only land battle in which Japanese and American forces fought in Arctic conditions.
</item 8>

<item 9>
Battle of Manila (1945)
The Battle of Manila (February 3, 1945 – March 3, 1945) was a major battle of the Philippine campaign of 1944-45, during the Second World War.

 It was fought by American and Filipino forces against Japanese troops in Manila, the capital city of the Philippines.

 The month-long battle, which resulted in the death of over 100,000 civilians and the complete devastation of the city, was the scene of the worst urban
fighting in the Pacific theater.

 Japanese forces committed mass murder against Filipino civilians during the battle.

 Along with massive loss of life, the battle also destroyed architectural and cultural heritage dating back to the city's foundation.

 The battle ended the almost three years of Japanese military occupation in the Philippines (1942–1945).

 The city's capture was marked as General Douglas MacArthur's key to victory in the campaign of reconquest.
</item 9>

<item 10>
Battle of Guam (1944)
The Second Battle of Guam (21 July – 10 August 1944) was the American recapture of the Japanese-held island of Guam, a U.S. territory in the Mariana Islands
captured by the Japanese from the U.S. in the 1941 First Battle of Guam during the Pacific campaign of World War II.
</item 10>
=== END ITEM id=54761ccc2fc4c6b0 ===

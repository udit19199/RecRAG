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

=== ITEM id=d681e04786bfe161 ===
Gold sentences and facts:
Sentence 0: Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.
  Fact 0.0: Hayley Nichole Williams was born on December 27, 1988.
  Fact 0.1: Hayley Nichole Williams is an American singer.
  Fact 0.2: Hayley Nichole Williams is an American songwriter.
  Fact 0.3: Hayley Nichole Williams is an American musician.
Sentence 1:  The band consists of Hayley Williams, Zac Farro and Taylor York.
  Fact 1.0: The band consists of Hayley Williams.
  Fact 1.1: The band consists of Zac Farro.
  Fact 1.2: The band consists of Taylor York.
Sentence 2: Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American heavy metal bands Lord Tracy and Pantera.
  Fact 2.0: Terrence Lee Glaze was born on November 29, 1964, in Columbus.
  Fact 2.1: Terrence Lee Glaze is known as Terry Lee.
  Fact 2.2: Terry Lee is a Texas-raised singer.
  Fact 2.3: Terry Lee is a Texas-raised guitarist.
  Fact 2.4: Terry Lee is most known for his work in the American heavy metal band Lord Tracy.
  Fact 2.5: Terry Lee is most known for his work in the American heavy metal band Pantera.

Retrieved items:
<item 1>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 1>

<item 2>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 2>

<item 3>
Billie Joe Armstrong
Billie Joe Armstrong (born February 17, 1972) is an American musician, singer, songwriter and actor who is best known as the lead vocalist, primary songwriter,
and guitarist of the punk rock band Green Day, which he co-founded with Mike Dirnt.

 He is also a guitarist and vocalist for the punk rock band Pinhead Gunpowder and provides lead vocals for Green Day's side projects Foxboro Hot Tubs and The
Network.
</item 3>

<item 4>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 4>

<item 5>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 5>

<item 6>
Jason White (musician)
Jason White (born November 11, 1973) is an American musician, best known as the current touring guitarist for the American punk rock band Green Day.

 He has played as a touring member of Green Day since 1999.

 However, in 2012, he was promoted to being an official member of the band before demoting himself back to his touring guitarist role in 2016.

 He is also the guitarist/vocalist for the Californian punk band Pinhead Gunpowder, lead guitarist/lead vocalist for California, and co-founder of Adeline
Records alongside Billie Joe Armstrong.
</item 6>

<item 7>
Hayley Williams (ice hockey)
Hayley Williams (born June 3, 1990) is an American professional ice hockey player for the Brampton Thunder of the Canadian Women's Hockey League (CWHL).

 Williams has previously played for the Buffalo Beauts in the National Women's Hockey League (NWHL).
</item 7>

<item 8>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
</item 8>

<item 9>
Paramore, formed in Franklin, Tennessee, in 2004, centers on Hayley Williams, guitarist Taylor York, and drummer Zac Farro, who rejoined in 2017. The band has
five studio albums, from *All We Know Is Falling* (2005) to *After Laughter* (2017), whose bright pop sound contrasts with themes of anxiety and depression.
Notable songs include “The Only Exception,” “Grow Up,” and “Hate to See Your Heart
</item 9>

<item 10>
Terry Glaze
Terrence Lee Glaze (born November 29, 1964, Columbus), known as Terry Lee, is a Texas-raised singer and guitarist who is most known for his work in American
heavy metal bands Lord Tracy and Pantera.
</item 10>
=== END ITEM id=d681e04786bfe161 ===

=== ITEM id=020678739d5dfa54 ===
Gold sentences and facts:
Sentence 0: Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the United States.
  Fact 0.0: Pago Pago International Airport has IATA code PPG.
  Fact 0.1: Pago Pago International Airport has ICAO code NSTU.
  Fact 0.2: Pago Pago International Airport has FAA LID PPG.
  Fact 0.3: Pago Pago International Airport is also known as Tafuna Airport.
  Fact 0.4: Pago Pago International Airport is a public airport.
  Fact 0.5: Pago Pago International Airport is located 7 miles (11.3 km) southwest of the central business district of Pago Pago.
  Fact 0.6: Pago Pago International Airport is located in the village and plains of Tafuna.
  Fact 0.7: Tafuna is on the island of Tutuila.
  Fact 0.8: Tutuila is in American Samoa.
  Fact 0.9: American Samoa is an unincorporated territory of the United States.
Sentence 1: Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business district of Hoonah, Alaska.
  Fact 1.0: Hoonah Airport has IATA code HNH.
  Fact 1.1: Hoonah Airport has ICAO code PAOH.
  Fact 1.2: Hoonah Airport has FAA LID HNH.
  Fact 1.3: Hoonah Airport is a state-owned public-use airport.
  Fact 1.4: Hoonah Airport is located one nautical mile (2 km) southeast of the central business district of Hoonah.
  Fact 1.5: Hoonah is in Alaska.

Retrieved items:
<item 1>
Tutuila
Tutuila is the largest and the main island of American Samoa in the archipelago of Samoan Islands.

 It is the third largest island in the Samoan Islands chain of the Central Pacific located roughly 4000 km northeast of Brisbane, Australia and over 1200 km
northeast of Fiji.

 It contains a large, natural harbor, Pago Pago Harbor, where Pago Pago, the capital of American Samoa is situated.

 Pago Pago International Airport is also located on Tutuila island.

 Its land expanse is about 68% of the total land area of American Samoa and with 56,000 people accounts for 95% of its population.

 The island has six terrestrial and three marine ecosystems.
</item 1>

<item 2>
Graph query row:
entity: Alaska
relationship: PART_OF
related: United States
</item 2>

<item 3>
Pago Pago International Airport
Pago Pago International Airport (IATA: PPG, ICAO: NSTU, FAA LID: PPG) , also known as Tafuna Airport, is a public airport located 7 miles (11.3 km) southwest of
the central business district of Pago Pago, in the village and plains of Tafuna on the island of Tutuila in American Samoa, an unincorporated territory of the
United States.
</item 3>

<item 4>
Inter Island Airways
Inter Island Airways (also known as "Inter Island Air") is a South Pacific regional airline based in Pago Pago, American Samoa.

 Inter Island Airways operates passenger and cargo flights in and between American Samoa, Independent Samoa and to neighboring Pacific island countries.

 Its main base of operations is at Pago Pago International Airport.
</item 4>

<item 5>
Veterans Memorial Stadium (Pago Pago)
Veterans Memorial Stadium is a football stadium located in Pago Pago Park, in Pago Pago, American Samoa.

 The 10,000 capacity venue is one of the smallest stadiums in Oceania, and serves as American Samoa's national stadium.

 It is the home venue of the American Samoa national football team, hosting all of their home games.

 It is currently used mostly for matches in various football codes, such as soccer, rugby league, and the territory's most popular code, American football.
</item 5>

<item 6>
Tafuna, American Samoa
Tafuna' is a village on the east coast of Tutuila Island, American Samoa.

 It is located on a peninsula a mile north of Pago Pago International Airport and one mile south of Nu'uuli, American Samoa.
</item 6>

<item 7>
Hoonah Airport
Hoonah Airport (IATA: HNH, ICAO: PAOH, FAA LID: HNH) is a state-owned public-use airport located one nautical mile (2 km) southeast of the central business
district of Hoonah, Alaska.
</item 7>

<item 8>
Nu'uuli, American Samoa
Nu'uuli is a village on the central east coast of Tutuila Island, American Samoa.

 It is located on a peninsula several miles up from Pago Pago International Airport.
</item 8>

<item 9>
WNG710
WNG710 (sometimes referred to as Pago Pago All Hazards) is a NOAA Weather Radio station that serves the entire United States territory island of American Samoa.

 It is programmed from the National Weather Service Forecast Office in Pago Pago.

 The station broadcasts weather and hazard information for the Eastern, Manu'a, and Western Districts, which are the three political divisions of American
Samoa.

 WNG710 has a sister station, WZ2529 at Mt. Olotele.

 Its transmitter is located at the opposite side of the island.
</item 9>

<item 10>
Pago Pago
Pago Pago ( ; Samoan: ] ) is the territorial capital of American Samoa.

 It is on the main island of American Samoa, Tutuila.

 The territory is served by Pago Pago International Airport at Tafuna, some 8 miles south west of Pago Pago.

 Tourism, entertainment, food, and tuna canning are its main industries.
</item 10>
=== END ITEM id=020678739d5dfa54 ===

=== ITEM id=1000c77667ecf329 ===
Gold sentences and facts:
Sentence 0:  He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems."
  Fact 0.0: The recipient was awarded the 2012 Nobel Prize in Physics.
  Fact 0.1: The recipient shared the 2012 Nobel Prize in Physics jointly with Serge Haroche.
  Fact 0.2: The 2012 Nobel Prize in Physics was awarded for ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems.
Sentence 1:  Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.
  Fact 1.0: Since 2001, Serge Haroche has been a Professor at the Collège de France.
  Fact 1.1: Since 2001, Serge Haroche has held the Chair of Quantum Physics.

Retrieved items:
<item 1>
Brian Josephson
Brian David Josephson, {'1': ", '2': ", '3': ", '4': "} (born 4 January 1940), is a Welsh theoretical physicist and professor emeritus of physics at the
University of Cambridge.

 Best known for his pioneering work on superconductivity and quantum tunnelling, he was awarded the Nobel Prize in Physics in 1973 for his prediction of the
Josephson effect, made in 1962 when he was a 22-year-old PhD student at Cambridge University.

 Josephson is the only Welshman to have won a Nobel Prize in Physics.

 He shared the prize with physicists Leo Esaki and Ivar Giaever, who jointly received half the award for their own work on quantum tunnelling.
</item 1>

<item 2>
List of Nobel laureates in Chemistry
The Nobel Prize in Chemistry (Swedish: "Nobelpriset i kemi" ) is awarded annually by the Royal Swedish Academy of Sciences to scientists in the various fields
of chemistry.

 It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel, who died in 1896.

 These prizes are awarded for outstanding contributions in chemistry, physics, literature, peace, and physiology or medicine.

 As dictated by Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that consists of five members elected by the Royal
Swedish Academy of Sciences.

 The first Nobel Prize in Chemistry was awarded in 1901 to Jacobus Henricus van 't Hoff, of the Netherlands.

 Each recipient receives a medal, a diploma and a monetary award prize that has varied throughout the years.

 In 1901, van 't Hoff received 150,782 SEK, which is equal to 7,731,004 SEK in December 2007.

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 2>

<item 3>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 3>

<item 4>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 4>

<item 5>
List of Nobel laureates in Physiology or Medicine
Each recipient receives a medal, a diploma and a monetary award that has varied throughout the years.

 In 1901, von Behring received 150,782 SEK, which is equal to 7,731,004 SEK in December 2008.

 In 2013, the prize was awarded to James E. Rothman, Randy W. Schekman and Thomas C. Südhof; they were recognised "after discovering how cells precisely
transport material".

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 5>

<item 6>
Graph query row:
professor: Serge Haroche
award: Nobel Prize in Physics
year: 2012
school: Collège de France
</item 6>

<item 7>
David J. Wineland
David Jeffrey Wineland (born February 24, 1944) is an American Nobel-laureate physicist at the National Institute of Standards and Technology (NIST) physics
laboratory.

 His work has included advances in optics, specifically laser cooling trapped ions and using ions for quantum computing operations.

 He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of
individual quantum systems."
</item 7>

<item 8>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 8>

<item 9>
Nobel Prize in Physics
The Nobel Prize in Physics (Swedish: "Nobelpriset i fysik" ) is a yearly award given by the Royal Swedish Academy of Sciences for those who conferred the most
outstanding contributions for mankind in the field of physics.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel
Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 9>

<item 10>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 10>
=== END ITEM id=1000c77667ecf329 ===

=== ITEM id=5796400f96ae6d43 ===
Gold sentences and facts:
Sentence 0: The University of California, Santa Barbara (commonly referred to as UC Santa Barbara or UCSB) is a public research university and one of the 10 campuses of the University of California system.
  Fact 0.0: The University of California, Santa Barbara is commonly referred to as UC Santa Barbara or UCSB
  Fact 0.1: The University of California, Santa Barbara is a public research university
  Fact 0.2: The University of California, Santa Barbara is one of the 10 campuses of the University of California system
Sentence 1:  Tracing its roots back to 1891 as an independent teachers' college, UCSB joined the University of California system in 1944 and is the third-oldest general-education campus in the system.
  Fact 1.0: UCSB traces its roots back to 1891
  Fact 1.1: UCSB's roots as an independent teachers' college date to 1891
  Fact 1.2: UCSB joined the University of California system in 1944
  Fact 1.3: UCSB is the third-oldest general-education campus in the University of California system
Sentence 2:  It was founded in 1991 by the renowned philanthropist Hakim Said of the Hamdard Foundation.
  Fact 2.0: UCSB was founded in 1991
  Fact 2.1: UCSB was founded by the renowned philanthropist Hakim Said
  Fact 2.2: Hakim Said is a philanthropist of the Hamdard Foundation

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Dalhousie University
Dalhousie University (commonly known as Dal) is a public research university in Nova Scotia, Canada, with three campuses in Halifax, a fourth in Bible Hill, and
medical teaching facilities in Saint John, New Brunswick.

 Dalhousie offers more than 4,000 courses and 180 degree programs in twelve undergraduate, graduate, and professional faculties.

 The university is a member of the U15, a group of research-intensive universities in Canada.
</item 2>

<item 3>
Santa Barbara International Marathon
The Santa Barbara International Marathon and Half Marathon is an annual race that takes place in Goleta, California and Santa Barbara, California.

 The inaugural race was on December 6, 2009.

 For the last five years the Santa Barbara City College has been the official host for the finish line and general operations.

 Rusty and June Snow are the co-directors of the race with Dan Campbell as the technical director.

 The Snow's moved from the East Coast and wanted to bring a marathon to Santa Barbara.

 After a few years of getting the race approved and organized it was finally set to happen in 2009, Rusty Snow states, “I learned to be stubborn from running
marathons.”

 The Santa Barbara International Marathon course serves as a Boston Marathon qualifier.
</item 3>

<item 4>
Santa Barbara High School
Santa Barbara Senior High School, "Home of the Dons," is situated on a 40 acre campus in Santa Barbara, California and is part of the Santa Barbara Unified
School District.

 One of the oldest high schools in California, it was established in 1875, and moved to its current site in 1924.

 Until the creation of two rival high schools in 1959 and 1966, it was the sole public high school serving the city of Santa Barbara.

 In November 2005, its 18th-century Spanish influenced design was named an official California and City historic landmark.

 Today, Santa Barbara High School has an enrollment of roughly 2,200 pupils and certified staff numbering 108.
</item 4>

<item 5>
Mission Santa Barbara
Mission Santa Barbara, also known as Santa Barbara Mission, is a Spanish mission founded by the Franciscan order near present-day Santa Barbara, California.

 It was founded by Padre Fermín Lasuén on December 4, 1786, the feast day of Saint Barbara, as the tenth mission for the religious conversion of the indigenous
local Chumash-Barbareño tribe of Native American people.

 The mission is the namesake of the city of Santa Barbara as well as of Santa Barbara County.
</item 5>

<item 6>
Santa Barbara City College
Santa Barbara City College (SBCC) is a two-year community college that opened in 1909 and is located on a 74 acre campus in the city of Santa Barbara,
California, USA.

 SBCC was named co-winner of the 2013 Aspen Institute Prize for Community College Excellence, earning it a #1 national ranking.

 The college was recognized for its quality and focus in four areas: facilitating underrepresented and minority student success, student learning outcomes,
degree completion and transfer rates, and labor market success in securing good jobs after college.

 Santa Barbara City College was established by the Santa Barbara High School District in 1909, making it one of the oldest community colleges in California.

 The college was discontinued shortly after World War I, and its work largely taken over by the Santa Barbara State Normal School, which became the Santa
Barbara State College, and later, the University of California, Santa Barbara.
</item 6>

<item 7>
University of California, Santa Barbara Library
The University of California, Santa Barbara Library is the university library system of the University of California, Santa Barbara in Santa Barbara,
California.

 The Library includes four facilities: Two libraries (the Main Library and the Music Library) and two annexes (Annex I and Annex II).

 The library has some three million print volumes, 30,000 electronic journals, 34,450 e-books, 900,055 digitized items, five million cartographic items
(including some 467,000 maps and 3.2 million satellite and aerial images), more than 3.7 million pieces of microform, 167,500 sound recordings, and 4,100
manuscripts.

 The Library states that it holds 3.2 mi of manuscript and archival collections.

 According to data collected by the American Library Association, as of July 2010 the UC Santa Barbara Library holds the 99th largest library collection in
America, reporting 2,948,999 volumes.
</item 7>

<item 8>
University of California, Santa Barbara
The University of California, Santa Barbara (commonly referred to as UC Santa Barbara or UCSB) is a public research university and one of the 10 campuses of the
University of California system.

 The main campus is located on a 1022 acre site near Isla Vista, California, United States, 8 mi from Santa Barbara and 100 mi northwest of Los Angeles.

 Tracing its roots back to 1891 as an independent teachers' college, UCSB joined the University of California system in 1944 and is the third-oldest
general-education campus in the system.
</item 8>

<item 9>
Hamdard University
Hamdard University (Urdu: ) is a private research university with campuses in Karachi and Islamabad, Pakistan.

 It was founded in 1991 by the renowned philanthropist Hakim Said of the Hamdard Foundation.

 Hamdard is one of the first and the oldest private institutions of higher education in Pakistan.

 In Karachi, Hamdard University is the largest private research university with a campus area of over 350 acres.
</item 9>

<item 10>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 10>
=== END ITEM id=5796400f96ae6d43 ===

=== ITEM id=23e6f8473a109ada ===
Gold sentences and facts:
Sentence 0: Brett Dallas McClure (born February 19, 1981 in Yakima, Washington) is a retired American gymnast.
  Fact 0.0: Brett Dallas McClure was born on February 19, 1981.
  Fact 0.1: Brett Dallas McClure was born in Yakima, Washington.
  Fact 0.2: Brett Dallas McClure is a retired American gymnast.
Sentence 1:  He won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia, and later helped his U.S. gymnastics team earn a silver in the team competition at the 2004 Summer Olympics in Athens.
  Fact 1.0: Brett Dallas McClure won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia.
  Fact 1.1: Brett Dallas McClure later helped his U.S. gymnastics team earn a silver medal in the team competition at the 2004 Summer Olympics in Athens.
Sentence 2:  The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.
  Fact 2.0: The competition was held in Brisbane, Queensland, Australia.
  Fact 2.1: The competition was held from 29 August to 9 September 2001.

Retrieved items:
<item 1>
2001 Goodwill Games
The 2001 Goodwill Games was the fifth and final edition of the international multi-sport event.

 The competition was held in Brisbane, Queensland, Australia from 29 August to 9 September 2001.

 A total around 1300 athletes took part in 14 sporting competitions.
</item 1>

<item 2>
Kevin Kelly (boxer)
Kevin "Bones" Kelly (born August 7, 1969 in La Perouse, New South Wales) is an Australian retired professional welter/light middle/middleweight boxer of the
1990s and 2000s who won the South Australia State welterweight title, New South Wales (Australia) State welterweight title, New South Wales (Australia) State
light middleweight title, Australian light middleweight title, Pan Asian Boxing Association (PABA) light middleweight title, World Boxing Union (WBU)
middleweight title, and Commonwealth light middleweight title (twice), and was a challenger for the World Boxing Association (WBA) World light middleweight
title against David Reid, and World Boxing Organization (WBO) Inter-Continental light middleweight title against Gary Lockett, his professional fighting weight
varied from 142 lb , i.e. welterweight to 158+1/4 lb , i.e. middleweight.
</item 2>

<item 3>
Nicole Gordon (badminton)
Nicole Gordon (born 17 March 1976 in Lower Hutt, New Zealand) is a female badminton player from New Zealand.

 At the 2002 Commonwealth Games she won a silver medal with Sara Petersen in the women's doubles, and a bronze medal in the mixed team.
</item 3>

<item 4>
Klete Keller
Klete D. Keller (born March 21, 1982) is an American former competition swimmer who won medals at the 2000 Summer Olympics and the 2004 Summer Olympics in the
400-meter freestyle and the 4×200-meter freestyle relay.

 In the 4×200-meter freestyle relay, Keller held off a charging Ian Thorpe in the anchor leg to win the race by 0.13 seconds.

 This was the first time Australia had been beaten in the event in over seven years.

 The American relay of Michael Phelps, Ryan Lochte, Peter Vanderkaay, and Keller are undefeated since the Athens games.

 Vanderkaay, Larsen Jensen, Erik Vendt, and Keller make up the core of the premier American mid-distance/distance freestyle swimmers.
</item 4>

<item 5>
Brett McClure
Brett Dallas McClure (born February 19, 1981 in Yakima, Washington) is a retired American gymnast.

 He won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia, and later helped his U.S. gymnastics team earn a silver in the
team competition at the 2004 Summer Olympics in Athens.

 During his sporting career, McClure has collected two more silver medals in the same program at the World Artistic Gymnastics Championships (2001 and 2003).

 McClure is currently the High Performance Director with USA men's gymnastics.
</item 5>

<item 6>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 6>

<item 7>
Anthony Biddle (Paralympian)
Anthony John Biddle, OAM (born 18 June 1975) is an Australian Paralympic tandem cyclist and athlete.

 He was born in the New South Wales city of Gosford.

 He competed in athletics without winning any medals at the 1996 Atlanta Games and the 2000 Sydney Games.

 At the 2004 Athens Games, he switched to cycling and won a gold medal in the Men's 1 km Time Trial Tandem B1–3 event, for which he received a Medal of the
Order of Australia, and a bronze medal in the Men's Sprint Tandem B1–3 event.

 Kial Stewart was his pilot for both events.
</item 7>

<item 8>
Brian MacLaren
Brian MacLaren (born 21 December 1943) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1972 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Bill Crothers, Ross MacKenzie, and Robert McLaren) and third in the 1967 Pan
American Games 800 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Ross MacKenzie and Bill Crothers.
</item 8>

<item 9>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 9>

<item 10>
2002 Commonwealth Games
The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25
July to 4 August 2002.

 The 2002 Games were to be hosted in the United Kingdom to coincide with the Golden Jubilee of Elizabeth II, head of the Commonwealth, and Manchester was
selected for the 2002 Games ahead of London.

 The XVII Commonwealth Games was, prior to the 2012 Summer Olympics, the largest multi-sport event ever to be held in the UK, eclipsing the London 1948 Summer
Olympics in numbers of teams and athletes participating.

 In terms of sports and events, the 2002 Games were the largest Commonwealth Games in history featuring 281 events across 17 sports.
</item 10>
=== END ITEM id=23e6f8473a109ada ===

=== ITEM id=bdbd9990744838e7 ===
Gold sentences and facts:
Sentence 0:  The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and Viv Albertine, who went on to play in The Slits.
  Fact 0.0: The band never played live.
  Fact 0.1: The band never released any recordings.
  Fact 0.2: Like London SS, the band is more famed for the number of band members who later became well known.
  Fact 0.3: Like Masters of the Backside, the band is more famed for the number of band members who later became well known.
  Fact 0.4: Sid Vicious is listed as a band member who later became well known as a member of the Sex Pistols.
  Fact 0.5: Keith Levene is listed as a band member who later became well known.
  Fact 0.6: Keith Levene was an early member of The Clash.
  Fact 0.7: Keith Levene was later a member of Public Image Ltd.
  Fact 0.8: Palmolive is listed as a band member who later became well known.
  Fact 0.9: Viv Albertine is listed as a band member who later became well known.
  Fact 0.10: Viv Albertine went on to play in The Slits.
Sentence 1:  Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
  Fact 1.0: The band was interviewed by a fanzine named SKUM despite never playing live.
  Fact 1.1: In the SKUM interview, Sid Vicious proclaimed 'I'll just be the yob that I am now'.
Sentence 2: Sid Vicious (born John Simon Ritchie, 10 May 1957 – 2 February 1979) was an English bassist and vocalist, most famous as a member of the influential punk rock band the Sex Pistols.
  Fact 2.0: Sid Vicious was born John Simon Ritchie.
  Fact 2.1: Sid Vicious was born on 10 May 1957.
  Fact 2.2: Sid Vicious died on 2 February 1979.
  Fact 2.3: Sid Vicious was an English bassist.
  Fact 2.4: Sid Vicious was an English vocalist.
  Fact 2.5: Sid Vicious was most famous as a member of the influential punk rock band the Sex Pistols.

Retrieved items:
<item 1>
The Flowers of Romance (band)
The Flowers of Romance were an early punk band, formed in mid-1976 by Jo Faull and Sarah Hall.

 The band never played live or released any recordings, and, like London SS and Masters of the Backside, are more famed for the number of band members that
later became well known, including: Sid Vicious of the Sex Pistols, Keith Levene (an early member of The Clash and later of Public Image Ltd) and Palmolive and
Viv Albertine, who went on to play in The Slits.

 Despite never playing live, they were interviewed by a fanzine named "SKUM", in which Sid Vicious proclaimed "I'll just be the yob that I am now".
</item 1>

<item 2>
Sid Vicious
Sid Vicious (born John Simon Ritchie, 10 May 1957 – 2 February 1979) was an English bassist and vocalist, most famous as a member of the influential punk rock
band the Sex Pistols.
</item 2>

<item 3>
Revolution in the Classroom
"Revolution in the Classroom" was a single released by the Ex Pistols; despite its packaging, the record has nothing to do with the Sex Pistols.

 The single was released in 1989 on the STP Records label, pressed on various colours of vinyl.

 The song, along with its B-side "Schools Are Prisons", were both included on the Ex-Pistols album, "Deny".

 They also appear disguised as genuine Sex Pistols recordings on two Sex Pistols retrospective albums by former producer Dave Goodman, "Pirates of Destiny" (I
Swirled Records, 1989) and "We Have Cum For Your Children" (Skyclad Records, 1988).

Under the title "Revolution" the song appears on a German Sex Pistols compilation "Anarchy in the USA" from 1992 released on MBC records.

 CD 084-60322 LP 008-60321
</item 3>

<item 4>
The entries profile rock, punk, and metal albums, bands, and musicians. Fuel’s *Puppet Strings* (2014) reunited singer Brett Scallions but featured none of the
prior album’s recording personnel. Jizzy Pearl sang on Quiet Riot’s *10* and L.A. Guns’ *Shrinking Violet*. Other subjects include J. Geils Band, Green Day side
projects and collaborators, Glassjaw and Daryl Palumbo, Clearview 77, Sons of Abraham,
</item 4>

<item 5>
Nancy Spungen
Nancy Laura Spungen (February 27, 1958 – October 12, 1978) was the American girlfriend of Sex Pistols bassist Sid Vicious and a figure of the 1970s punk rock
scene.

 Spungen's life and death have been the subject of controversy among music historians and fans of the Sex Pistols.
</item 5>

<item 6>
The entries chiefly cover Sex Pistols history: Glen Matlock’s songwriting, Sid Vicious and Nancy Spungen, the Avengers’ 1978 Winterland set, and controversial
songs and releases. They also describe the deceptive Ex Pistols, early punk band the Flowers of Romance, and numerous artists spanning post-punk, new wave, pop,
psychedelic, dance, and alternative rock—including Talking Heads, Berlin, Afrodiziak, This Mortal Coil, and Age of Chance.
</item 6>

<item 7>
Bodies (Sex Pistols song)
"Bodies" is a Sex Pistols song about the shock of abortion from the 1977 album "Never Mind the Bollocks, Here's the Sex Pistols".

 The lyrics contain very graphic imagery about a terminated fetus and feature a great deal of profanity for the time: the third and final verse begins with a
couplet in which the word "fuck" is repeated five times in rapid succession.

 Along with the later "Belsen Was a Gas," it is probably the most graphic and controversial Sex Pistols song in both its subject matter and style.

 Musically, it is also the fastest and heaviest song in the Sex Pistols canon — characterized by thudding drums, droning buzzsaw guitar, and shouted vocals.

 As such, it can be considered a significant antecedent to the genres of hardcore, thrash, and speed metal that was to emerge in the 1980s.
</item 7>

<item 8>
Ex Pistols
The Ex Pistols were a scam act introduced in 1979 by one-time Sex Pistols producer Dave Goodman after his services were substituted for those of professional
producers.

 The Ex Pistols existed as a sound-alike band meant to mislead fans due to the similar music, name, and artwork, that led people to believe their songs were
actually new or lost Sex Pistols recordings.
</item 8>

<item 9>
The Idols with Sid Vicious
The Idols with Sid Vicious is a concert album of former Sex Pistols bassist Sid Vicious performing with The Idols; though recorded in September 1978, it wasn't
released until 1993.

 Steve Dior provided a recording of this performance to the New Rose record label.

 It was originally released by the Fan Club division of New Rose.

 It was also released in Japan by Teichiku Records, under license from New Rose.
</item 9>

<item 10>
Glen Matlock
Glen Matlock (born 27 August 1956) is an English musician best known for being the bass guitarist in the original line-up of the punk rock band the Sex Pistols.

 He is sometimes credited as a co-author on 10 of the 12 songs on "Never Mind the Bollocks, Here's the Sex Pistols", although he had left the band by the time
the record was released and the bass was played by Steve Jones.

 He also continues to make his own records and tour with various bands, including the Sex Pistols.
</item 10>
=== END ITEM id=bdbd9990744838e7 ===

=== ITEM id=ce4bda5b67c9866f ===
Gold sentences and facts:
Sentence 0: John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.
  Fact 0.0: John Kingsley Orton, known as Joe Orton, was born on 1 January 1933.
  Fact 0.1: Joe Orton died on 9 August 1967.
  Fact 0.2: Joe Orton was an English playwright.
  Fact 0.3: Joe Orton was an English author.
Sentence 1:  His public career was short but prolific, lasting from 1964 until his death three years later.
  Fact 1.0: Joe Orton's public career was short but prolific.
  Fact 1.1: Joe Orton's public career lasted from 1964 until his death.
  Fact 1.2: Joe Orton died three years after his public career began in 1964.
Sentence 2:  During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.
  Fact 2.0: During his brief public career, Joe Orton shocked audiences with his scandalous black comedies.
  Fact 2.1: During his brief public career, Joe Orton outraged audiences with his scandalous black comedies.
  Fact 2.2: During his brief public career, Joe Orton amused audiences with his scandalous black comedies.
Sentence 3:  The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
  Fact 3.0: The adjective 'Ortonesque' is sometimes used to refer to work characterised by a dark yet farcical cynicism.
  Fact 3.1: The cynicism is similarly dark yet farcical.
Sentence 4: Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary modernism, and is considered by many to be among the greatest novelists in the English language.
  Fact 4.0: Henry James, OM, was an American author.
  Fact 4.1: Henry James was born in 1843.
  Fact 4.2: Henry James died in 1916.
  Fact 4.3: Henry James was regarded as a key transitional figure between literary realism and literary modernism.
  Fact 4.4: Henry James is considered by many to be among the greatest novelists in the English language.
Sentence 5:  He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
  Fact 5.0: Henry James was the son of Henry James, Sr.
  Fact 5.1: Henry James was the brother of the renowned philosopher and psychologist William James.
  Fact 5.2: Henry James was the brother of diarist Alice James.

Retrieved items:
<item 1>
The Henry James Review
The Henry James Review is a triannual peer-reviewed academic journal established in 1979 and is the official publication of the Henry James Society.

 It is dedicated to the scholarly, critical, and theoretical study of the American writer Henry James.

 Each issue focuses on a specific theme of interest and seeks to promote understanding and study of James' contributions.

 The journal is published by the Johns Hopkins University Press and the current editor-in-chief is Susan M. Griffin (University of Louisville).
</item 1>

<item 2>
Henry James
Henry James, OM ((1843--)15 1843 – (1916--)28 1916 ) was an American author regarded as a key transitional figure between literary realism and literary
modernism, and is considered by many to be among the greatest novelists in the English language.

 He was the son of Henry James, Sr. and the brother of renowned philosopher and psychologist William James and diarist Alice James.
</item 2>

<item 3>
Joe Orton
John Kingsley "Joe" Orton (1 January 1933 – 9 August 1967) was an English playwright and author.

 His public career was short but prolific, lasting from 1964 until his death three years later.

 During this brief period he shocked, outraged, and amused audiences with his scandalous black comedies.

 The adjective "Ortonesque" is sometimes used to refer to work characterised by a similarly dark yet farcical cynicism.
</item 3>

<item 4>
Prick Up Your Ears
Prick Up Your Ears is a 1987 British film, directed by Stephen Frears, about the playwright Joe Orton and his lover Kenneth Halliwell.

 The screenplay was written by Alan Bennett, based on the biography by John Lahr.

 The film stars Gary Oldman as Orton, Alfred Molina as Halliwell, Wallace Shawn as Lahr and Vanessa Redgrave as Peggy Ramsay.
</item 4>

<item 5>
Simon Bent
Simon Bent is a British screenwriter and playwright, notable for work including BBC TV drama "" (2006), the screenplay for the feature film "Christie Malry's
Own Double-Entry" (2000), and the Joe Orton biographical play "Prick Up Your Ears" based on John Lahr's book.
</item 5>

<item 6>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 6>

<item 7>
The Ruffian on the Stair
The Ruffian On the Stair is a play by British playwright Joe Orton which was first broadcast on BBC Radio in August 1964.

 It is an unsympathetic yet comedic one-act portrayal of working class England, as played out by a couple and a mysterious young man who toys with their lives.

 It was based on "The Boy Hairdresser", a novel by Orton and his lover Kenneth Halliwell.

 The title and play are based on a few lines from poet and dramatist William Ernest Henley: "Madam Life's a piece in bloom, / Death goes dogging everywhere: /
She's the tenant of the room, / He's the ruffian on the stair."

 "Ruffian" is not as renowned as other works such as "Loot" and "What the Butler Saw", but it is still staged on occasion.
</item 7>

<item 8>
What the Butler Saw (play)
What the Butler Saw is a farce written by the English playwright Joe Orton.

 It was premièred at the Queen's Theatre in London on 5 March 1969.

 It was Orton's final play and the second to be performed after his death, following "Funeral Games" in 1968.
</item 8>

<item 9>
Prick Up Your Ears (play)
Prick Up Your Ears is a play by Simon Bent, based on the life of playwright Joe Orton.

 Produced by Sonia Friedman it opened at the Comedy Theatre in London's West End on 30 September 2009 following previews from 17 September.

 It starred Chris New as Joe Orton and Matt Lucas as Orton's lover and murderer, Kenneth Halliwell.
</item 9>

<item 10>
George Orwell
Eric Arthur Blair (25 June 1903 – 21 January 1950), better known by his pen name George Orwell, was an English novelist, essayist, journalist, and critic.

 His work is marked by lucid prose, awareness of social injustice, opposition to totalitarianism, and outspoken support of democratic socialism.
</item 10>
=== END ITEM id=ce4bda5b67c9866f ===

=== ITEM id=0719129d5ae4b3fb ===
Gold sentences and facts:
Sentence 0: Ravi Sethi (born 1947) is an Indian computer scientist retired from Bell Labs and president of Avaya Labs Research.
  Fact 0.0: Ravi Sethi was born in 1947.
  Fact 0.1: Ravi Sethi is an Indian computer scientist.
  Fact 0.2: Ravi Sethi retired from Bell Labs.
  Fact 0.3: Ravi Sethi is president of Avaya Labs Research.
Sentence 1: Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company, owned by Finnish company Nokia.
  Fact 1.0: Nokia Bell Labs was formerly named AT&T Bell Laboratories.
  Fact 1.1: Nokia Bell Labs was formerly named Bell Telephone Laboratories.
  Fact 1.2: Nokia Bell Labs was formerly named Bell Labs.
  Fact 1.3: Nokia Bell Labs is an American research and scientific development company.
  Fact 1.4: Nokia Bell Labs is owned by the Finnish company Nokia.
Sentence 2:  Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
  Fact 2.0: The headquarters of Nokia Bell Labs are located in Murray Hill, New Jersey.
  Fact 2.1: Nokia Bell Labs has other laboratories around the rest of the United States.
  Fact 2.2: Nokia Bell Labs has laboratories in other countries.

Retrieved items:
<item 1>
Sethi–Ullman algorithm
In computer science, the Sethi–Ullman algorithm is an algorithm named after Ravi Sethi and Jeffrey D. Ullman, its inventors, for translating abstract syntax
trees into machine code that uses as few registers as possible.
</item 1>

<item 2>
Richard Yoo
Richard Yoo is an American entrepreneur, the co-founder and former CEO of the web hosting company Rackspace, and the founder and former CEO of web hosting
company ServerBeach.
</item 2>

<item 3>
Rinku Singh
Rinku Singh (born August 8, 1988 in Lucknow, India) is a left-handed baseball pitcher who is currently a free agent.

 Singh was signed by the Pittsburgh Pirates organization after he won a pitching contest on a 2008 reality television show, "The Million Dollar Arm".

 He was the first Indian to play professional baseball and has spent several seasons in the minor leagues, only reaching as high as the A level.

 He is the subject of the movie "Million Dollar Arm".
</item 3>

<item 4>
Pharma Medica
Pharma Medica Research Inc. is an American research and development company that works with pharmaceutical companies and performs clinical trials in the
pharmaceutical and biotechnology industry.

 The company is based in Toronto, Canada with headquarters in Mississauga, Ontario.

 Pharma Medica has locations in Scarborough, Ontario and St. Charles, Missouri.
</item 4>

<item 5>
Genpact
Genpact is a global business process management and services and IT corporation with key offices in New York City, United States.

 It’s listed on the New York Stock Exchange under the symbol G.In 2016, the company reported net revenues of US$2.57 billion with more than 77,000 employees in
20 countries.

 Genpact provides consulting services to hundreds of clients, including approximately one-fifth of the Fortune Global 500.
</item 5>

<item 6>
Joshua Bloom
Joshua Simon Bloom (born June 8, 1974 in Washington, D.C.) is an American astrophysicist, full professor of astronomy at the University of California, Berkeley,
and was the CTO and co-founder of the machine-learning company wise.io (acquired by General Electric, 2016).

 He received a Bachelor of Arts in astronomy and astrophysics and physics from the Harvard College in 1996, an M.Phil from Cambridge University in 1997, and a
PhD in astronomy from the California Institute of Technology in 2002.

 He was a Junior Fellow of the Harvard Society of Fellows from 2002 to 2005.

 His astronomy research focuses on gamma-ray bursts and other astrophysical transients such as supernovae and tidal disruption events.

 He is author of the book "What Are Gamma-Ray Bursts?"

 published by Princeton University Press in 2011.
</item 6>

<item 7>
Bell Labs
Nokia Bell Labs (formerly named AT&T Bell Laboratories, Bell Telephone Laboratories and Bell Labs) is an American research and scientific development company,
owned by Finnish company Nokia.

 Its headquarters are located in Murray Hill, New Jersey, in addition to other laboratories around the rest of the United States and in other countries.
</item 7>

<item 8>
Ravi Sethi
Ravi Sethi (born 1947) is an Indian computer scientist retired from Bell Labs and president of Avaya Labs Research.

 He also serves as a member of the National Science Foundation's Computer and Information Science and Engineering (CISE) Advisory Committee.

 He is best known as one of three authors of the classic computer science textbook "", also known as the "Dragon Book".
</item 8>

<item 9>
Umesh Varshney
Umesh Varshney (born 1957) is an Indian molecular biologist, academician and the head of the "Prof.Umesh Varshney's Lab" at the Indian Institute of Science,
Bengaluru.

 He is a J. C. Bose National Fellow of the Department of Science and Technology and is known for his studies on protein synthesis and DNA repair in "Escherichia
coli" and Mycobacterium tuberculosis.

 An elected fellow of the Indian Academy of Sciences, Indian National Science Academy and the National Academy of Sciences (India), he is also a recipient of
the National Bioscience Award for Career Development of the Government of India.

 The Council of Scientific and Industrial Research, the apex agency of the Government of India for scientific research, awarded him the Shanti Swarup Bhatnagar
Prize for Science and Technology, one of the highest Indian science awards, in 2001, and then in 2014 with the G. N. Ramachandran Gold Medal for Excellence in
Biological Sciences & Technology for his contributions to biological sciences.
</item 9>

<item 10>
Von Braun Center for Science &amp; Innovation
The Von Braun Center for Science & Innovation (VCSI ) is a non-profit research and development organization based in Huntsville, Alabama and is named for
pioneering aerospace engineer Wernher von Braun.

 VCSI is affiliated with NASA, Department of Defense and other federal government agencies.
</item 10>
=== END ITEM id=0719129d5ae4b3fb ===

=== ITEM id=11d64437909dc804 ===
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
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 1>

<item 2>
Karl Wilhelm Friedrich Schlegel
Karl Wilhelm Friedrich (after 1814: von) Schlegel (10 March 1772 – 12 January 1829), usually cited as Friedrich Schlegel, was a German poet, literary critic,
philosopher, philologist and Indologist.

 With his older brother, August Wilhelm Schlegel, he was one of the main figures of the Jena romantics.

 He was a zealous promoter of the Romantic movement and inspired Samuel Taylor Coleridge, Adam Mickiewicz and Kazimierz Brodziński.

 Schlegel was a pioneer in Indo-European studies, comparative linguistics, morphological typology, and was the first to notice what became known as Grimm's law.
</item 2>

<item 3>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 3>

<item 4>
Hans von Hentig
Hans von Hentig (9 June 1887 in Berlin – 6 July 1974 in Bad Tölz) was a German criminal psychologist and politician.He was the second son of lawyer Otto von
Hentig (1852–1934) .

His older brother was later diplomat Werner Otto von Hentig.

 Otto von Hentig was one of the leading lawyers in Berlin.

 Hans von Hentig was instrumental in the setting up of a short-lived Bavarian Soviet Republic in 1919.

 During the 1920s he was a prominent exponent of National Bolshevism.

 He emigrated to United States in 1935.

 Hans von Hentig worked on Yale University.
</item 4>

<item 5>
Hermann Knoblauch
Karl Hermann Knoblauch (11 April 1820 – 30 June 1895) was a German physicist.

 He is most notable for his studies of radiant heat.

 He was one of the six founding members of the Deutsche Physikalische Gesellschaft at Berlin on 14 January 1845.
</item 5>

<item 6>
Gassenhauer
Gassenhauer nach Hans Neusiedler (1536), commonly known as Gassenhauer (] ), is a short piece from Carl Orff's Schulwerk.

 As the full title indicates, it is either an arrangement of, or inspired by, a much older work by the lutenist Hans Neusiedler from 1536.

 It (along with several other pieces) is in fact credited to Orff's longtime collaborator, Gunild Keetman, on at least one recent release of the Schulwerk.

 As with many other pieces from the Schulwerk, it has been used multiple times on television, radio, and in films, including the films "Badlands" (1973),
"Ratcatcher" (1999), "Finding Forrester" (2000), "Monster" (2003), and, most recently, Michael Moore's "" (2009) and "The Simpsons"′ 22nd-season episode "The
Scorpion's Tale" (2011).

 The piece was used as the theme music for an afternoon radio program also titled "Gassenhauer" on the classical music station WCLV in Cleveland, Ohio, in the
1970s.
</item 6>

<item 7>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 7>

<item 8>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 8>

<item 9>
Søren Kierkegaard
Søren Aabye Kierkegaard ( or ; ] ; 5 May 1813 – 11 November 1855) was a Danish philosopher, theologian, poet, social critic and religious author who is widely
considered to be the first existentialist philosopher.

 He wrote critical texts on organized religion, Christendom, morality, ethics, psychology, and the philosophy of religion, displaying a fondness for metaphor,
irony and parables. Much of his philosophical work deals with the issues of how one lives as a "single individual", giving priority to concrete human reality
over abstract thinking and highlighting the importance of personal choice and commitment.

 He was against literary critics who defined idealist intellectuals and philosophers of his time, and thought that Swedenborg, Hegel, Goethe, Fichte, Schelling,
Schlegel and Hans Christian Andersen were all "understood" far too quickly by "scholars".
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
=== END ITEM id=11d64437909dc804 ===

=== ITEM id=6e40b089c813d766 ===
Gold sentences and facts:
Sentence 0:  Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
  Fact 0.0: The unnamed person in this passage has other voice-over roles
  Fact 0.1: The unnamed person voiced Dr. Drakken in Kim Possible
  Fact 0.2: The unnamed person voiced Motor Ed in Kim Possible
  Fact 0.3: The unnamed person voiced Brother Blood in Teen Titans
  Fact 0.4: The unnamed person voiced Aquaman in an unnamed work
  Fact 0.5: The unnamed person voiced Rico in The Penguins of Madagascar
  Fact 0.6: The unnamed person voiced Ogre in an unnamed work
  Fact 0.7: The unnamed person voiced Niblet in Pound Puppies
  Fact 0.8: The unnamed person voiced the Scotsman in Samurai Jack
  Fact 0.9: The unnamed person voiced Shnitzel in Chowder
Sentence 1: Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.
  Fact 1.0: Chowder is an American animated television series
  Fact 1.1: Chowder was created by C. H. Greenblatt
  Fact 1.2: Chowder was created for Cartoon Network

Retrieved items:
<item 1>
Townsend Coleman
Townsend Putnam Coleman III (born May 28, 1954) is an American voice actor who performed in many animated series and TV commercials beginning in the early
1980s.

 Among his most notable roles are Michaelangelo from "Teenage Mutant Ninja Turtles", Wayne Gretzky on "ProStars" and the title character in "Where's Waldo?

" and the eponymous "The Tick" and he also did additional voices in "Dexter's Laboratory", "The Powerpuff Girls", "Adventure Time", "Animaniacs", "The Buzz on
Maggie", "Camp Lazlo", "CatDog", "Catscratch", "Danny Phantom", "Dave the Barbarian", "", "The Fairly OddParents", "Foster's Home for Imaginary Friends",
"Gravity Falls", "Hi Hi Puffy AmiYumi", "", "The Loud House", "The Mighty B!

", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia
2000" (1999) and "Sing" (2016)
</item 1>

<item 2>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 2>

<item 3>
Dwight Schultz
William Dwight Schultz (born November 24, 1947) is an American actor and voice artist.

 He is known for his roles as Captain "Howling Mad" Murdock on the 1980s action series "The A-Team", and as Reginald Barclay in "", "" and the film "".

 He is also well known in animation as the mad scientist Dr. Animo in the "Ben 10" series, Chef Mung Daal in the children's cartoon "Chowder", and Eddie the
Squirrel in "CatDog".
</item 3>

<item 4>
John DiMaggio
John William DiMaggio (born September 4, 1968) is an American voice actor and comedian, known for his gruff voice and for his work as Bender from the television
show "Futurama", Jake the Dog on "Adventure Time", and Marcus Fenix in the hit Xbox video game "Gears Of War".

 Other voice-over roles of his include Dr. Drakken and Motor Ed on "Kim Possible", Brother Blood on "Teen Titans", Aquaman in "", Rico in "The Penguins of
Madagascar", and Ogre In "", Niblet on "Pound Puppies", the Scotsman on "Samurai Jack", and Shnitzel on "Chowder".
</item 4>

<item 5>
Harvey Beaks
Harvey Beaks is an American animated television series created by C.H. Greenblatt for Nickelodeon.

 The series premiered on March 28, 2015 following the 2015 Kids' Choice Awards.
</item 5>

<item 6>
Adventure Time
Adventure Time is an American animated television series created by Pendleton Ward for Cartoon Network.

 It follows the adventures of a boy named Finn (voiced by Jeremy Shada) and his best friend and adoptive brother Jake (John DiMaggio)— a dog with the magical
power to change shape and size at will.

 Finn and Jake live in the post-apocalyptic Land of Ooo, where they interact with Princess Bubblegum (Hynden Walch), the Ice King (Tom Kenny), Marceline the
Vampire Queen (Olivia Olson), BMO (Niki Yang), and others.

 The series is based on a 2007 short produced for Nicktoons and Frederator Studios' animation incubator series "Random!

 Cartoons".

 After the short became a viral hit on the Internet, Cartoon Network commissioned a full-length series, which previewed on March 11, 2010, and officially
premiered on April 5, 2010.
</item 6>

<item 7>
Chowder (TV series)
Chowder is an American animated television series created by C. H. Greenblatt for Cartoon Network.

 The series follows an aspiring young child named Chowder and his day-to-day adventures as an apprentice in Chef Mung Daal's catering company.

 Although he means well, Chowder often finds himself in predicaments due to his perpetual appetite and his nature as a scatterbrain.

 It is animated with both traditional animation as well as short stop motion and puppet sequences that are inter-cut into the episodes, and that run over the
end credits.

 Stop motion sequences are produced by Screen Novelties.

 It is also the first Cartoon Network original series to premiere during Stuart Snyder's tenure as Jim Samples had resigned months before the series premiere.
</item 7>

<item 8>
Sherry Alberoni
Sharyn Eileen "Sherry" Alberoni (born December 4, 1946) is an American actress and voice artist.

 Alberoni got her start as a Mouseketeer on the weekday ABC television program "The Mickey Mouse Club".

 As an adult, she became a voice artist for Hanna-Barbera Productions.

 Besides providing voices for numerous incidental characters in series such as "Jeannie", Alberoni is best known as the voice of nasty rich-girl Alexandra Cabot
from "Josie and the Pussycats" , "superhero-in-training" Wendy from the first season of "Super Friends", the heroic robot, Bo in "Mighty Orbots" and
Glumdalclitch in "The Three Worlds of Gulliver".

 In 1971, she starred alongside Patty Andrews in the Sherman Brothers stage musical, "Victory Canteen".
</item 8>

<item 9>
Adventure Time (pilot)
"Adventure Time" is an animated short created by Pendleton Ward, as well as the pilot to the Cartoon Network series of the same name.

 The short follows the adventures of Pen (voiced by Zack Shada), a human boy, and his best friend Jake (voiced by John DiMaggio), a dog with magical powers to
change shape and grow and shrink at will.

 In this episode, Pen and Jake have to rescue Princess Bubblegum (voiced by Paige Moss) from the antagonistic Ice King (voiced by John Kassir).
</item 9>

<item 10>
Right Now Kapow
Right Now Kapow is an American animated sketch comedy television series produced by Warner Bros.

 Animation for Disney XD.

 It is the first collaboration between Warner Bros.

 Animation and Disney.

 The series premiered on September 19, 2016 and ended on May 31, 2017.

 The series was created by Justin Becker and Marly Halpern-Graser, who previously worked on the Cartoon Network series "Mad".

 Becker also worked on Adult Swim infomercials, and Halpern-Graser also worked on Cartoon Network's "DC Nation".

 The series follows Dog, Candy, Ice Cream, Diamond, Plant, and Moon going on new adventures everyday...and find themselves in mischief.

 Every episode of the series has 3 main parts in every 11-minute segment and other random shorts.

 The series has a style of humor similar to Cartoon Network's previous series, "Mad".
</item 10>
=== END ITEM id=6e40b089c813d766 ===

=== ITEM id=4513ffaf38737d87 ===
Gold sentences and facts:
Sentence 0:  Players compete in races on tracks inspired by attractions such as the Haunted Mansion, Pirates of the Caribbean, Space Mountain and Big Thunder Mountain Railroad to acquire missing parts for the park's fireworks machine, which was accidentally destroyed by Chip 'n' Dale while they were gathering acorns.
  Fact 0.0: Players compete in races.
  Fact 0.1: The race tracks are inspired by attractions such as the Haunted Mansion.
  Fact 0.2: The race tracks are inspired by attractions such as Pirates of the Caribbean.
  Fact 0.3: The race tracks are inspired by attractions such as Space Mountain.
  Fact 0.4: The race tracks are inspired by attractions such as Big Thunder Mountain Railroad.
  Fact 0.5: Players race on tracks to acquire missing parts for the park's fireworks machine.
  Fact 0.6: The park's fireworks machine was accidentally destroyed by Chip 'n' Dale.
  Fact 0.7: Chip 'n' Dale destroyed the fireworks machine while they were gathering acorns.
Sentence 1:  Opened on January 15, 1975, Space Mountain is the oldest operating roller coaster in the state of Florida, and is the original version of the iconic attraction that has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide, except for the Shanghai Disneyland Resort.
  Fact 1.0: Space Mountain opened on January 15, 1975.
  Fact 1.1: Space Mountain is the oldest operating roller coaster in the state of Florida.
  Fact 1.2: Space Mountain is the original version of an iconic attraction.
  Fact 1.3: The original version of Space Mountain has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide.
  Fact 1.4: The replication of Space Mountain excludes the Shanghai Disneyland Resort.

Retrieved items:
<item 1>
Walt Disney World Dolphin
The Walt Disney World Dolphin is a resort hotel designed by architect Michael Graves located between Epcot and Disney's Hollywood Studios in the Walt Disney
World Resort in Bay Lake, Florida, next to Disney's BoardWalk Resort area.

 It opened on June 1, 1990 and is joined to its sister hotel, the Walt Disney World Swan (also designed by Graves) by a palm-tree lined covered walkway crossing
a lagoon.

 The Walt Disney World Swan and Dolphin is a joint venture between the Walt Disney Company, Tishman Hotel Corporation, MetLife and Starwood Hotels and Resorts.

 The land the resort occupies is owned by the Walt Disney Company, while the buildings themselves are leased by Disney to the Tishman Hotel Corporation and
MetLife but operated by Starwood Hotels & Resorts Worldwide under the Sheraton Hotels brand.

 The Walt Disney World Swan and Dolphin are a part of the Walt Disney Collection of resorts; because of this they are Disney branded and guests of the resort
have access to special Disney benefits available to Disney Resort Hotel guests only.
</item 1>

<item 2>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 2>

<item 3>
Millennium Force
Millennium Force is a steel roller coaster built by Intamin at Cedar Point amusement park in Sandusky, Ohio, United States.

 It was the fourteenth roller coaster to be built at the park since Blue Streak opened in 1964.

 Upon completion in 2000, Millennium Force broke six world records and was the world's first Giga Coaster, a roller coaster that exceeds 300 ft in height and
completes a full circuit.

 It was briefly the tallest and fastest in the world until Steel Dragon 2000 opened later the same year.

 The ride is also the third-longest roller coaster in North America after The Beast at Kings Island and Fury 325 at Carowinds.

 It was the first roller coaster to use a cable lift system rather than a traditional chain lift.

 The coaster has a 310 ft , 45-degree lift hill with a 300 ft drop and features two tunnels, three overbanked turns, and four hills.

 It has a top speed of 93 mph .
</item 3>

<item 4>
Walt Disney World Quest: Magical Racing Tour
Walt Disney World Quest: Magical Racing Tour is a 2000 go-kart racing video game based on attractions at the Walt Disney World Resort.

 Players compete in races on tracks inspired by attractions such as the Haunted Mansion, Pirates of the Caribbean, Space Mountain and Big Thunder Mountain
Railroad to acquire missing parts for the park's fireworks machine, which was accidentally destroyed by Chip 'n' Dale while they were gathering acorns.

 The game was developed by Crystal Dynamics and published by Eidos Interactive.

 The Game Boy Color version was developed by Prolific Publishing.
</item 4>

<item 5>
Primeval Whirl
Primeval Whirl is a steel wild mouse roller coaster at Disney's Animal Kingdom at the Walt Disney World Resort.

 The ride is a spinning roller coaster purchased from Reverchon Industries.

 The ride is part of Chester and Hester's Dino-Rama, which is itself part of DinoLand U.S.A. It is a roller coaster in the "Mild But Wild Thrills" category.

 It has cars that spin in circles while traveling on tracks, permitting the ride experience to vary greatly each time it is ridden.

 The ride features 13 cars, each seating up to 4 riders.
</item 5>

<item 6>
Space Mountain (Disneyland)
Space Mountain (known as Hyperspace Mountain from 2015 to 2017) is an indoor, space-themed roller coaster in Tomorrowland at Disneyland in Anaheim, California.

 Opened on May 27, 1977, it was the second roller coaster built at Disneyland, and was the second of the five versions of Space Mountain built by The Walt
Disney Company.

 Its exterior façade is one of Disneyland's four "mountain" structures that serve as park landmarks.
</item 6>

<item 7>
Rock 'n' Roller Coaster Starring Aerosmith
Rock 'n' Roller Coaster Starring Aerosmith is an enclosed launched steel roller coaster at Disney's Hollywood Studios at the Walt Disney World Resort and at the
Walt Disney Studios Park in Disneyland Paris (where it is called "Rock 'n' Roller Coaster avec Aerosmith").

 As the attraction's name suggests, the coaster features Aerosmith members, Steven Tyler, Joe Perry, Tom Hamilton, Joey Kramer, and Brad Whitford.
</item 7>

<item 8>
Best Western Lake Buena Vista Resort Hotel
The Best Western Lake Buena Vista Resort Hotel is a 325-room resort that is on the property of Walt Disney World Resort in Lake Buena Vista, Florida.

 The resort is located across from the Disney Springs area.

 The hotel originally opened in November 1972 as the Travelodge at Lake Buena Vista.

 It was operated by Travelodge from 1972 through 1983, then in 1984 the resort became the Viscount Hotel and operated as such until 1988.

 In 1989, it was reacquired by Travelodge.

 In 2000, the Travelodge hotel switched over to become the Best Western Lake Buena Vista Resort Hotel.

 The hotel was renovated in 2004 and has 2 outdoor pools.
</item 8>

<item 9>
Space Mountain (Magic Kingdom)
Space Mountain is an indoor dark outer space-themed steel roller coaster at the Magic Kingdom theme park at the Walt Disney World Resort in Lake Buena Vista,
Florida.

 Opened on January 15, 1975, Space Mountain is the oldest operating roller coaster in the state of Florida, and is the original version of the iconic attraction
that has since been replicated at all of The Walt Disney Company's Magic Kingdom-style theme parks worldwide, except for the Shanghai Disneyland Resort.

 RCA helped fund Space Mountain's construction and sponsored the ride from 1975 to 1993; FedEx sponsored Space Mountain from 1994 to 2004.
</item 9>

<item 10>
Space Mountain
Space Mountain is the name of a space-themed indoor roller coaster attraction located at five of the Magic Kingdom-style Disney Parks (all except for Shanghai
Disneyland Park).

 Although all five versions of the attraction are different in nature, all have a similar domed exterior façade that is a landmark for the respective park.

 The first Space Mountain ride opened in 1975 at Walt Disney World Resort in Florida, and similar attractions were then eventually built at the other Disney
parks.
</item 10>
=== END ITEM id=4513ffaf38737d87 ===

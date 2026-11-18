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

=== ITEM id=b2080bba53aff01b ===
Gold sentences and facts:
Sentence 0: Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.
  Fact 0.0: Semyon Aronovich Gershgorin was born on August 24, 1901.
  Fact 0.1: Semyon Aronovich Gershgorin died on May 30, 1933.
  Fact 0.2: Semyon Aronovich Gershgorin was Soviet.
  Fact 0.3: Semyon Aronovich Gershgorin was born in Pruzhany, Belarus, Russian Empire.
  Fact 0.4: Semyon Aronovich Gershgorin was a mathematician.
Sentence 1:  He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after him.
  Fact 1.0: Semyon Aronovich Gershgorin was a mathematician.
  Fact 1.1: Semyon Aronovich Gershgorin was noted for his work on the decidability of various algebraic groups.
  Fact 1.2: Malcev algebras are generalisations of Lie algebras.
  Fact 1.3: Malcev algebras are named after Semyon Aronovich Gershgorin.

Retrieved items:
<item 1>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 1>

<item 2>
A diverse collection profiles artists, performers, filmmakers, writers, scientists, and political figures across Russian, Soviet, Swedish, Australian, and other
cultures. Topics include AES+F, Lev Yilmaz, Lev Kuleshov’s montage theory, Hermann Brunn’s mathematics, and Russian revolutionaries and activists. Film entries
span silent-era landmarks, Soviet cinema and animation, and Hollywood, featuring figures such as Greta Garbo, Mauritz Stiller, Lois Weber, and Gene Hackman.
</item 2>

<item 3>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 3>

<item 4>
Lev Kuleshov (1899–1970), Soviet filmmaker and theorist, pioneered Soviet montage, the Kuleshov effect, and creative geography. Hermann Brunn (1862–1939),
German mathematician, contributed to convex geometry and knot theory. Mercury’s 165-km Chaikovskij crater honors composer Pyotr Tchaikovsky; a 1970 Soviet
biopic about him earned two Oscar nominations. Søren Kierkegaard (
</item 4>

<item 5>
The entries profile Russian and Soviet figures across science, arts, politics, and activism: entomologist and prolific author Pavel Marikovsky; film director
Aleksandr Ivanovsky; animator Vladimir Danilevich; actors and collectors Ekaterina and Vladimir Semenikhin; and actor Aleksei Petrenko. They also describe the
anti-tsarist Petrashevsky Circle and member Nikolay Speshnev, plus Marxist revolutionaries, thinkers, and officials including Bazarov, Bag
</item 5>

<item 6>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 6>

<item 7>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 7>

<item 8>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 8>

<item 9>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 9>

<item 10>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 10>
=== END ITEM id=b2080bba53aff01b ===

=== ITEM id=f0c364f8d29975fe ===
Gold sentences and facts:
Sentence 0:  Her favorite events are 800 metres and 1500 metres.
  Fact 0.0: This person's favorite events are the 800 metres.
  Fact 0.1: This person's favorite events are the 1500 metres.
Sentence 1:  The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.
  Fact 1.0: The distance has been contested at the Summer Olympics since 1896.
  Fact 1.1: The distance has been contested at the World Championships in Athletics since 1983.

Retrieved items:
<item 1>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 1>

<item 2>
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
</item 2>

<item 3>
Nicolas N’Godrela won three tennis golds at the 2015 Pacific Games. Aerobic gymnast Remus Nicolai earned 11 world and European medals, then coached in Romania.
Brett McClure won Olympic team silver and is USA Gymnastics’ men’s High Performance Director. Masayoshi Yamamoto won 2014 Asian Games pommel horse gold. Louis
Smith won five Olympic medals across three Games and a 2015 European pommel horse title. Luke
</item 3>

<item 4>
Sinimole Paulose
Sinimol Paulose (born 24 June 1983) is an Indian middle distance runner.

 Her favorite events are 800 metres and 1500 metres.

 In 2009, she received Arjuna Award for her contribution to the Indian athletics.

 Sinimol was born in Piravom in Ernakulam district of Kerala state.

 She represented India at Beijing Olympics.
</item 4>

<item 5>
K. M. Binu
Kalayathumkuzhi Mathews Binu (born 20 December 1980) is an Indian track and field athlete from Kerala who specializes in 400 metres and 800 metres.

 He holds the current 400 metres national record of 45.48 s set at the 2004 Athens Olympics on 20 August 2004.

 He broke the 44-year-old Olympics mark (by an Indian) held by Milkha Singh who set an Indian National Record with a timing of 45.73 s at the 1960 Rome
Olympics.

 He and his elder sister K. M. Beenamol made history when they became the first Indian siblings to win medals in a major international competition.

 They won medals at the Busan Asian Games (2002).

 While Binu won the men's 800 metres silver, his sister won the gold medal in the women's event.

 Binu received the Arjuna Award for the year 2006 for his achievements in the Indian athletics.
</item 5>

<item 6>
The European 10,000m Cup, organized annually by European Athletics since 1997, grew from the Iberian championships and was renamed in 2005. Track 10,000m races
cover 25 laps; the 1500m is a major Olympic middle-distance event. Charlottesville’s annual Bill Steers Men’s 4-Miler, held since 2004, promotes men’s health.
European under-20 championships evolved from junior games (1964, 1968)
</item 6>

<item 7>
1500 metres
The 1500 metres or 1,500-metre run (typically pronounced 'fifteen-hundred metres') is the foremost middle distance track event in athletics.

 The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.

 It is equivalent to 1.5 kilometers or approximately ⁄  miles.
</item 7>

<item 8>
Siegmar Ohlemann
Siegmar ("Sig") Ohlemann (born May 11, 1938) is a retired middle distance runner, who represented Canada in three events (800 metres, 4 × 100 metres and 4 × 400
metres) at the 1960 Summer Olympics in Rome, Italy.

 He claimed the silver medal in 800 metres at the 1963 Pan American Games in Brazil, behind fellow Canadian Don Bertoia.
</item 8>

<item 9>
Canadian athletes Ross MacKenzie and Brian MacLaren earned 1966 Commonwealth and 1967 Pan American relay silver; both competed in Olympic 400m. Guyanese Harry
Prowell set a marathon record and remains Guyana’s sole Olympic marathoner. Canadian miler Dave Bailey broke four minutes and won Pan Am bronze; Siegmar
Ohlemann won Pan Am 800m silver. Geri Donnelly and China’s Liu Ailing were World Cup footballers; Liu also
</item 9>

<item 10>
John Lindsay (Paralympian)
John Lindsay, OAM (born 29 January 1970) is an Australian Paralympic athlete from Melbourne.

 He competed in the 1988 Seoul games in distances ranging from 100 m to 800 m, but did not win any medals.

 At the 1992 Barcelona Games, he won a gold medal in the Men's 200 m TW3 event, for which he received a Medal of the Order of Australia, a silver medal in the
Men's 100 m TW3 event and a bronze medal in the Men's 400 m TW3 event.

 That year, he had a Victorian Institute of Sport scholarship.
</item 10>
=== END ITEM id=f0c364f8d29975fe ===

=== ITEM id=4a380b82bea459e4 ===
Gold sentences and facts:
Sentence 0: St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.
  Fact 0.0: St. Elizabeths Hospital opened in 1855.
  Fact 0.1: St. Elizabeths Hospital opened as the first federally operated psychiatric hospital in the United States.
Sentence 1:  Founded in 1861, it is the longest continuously operating hospital in the District.
  Fact 1.0: St. Elizabeths Hospital was founded in 1861.
  Fact 1.1: St. Elizabeths Hospital is the longest continuously operating hospital in the District.

Retrieved items:
<item 1>
History of psychiatric hospitals in Pittsburgh
Pittsburgh, Pennsylvania, United States, is home to a plethora of hospitals, be it for the physically ill or, what was defined as the insane and feeble minded.

 In 1891, the St. Margaret Memorial Hospital opened under John H. Shoenberger.

 In 1908, the Pittsburgh Hospital was run under C. A. Fagan, and on June 1, 1889, the South Side Hospital opened under Anna M. Rindlaub.

 Pittsburgh, commonly known as the "Steel City," also built up a reputation for its abundance of hospitals that had opened up within a fairly close time period
of each other.

 During this time, many new mental hospitals were established due to overcrowding.

 In 1899, a building to house the mentally insane, Mayview State Hospital, was constructed; but it was not long before the complex had a reputation “as a place
of sorrow” due to the lack of proper health care patients received during their stay.
</item 1>

<item 2>
St Anne Within the Liberty of Westminster
St Anne Within the Liberty of Westminster, also known as St Anne Soho, was a civil parish in the metropolitan area of London, England.

 The creation of the parish accompanied the building of St Anne's Church, Soho to meet the demands of the growing population.

 The parish was formed in 1687 from part of the ancient parish of St Martin in the Fields in the Liberty of Westminster and county of Middlesex.

 It included the eastern section of the contemporary districts of Soho to the north of Shaftesbury Avenue and Chinatown to the south of it.

 Initially controlled by a select vestry, the parish was governed by an open vestry of all inhabitants until 1855, when the vestry was superseded for most
purposes by the Strand District Board of Works.

 In 1889 the parish became part of the County of London and in 1900 the local authority became Westminster City Council.

 The parish continued to have nominal existence until 1922.
</item 2>

<item 3>
Heritage Hospital
Heritage Hospital is a hospital located in Tarboro, North Carolina.

 It is a part of the University Health Systems of Eastern Carolina (UHSEC).

 Edgecombe General Hospital opened as a county-owned hospital in 1916.

 It succeeded Pittman Hospital, which opened in 1901.

 In 1959, the Hill-Burton Act helped combine Edgecombe General Hospital, with three other facilities.

 Edgecombe County sold the hospital to Hospital Corporation of America (HCA) in 1982.

 HCA opened a 127-bed facility in 1985, named Heritage Hospital.

 UHSEC bought Heritage Hospital in 1998 from HCA.

 The hospitals focus is as a community hospital.

 The hospital has 101 general and 16 rehabilitation hospital beds.

 It has five Shared Inpatient/Ambulatory Surgery, two Endoscopy, and one C-Section operating rooms.
</item 3>

<item 4>
St. Elizabeths Hospital
St. Elizabeths Hospital opened in 1855 as the first federally operated psychiatric hospital in the United States.

 Housing over 8,000 patients at its peak in the 1950s, the hospital at one point had a fully functioning medical-surgical unit, a school of nursing, and
accredited internships and psychiatric residencies.

 Its campus was designated a National Historic Landmark in 1990.
</item 4>

<item 5>
Douglass (Washington, D.C.)
Douglass is a residential neighborhood in Southeast Washington, D.C., on the eastern side of St. Elizabeths Hospital, on the border of the Congress Heights
Metro Station.

 It is bounded by Suitland Parkway to the north and east, Alabama Avenue to the south, and the St. Elizabeths campus to the west.
</item 5>

<item 6>
St. Mary's Church (Swormville, New York)
St. Mary's Catholic Church is a church located on the corner of Transit (New York State Route 78) and Stahley roads in Swormville, New York, in the United
States.

 Originally known as St. Mary’s Church of the Assumption at Transit, the building was constructed of about 260,000 bricks.

 John Nepomucene Neumann (later to be named St. John Neumann) started to visit Swormville, celebrating mass in local homes, barns and fields.

 In 1839 he instructed that a small house would be made to hold mass.

 Bishop John Timon established the first catholic church (St. Mary's) of the town of Clarence, New York.

 In 1861, Father Michael Schinabeak made plans for a church to be constructed.

 Construction began in 1862 and finished in fall of 1865.

 The official opening of the Church was in January 1866.

 In 2010 a new Church was constructed behind the old St. Mary's.
</item 6>

<item 7>
Loreta Janeta Velázquez
Loreta Janeta Velázquez (June 26, 1842 – 1923), was a Cuban-born woman who masqueraded as a male Confederate soldier during the American Civil War.

 After her soldier husband's accidental death, she enlisted in the Confederate States Army in 1861.

 She then fought at Bull Run, Ball's Bluff, and Fort Donelson, but was discharged when her gender was discovered while in New Orleans.

 Undeterred, she reenlisted and fought at Shiloh, until unmasked once more.

 She then became a Confederate spy, working in both male and female guises, and as a double agent also reporting to the U.S. Secret Service.

 She remarried three more times, being widowed in each instance.

 According to William C Davis, she died in January 1923 under the name Loretta J. Beard after many years away from the public eye in a public psychiatric
facility, St. Elizabeths Hospital.

 She spied on the Union for about 5 years.
</item 7>

<item 8>
Providence Hospital (Washington, D.C.)
Providence Hospital is a 408 bed hospital located in the District of Columbia.

 Founded in 1861, it is the longest continuously operating hospital in the District.

 Providence Hospital is a member of Ascension Health, the largest non-profit health care organization in the United States.
</item 8>

<item 9>
Winfred Overholser
Winfred Overholser (1892 – October 6, 1964) was an American psychiatrist, president of the American Psychiatric Association, and for 25 years the superintendent
of St. Elizabeths Hospital, a federal institution for the mentally ill in Washington, D.C.
</item 9>

<item 10>
Congress Heights station
Congress Heights is an island platformed Washington Metro station in the Congress Heights neighborhood of Washington, D.C., United States.

 The station was opened on January 13, 2001, and is operated by the Washington Metropolitan Area Transit Authority (WMATA).

 Providing service for only the Green Line, the station is located at Alabama Avenue and 13th Street, lying under St. Elizabeths Hospital.

 Congress Heights is the last Green Line station in the District of Columbia going southeast.
</item 10>
=== END ITEM id=4a380b82bea459e4 ===

=== ITEM id=7dda7778e42aa4a8 ===
Gold sentences and facts:
Sentence 0: Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
  Fact 0.0: Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book.
  Fact 0.1: Lawrence Weschler wrote Mr. Wilson's Cabinet of Wonder.
  Fact 0.2: Mr. Wilson's Cabinet of Wonder is primarily about the Museum of Jurassic Technology.
  Fact 0.3: Mr. Wilson's Cabinet of Wonder is more broadly about the history and role of museums.
Sentence 1: The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.
  Fact 1.0: The Museum of Jurassic Technology is a museum.
  Fact 1.1: The Museum of Jurassic Technology is located at 9341 Venice Boulevard.
  Fact 1.2: The Museum of Jurassic Technology is in the Palms district of Los Angeles, California.

Retrieved items:
<item 1>
- Glacier National Park Fund (1999, West Glacier) funds park experience, education, ecosystem research and Centennial projects; supported bear-aversion research
and a $10,000 bighorn sheep study; mascot Billy Bowman.
- Los Angeles’s Museum of Jurassic Technology was founded in 1988 by David and Diana Wilson.
- Snowkiting uses kites to ski or snowboard across snow and ice, including uphill travel; it’s growing worldwide but can be hazardous.
- Downtown Grand
</item 1>

<item 2>
Gate of Ivory, Gate of Horn
Gate of Ivory, Gate of Horn is a fantasy novel by British author Robert Holdstock.

 It was originally published in the United States in 1997 (and in the United Kingdom under the title "Gate of Ivory" in 1998.)

 The story is a prequel to "Mythago Wood" and explores Christian Huxley's quest into Ryhope Wood and the apparent suicide of his mother, Jennifer Huxley.

 The title of the book refers to the gates of horn and ivory described in both Homer's "Odyssey" and Virgil's "Aeneid".
</item 2>

<item 3>
Mr. Wilson's Cabinet of Wonder
Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily
about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
</item 3>

<item 4>
Merlin's Wood
Merlin's Wood; or, The Vision of Magic is a short novel by British writer Robert Holdstock, first published in the United Kingdom in 1994.

 The novel is considered part of the "Mythago Wood" cycle, but takes place in Brittany, France instead of Herefordshire, England.

 The work has all new characters and focuses on the mythical birthplace and burial site of Merlin, the magical wood Brocéliande.

 Brocéliande is a smaller version of Ryhope wood where British myth predominates.
</item 4>

<item 5>
A wide-ranging reference collection spans animals, food, forests, film, television, animation, games, music, theatre, and literature. It notes Wire Fox Terrier
Signal Circuit of Halleston’s 1926 Westminster win, the show’s 50th, and the breed’s fourth Best in Show; covers terriers, hounds, dogs, cheese, and wine; and
profiles many works and creators, from Batman, Superman, Disney, Pixar, and *Adventure Time* to classic films
</item 5>

<item 6>
Museum of Jurassic Technology
The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.

 It was founded by David Hildebrand Wilson and Diana Drake Wilson (husband and wife) in 1988.
</item 6>

<item 7>
John G. Wilson
John Guy Wilson (1842–1892) was a Manchester, UK patent agent and the founder of Wilson Gunn, one of the oldest firms of Patent and Trade Mark Attorneys in the
UK.

 John G. Wilson built up the business until his death in the 1890s.

 As a side line he acted as agent to the Alliance Insurance Company.

 There is no record that Mr. Wilson ever became a member of the Chartered Institute of Patent Agents.
</item 7>

<item 8>
Primeval Whirl
Primeval Whirl is a steel wild mouse roller coaster at Disney's Animal Kingdom at the Walt Disney World Resort.

 The ride is a spinning roller coaster purchased from Reverchon Industries.

 The ride is part of Chester and Hester's Dino-Rama, which is itself part of DinoLand U.S.A. It is a roller coaster in the "Mild But Wild Thrills" category.

 It has cars that spin in circles while traveling on tracks, permitting the ride experience to vary greatly each time it is ridden.

 The ride features 13 cars, each seating up to 4 riders.
</item 8>

<item 9>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 9>

<item 10>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 10>
=== END ITEM id=7dda7778e42aa4a8 ===

=== ITEM id=845b9be1c99bb573 ===
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
Ural Federal University
The links between the university and intermediate educational institutions are actively developed.

 The University also performs consistent work on establishing strategic partnership with employers for the sake of development of the regional economy:
employers are involved in the educational environment as direct participants, large-scale enployers-sponsored and corporate education is carried out based on
the client-centered approach, joint business projects in economic and social areas are realized.
</item 1>

<item 2>
Cal Poly Pomona Broncos
The Cal Poly Pomona Broncos or Cal Poly Broncos are the athletic sports teams for the California State Polytechnic University, Pomona (Cal Poly Pomona).

 Cal Poly Pomona has 10 varsity sports teams and offers student participation in a wide range of sports including soccer, volleyball, track and field,
basketball, softball, and baseball.

 Cal Poly Pomona participates at the NCAA's Division II (DII) level in the California Collegiate Athletic Association (CCAA).

 The Broncos have 65 CCAA championships and 14 NCAA National Championships.

 Current and former Cal Poly athletes have won 7 Olympic medals (3 gold, 1 silver, and 3 bronze).
</item 2>

<item 3>
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
</item 3>

<item 4>
Cal Poly Pomona College of Environmental Design
The Cal Poly Pomona College of Environmental Design (CENV) is a college part of the California State Polytechnic University, Pomona (Cal Poly Pomona).

 The college houses over 1,600 students; making it one of largest environmental design programs in the United States.

 The college offers bachelor's degrees in five departments, as well as three master's degree programs.

 It is the only academic unit within the California State University system to be associated with a Pritzker Prize laureate (often referred to as "The Nobel
Prize in Architecture").
</item 4>

<item 5>
Ural Federal University
The Ural Federal University named after the first President of Russia B. N. Yeltsin (Russian: Уральский федеральный университет имени первого Президента России
Б.Н.

 Ельцина, "Uralʹskiĭ federalʹnyĭ universitet imeni pervogo Prezidenta Rossii B.N. Yelʹtsina", often shortened to UrFU, УрФУ ) (formed by a merger of the Ural
State Technical University and Ural State University) is one of the leading educational institutions in the Ural region.

 Ural Federal University acts as a research and innovation center of the Ural region and has close cooperation with the Russian Academy of Sciences.

 Training of students is carried out in four main areas of knowledge and 108 academic majors.

 The links between the university and intermediate educational institutions are actively developed.
</item 5>

<item 6>
California State Polytechnic University, Pomona
California State Polytechnic University, Pomona (CPP, Cal Poly Pomona, or Cal Poly) is a public polytechnic university located in Pomona, California, United
States.

 It is one of two polytechnics in the California State University system.
</item 6>

<item 7>
Cal Poly Pomona Broncos women's basketball
The Cal Poly Pomona Broncos women's basketball team is the women's basketball team that represents California State Polytechnic University, Pomona, in Pomona,
California.

 The school's team currently competes in the California Collegiate Athletic Association.
</item 7>

<item 8>
Cal Poly Pomona University Library
The Cal Poly Pomona University Library is the main library on the campus of California State Polytechnic University, Pomona (Cal Poly Pomona).

 It has in its collection 670,580 books; 6,883 serial subscriptions and 10,417 audiovisual materials.
</item 8>

<item 9>
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
</item 9>

<item 10>
1981 Cal Poly Pomona Broncos football team
The 1981 Cal Poly Pomona Broncos football team represented California State Polytechnic University, Pomona during the 1981 NCAA Division II football season.

 Cal Poly Pomona competed in the California Collegiate Athletic Association (CCAA).
</item 10>
=== END ITEM id=845b9be1c99bb573 ===

=== ITEM id=a4c951de0e1578fb ===
Gold sentences and facts:
Sentence 0:  It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
  Fact 0.0: The area occupies 40 km (15 sq mi).
  Fact 0.1: The area extends into each of the six municipalities on the island.
Sentence 1: Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the fifth largest in Spain's national parks system.
  Fact 1.0: Sierra de Guadarrama National Park is a national park in Spain.
  Fact 1.1: The Spanish name of Sierra de Guadarrama National Park is Parque Nacional de la Sierra de Guadarrama.
  Fact 1.2: Sierra de Guadarrama National Park covers nearly 34,000 hectares.
  Fact 1.3: Sierra de Guadarrama National Park is the fifth largest in Spain's national parks system.

Retrieved items:
<item 1>
North Cascades National Park
North Cascades National Park is a U.S. National Park located in the state of Washington.

 The park is the largest of the three National Park Service units that comprise the North Cascades National Park Service Complex.

 Several national wilderness areas and British Columbia parkland adjoin the National Park.

 The park features rugged mountain peaks and protects portions of the North Cascades range.
</item 1>

<item 2>
Golf La Moraleja
Golf La Moraleja is one of Spain’s most important clubs.

 It is the largest Spanish golf club for its number of its holes and the largest in Europe located in a metropolitan area.

 Founded in 1973, it currently has four 18-hole golf courses, all rated par 72, designed by Jack Nicklaus.

 It also contains a 9-hole short course, 8 tennis courts, 12 paddle tennis courts, 2 squash courts, one covered and three open-air swimming pools, a gym and a
spa.

 Added to these are two clubhouses, a children’s chalet and a tennis clubhouse (on 1 and 2 courses), plus a prebuilt modular clubhouse for courses 3 and 4.

 Courses 1 and 2 are located at La Moraleja residential estate (Alcobendas, Madrid).

 Courses 3 and 4 are located in the municipality of Algete.

 It has 6,000 shareholder members.
</item 2>

<item 3>
Gates of the Arctic National Park and Preserve
Gates of the Arctic National Park and Preserve is a U.S. National Park in Alaska.

 It is the northernmost national park in the U.S. (the entirety of the park lies north of the Arctic Circle) and the second largest at 8,472,506 acres ,
slightly larger in area than Belgium.

 The park consists primarily of portions of the Brooks Range of mountains.

 It was first protected as a U.S. National Monument on December 1, 1978, before becoming a national park and preserve two years later in 1980 upon passage of
the Alaska National Interest Lands Conservation Act.

 A large part of the park is protected in the Gates of the Arctic Wilderness which covers 7,167,192 acre .

 The wilderness area adjoins the Noatak Wilderness Area and together they form the largest contiguous wilderness in the United States.
</item 3>

<item 4>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 4>

<item 5>
Pirin National Park
Pirin National Park (Bulgarian: Национален парк "Пирин" ) is a national park that encompasses the larger part of the Pirin Mountains in south-west Bulgaria,
spanning an area of 403.56 km2 .

 It is one of the three national parks in the country, the others being Rila National Park and Central Balkan National Park.

 The park was established in 1962 and its territory was expanded several times since then.

 Pirin National Park was declared a UNESCO World Heritage Site in 1983.

 The altitude varies from 950 m to 2,914 m at Vihren, Bulgaria's second highest summit and the Balkans' third.
</item 5>

<item 6>
Badiar National Park
Badiar National Park (French: "Parc National du Badiar" ) is a national park in Guinea, on the border with Senegal and contiguous with Senegal's much larger
Niokolo-Koba National Park.

 It was established on 30 May 1985 (by ordonnance N°124/PRG/85), partly in response to Senegal's concern about poaching in Niokolo-Koba National Park.

 Badiar is an International Union for Conservation of Nature Category II park.
</item 6>

<item 7>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 7>

<item 8>
Landes de Gascogne Regional Natural Park
Landes de Gascogne Regional Natural Park (French: "Parc naturel régional des Landes de Gascogne") is a protected area of pine forest, wetland and oceanic
coastline located in the Aquitaine region of southwestern France.
</item 8>

<item 9>
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
</item 9>

<item 10>
Guadarrama National Park
Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the
fifth largest in Spain's national parks system.

 The Guadarrama mountain range ("Sistema Central") contains some ecologically valuable areas, located in the Community of Madrid and Castile and León (provinces
of Segovia and Ávila).

 The law that regulates the recently approved national park was published in the BOE in  26, 2013 (2013--) .
</item 10>
=== END ITEM id=a4c951de0e1578fb ===

=== ITEM id=787d214ea2df2c4d ===
Gold sentences and facts:
Sentence 0: "For the sedan variant of the car, see Volvo S40."
  Fact 0.0: The sedan variant of the car is described under Volvo S40.
Sentence 1: The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
  Fact 1.0: The Volvo S40 is a compact luxury car.
  Fact 1.1: The Volvo S40 is marketed by Volvo Cars.
  Fact 1.2: The Volvo S40 is produced by Volvo Cars.
  Fact 1.3: Volvo Cars is a Swedish manufacturer.
  Fact 1.4: The Volvo S40 was marketed and produced from 1995 to 2012.
  Fact 1.5: The Volvo S40 spanned two generations.

Retrieved items:
<item 1>
Volvo 300 Series
The Volvo 300 Series is a rear wheel drive small family car sold as both a hatchback and (later) a conventional saloon from 1976 to 1991.

 It was launched in the Netherlands shortly after Volvo acquired a major stake in the passenger car division of DAF in 1973.

 The series consisted of the Volvo 340 (previously 343/345) and the later Volvo 360.
</item 1>

<item 2>
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 2>

<item 3>
Volvo 440/460
The Volvo 440 and 460 are versions of a small family car produced by the Swedish manufacturer Volvo between 1987 and 1996.

 The 440 was introduced in 1987, whilst the 460 followed in 1989.

 They were built at the NedCar factory in Born, The Netherlands.

 The 440 was a five door front wheel drive hatchback and the 460 a saloon.

 They shared many components with the already successful Volvo 480, including floorpan, front and rear suspension, engines, transmission and braking systems.
</item 3>

<item 4>
Volvo 480
The Volvo 480 is a compact car that was produced in Born, Netherlands, by Volvo from 1986 to 1995.

 It was the first front-wheel drive car made by the automaker.

 The 480 was available in only one body style on a automobile platform related to the Volvo 440/460 five door hatchback and four door sedan models.
</item 4>

<item 5>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 5>

<item 6>
Volvo C30
The Volvo C30 is a near-luxury compact hatchback, manufactured and marketed by Volvo Cars for model years 2006-2013, available across its single generation as a
three-door hatchback.

 Powered by inline-four and straight-five engines, the C30 is variant of the Volvo S40/V50/C70 range, sharing the same Ford C1/Volvo P1 platform.

 Volvo marketed the C30 as a premium hatchback / sports coupe.
</item 6>

<item 7>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 7>

<item 8>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 8>

<item 9>
Volvo V50
"For the sedan variant of the car, see Volvo S40."
</item 9>

<item 10>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 10>
=== END ITEM id=787d214ea2df2c4d ===

=== ITEM id=f9db86f91a28223d ===
Gold sentences and facts:
Sentence 0: Edward Sharpe and the Magnetic Zeros is an American folk rock band formed in Los Angeles, California in 2007.
  Fact 0.0: Edward Sharpe and the Magnetic Zeros is an American folk rock band.
  Fact 0.1: Edward Sharpe and the Magnetic Zeros was formed in Los Angeles, California.
  Fact 0.2: Edward Sharpe and the Magnetic Zeros was formed in 2007.
Sentence 1: Louis XIV is an American rock band from San Diego, California.
  Fact 1.0: Louis XIV is an American rock band.
  Fact 1.1: Louis XIV is from San Diego, California.
Sentence 2:  The band has released four EPs between 2003 and 2007, and three albums between 2003 and 2008, the latter two of which were distributed by Atlantic Records.
  Fact 2.0: The band released four EPs between 2003 and 2007.
  Fact 2.1: The band released three albums between 2003 and 2008.
  Fact 2.2: Two of the band's three albums, the latter two, were distributed by Atlantic Records.

Retrieved items:
<item 1>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 1>

<item 2>
PersonA
PersonA (pronounced "Persona") is the fourth studio album by the band Edward Sharpe and the Magnetic Zeros.

 It was released on April 15, 2016.
</item 2>

<item 3>
Leeds band Vib Gyor took its name from rainbow colors ordered by wavelength. Its debut single “Fallen” (July 2006) gained acclaim and radio play. The Secret EP
(March 2007), recorded in Bath by Paul Corkett, includes a hidden instrumental; The White EP followed in November, recorded at Rockfield by Ken Thomas. Leeds
alternative-rock band Distortion Mirrors formed in 2011. Other releases include Edward Sharpe and the Magnetic Zeros’ 201
</item 3>

<item 4>
The collection surveys music albums, artists, songs, and genres across rock, pop, metal, and electronic music. Highlights include Patty Griffin’s 2004 album
*Impossible Dream*; Edward Sharpe and the Magnetic Zeros’ 2013 album and 2010 single; the White Stripes’ rise, breakup, acclaimed *Elephant* (2003), and
Grammy-winning “Seven Nation Army”; and biographies of Meg White and Jack White. Other entries cover releases by Steve Winwood
</item 4>

<item 5>
Lingdian
Zero Band () are a Chinese rock band from Inner Mongolia.

 The lead singer is Zhou Xiaoou (周晓鸥).

 The English name "Zero Band" was used in Beijing concert advertisements, though the band did not originally use an English name; the name has also been
translated as "Point Zero" in some older English publications.

 The band appeared on the Jingwen label in China, but were licensed outside China by JVC.

 One of the top Beijing bands at the end of the 1990s, Lingdian originally suffered the same resistance from government-owned venues as other indigenous PRC
rock bands.
</item 5>

<item 6>
Edward Sharpe and the Magnetic Zeros
Edward Sharpe and the Magnetic Zeros is an American folk rock band formed in Los Angeles, California in 2007.

 The group is led by lead singer Alex Ebert.

 The band's name is based on a story Ebert wrote in his youth, about a messianic figure named Edward Sharpe.

 Drawing from roots rock, folk, gospel, and psychedelic music, the band's image and sound evoke the hippie movement of the 1960s and 1970s.

 The group's first show was played July 18, 2007 at The Troubadour in West Hollywood, California.

 Their first studio album, "Up from Below", was released on July 7, 2009 on Community Records and featured the popular single "Home".

 The group released their second full-length album, "Here", on May 29, 2012, and third album, "Edward Sharpe and the Magnetic Zeros", on July 23, 2013.

 Their fourth studio album, "PersonA", was released in April 2016.
</item 6>

<item 7>
Louis XIV (band)
Louis XIV is an American rock band from San Diego, California.

 The band has released four EPs between 2003 and 2007, and three albums between 2003 and 2008, the latter two of which were distributed by Atlantic Records.

 The band broke up in 2009, but in 2013, in an interview with The Reno Dispatch, Jason Hill confirmed that the band had decided to reunite.
</item 7>

<item 8>
Alexander (Alex Ebert album)
Alexander is the first solo album by American musician Alex Ebert, lead singer of Edward Sharpe and the Magnetic Zeros and Ima Robot.

 It was released on March 1, 2011 under the Community Records label.

 The song "Truth" was featured in the season 4 premiere of AMC's "Breaking Bad" titled "Box Cutter".
</item 8>

<item 9>
Edward Sharpe and the Magnetic Zeros (album)
Edward Sharpe and the Magnetic Zeros is the third album by Edward Sharpe and the Magnetic Zeros.

 It was released on July 23, 2013 in North America and was released on July 29, 2013 around the world through Vagrant Records, Rough Trade Records and Communion
Records.

 Frontman Alex Ebert stated that "These songs mean everything to me - It's the rawest, most liberated, most rambunctious stuff we've done."
</item 9>

<item 10>
Orpheo McCord
Orpheo McCord (born 1979) is an American drummer and percussionist.

 He is an original member of Edward Sharpe and the Magnetic Zeros, and a former member of The Fall, Fool's Gold and Cass McCombs.
</item 10>
=== END ITEM id=f9db86f91a28223d ===

=== ITEM id=83f1f3fec2f7ea20 ===
Gold sentences and facts:
Sentence 0: Rivermaya is a Filipino rock band.
  Fact 0.0: Rivermaya is a Filipino rock band.
Sentence 1: Kodaline ( ) are an Irish band.
  Fact 1.0: Kodaline is an Irish band.

Retrieved items:
<item 1>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 1>

<item 2>
Catfish and the Bottlemen
Catfish and the Bottlemen are a British rock band, formed in Llandudno, Wales in 2007.

 Their debut album, "The Balcony", reached number 10 in the UK Albums Chart and achieved Platinum status on 30 December 2016.

 The band have toured in South America, Japan, UK, Europe, North America, and Australia and have featured in a number of festivals including Lollapalooza,
Glastonbury, Latitude, Community festival, Reading and Leeds, T in the Park, Governors Ball and Bonnaroo.

 They won a Brit Award for British Breakthrough act on 24 February 2016.

 On 27 May 2016, they released their second album, "The Ride".
</item 2>

<item 3>
Kodaline
Kodaline ( ) are an Irish band.

 Originally known as 21 Demands, in 2012 the band changed their name to Kodaline.
</item 3>

<item 4>
Honest (Kodaline song)
"Honest" is a song by Dublin-based alternative rock quartet Kodaline.

 The song was released on 1 February 2015 as the lead single from the band's second studio album, "Coming Up for Air" (2015).

 "Honest" became the band's third top 10 single in their home country, Ireland, following "High Hopes" (2013) and "Love Like This" (2013).
</item 4>

<item 5>
Inkwell (band)
Inkwell is an indie rock band from Winter Park, Florida.

 They have released three full length albums and one EP, as well as a collaboration with former techno artist Floorboard.

 They are currently signed with One Eleven Records.

 The band consists of two members, Travis Adams, former frontman from My Hotel Year and Davey Pierce who has worked with of Montreal.

 The band recently performed a small tour with of Montreal along the east coast.

 Their most recent album "Rivers of Blood and Sadness, or Maybe Happy" was released on iTunes music store April 21, 2009.
</item 5>

<item 6>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 6>

<item 7>
Raindancer
Raindancer was an electroacoustic rock band formed in Newport, Wales in the early 1990s.

 The band featured Grant Nicholas and Jon Lee, who later became founding members of Feeder.
</item 7>

<item 8>
Rivermaya
Rivermaya is a Filipino rock band.

 Formed in 1994, it is one of several bands who spearheaded the 1990s Philippine alternative rock explosion.

 Rivermaya is currently composed of original members Mark Escueta and Nathan Azarcon, together with Mike Elgar and Ryan Peralta.

 Former original members include Rico Blanco, who had been the original songwriter of the band and vocalist Bamboo Mañalac, who later formed the band Bamboo and
later went on his solo career.

 Rivermaya is listed as the twentieth biggest-selling artists/act in the Philippines as of present.
</item 8>

<item 9>
Vib Gyor
Vib Gyor was a band that was formed in Leeds, England.

 The name was derived from the first letters of all the colours in the rainbow, in order from the shortest to longest wavelengths.
</item 9>

<item 10>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 10>
=== END ITEM id=83f1f3fec2f7ea20 ===

=== ITEM id=1b47b0c8601ef1d2 ===
Gold sentences and facts:
Sentence 0: Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.
  Fact 0.0: Return to Never Land is also known as Peter Pan 2.
  Fact 0.1: Return to Never Land is also known as Peter Pan In: Return to Never Land.
  Fact 0.2: Return to Never Land is a 2002 American film.
  Fact 0.3: Return to Never Land is an animated musical fantasy-adventure film.
  Fact 0.4: Return to Never Land was produced by Walt Disney Television Animation.
  Fact 0.5: Return to Never Land was released by Walt Disney Pictures.
  Fact 0.6: Return to Never Land was distributed by Walt Disney Studios Motion Pictures.
Sentence 1:  The film was co-produced by Walt Disney Pictures and Mayhem Pictures.
  Fact 1.0: Return to Never Land was co-produced by Walt Disney Pictures.
  Fact 1.1: Return to Never Land was co-produced by Mayhem Pictures.

Retrieved items:
<item 1>
Peter Pan: Adventures in Never Land
Peter Pan: Adventures in Never Land (also known as Peter Pan in Disney's Return to Never Land) is a 2002 video game based on the Peter Pan Disney franchise.
</item 1>

<item 2>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 2>

<item 3>
Jake and the Never Land Pirates
Jake and the Never Land Pirates (also known as Captain Jake and the Never Land Pirates in the fourth season and associated merchandise) is an Annie
Award-winning musical and interactive American children's animated television series shown on Disney Junior.

 It is based on Disney's "Peter Pan" franchise, which in turn is based on the famous book and play by British author J. M. Barrie.

 It is the first Disney Junior original show following the switch from Playhouse Disney.

 It stars Sean Ryan Fox from "Henry Danger", Megan Richie, Jadon Sand, David Arquette, Corey Burton, Jeff Bennett, Loren Hoskins and Dee Bradley Baker.

 The title character Jake was previously voiced by Colin Ford, and then later by Cameron Boyce, while Izzy was voiced for the first three seasons by Madison
Pettis and Cubby was voiced by Jonathan Morgan Heit.

 The series is created by Disney veteran Bobs Gannaway, whose works include another Disney Junior series, "Mickey Mouse Clubhouse", and films such as "Secret of
the Wings", "The Pirate Fairy" and "".

 The last episode aired on November 6, 2016.
</item 3>

<item 4>
Home on the Range (2004 film)
Home on the Range is a 2004 American animated musical western comedy film produced by Walt Disney Feature Animation and released by Walt Disney Pictures.

 The 45th Disney animated feature film, it was the last 2D animated Disney film released until "The Princess and the Frog" in 2009.

 Named after the popular country song of the same name, "Home on the Range" features the voices of Roseanne Barr, Judi Dench, Jennifer Tilly, Cuba Gooding Jr.,
Randy Quaid, and Steve Buscemi.

 The film is set in the Old West, and centers on a mismatched trio of dairy cows—brash, adventurous Maggie; prim, proper Mrs. Caloway; and ditzy, happy-go-lucky
Grace.

 The three cows must capture an infamous cattle rustler named Alameda Slim for his bounty in order to save their idyllic farm from foreclosure.

 Aiding them in their quest is Lucky Jack, a feisty, peg-legged rabbit, but a selfish horse named Buck, eagerly working in the service of Rico, a famous bounty
hunter, seeks the glory for himself.
</item 4>

<item 5>
Josann McGibbon
Josann McGibbon is an American screenwriter working in partnership with Sara Parriott.

 The team's first major success as a screenwriter was the early Brad Pitt film, "The Favor".

 Their biggest hits since then include "Three Men and a Little Lady" and "Runaway Bride".

 In 2007, McGibbon and Parriott co-wrote and produced the hit Debra Messing miniseries, "The Starter Wife".

 "The Starter Wife" received 10 Emmy nominations in 2007, including for best screenwriting, and won one Emmy Award.

 It was also nominated for Golden Globe and Writers Guild awards, and was then produced as a series, also on USA Network.

 McGibbon and Parriott wrote and co-produced the Disney Channel movie, "Descendants" which was directed by Kenny Ortega and premiered in July, 2015.

 On February, 2013, it won the Writers Guild of America Award in television for Outstanding Children's Long Form.

 They then wrote and executive-produced "Descendants 2", also directed by Kenny Ortega, which was simulcast on the Disney Channel, ABC, and the other cable
channels owned by Disney-ABC in July, 2017.
</item 5>

<item 6>
McFarland, USA
McFarland, USA (also known as McFarland) is a 2015 American sports drama film directed by Niki Caro, produced by Mark Ciardi and Gordon Gray, written by
Christopher Cleveland, Bettina Gilois and Grant Thompson with music composed by Antônio Pinto.

 The film was co-produced by Walt Disney Pictures and Mayhem Pictures.

 Based on the true story of a 1987 cross country team from a mainly Latino high school in McFarland, California, the film stars Kevin Costner as Jim White, the
school's coach, who leads the team to win a state championship.

 The film also stars Maria Bello and Morgan Saylor.
</item 6>

<item 7>
Disney Renaissance
In the history of The Walt Disney Company, the Disney Renaissance refers to the era from 1989 to 1999 during which Walt Disney Feature Animation (renamed Walt
Disney Animation Studios in 2006) experienced a creative resurgence in producing successful animated films based on well-known stories, which restored public
and critical interest in The Walt Disney Company as a whole.

 During this era, the studio produced and released ten animated films: "The Little Mermaid" (1989), "The Rescuers Down Under" (1990), "Beauty and the Beast"
(1991), "Aladdin" (1992), "The Lion King" (1994), "Pocahontas" (1995), "The Hunchback of Notre Dame" (1996), "Hercules" (1997), "Mulan" (1998) and "Tarzan"
(1999).
</item 7>

<item 8>
Return to Never Land
Return to Never Land (also known as Peter Pan 2 or Peter Pan In: Return to Never Land) is a 2002 American animated musical fantasy-adventure film produced by
Walt Disney Television Animation, and released by Walt Disney Pictures and distributed by Walt Disney Studios Motion Pictures.

 The film is a sequel to Walt Disney Feature Animation 1953 film "Peter Pan", It is based on J. M. Barrie's novel "Peter and Wendy", and had a worldwide gross
of $109 million.
</item 8>

<item 9>
Never Land Books
The Never Land Books or Never Land Adventures are a series of short chapter books set in Never Land, the home of Peter Pan.

 They are based on the situations and characters established in the novel "Peter and the Starcatchers" and its sequels.

 Like the novels, they are written by Pulitzer Prize-winning humorist Dave Barry and suspense novelist Ridley Pearson, and illustrated by Greg Call.

 Although five books were planned, only three were published, in 2006–2008.

 The stories focus on supporting characters from the novels, such as the Indians, mermaids, pirates, and Lost Boys.

 They include:
</item 9>

<item 10>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 10>
=== END ITEM id=1b47b0c8601ef1d2 ===

=== ITEM id=a0055e1f49f5b627 ===
Gold sentences and facts:
Sentence 0: The Great War is a 26-episode documentary series from 1964 on the First World War.
  Fact 0.0: The Great War is a documentary series.
  Fact 0.1: The Great War has 26 episodes.
  Fact 0.2: The Great War is a documentary series from 1964.
  Fact 0.3: The Great War is about the First World War.
Sentence 1:  The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian Broadcasting Commission.
  Fact 1.0: The Great War was a co-production of the Imperial War Museum.
  Fact 1.1: The Great War was a co-production of the British Broadcasting Corporation.
  Fact 1.2: The Great War was a co-production of the Canadian Broadcasting Corporation.
  Fact 1.3: The Great War was a co-production of the Australian Broadcasting Commission.
Sentence 2:  The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.
  Fact 2.0: Michael Redgrave was the narrator of The Great War.
  Fact 2.1: Marius Goring gave readings in The Great War.
  Fact 2.2: Ralph Richardson gave readings in The Great War.
  Fact 2.3: Cyril Luckham gave readings in The Great War.
  Fact 2.4: Sebastian Shaw gave readings in The Great War.
  Fact 2.5: Emlyn Williams gave readings in The Great War.
Sentence 3: Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.
  Fact 3.0: Imperial War Museums (IWM) is a British national museum organisation.
  Fact 3.1: Imperial War Museums has branches at five locations in England.
  Fact 3.2: Three of the five Imperial War Museums branches are in London.

Retrieved items:
<item 1>
Ken Barrie
Leslie Hulme (9 January 1933 – 29 July 2016), known professionally as Ken Barrie, was an English voice actor and singer best known for narrating, and singing
the theme tune of the BBC television programmes "Postman Pat" and "Charlie Chalk".

 He was also responsible for providing the voices of several of the series' characters.
</item 1>

<item 2>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 2>

<item 3>
British National Bibliography
British National Bibliography (BNB) was established to publish a list of the books, journals and serials that are published in the United Kingdom and Republic
of Ireland.

 It also includes information on forthcoming titles.

 New books and serials have been recorded in the British National Bibliography (BNB) since 1950.

 The British National Bibliography was first established at the British Museum.

 This is the single most comprehensive listing of UK titles.

 UK and Irish publishers are obliged by legal deposit to send a copy of all new publications, including serial titles to BNB for listing.

 The BNB publish the list weekly in electronic form: the last printed weekly list appeared in December 2011.
</item 3>

<item 4>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 4>

<item 5>
First Life (TV series)
First Life is a 2010 British nature documentary series written and presented by David Attenborough, also known by the expanded titles David Attenborough's First
Life (UK) and First Life with David Attenborough (USA).

 It was first broadcast in the USA as a two-hour special on the Discovery Channel on 24 October 2010.

 In the United Kingdom it was broadcast as a two-part series on BBC Two on 5 November 2010.

 "First Life" sees Attenborough tackle the subject of the origin of life on Earth.

 He investigates the evidence from the earliest fossils, which suggest that complex animals first appeared in the oceans around 540 million years ago, an event
known as the Cambrian Explosion.

 Trace fossils of multicellular organisms from an even earlier period, the Ediacaran biota, are also examined.

 Attenborough travels to Canada, Morocco and Australia, using some of the latest fossil discoveries and their nearest equivalents amongst living species to
reveal what life may have been like at that time.

 Visual effects and computer animation are used to reconstruct and animate the extinct life forms.
</item 5>

<item 6>
Mr. Wilson's Cabinet of Wonder
Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily
about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
</item 6>

<item 7>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 7>

<item 8>
The Great War (documentary)
The Great War is a 26-episode documentary series from 1964 on the First World War.

 The documentary was a co-production of the Imperial War Museum, the British Broadcasting Corporation, the Canadian Broadcasting Corporation and the Australian
Broadcasting Commission.

 The narrator was Michael Redgrave, with readings by Marius Goring, Ralph Richardson, Cyril Luckham, Sebastian Shaw and Emlyn Williams.

 Each episode is  40 minutes long.
</item 8>

<item 9>
Norman Gilliland
Norman Gilliland has been a producer on Wisconsin Public Radio since 1984, where he hosts classical music broadcasts, produces the interview program "University
of the Air," and reads for "Chapter A Day."

 He holds degrees in English and Broadcasting from the University of Florida and attended graduate school in English at Duke University, where he developed an
interest in broadcasting.

 He is also an active author with four published books, the historical novel "Sand Mansions" and its stand-alone sequel "Midnight Catch," "Downeast Ledge"
(2013), plus two nonfiction books about classical music--"Grace Notes for a Year" and "Scores to Settle."

 He has produced an audio drama based upon Dick Ringler's modern English translation of the Old English narrative Beowulf titled "Beowulf: The Complete Story—A
Drama" (ISBN  ).

 He was one of a handful of experts interviewed in the Academy Award winning short documentary .
</item 9>

<item 10>
Gordon Burns
Gordon Henry Burns (born 10 June 1942) is a Northern Irish journalist and broadcaster.

 He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West
Tonight" from January 1997 to October 2011.

 In November 2011 Gordon moved back to Belfast were he was born.
</item 10>
=== END ITEM id=a0055e1f49f5b627 ===

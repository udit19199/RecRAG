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

=== ITEM id=914b7dfb6484b1a5 ===
Gold sentences and facts:
Sentence 0:  He was awarded with the highest national military decoration: the Argentine Nation to the Heroic Valour in Combat Cross, the Argentine Congressional Medal, and the Highest Distinction of the Argentine Air Force.
  Fact 0.0: He was awarded the highest national military decoration, the Argentine Nation to the Heroic Valour in Combat Cross.
  Fact 0.1: He was awarded the Argentine Congressional Medal.
  Fact 0.2: He was awarded the Highest Distinction of the Argentine Air Force.
Sentence 1:  In 2010 it had 14,600 military personnel and 6,900 civilian personnel.
  Fact 1.0: In 2010, the entity referred to here had 14,600 military personnel.
  Fact 1.1: In 2010, the entity referred to here had 6,900 civilian personnel.

Retrieved items:
<item 1>
Teodoro Waldner
Teodoro Guillermo Waldner (born 1927) is a retired Argentine Air Force commander who served as Air Force Chief of the General Staff from 1983 to 1985.
</item 1>

<item 2>
Argentine Air Force Mobile Field Hospital
The Argentine Air Force Mobile Field Hospital (Spanish: "Hospital Reubicable de la Fuerza Aérea" ) is a field hospital operated by the Argentine Air Force.

 It is one of three health centers of its kind worldwide.
</item 2>

<item 3>
Hector Fautario
Hector Luis Fautario was a retired Argentine Air Force general.

 He was the General Commander of the Argentine Air Force from 1973 to 1975.

 Shortly prior to the 1976 Argentine coup d'état, he was the government's only remaining loyal officer.

 He had drawn harsh criticism towards himself from the Army and Navy due to his vehement opposition to their repressive plans, and for his refusal to mobilize
the Air Force against the guerrillas' strongholds in the north.

 He was General Jorge Rafael Videla's last obstacle on the way to power.
</item 3>

<item 4>
Argentine Air Force
The Argentine Air Force (Spanish: Fuerza Aérea Argentina , or simply FAA) is the national aviation branch of the Armed Forces of the Argentine Republic.

 In 2010 it had 14,600 military personnel and 6,900 civilian personnel.
</item 4>

<item 5>
South America air forces maneuvers
The South American air forces performs several joint aerial combat training exercises.

 Among the more important are Cruzex ( Portuguese: "Exercício Cruzeiro do Sul" ) which is hosted by the Brazilian Air Force, Salitre ( English: Saltpeter )
hosted by the Chilean Air Force and Ceibo ( English: Erythrina crista-galli flower ) hosted by the Argentine Air Force.

 The goal is to train together in order to respond to a crisis or integrate into United Nations peacekeeping operations as a unified team; the United States Air
Force and the Royal Canadian Air Force have also participated in some of these exercises in recent years.
</item 5>

<item 6>
Cuerpo de Fuerzas Especiales
The Mexican Cuerpo de Fuerzas Especiales (Special Forces Corps) is a special forces unit of the Mexican Army.

 Formerly the "GAFE" (Grupo Aeromóvil de Fuerzas Especiales | Special-Forces Airmobile Group), the SF Corps has six battalions; one is the "Fuerza especial de
reaccion", a quick-response unit, and one is assigned to the Paratroopers Rifle Brigade; the motto of the SF Corps is "Todo por México" (Everything for Mexico).

 Within the SF Corps, there are regular, intermediate, and veteran -service troops.

 The regular-service soldiers usually operate as light infantry.

 The intermediate-service soldiers (lieutenants and captains) usually are instructors.

 The veteran-service soldiers of the Grupo Aeromóvil de Fuerzas Especiales del Alto Mando (GAFE High Command) handle Black-Ops missions.

 Also known as the COIFE, the Special Forces Corps of the Mexican Army is equivalent to the U.S. Army Special Forces.
</item 6>

<item 7>
Servicio de Inteligencia de la Fuerza Aérea
Servicio de Inteligencia de la Fuerza Aérea ("Air Force Intelligence Service", SIFA) is the intelligence agency of the Argentine Air Force.

 It is part of J-2.

 Its director is Commodore García.
</item 7>

<item 8>
Pablo Carballo
Commodore Pablo Marcos Rafael Carballo (b. 11 December 1947 Buenos Aires) is a retired member of the Argentine Air Force - the "Fuerza Aérea Argentina" (FAA) -
who fought in the 1982 Falklands War ( Spanish: "Guerra de las Malvinas" ) where he participated in actions that led to the sinking of three Royal Navy ships.

 He was awarded with the highest national military decoration: the Argentine Nation to the Heroic Valour in Combat Cross, the Argentine Congressional Medal, and
the Highest Distinction of the Argentine Air Force.
</item 8>

<item 9>
Filiberto Ojeda Ríos
Filiberto Ojeda Ríos (April 26, 1933 – September 23, 2005) was the commander-in-chief ("Responsable General") of the Boricua Popular Army ("Ejército Popular
Boricua, a.k.a., Los Macheteros").

 According to an unsigned article in the "Los Angeles Times", Los Macheteros was “a terrorist group seeking Puerto Rico's independence.”

 The group campaigned for, and supported, the independence of Puerto Rico from the United States.

 In 2001, FBI Director Louis J. Freeh claimed the group was linked to acts of terrorism, but Ronald Fernandez, scholar of Puerto Rican history, suggests such
labeling was an act of political convenience by the United States Government, intended to "shift the blame for any attacks on U.S. policy or personnel from us
to them".

 Ríos was also a founder of the FALN.

 In a 1983 New York Times article, Robert McFadden described the FALN as a Puerto Rican terrorist organization responsible for bombings during the 1970s and
early 1980s "in the name of Puerto Rican independence".
</item 9>

<item 10>
Cicaré CK.1
The Cicaré CK.1 (originally, the CH.III Colibrí) was a light helicopter developed in Argentina in the 1970s.

 It was a small, single-rotor aircraft of pod-and-boom configuration with a fully enclosed bubble canopy that could seat three people side-by-side.

 Cicaré's previous helicopter designs had attracted the attention of the Argentine Air Force, which in 1974 contracted him to develop a light helicopter for
training and also marketed for agricultural use.

 A prototype, registered "LV-X62" flew in September 1976, and the Air Force placed an order for five pre-production machines.

 However, development was terminated at this point.
</item 10>
=== END ITEM id=914b7dfb6484b1a5 ===

=== ITEM id=e4719ddea60acbc6 ===
Gold sentences and facts:
Sentence 0: Baghdad ER is a documentary released by HBO on May 21, 2006.
  Fact 0.0: Baghdad ER is a documentary.
  Fact 0.1: Baghdad ER was released by HBO.
  Fact 0.2: Baghdad ER was released on May 21, 2006.
Sentence 1: Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium in Berlin, Germany.
  Fact 1.0: Olympia is a documentary film.
  Fact 1.1: Olympia was released in 1938.
  Fact 1.2: Olympia is a German film.
  Fact 1.3: Olympia was written by Leni Riefenstahl.
  Fact 1.4: Olympia was directed by Leni Riefenstahl.
  Fact 1.5: Olympia was produced by Leni Riefenstahl.
  Fact 1.6: Olympia documents the 1936 Summer Olympics.
  Fact 1.7: The 1936 Summer Olympics were held in the Olympic Stadium in Berlin, Germany.

Retrieved items:
<item 1>
Olympia (1938 film)
Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium
in Berlin, Germany.

 The film was released in two parts: "Olympia 1.

 Teil — Fest der Völker" (Festival of Nations) and "Olympia 2.

 Teil — Fest der Schönheit" (Festival of Beauty).

 It was the first documentary feature film of the Olympic Games ever made.

 Many advanced motion picture techniques, which later became industry standards but which were groundbreaking at the time, were employed —including unusual
camera angles, smash cuts, extreme close-ups, placing tracking shot rails within the bleachers, and the like.

 The techniques employed are almost universally admired, but the film is controversial due to its political context.

 Nevertheless, the film appears on many lists of the greatest films of all time, including "Time" magazine's "All-Time 100 Movies."
</item 1>

<item 2>
Baghdad Jewish Arabic
Baghdad Jewish Arabic is the Arabic dialect spoken by the Jews of Baghdad and other towns of Southern Iraq.

 This dialect differs from the dialect spoken by the Jews in Northern Iraq, such as Mosul and 'Ana.

 The Baghdadi and Northern dialects may be regarded as subvarieties of Judeo-Iraqi Arabic.

 As with most Judeo-Arab communities, there are likely to be few, if any, speakers of the Judeo-Iraqi Arabic dialects who still reside within Iraq.

 Rather these dialects have been maintained or are facing critical endangerment within respective Judeo-Iraqi diasporas, namely those of Israel and the United
States.

 In 2014, the film "Farewell Baghdad" (Arabic: مطير الحمام; Hebrew: מפריח היונים, lit.

 "The Dove Flyer"), which is performed mostly in Jewish Baghdadi Arabic dialect, became the first film to be almost completely performed in Judeo-Iraqi Arabic.
</item 2>

<item 3>
The entries survey international cinema from silent films to the 1960s, including biographies of composers Giuseppe Verdi and author Émile Zola; several films
about Lucrezia Borgia; and works by Jacques Feyder and Max Ophüls. Ophüls’s films span romantic, historical, and psychological drama, while his son Marcel made
the documentary *The Sorrow and the Pity* about French collaboration during World War II. Other highlights include *The Earrings of
</item 3>

<item 4>
List of accolades received by The Hurt Locker
"The Hurt Locker" is a 2009 Iraq War film written by Mark Boal and directed by Kathryn Bigelow.

 The film premiered on September 4, 2008 at the 65th Venice International Film Festival, where it competed with "The Wrestler" for the Golden Lion award.

 It was released in Italy by Warner Bros.

 Pictures on October 10, 2008.

 Following a showing at the 33rd Toronto International Film Festival, Summit Entertainment picked the film up for distribution in the United States.

 "The Hurt Locker" was released in the United States on June 26, 2009, with a limited release at four theaters in Los Angeles and New York City.

 The film grossed over $145,000 on its opening weekend, averaging around $36,000 per theater.

 The following weekend, beginning July 3, the film grossed over $131,000 at nine theaters.

 It held the highest per-screen average of any movie playing theatrically in the United States for the first two weeks of its release, gradually moving into the
top 20 chart.

 Summit Entertainment then released "The Hurt Locker" to more screens.
</item 4>

<item 5>
The entries cover international cinema from silent films through the 1960s, including works about Verdi, Zola and Lucrezia Borgia, and films by Jacques Feyder,
Max Ophüls and Mauritz Stiller. Later highlights include John Steiner’s Italian genre films; Ruggero Deodato’s disputed role directing *Hercules, Prisoner of
Evil* and controversial *Cannibal Holocaust*; Stefan Ruzowitzky’s Oscar-winning *The Counterfeit
</item 5>

<item 6>
Matthew O'Neill (filmmaker)
Matthew O'Neill is a documentary filmmaker best known for his work on the HBO film "Baghdad ER", for which he and co-creator Jon Alpert won three Emmy Awards.
</item 6>

<item 7>
The entries survey documentaries, films, TV, and film-related concepts: Los Angeles hip-hop, Aboriginal country music, Latin American politics, oceans, nerd
culture, Iraqi life and war, Bougainville, education, disability, poetry, surfing, and family relationships. Featured works include *Gasland*, *Baghdad ER*,
*War/Dance*, *First Life*, and *The Coconut Revolution*. Several received major awards or nominations. The material also profiles filmmaker Wayne Coles-Janess
and defines
</item 7>

<item 8>
Baghdad ER
Baghdad ER is a documentary released by HBO on May 21, 2006.

 It shows the Iraq war from the perspective of a military hospital in Baghdad.

 It has some relatively disturbing scenes in it (e.g. amputations), therefore the U.S. Army is officially warning that military personnel watching it could
experience symptoms of post-traumatic stress disorder (PTSD).
</item 8>

<item 9>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 9>

<item 10>
The entries cover diverse films and careers: crime drama *Deadfall* and its *Arsenal* follow-up; *Rogue One*’s rebels stealing Death Star plans; Mel Gibson,
Nicolas Cage, James Garner and Tom Cruise filmographies; Everest disaster drama *Everest*; integration-themed sports films *Remember the Titans* and *McFarland,
USA*; acclaimed *True Grit* and *The Hurt Locker*; espionage film *The Ipcress File*; Vietnam
</item 10>
=== END ITEM id=e4719ddea60acbc6 ===

=== ITEM id=645f8a11733434c5 ===
Gold sentences and facts:
Sentence 0: Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.
  Fact 0.0: Cassiope is a genus of 9 to 12 small shrubby species.
  Fact 0.1: Cassiope is in the family Ericaceae.
Sentence 1: Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to Japan and the Philippines), and Central America and also Europe.
  Fact 1.0: Deutzia is a genus of about 60 species of flowering plants.
  Fact 1.1: Deutzia is in the family Hydrangeaceae.
  Fact 1.2: Deutzia is native to eastern Asia.
  Fact 1.3: Deutzia is native to central Asia.
  Fact 1.4: Deutzia is native from the Himalayas east to Japan and the Philippines.
  Fact 1.5: Deutzia is native to Central America.
  Fact 1.6: Deutzia is also native to Europe.

Retrieved items:
<item 1>
Cassiope mertensiana
Cassiope mertensiana is a species of flowering plant known by the common names western moss heather and white mountain heather.
</item 1>

<item 2>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 2>

<item 3>
Cassiope lycopodioides
Cassiope lycopodioides, Haida Gwaii mountain-heather or clubmoss mountain heather, is a plant species native to southern Alaska, British Columbia, and the US
State of Washington.

 It is found on rocky slopes in arctic and alpine tundra at elevations up to 2000 m.

 In Washington, it is reported only from King County.

 The specific epithet ""lycopodioides"" refers to the plant's superficial resemblance to some species of clubmoss "(Lycopodium" sensu lato).
</item 3>

<item 4>
Deutzia ningpoensis
Deutzia ningpoensis is a shrub in the family Hydrangeaceae.

 The species is endemic to China.

 It grows to between 1 and 2.5 metres high and produces panicles of white flowers from May to July in its native range.
</item 4>

<item 5>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 5>

<item 6>
Aizoaceae
Aizoaceae (the fig-marigold family) is a large family of dicotyledonous flowering plants containing 135 genera and about 1900 species.

 They are commonly known as stone plants or carpet weeds.

 They are often called vygies in South Africa and New Zealand.

 Species that resemble stones or pebbles are sometimes called mesembs.

 Several species are known as ice plants because of the glistening globular bladder cells covering their stems, fruit and leaves, "... they sparkle like ice
crystals."
</item 6>

<item 7>
Cassiope
Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.

 They are native to the Arctic and north temperate montane regions.

 The genus is named after Cassiopeia of Greek mythology.

 Common names, shared with several other similar related genera, include heather and heath.

 They have scale-like leaves lying against the stems, and produce solitary bell-shaped flowers in late spring.

 Though hardy, flowers can be damaged by late frosts.
</item 7>

<item 8>
Deutzia silvestrii
Deutzia silvestrii is a plant species in the family Hydrangeaceae.
</item 8>

<item 9>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 9>

<item 10>
Dieffenbachia
Dieffenbachia is a genus of tropical flowering plants in the family Araceae.

 It is native to the New World Tropics from Mexico and the West Indies south to Argentina.

 Some species are widely cultivated as ornamental plants, especially as houseplants, and have become naturalized on a few tropical islands.
</item 10>
=== END ITEM id=645f8a11733434c5 ===

=== ITEM id=2ad512c93f655afb ===
Gold sentences and facts:
Sentence 0: MiMA, which stands for "Middle of Manhattan", is a mixed-use building located at 450 West 42nd Street between Dyer and 10th Avenues in the Hell's Kitchen neighborhood of Manhattan, New York City.
  Fact 0.0: MiMA stands for 'Middle of Manhattan'.
  Fact 0.1: MiMA is a mixed-use building.
  Fact 0.2: MiMA is located at 450 West 42nd Street.
  Fact 0.3: MiMA is located between Dyer and 10th Avenues.
  Fact 0.4: MiMA is located in the Hell's Kitchen neighborhood of Manhattan, New York City.
Sentence 1:  Ground was broken in 2007 and topping out occurred in early August 2010.
  Fact 1.0: Ground was broken on MiMA in 2007.
  Fact 1.1: Topping out of MiMA occurred in early August 2010.
Sentence 2:  It was designed by the Miami-based architecture firm of Arquitectonica, and has 43 floors of luxury rentals on floors 7 to 50, twelve floors of condominiums on floors 51 to 63, and a Yotel hotel on the lower levels.
  Fact 2.0: MiMA was designed by the Miami-based architecture firm of Arquitectonica.
  Fact 2.1: MiMA has 43 floors of luxury rentals on floors 7 to 50.
  Fact 2.2: MiMA has twelve floors of condominiums on floors 51 to 63.
  Fact 2.3: MiMA has a Yotel hotel on the lower levels.
Sentence 3:  The building's height is 204 meters (670 feet).
  Fact 3.0: The height of MiMA is 204 meters (670 feet).
Sentence 4: 270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan, New York City.
  Fact 4.0: 270 Park Avenue is also known as the JPMorgan Chase Tower.
  Fact 4.1: 270 Park Avenue was formerly known as the Union Carbide Building.
  Fact 4.2: 270 Park Avenue is a high-rise office building.
  Fact 4.3: 270 Park Avenue is located in Midtown Manhattan, New York City.

Retrieved items:
<item 1>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 1>

<item 2>
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 2>

<item 3>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 3>

<item 4>
MiMA (building)
MiMA, which stands for "Middle of Manhattan", is a mixed-use building located at 450 West 42nd Street between Dyer and 10th Avenues in the Hell's Kitchen
neighborhood of Manhattan, New York City.

 Ground was broken in 2007 and topping out occurred in early August 2010.

 It was designed by the Miami-based architecture firm of Arquitectonica, and has 43 floors of luxury rentals on floors 7 to 50, twelve floors of condominiums on
floors 51 to 63, and a Yotel hotel on the lower levels.

 The building's height is 204 meters (670 feet).
</item 4>

<item 5>
Park Avenue House
The Park Avenue House is a high rise residential building located at 2305 Park Avenue in the Park Avenue Historic District in Downtown Detroit, Michigan.

 It was listed on the National Register of Historic Places in 1996.

 It should not be confused with the nearby Park Avenue Hotel, which was demolished in 2015.
</item 5>

<item 6>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 6>

<item 7>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 7>

<item 8>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 8>

<item 9>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 9>

<item 10>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 10>
=== END ITEM id=2ad512c93f655afb ===

=== ITEM id=0d5da4df09034084 ===
Gold sentences and facts:
Sentence 0:  It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
  Fact 0.0: This song is most known through the 1989 single by Kathy Mattea.
  Fact 0.1: The 1989 single by Kathy Mattea was released in conjunction with her album Willow in the Wind.
  Fact 0.2: This song was first recorded and released on the 1987 Don Williams album Traces.
  Fact 0.3: This song was also released in 1988 by Clark's husband on his album Old Friends.
Sentence 1: This is a detailed discography for American country music artist Don Williams.
  Fact 1.0: The discography is detailed.
  Fact 1.1: The discography is for American country music artist Don Williams.

Retrieved items:
<item 1>
Every Beat of My Heart (song)
"Every Beat of My Heart" is a rhythm and blues song by Johnny Otis.

 It was first recorded in 1954 by his group, The Royals (later to be known as The Midnighters).

 In 1961 Gladys Knight & the Pips recorded the song for their debut single on the Vee-Jay label.

 Credited to The Pips, it was the first of eleven releases by the group to make it to number one on the R&B/soul chart.

 It was also the group's first top ten on the "Billboard" Hot 100.

 "Every Beat of My Heart" was first recorded for the Huntom label, who later sold the master to Vee-Jay.

 At the time of the song's release, The Pips were on the Fury label where they re-recorded the song without piano.

 In an unusual occurrence, the Fury recording of the song also made the top twenty on the R&B Sides chart and also made the Hot 100.
</item 1>

<item 2>
Hjärtats trakt
Hjärtats trakt was released on 20 April 1993 and is a compilation album from Swedish pop artist Per Gessle.

 The album includes songs taken from Gessle's solo albums "Per Gessle" and "Scener".

 It was also released under license by the British record company Pickwick and it is the only Gessle album released by another label than EMI.

 In 1998 the album was re-released by the Dutch record company Disky Communications B.V. which is a subsidiary of EMI.
</item 2>

<item 3>
Streets of Heaven (song)
"Streets of Heaven" is a song co-written and recorded by Australian country music artist Sherrié Austin.

 It was released in June 2003 as the first single and title track from her album "Streets of Heaven".

 Austin wrote the song with Paul Duncan and Al Kasha.
</item 3>

<item 4>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 4>

<item 5>
Leave Them Boys Alone
"Leave Them Boys Alone" is a song recorded by American country music artist Hank Williams Jr. with Waylon Jennings and Ernest Tubb.

 It was released in May 1983 as the second single from Williams' album "Strong Stuff".

 The song reached number 6 on the "Billboard" Hot Country Singles chart.

 It was written by Williams, Dean Dillon, Gary Stewart and Tanya Tucker.

 The song is notable for its combination of two singers associated with the outlaw movement with a country legend from the honky tonk days and golden age of the
Grand Ole Opry.

 Outlaw singers like Williams and Jennings saw themselves as taking country music back to its raw, honky tonk roots, and recording an up tempo song with Tubb
(who would never have received radio airplay in the late 1970s and early 80's) and reaching #6 was a slap in the face to the proponents of the country pop
sound.

 The lyrics of the song, much like Williams' "Family Tradition" echo the sentiment that the outlaw singers and their current escapades were predated by the hard
living honky tonkers of the 1950s such as Hank Williams, Sr.
</item 5>

<item 6>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 6>

<item 7>
Don Williams discography
This is a detailed discography for American country music artist Don Williams.
</item 7>

<item 8>
Hate to See Your Heart Break
"Hate to See Your Heart Break" is a song by American rock band Paramore, recorded for their 2013 self-titled fourth album "Paramore".

 It was re-recorded to feature vocals by Joy Williams (formerly of The Civil Wars) for the 2014 deluxe edition of the album, the first time Paramore has
collaborated with another artist or group on a studio recording.
</item 8>

<item 9>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 9>

<item 10>
Night Is Fallin' in My Heart
"Night Is Fallin' in My Heart" is a song written by Dennis Linde.

 The song was first recorded by Country music artist J.P. Pennington for his 1991 album, "Whatever It Takes".

 It was later recorded and released as a single in October 1994 by American country music group Diamond Rio.

 It was the second single released from their third album, "Love a Little Stronger".

 It peaked at number 9 in the United States, and number 6 in Canada.
</item 10>
=== END ITEM id=0d5da4df09034084 ===

=== ITEM id=4aceed1e3f696e15 ===
Gold sentences and facts:
Sentence 0: Janet Williams is an American soprano who has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera, Théâtre des Champs-Élysées, Opera de Lyon, Nice Opera, Théâtre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre as well as in concerts throughout Europe, North America, Canada, Israel and Japan with conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, René Jacobs, Marek Janowski, Neeme Järvi, Raymond Leppard, Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
  Fact 0.0: Janet Williams is an American soprano.
  Fact 0.1: Janet Williams has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera, Theatre des Champs-Elysees, Opera de Lyon, Nice Opera, Theatre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre.
  Fact 0.2: Janet Williams has performed in concerts throughout Europe, North America, Canada, Israel and Japan.
  Fact 0.3: Janet Williams has performed with conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, Rene Jacobs, Marek Janowski, Neeme Jarvi, Raymond Leppard, Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
Sentence 1: Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized solo pianist, chamber music performer, and conductor.
  Fact 1.0: Vladimir Davidovich Ashkenazy's Russian name is Vladimir Davidovich Ashkenazi.
  Fact 1.1: Vladimir Davidovich Ashkenazy was born on 6 July 1937.
  Fact 1.2: Vladimir Davidovich Ashkenazy is an internationally recognized solo pianist.
  Fact 1.3: Vladimir Davidovich Ashkenazy is an internationally recognized chamber music performer.
  Fact 1.4: Vladimir Davidovich Ashkenazy is an internationally recognized conductor.
Sentence 2:  His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
  Fact 2.0: Vladimir Davidovich Ashkenazy's recordings have earned him five Grammy awards.
  Fact 2.1: Vladimir Davidovich Ashkenazy has received Iceland's Order of the Falcon.

Retrieved items:
<item 1>
Elton John
Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.

 He has worked with lyricist Bernie Taupin as his songwriting partner since 1967; they have collaborated on more than 30 albums to date.

 In his five-decade career Elton John has sold more than 300million records, making him one of the best-selling music artists in the world.

 He has more than fifty Top 40 hits, including seven consecutive No. 1 US albums, 58 "Billboard" Top 40 singles, 27 Top 10, four No. 2 and nine No. 1.

 For 31 consecutive years (1970–2000) he had at least one song in the "Billboard" Hot 100.

 His tribute single, re-penned in dedication to the late Princess Diana, "Candle in the Wind 1997" sold over 33million copies worldwide and is the best-selling
single in the history of the UK and US singles charts.

 He has also composed music, produced records, and has occasionally acted in films.
</item 1>

<item 2>
The collection surveys music albums, artists, songs, and genres across rock, pop, metal, and electronic music. Highlights include Patty Griffin’s 2004 album
*Impossible Dream*; Edward Sharpe and the Magnetic Zeros’ 2013 album and 2010 single; the White Stripes’ rise, breakup, acclaimed *Elephant* (2003), and
Grammy-winning “Seven Nation Army”; and biographies of Meg White and Jack White. Other entries cover releases by Steve Winwood
</item 2>

<item 3>
Fred Hersch
Fred Hersch (born October 21, 1955) is an American jazz pianist and educator.

 He has performed solo and led his own groups, including the Pocket Orchestra consisting of piano, trumpet, voice, and percussion.

 He was the first person to play weeklong engagements as a solo pianist at the Village Vanguard in New York City.

 He has recorded more than 70 of his jazz compositions.

 Hersch has been nominated for several Grammy Awards, and, as of December 2014, had been on the Jazz Studies faculty of the New England Conservatory since 1980
(with breaks).
</item 3>

<item 4>
Janet Williams (soprano)
Janet Williams is an American soprano who has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera,
Théâtre des Champs-Élysées, Opera de Lyon, Nice Opera, Théâtre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco
Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre as well as in concerts throughout Europe, North America, Canada, Israel and Japan with
conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, René Jacobs, Marek Janowski, Neeme Järvi, Raymond Leppard,
Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
</item 4>

<item 5>
Robert Glasper
Robert Glasper (born April 6, 1978, in Houston, Texas) is an American pianist and record producer.

 He has been nominated for 6 Grammys, has won 3 Grammy Awards and is currently nominated for an Emmy Award.

 His 2012 album "Black Radio" won the Grammy Award for Best R&B Album at the 55th Grammy Awards.

 His 2014 album "Black Radio 2" won the Grammy Award for Best Traditional R&B Performance at the 56th Grammy Awards.

 The song "These Walls" from Kendrick Lamar's album "To Pimp A Butterfly" won Best Rap/Sung Collaboration at the 57th Grammy Awards, on which Glasper plays
keys.

 The soundtrack for the film "Miles Ahead" won Best Soundtrack Compilation at the 58th Grammy Awards, for which Glasper was a producer.

 The song "Letter To The Free", written with Common, is nominated for an Emmy Award for Best Original Song in the Ava Duvernay documentary film "13th" (Netflix)
at the 2017 Emmys.
</item 5>

<item 6>
Premio Lo Nuestro 2003 marked the awards’ 15th anniversary, hosted by Marco Antonio Regil and Adal Ramones. Juanes and Celia Cruz each won four awards; Pilar
Montenegro won three. Highlights included a 15-year Latin-hits medley and Fania All-Stars’ tribute to Cruz. The remaining entries cover Madonna songs and
videos; Gloria Trevi’s career and albums; Luis Miguel’s *Todos Los Romances*; chart histories and
</item 6>

<item 7>
Profiles span musicians across jazz, rock, metal, electronic, gospel, and pop. Robert Glasper has three Grammys; Carlos Santana, ten Grammys and a landmark
Latin-rock fusion career. Daron Malakian (System of a Down), Gaahl (Gorgoroth), Roy Khan (Kamelot), Toshi (X Japan), and Rivermaya’s former guitarist Perfecto
de Castro are noted rock figures. Relient K’s Matt Thiessen, electronic artists Greg Davis and
</item 7>

<item 8>
The Impeccable Mr. Wilson
The Impeccable Mr. Wilson is an album by American jazz pianist Teddy Wilson featuring performances recorded in 1956 for the Norgran label.
</item 8>

<item 9>
David M. Arden
David M. Arden (born September 6, 1949) is an American concert pianist whose performing and recording career has focused predominantly on contemporary and
American classical repertoire, including premiere performances and first recordings of piano works by a number of notable contemporary composers, such as Henryk
Górecki, Luciano Berio, Earle Brown, Carson Kievman and David Lang.

 The Russian composer Galina Ustvolskaya named Arden's recording of her 12 Preludes for Piano as her preferred recording of that work.
</item 9>

<item 10>
The entries cover diverse music milestones: Marc Ribot’s 2005 Albert Ayler tribute, *Spiritual Unity*; comic duo Collins & Harlan, whose 1916 recording helped
introduce “jas,” precursor to “jazz”; and reggae singer Cedric Myton, a Royal Rasses and Congos member whose voice features on *Humanity* and *Heart of the
Congos*. They also describe early Dean Martin, Chico Hamilton, Shelly Manne, and Maynard
</item 10>
=== END ITEM id=4aceed1e3f696e15 ===

=== ITEM id=4c497d924623570e ===
Gold sentences and facts:
Sentence 0: Liu Ailing (; born June 2, 1967) is a Chinese former footballer who played for the China national team at the 1991, 1995 and 1999 editions of the FIFA Women's World Cup.
  Fact 0.0: Liu Ailing was born on June 2, 1967.
  Fact 0.1: Liu Ailing is a Chinese former footballer.
  Fact 0.2: Liu Ailing played for the China national team at the 1991 FIFA Women's World Cup.
  Fact 0.3: Liu Ailing played for the China national team at the 1995 FIFA Women's World Cup.
  Fact 0.4: Liu Ailing played for the China national team at the 1999 FIFA Women's World Cup.
Sentence 1: The FIFA Women's World Cup is an international football competition contested by the senior women's national teams of the members of "Fédération Internationale de Football Association" (FIFA), the sport's global governing body.
  Fact 1.0: The FIFA Women's World Cup is an international football competition.
  Fact 1.1: The FIFA Women's World Cup is contested by the senior women's national teams of the members of FIFA.
  Fact 1.2: FIFA stands for Fédération Internationale de Football Association.
  Fact 1.3: FIFA is the sport's global governing body.

Retrieved items:
<item 1>
FIFA Women's World Cup
The FIFA Women's World Cup is an international football competition contested by the senior women's national teams of the members of "Fédération Internationale
de Football Association" (FIFA), the sport's global governing body.

 The competition has been held every four years since 1991, when the inaugural tournament, then called the Women's World Championship, was held in China.
</item 1>

<item 2>
Luan Zhili
Luan Zhili (born 6 January 1973) is a female discus thrower from PR China.

 She finished ninth at the 1997 World Championships and won the 1998 Asian Games, the latter in a personal best throw of 63.43 metres.

 The Chinese, and Asian, record is currently held by Xiao Yanling with 71.68 metres.
</item 2>

<item 3>
EAFF E-1 Football Championship
EAFF E-1 Football Championship, known as the East Asian Football Championship from 2003 to 2010, and the EAFF East Asian Cup for the 2013 and 2015 editions, is
a men's international football competition in East Asia for member nations of the East Asian Football Federation (EAFF).

 Before the EAFF was founded in 2002, the Dynasty Cup was held between the East Asian top four teams, and was regarded as the unofficial East Asian
Championship.

 There is a separate competition for both men (first held in 2003) and women (first held in 2005).

 There was also a combined points competition in 2005, where the results of the men's and women's teams are added together (not including qualifiers).
</item 3>

<item 4>
Liu Ailing
Liu Ailing (; born June 2, 1967) is a Chinese former footballer who played for the China national team at the 1991, 1995 and 1999 editions of the FIFA Women's
World Cup.

 She won a silver medal at the 1996 Atlanta Olympics and participated at the 2000 Sydney Olympics.

 A playmaking midfielder, she played professional club football in Japan and the United States.
</item 4>

<item 5>
Gambia women's national football team
The Gambia women's national football team represents the Gambia in international football competition.

 The team, however, has not competed in a match recognised by FIFA, the sport's international governing body, despite that organised women's football has been
played in the country since 1998.

 The Gambia has two youth teams, an under-17 side that has competed in FIFA U-17 Women's World Cup qualifiers, and an under-19 side that withdrew from regional
qualifiers for an under-19 World Cup.

 The development of a national team faces challenges similar to those across Africa, although the national football association has four staff members focusing
on women's football.
</item 5>

<item 6>
Women's football at the 2015 Pacific Games
The 2015 Pacific Games women's football tournament was the 4th edition of the Pacific Games women's football tournament.

 The women's football tournament was held in Port Moresby, Papua New Guinea between 6–16 July 2015 as part of the 2015 Pacific Games.

 The tournament was open to full women's national teams (unlike the men's tournament, which was age-restricted).
</item 6>

<item 7>
EAFF E-1 Football Championship (women)
EAFF E-1 Football Championship is an international football competition in East Asia for national teams of the East Asian Football Federation (EAFF).

 The competition between women's national teams is held alongside men's competition.
</item 7>

<item 8>
Australia women's national soccer team
The Australian women's national soccer team represents Australia in international women's soccer at the senior level.

 The team is overseen by the governing body for soccer in Australia, Football Federation Australia (FFA), which is currently a member of the Asian Football
Confederation (AFC) and the regional ASEAN Football Federation (AFF) since leaving the Oceania Football Confederation (OFC) in 2006.

 The team's official nickname is the Matildas (from the Australian folk song "Waltzing Matilda"), having been known as the Female Socceroos before 1995.

 The current team manager is Alen Stajcic.
</item 8>

<item 9>
Women's Nordic Football Championship
Women's Nordic Football Championship was an international football competition contested by the women's national football teams of the Nordic countries.

 The tournament was held annually between 1974 and 1982.

 Finland, Denmark and Sweden competed from the start, Norway joined the tournament in 1978.

 Iceland and Faroe Islands did not take part at the competition.
</item 9>

<item 10>
The collection surveys football cups, national teams, clubs and collegiate leagues, plus surfing and baseball events and the European Environment Agency.
Highlights include the 92-club English EFL Cup; Trinidad and Tobago’s First Citizens Cup; East Asian men’s and women’s championships; the FIFA Women’s World
Cup; and histories of women’s teams from Gambia, Togo and Australia. It also covers Nordic women’s championships, the 1929–30 Home Championship, Real Sociedad
and its
</item 10>
=== END ITEM id=4c497d924623570e ===

=== ITEM id=722f20e7f0952126 ===
Gold sentences and facts:
Sentence 0: Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)
  Fact 0.0: Julia Compton Moore's dates are February 10, 1929 to April 18, 2004.
  Fact 0.1: Julia Compton Moore was the wife of Lieutenant General (Ret.).
Sentence 1:  Hal Moore, a United States Army officer.
  Fact 1.0: Hal Moore is a United States Army officer.
Sentence 2: Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.
  Fact 2.0: Harold Gregory "Hal" Moore, Jr. lived from February 13, 1922 to February 10, 2017.
  Fact 2.1: Harold Gregory "Hal" Moore, Jr. was a United States Army lieutenant general.
  Fact 2.2: Harold Gregory "Hal" Moore, Jr. was an author.
Sentence 3:  He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point class (1945) to be promoted to brigadier general, major general, and lieutenant general.
  Fact 3.0: Hal Moore was a recipient of the Distinguished Service Cross.
  Fact 3.1: The Distinguished Service Cross is the U.S. military's second highest decoration for valor.
  Fact 3.2: Hal Moore was the first of his West Point class (1945) to be promoted to brigadier general.
  Fact 3.3: Hal Moore was the first of his West Point class (1945) to be promoted to major general.
  Fact 3.4: Hal Moore was the first of his West Point class (1945) to be promoted to lieutenant general.

Retrieved items:
<item 1>
Wayne W. Lambert
Wayne W. Lambert (born 1936) was an Air force Brigadier General (United States).

 He attended the United States Military Academy at West Point, New York, after already having served a year in the US Army.

 Lambert graduated from the Academy in 1959 and was commissioned a 2nd Lieutenant in the USAF that same year.

 He trained as a bomber pilot, receiving his Aviator badge in 1960.

 He has logged over 5,800 flying hours with the US Air Force.

 A seasoned combat veteran, Lambert flew over 225 combat missions in South East Asia as a B-52 crew member.

 He was awarded the Distinguished Flying Cross (United States) and received 10 awards of the Air Medal for his service during the Vietnamese War.

 After serving in a succession of lesser command positions, Lambert was promoted to the rank of Brigadier General in 1983, assuming command of the former
Strategic Air Command’s 7th Air Division, headquartered then at Ramstein Air Base in West Germany, and overseeing SAC operations throughout the European
theater.

 The Brigadier General retired from the US Air Force in 1989.
</item 1>

<item 2>
Benedict Arnold
Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American
Continental Army, and later defected to the British Army.

 While a general on the American side, he obtained command of the fortifications at West Point, New York (which after 1802 would become the site of the U.S.
Military Academy), overlooking the cliffs at the Hudson River (upriver from British-occupied New York City), and planned to surrender them to British forces.

 This plan was exposed in September 1780.

 He was commissioned into the British Army as a brigadier general.
</item 2>

<item 3>
Nadja West
Nadja Y. West (born 1961) is a United States Army lieutenant general and the 44th U.S. Army Surgeon General and Commanding General of the U.S. Army Medical
Command.

 West is the first black Army Surgeon General, and was the first black female, active-duty, major general and the first black female major general in Army
Medicine.

 West is also the first Army black female lieutenant general.

 She is the highest ranking female to have graduated from the United States Military Academy.

 She received the Army Distinguished Service Medal, the Defense Superior Service Medal, the Legion of Merit, and others.
</item 3>

<item 4>
Distinguished Service Cross (United States)
The Distinguished Service Cross is the second highest military award that can be given to a member of the United States Army (and previously, the United States
Army Air Forces and the United States Air Force), for extreme gallantry and risk of life in actual combat with an armed enemy force.

 Actions that merit the Distinguished Service Cross must be of such a high degree that they are above those required for all other U.S. combat decorations but
do not meet the criteria for the Medal of Honor.

 The Distinguished Service Cross is equivalent to the Navy Cross (Navy and Marine Corps), the Air Force Cross (Air Force), and the Coast Guard Cross (Coast
Guard).
</item 4>

<item 5>
Richard E. Cavazos
Richard Edward Cavazos (born January 31, 1929), a Korean War recipient of the Distinguished Service Cross as a first lieutenant, who advanced in rank to become
the United States Army's first Hispanic four-star general.

 During the Vietnam War, as a lieutenant colonel, Cavazos was awarded a second Distinguished Service Cross.

 In 1976, Cavazos became the first Mexican American to reach the rank of brigadier general in the U.S. Army.

 Cavazos served with great distinction for thirty-three years, with his final command as head of the U.S. Army Forces Command.
</item 5>

<item 6>
Julia Compton Moore
Julia Compton Moore (February 10, 1929April 18, 2004) was the wife of Lieutenant General (Ret.)

 Hal Moore, a United States Army officer.

 Her efforts and complaints in the aftermath of the Battle of Ia Drang prompted the U.S. Army to set up survivor support networks and casualty notification
teams consisting of uniformed officers, which are still in use.
</item 6>

<item 7>
Charles Pede
Lieutenant General Charles N. Pede is a military lawyer who currently serves as the 40th Judge Advocate General of the United States Army.

 General Pede was promoted from the rank of Brigadier General to the rank of Lieutenant General, bypassing the rank of Major General on July 26, 2017.
</item 7>

<item 8>
Hal Moore
Harold Gregory "Hal" Moore, Jr. (February 13, 1922 – February 10, 2017) was a United States Army lieutenant general and author.

 He was a recipient of the Distinguished Service Cross, which is the U.S. military's second highest decoration for valor, and was the first of his West Point
class (1945) to be promoted to brigadier general, major general, and lieutenant general.
</item 8>

<item 9>
Myron F. Diduryk
Myron F. Diduryk (July 15, 1938 to April 24, 1970) was a Ukrainian-American United States Army Major, who played a key role as an infantry company commander in
the Battle of Ia Drang, the first major battle of the Vietnam War.

 His exploits in that battle were described by Hal Moore in, "We Were Soldiers Once and Young".

 Moore said that Diduryk was, “… the finest battlefield company commander I had ever seen, bar none.”

 Diduryk was killed in action on his second tour in Vietnam.
</item 9>

<item 10>
Major general (United States)
In the United States Army, United States Marine Corps, and United States Air Force, major general is a two-star general-officer rank, with the pay grade of O-8.

 Major general ranks above brigadier general and below lieutenant general.

 A major general typically commands division-sized units of 10,000 to 15,000 soldiers.

 Major general is equivalent to the two-star rank of rear admiral in the United States Navy and United States Coast Guard and is the highest permanent rank
during peacetime in the uniformed services.

 (The higher ranks are temporary ranks linked to specific positions, although virtually all officers who have been promoted to those ranks are approved to
retire at their highest earned rank.)
</item 10>
=== END ITEM id=722f20e7f0952126 ===

=== ITEM id=103c2497f98fab17 ===
Gold sentences and facts:
Sentence 0:  Editors are based in Silver Spring, Maryland and Seoul, Korea.
  Fact 0.0: Editors are based in Silver Spring, Maryland.
  Fact 0.1: Editors are based in Seoul, Korea.
Sentence 1:  The magazine is headquartered in Baltimore, Maryland.
  Fact 1.0: The magazine is headquartered in Baltimore, Maryland.

Retrieved items:
<item 1>
Adventist World
Adventist World is a monthly international magazine of the Seventh-day Adventist Church published by the Review and Herald Publishing Association.

 Editors are based in Silver Spring, Maryland and Seoul, Korea.
</item 1>

<item 2>
The entries cover diverse schools, hospitals, and community institutions. They include Jesuit and other schools in the U.S., Sweden, India, Nepal, Belize, and
Japan-related programs in the U.S.; several Indian schools are fully residential, while Japanese supplementary schools serve expatriate children. U.S. district
profiles describe grade structures and administrators. Health organizations include historic and regional hospitals, Adventist HealthCare, and Common Hope’s
support for Guatemalan children. Other sites include St. John Cemetery
</item 2>

<item 3>
Adventist HealthCare
Adventist HealthCare is a not-for-profit health services organization based in Gaithersburg, Maryland that employs more than 6,200 people and provides
healthcare for more than 400,000 individuals in the community each year.

 The primary service area for Adventist HealthCare is the Washington, D.C. metropolitan area.

 Despite similar names, it is not a part of the California-based Adventist Health, or the Florida-based Adventist Health System.
</item 3>

<item 4>
Adventist Review
The Adventist Review is the official newsmagazine of the Seventh-day Adventist Church.

 Commonly known as the Review, it is published weekly by the Review and Herald Publishing Association.

 The Review and Herald also publishes a sister magazine, "Adventist World".

 The magazine is headquartered in Silver Spring, Maryland.

 The current editor of the "Adventist Review" is Bill Knott.

 The magazine currently has nearly 30,000 paying subscribers.

 Its library reference number is OCLC 9572173  .
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
Hänt Bild (Swedish weekly, 2003–, Aller); Hänt Extra (Swedish celebrity/youth weekly); Girlfriends (lesbian culture, 1993–2006); Rap-Up (hip-hop/R&B quarterly,
2001–); Sculptural Pursuit (quarterly art); DIVA (UK lesbian monthly, 1994–); Elle (global fashion/lifestyle, founded 1945); Girls’ Life (US teen, 1994–
</item 6>

<item 7>
The entries cover magazines, journalists, writers, and the meaning of “review.” Children’s titles include bimonthly Jack and Jill and National Wildlife
Federation nature magazines Ranger Rick, Ranger Rick Jr., and Ranger Rick Cub. Women’s magazines include Macedonian Tea Moderna and Danish ALT for Damerne.
Other publications range from Adventist Review and conservative National Review to the Human Life Review, Henry James Review, and defunct American Mercury and
Insurgent49. Profiles include Rosie O’Donnell
</item 7>

<item 8>
Seventh-day Adventist education
The Seventh-day Adventist educational system is part of the Seventh-day Adventist Church and overseen by the General Conference of Seventh-day Adventist located
in Silver Spring, Maryland.The educational system is the second-largest Christian school-system in the world, after the Roman Catholic system.
</item 8>

<item 9>
Girls' Life (magazine)
Girls' Life is an American teen magazine.

 It was launched in 1994 by Monarch Services.

 The magazine is headquartered in Baltimore, Maryland.
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
=== END ITEM id=103c2497f98fab17 ===

=== ITEM id=72fcd5967b7ddd28 ===
Gold sentences and facts:
Sentence 0: While studying at the University of Michigan, he played with visiting musicians such as Dizzy Gillespie, Miles Davis, Thad Jones and Billy Mitchell, before going on to play with Sonny Stitt, Count Basie and Al McKibbon, Cannonball Adderley, Percy Heath, Philly Joe Jones, Lou Donaldson, Donald Byrd, Kenny Dorham and Zoot Sims.
  Fact 0.0: The musician studied at the University of Michigan.
  Fact 0.1: While studying at the University of Michigan, the musician played with visiting musician Dizzy Gillespie.
  Fact 0.2: While studying at the University of Michigan, the musician played with visiting musician Miles Davis.
  Fact 0.3: While studying at the University of Michigan, the musician played with visiting musician Thad Jones.
  Fact 0.4: While studying at the University of Michigan, the musician played with visiting musician Billy Mitchell.
  Fact 0.5: After that, the musician played with Sonny Stitt, Count Basie and Al McKibbon.
  Fact 0.6: After that, the musician played with Cannonball Adderley, Percy Heath, Philly Joe Jones and Lou Donaldson.
  Fact 0.7: After that, the musician played with Donald Byrd, Kenny Dorham and Zoot Sims.
Sentence 1: Miles Dewey Davis III (May 26, 1926September 28, 1991) was an American jazz trumpeter, bandleader, and composer.
  Fact 1.0: Miles Dewey Davis III was born on May 26, 1926.
  Fact 1.1: Miles Dewey Davis III died on September 28, 1991.
  Fact 1.2: Miles Dewey Davis III was an American jazz trumpeter.
  Fact 1.3: Miles Dewey Davis III was an American jazz bandleader.
  Fact 1.4: Miles Dewey Davis III was an American jazz composer.

Retrieved items:
<item 1>
Don Ellis
Donald Johnson Ellis (July 25, 1934 – December 17, 1978) was an American jazz trumpeter, drummer, composer, and bandleader.

 He is best known for his extensive musical experimentation, particularly in the area of time signatures. Later in his life he worked as a film composer,
contributing a score to 1971's "The French Connection" and 1973's "The Seven-Ups".
</item 1>

<item 2>
Louis Smith (musician)
While studying at the University of Michigan, he played with visiting musicians such as Dizzy Gillespie, Miles Davis, Thad Jones and Billy Mitchell, before
going on to play with Sonny Stitt, Count Basie and Al McKibbon, Cannonball Adderley, Percy Heath, Philly Joe Jones, Lou Donaldson, Donald Byrd, Kenny Dorham and
Zoot Sims.
</item 2>

<item 3>
Paul Smith (defensive end)
Paul Edward Smith (August 13, 1945 – March 14, 2000) was a collegiate and professional American football defensive end.

 Smith was selected in the 1968 Common Draft by the American Football League's Denver Broncos and played in the National Football League for the Broncos and
Washington Redskins.
</item 3>

<item 4>
Thad Jones
Thaddeus Joseph Jones (March 28, 1923 – August 20, 1986) was an American jazz trumpeter, composer, and bandleader who has been called "one of the all-time
greatest jazz trumpet soloists."
</item 4>

<item 5>
George Albert Smith
George Albert Smith Sr. (April 4, 1870 – April 4, 1951) was an American religious leader who served as the eighth president of The Church of Jesus Christ of
Latter-day Saints (LDS Church).
</item 5>

<item 6>
Here Comes Louis Smith
Here Comes Louis Smith is the debut album by American trumpeter Louis Smith recorded in 1958 and released on the Blue Note label.

 Originally recorded for the Transition label, the company went out of business shortly afterwards and before the recording could be released.

 The album masters were acquired by Blue Note's Alfred Lion.
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
Miles Davis
Miles Dewey Davis III (May 26, 1926September 28, 1991) was an American jazz trumpeter, bandleader, and composer.

 He is among the most influential and acclaimed figures in the history of jazz and 20th century music.

 Davis adopted a variety of musical directions in his five-decade career which kept him at the forefront of a number of major stylistic developments in jazz.
</item 8>

<item 9>
Marty Marsala
Marty Marsala (2 April 1909 – 27 April 1975) was an American jazz trumpeter born in Chicago, perhaps best known for working from 1926-1946 with his brother Joe
Marsala in a big band in New York City and Chicago.

 He had also toured with various artists, such as Chico Marx and Miff Mole, to name a few.

 During the 1940s Marsala was a celebrated West Coast jazz trumpeter, commuting back and forth from Chicago to San Francisco frequently.

 In various club settings Marsala shared stages with Earl Hines and Sidney Bechet.
</item 9>

<item 10>
Louis Smith (gymnast)
Louis Antoine Smith, MBE (born 22 April 1989) is a British artistic gymnast.

 He received a bronze medal and a silver medal on the pommel horse at the 2008 Beijing Olympics and the 2012 London Olympics respectively, with the former
marking the first time a British gymnast had placed in an Olympic event since 1928.

 He followed this up with a second consecutive silver medal on the pommel horse at the 2016 Rio Olympics, finishing behind teammate Max Whitlock.

 Smith was part of the Great Britain team that took the bronze in the men's artistic team all-around at the 2012 London Olympics.

 He is the only British gymnast to win Olympic medals in three separate Games.

 In 2015 he became the European champion on Pommel Horse.
</item 10>
=== END ITEM id=72fcd5967b7ddd28 ===

=== ITEM id=8a06881e876e1589 ===
Gold sentences and facts:
Sentence 0: Rosario Castellanos Figueroa (] ; 25 May 1925 – 7 August 1974) was a Mexican poet and author.
  Fact 0.0: Rosario Castellanos Figueroa was born on 25 May 1925.
  Fact 0.1: Rosario Castellanos Figueroa died on 7 August 1974.
  Fact 0.2: Rosario Castellanos Figueroa was a Mexican poet.
  Fact 0.3: Rosario Castellanos Figueroa was a Mexican author.
Sentence 1: Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a series of novels beginning in 1908 with "Anne of Green Gables".
  Fact 1.0: Lucy Maud Montgomery was born on November 30, 1874.
  Fact 1.1: Lucy Maud Montgomery died on April 24, 1942.
  Fact 1.2: Lucy Maud Montgomery published as L.M. Montgomery.
  Fact 1.3: Lucy Maud Montgomery was a Canadian author.
  Fact 1.4: Lucy Maud Montgomery is best known for a series of novels.
  Fact 1.5: The series of novels began in 1908 with Anne of Green Gables.

Retrieved items:
<item 1>
Rosario Castellanos
Rosario Castellanos Figueroa (] ; 25 May 1925 – 7 August 1974) was a Mexican poet and author.

 Along with the other members of the Generation of 1950 (the poets who wrote following the Second World War, influenced by César Vallejo and others), she was
one of Mexico's most important literary voices in the last century.

 Throughout her life, she wrote eloquently about issues of cultural and gender oppression, and her work has influenced feminist theory and cultural studies.

 Though she died young, she opened the door of Mexican literature to women, and left a legacy that still resonates today.
</item 1>

<item 2>
The entries profile women in literature, theatre, music, visual art, business, and activism. Highlights include Pearson CEO Marjorie Scardino, the first woman
to lead a FTSE 100 company; Nobel-winning short-story writer Alice Munro; actress Elizabeth Taylor; pioneering Restoration writer Aphra Behn; and Aboriginal
activist Mum Shirl. They also cover Lucy Maud Montgomery’s Anne novels, Jean Webster’s *When Patty Went to College*, Margaret Landon’s *Anna and the King
</item 2>

<item 3>
Rea Wilmshurst
Rea Wilmshurst (August 10, 1941 – March 22, 1996) graduated from the University of Toronto with a degree in English in 1970.

 She went on to edit eight volumes of Lucy Maud Montgomery's previously unknown short stories and publish them through McClelland & Stewart.

 In 1985, she published a bibliography of Montgomery's short stories, poems, and articles.

 Wilmshurst was also an editorial assistant for the projects that compiled the Collected Works of John Stuart Mill and the Collected Works of Samuel Taylor
Coleridge.
</item 3>

<item 4>
L.M. Montgomery's Anne of Green Gables
L.M. Montgomery's Anne of Green Gables is a Canadian television film based on Lucy Maud Montgomery's novel of the same name.

 It first aired on YTV on February 15, 2016 and starred Ella Ballentine, Martin Sheen and Sara Botsford.

 Montgomery's granddaughter, Kate Macdonald Butler, was one of the film's executive producers.

 The film's world premiere was held February 2, 2016 at the Canadian Museum of History.
</item 4>

<item 5>
Lucy Maud Montgomery
Lucy Maud Montgomery {'1': ", '2': ", '3': ", '4': "} (November 30, 1874 – April 24, 1942) published as L.M. Montgomery, was a Canadian author best known for a
series of novels beginning in 1908 with "Anne of Green Gables".

 The book was an immediate success.

 The central character, Anne Shirley, an orphaned girl, made Montgomery famous in her lifetime and gave her an international following.
</item 5>

<item 6>
Anne of Ingleside
Anne of Ingleside is a children's novel by Canadian author Lucy Maud Montgomery.

 It was first published in July 1939 by McClelland and Stewart (Toronto) and the Frederick A. Stokes Company (New York).

 It is the tenth of eleven books that feature the character of Anne Shirley, and Montgomery's final published novel.

 (Two novels that occur later in the "Anne" chronology were actually published years earlier.

 As well, the short story collection "The Blythes Are Quoted", written in 1941/42, but not published until 2009, concludes the Anne chronology.)
</item 6>

<item 7>
Alice Munro
Alice Ann Munro ( , "née" Laidlaw ; born 10 July 1931) is a Canadian short story writer and Nobel Prize winner.

 Munro's work has been described as having revolutionized the architecture of short stories, especially in its tendency to move forward and backward in time.

 Her stories have been said to "embed more than announce, reveal more than parade."
</item 7>

<item 8>
Emily's Quest
Emily's Quest is a novel and the last of the "Emily" trilogy by Lucy Maud Montgomery.

 After finishing "Emily Climbs", Montgomery suspended writing "Emily's Quest" and published "The Blue Castle"; she resumed writing and published in 1927.
</item 8>

<item 9>
Anne of Green Gables
Anne of Green Gables is a 1908 novel by Canadian author Lucy Maud Montgomery (published as L. M. Montgomery).

 Written for all ages, it has been considered a children's novel since the mid-twentieth century.

 It recounts the adventures of Anne Shirley, an 11-year-old orphan girl who is mistakenly sent to Matthew and Marilla Cuthbert, a middle-aged brother and sister
who had intended to adopt a boy to help them on their farm in the fictional town of Avonlea on Prince Edward Island.

 The novel recounts how Anne makes her way with the Cuthberts, in school, and within the town.
</item 9>

<item 10>
Anne Shirley
Anne Shirley is a fictional character introduced in the 1908 novel "Anne of Green Gables" by Lucy Maud Montgomery.

 Montgomery wrote in her journal that the idea for Anne's story came from relatives who, planning to adopt an orphaned boy, received a girl instead.

 Anne Shirley's appearance was inspired by a photograph which Montgomery clipped from the Metropolitan Magazine and kept, unaware of the model's identity as the
1900s Gibson Girl Evelyn Nesbit.
</item 10>
=== END ITEM id=8a06881e876e1589 ===

=== ITEM id=f00846acd0abf3c7 ===
Gold sentences and facts:
Sentence 0:  It is known for producing the long-running series "Tom Goes to the Mayor", "Nathan For You", "Tim and Eric Awesome Show, Great Job!
  Fact 0.0: The entity referred to as 'It' is known for producing the long-running series 'Tom Goes to the Mayor'.
  Fact 0.1: The entity referred to as 'It' is known for producing the long-running series 'Nathan For You'.
  Fact 0.2: The entity referred to as 'It' is known for producing the long-running series 'Tim and Eric Awesome Show, Great Job!'.
Sentence 1: " and "Check It Out!
  Fact 1.0: 'Check It Out!' is a series title.
Sentence 2:  with Dr. Steve Brule".
  Fact 2.0: The series title is 'Check It Out! with Dr. Steve Brule'.
Sentence 3: Check It Out!
  Fact 3.0: Check It Out! is a title.
Sentence 4:  with Dr. Steve Brule is an American sketch comedy television series that is a spin-off of "Tim and Eric Awesome Show, Great Job!
  Fact 4.0: Check It Out! with Dr. Steve Brule is an American sketch comedy television series.
  Fact 4.1: Check It Out! with Dr. Steve Brule is a spin-off of 'Tim and Eric Awesome Show, Great Job!'.

Retrieved items:
<item 1>
Bagboy (TV special)
Bagboy is a 2015 American television special produced for Adult Swim, and aired on February 21, 2015, to positive critical reception.

 Written and directed by Tim Heidecker and Eric Wareheim for Abso Lutely Productions and John C. Reilly, the special is a fictional sitcom pilot in the universe
of "Tim and Eric Awesome Show, Great Job!

", written by and starring Reilly's recurring "Tim and Eric" character Dr. Steve Brule.

 This is the second "Tim and Eric" spin-off to feature the Dr. Steve Brule character after "Check it Out!

 with Dr. Steve Brule".
</item 1>

<item 2>
Upright Citizens Brigade (TV series)
Upright Citizens Brigade is an American sketch comedy television series that premiered on August 19, 1998 on Comedy Central.

 The show aired for three seasons with each season consisting of ten episodes.

 The series featured four members of Upright Citizens Brigade, an improvisational sketch comedy group.

 The cast included Matt Besser, Amy Poehler, Ian Roberts, and Matt Walsh.

 The cast would later reunite for another series of a similar format that premiered in 2016 on Seeso.
</item 2>

<item 3>
Abso Lutely Productions
Abso Lutely Productions is a film and television production company owned by actor and producers Tim Heidecker, Eric Wareheim and producer Dave Kneebone.

 It is known for producing the long-running series "Tom Goes to the Mayor", "Nathan For You", "Tim and Eric Awesome Show, Great Job!

" and "Check It Out!

 with Dr. Steve Brule".
</item 3>

<item 4>
Check It Out! with Dr. Steve Brule
Check It Out!

 with Dr. Steve Brule is an American sketch comedy television series that is a spin-off of "Tim and Eric Awesome Show, Great Job!

" starring John C. Reilly as Dr. Steve Brule.

 The series premiered on Cartoon Network's late night programming block, Adult Swim, on May 16, 2010.

 The program follows Brule as he examines different facets of living.

 His severe naivete and social awkwardness generally land him in embarrassing situations, though he largely remains ignorant of any embarrassment he's causing
himself.

 As the series progresses, he reveals shocking and sometimes horrifying details about his past and personal life.
</item 4>

<item 5>
David Jablin
David Jablin is an award winning comedy producer and director and one of the pioneers of original television programming for Pay Television.

 In 1981 he created and produced the innovative comedy anthology series "Likely Stories" for HBO/Cinemax, which featured the directing debuts of Rob Reiner,
Danny DeVito, Christopher Guest, and Billy Crystal.
</item 5>

<item 6>
And There's More
And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
</item 6>

<item 7>
Graph query row:
series: Check It Out! with Dr. Steve Brule
relationship: CREATED_BY
connected_entity: Abso Lutely Productions
stars_john_c_reilly: False
</item 7>

<item 8>
In Living Color
In Living Color is an American sketch comedy television series that originally ran on Fox from April 15, 1990, to May 19, 1994.

 Brothers Keenen and Damon Wayans created, wrote and starred in the program.

 The show was produced by Ivory Way Productions in association with 20th Century Fox Television and was taped at stage 7 at the Fox Television Center on Sunset
Boulevard in Hollywood, California.

 The title of the series was inspired by the NBC announcement of broadcasts being presented "in living color" during the 1960s, prior to mainstream color
television.

 It also refers to the fact that most of the show's cast were black, unlike other sketch comedy shows such as "Saturday Night Live" whose casts are mostly
white.

 It was controversial due to the Wayans' decision to portray African-American humor from the ghetto in a time when mainstream American tastes regarding black
comedy had been set by more upscale shows such as "The Cosby Show", causing an eventual feud for control between Fox executives and the Wayans.
</item 8>

<item 9>
Friends of the People
Friends of the People is an American sketch comedy television series.

 It was slated to premiere on TruTV in summer 2014, but was pushed to October 28, 2014, as part of the network's shift in their programming direction.

 Many of the cast members (Jennifer Bartels, Jermaine Fowler, and Lil Rel Howery) were originally reported to be cast members of a planned revival of In Living
Color which never materialized.

 The show's first season consists of 10 episodes.

 This makes it the network's first sketch comedy show.

 The series holds a TV-14 rating, though select episodes are rated TV-MA--also a first for the truTV network.
</item 9>

<item 10>
Dave Kneebone
Dave Kneebone is an American producer.

 Along with Eric Wareheim and Tim Heidecker he runs Abso Lutely Productions which produces a variety of television and films.

 His role has been described as the business chief and "straight man" at Abso Lutely.

 Kneebone has worked as a producer on a variety of television shows including "Comedy Bang!

 Bang!

", "Nathan for You", and "Tim and Eric Awesome Show, Great Job!

".

 He is also listed a producer on the feature film "Tim and Eric's Billion Dollar Movie".
</item 10>
=== END ITEM id=f00846acd0abf3c7 ===

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

=== ITEM id=80710f26dc100c0d ===
Gold sentences and facts:
Sentence 0:  He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink, T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
  Fact 0.0: This producer has produced songs for Justin Bieber.
  Fact 0.1: This producer has produced songs for Monica.
  Fact 0.2: This producer has produced songs for Britney Spears.
  Fact 0.3: This producer has produced songs for Usher.
  Fact 0.4: This producer has produced songs for Keri Hilson.
  Fact 0.5: This producer has produced songs for T.I.
  Fact 0.6: This producer has produced songs for Nelly Furtado.
  Fact 0.7: This producer has produced songs for Kevin Cossom.
  Fact 0.8: This producer has produced songs for Ciara.
  Fact 0.9: This producer has produced songs for Mariah Carey.
  Fact 0.10: This producer has produced songs for Timbaland.
  Fact 0.11: This producer has produced songs for Madonna.
  Fact 0.12: This producer has produced songs for J. Cole.
  Fact 0.13: This producer has produced songs for Kanye West.
  Fact 0.14: This producer has produced songs for Whitney Houston.
  Fact 0.15: This producer has produced songs for Missy Elliott.
  Fact 0.16: This producer has produced songs for M.I.A.
  Fact 0.17: This producer has produced songs for Justin Timberlake.
  Fact 0.18: This producer has produced songs for JoJo.
  Fact 0.19: This producer has produced songs for Joe Jonas.
  Fact 0.20: This producer has produced songs for Simple Plan.
  Fact 0.21: This producer has produced songs for The Clutch.
  Fact 0.22: This producer has produced songs for Pink.
  Fact 0.23: This producer has produced songs for T-Pain.
  Fact 0.24: This producer has produced songs for Diddy.
  Fact 0.25: This producer has produced songs for Meek Mill.
  Fact 0.26: This producer has produced songs for Bjork.
  Fact 0.27: This producer has produced songs for Duran Duran.
Sentence 1:  Raised in Fayetteville, North Carolina, Cole initially gained recognition as a rapper following the release of his debut mixtape, "The Come Up", in early 2007.
  Fact 1.0: Cole was raised in Fayetteville, North Carolina.
  Fact 1.1: Cole initially gained recognition as a rapper.
  Fact 1.2: Cole initially gained recognition as a rapper after the release of his debut mixtape.
  Fact 1.3: Cole's debut mixtape is titled The Come Up.
  Fact 1.4: Cole's debut mixtape The Come Up was released in early 2007.

Retrieved items:
<item 1>
Wayne Coles-Janess
Wayne Coles-Janess is an Australian producer, writer and director of drama and documentary film and TV programs.

 Based in Melbourne, Australia, he has produced documentaries about frontier places in the country.

 He has also made some documentaries in several international locations, including during times of war.
</item 1>

<item 2>
Daniel Boone Native Gardens
Daniel Boone Native Gardens, located in Boone, North Carolina, United States, has a collection of North Carolina native plants in an informal landscaped design.

 The gardens are open daily from May to October.
</item 2>

<item 3>
Raleigh Moncrief
Raleigh Moncrief is an American electronic musician, record producer and recording engineer who has worked with Zach Hill, Dirty Projectors and !!!

.
</item 3>

<item 4>
Daron Jones
Daron Tavaris Jones (born December 27, 1976) is an American singer-songwriter, record producer.

 He is a member in Bad Boy Records group 112.

 His debut solo album titled "Uncensored" was released December 5, 2010.

 The first single from the album is the digitally-released "Money."

 Daron digitally released "Christmas in Atlanta" in December 2014, which reached the top 30 on the iTunes R&B chart.

 Daron is currently in the studio working on new music.
</item 4>

<item 5>
Anna Graceman
Anna Graceman is a young American singer-songwriter and pianist whose self-penned songs and music have gained attention at home and abroad.

 Her style of music incorporates elements of folk, soul and rock.

 As of 2016 she had written and performed over 70 of her own published songs; writing as well as producing her releases.

 Anna continues to divide her time between her solo projects and her new project - the ‘Graceman’ Band - a trio that includes Allie Graceman (Rhythm/Lead
Guitar) and Landon Graceman (Drums).
</item 5>

<item 6>
Kaydean
Kaydean is a British-American record and TV producer, composer, DJ, record label executive and artist manager.

 Kaydean produces, composes and arranges music in the Dance, Electronica, Latin Pop, Contemporary R&B, and Electronica genres.

 Born in London, he has worked with many artists including Janet Jackson, Maxi Priest, Robyn, Man Parrish, Angel "Cucco" Peña, Baron Lopez, Raekwon, Ky-Mani
Marley, Frankie Cutlass and Nocera amongst others.

 He is also known for discovering Latin Pop Star Janid and managing her career.

 He produced her Spanish album "La Magia" under the Sony Music Latin imprint label, Handshake and her Hit Single "Penicilina".

 In 2015 he debuted as a TV executive producer with Janid's reality show on Mega TV, The first episode aired on September 6, 2015.
</item 6>

<item 7>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 7>

<item 8>
J. Cole
Jermaine Lamarr Cole (born January 28, 1985), better known by his stage name J. Cole, is an American hip hop recording artist and record producer.

 Raised in Fayetteville, North Carolina, Cole initially gained recognition as a rapper following the release of his debut mixtape, "The Come Up", in early 2007.

 Intent on further pursuing a solo career as a rapper, he went on to release two additional mixtapes after signing to Jay Z's Roc Nation imprint in 2009.
</item 8>

<item 9>
Leah Flanagan
Leah Flanagan is a singer-songwriter from Sydney, NSW.

 Leah has appeared on Australian TV shows Spicks & Specks and RockQuiz, has released 2 albums and toured extensively through Australia with her music and part
of festival ensembles.

 Leah’s status as an Indigenous musician is one of pride and respect; she has helped make history with such recent essential albums as Archie Roach’s 25th
anniversary reissue of Charcoal Lane and Buried Country 1.5, The Story of Aboriginal Country Music.

 Her latest album Saudades was recorded at the studio of Midnight Oil's Jim Moginie and mixed by Paul McKercher (You Am I, Augie March)
</item 9>

<item 10>
Danja (record producer)
Floyd Nathaniel "Nate" Hills (born February 22, 1982), professionally known as Danja , is an American record producer and songwriter from Virginia Beach,
Virginia.

 He has produced songs for prominent artists such as Justin Bieber, Monica, Britney Spears, Usher, Keri Hilson, T.I., Nelly Furtado, Kevin Cossom, Ciara, Mariah
Carey, Timbaland, Madonna, J. Cole, Kanye West, Whitney Houston, Missy Elliott, M.I.A., Justin Timberlake, JoJo, Joe Jonas, Simple Plan, The Clutch, Pink,
T-Pain, Diddy, Meek Mill, Björk and Duran Duran.
</item 10>
=== END ITEM id=80710f26dc100c0d ===

=== ITEM id=7e0b68143a18a037 ===
Gold sentences and facts:
Sentence 0:  His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files" (1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
  Fact 0.0: The subject of this sentence portrayed Bret Maverick in the series Maverick from 1957 to 1962.
  Fact 0.1: The subject of this sentence portrayed Bret Maverick in the series Bret Maverick from 1981 to 1982.
  Fact 0.2: The subject of this sentence portrayed Jim Rockford in the series The Rockford Files from 1974 to 1980.
  Fact 0.3: The subject of this sentence portrayed Jim Egan in the series 8 Simple Rules from 2003 to 2005.
Sentence 1: The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.
  Fact 1.0: The Notebook is a 2004 film.
  Fact 1.1: The Notebook is an American romantic drama film.
  Fact 1.2: The Notebook was directed by Nick Cassavetes.
  Fact 1.3: The Notebook is based on a 1996 novel.
  Fact 1.4: The 1996 novel has the same name as the film The Notebook.
  Fact 1.5: The 1996 novel was written by Nicholas Sparks.

Retrieved items:
<item 1>
The New Maverick
The New Maverick is a 1978 made-for-TV movie based on the 1957 television series "Maverick", with James Garner as Bret Maverick, Charles Frank as newcomer
cousin Ben Maverick (son of Beau Maverick), Jack Kelly as Bart Maverick, and Susan Sullivan as Poker Alice Ivers.

 Garner had been 29 years old at the beginning of the original series and was 50 while filming "The New Maverick".

 The TV-movie was a pilot for the series "Young Maverick", which featured Frank and only lasted a few episodes.

 Directed by Hy Averback and written by Juanita Bartlett, the movie was filmed while Garner's series "The Rockford Files" was on hiatus.

 Garner would later star in "Bret Maverick", another attempt at a television series revival inspired by this TV-movie, for the 1981-82 season.
</item 1>

<item 2>
James Garner filmography
(1969) with Walter Brennan, "Murphy's Romance" (1985) with Sally Field, "Sunset" (1988) with Bruce Willis, "Maverick" (1994) with Mel Gibson and Jodie Foster,
"Divine Secrets of the Ya-Ya Sisterhood" (2002), and "The Notebook" (2004) with Ryan Gosling and Rachel McAdams.

 His television series credits include portraying Bret Maverick in "Maverick" (1957–1962) and "Bret Maverick" (1981–1982), Jim Rockford in "The Rockford Files"
(1974–1980), and Jim Egan in "8 Simple Rules" (2003–2005).
</item 2>

<item 3>
List of American films of 2004
A list of American films released in 2004. "

Million Dollar Baby" won the 2004 Academy Award for Best Picture and "Crash" won the 2005 Academy Award for Best Picture. "

The Aviator" won the BAFTA Award for Best Film and the Golden Globe Award for Best Motion Picture – Drama. "

Sideways" won the Golden Globe Award for Best Motion Picture – Musical or Comedy and the Satellite Award for Best Film – Musical or Comedy. "

Hotel Rwanda" won the Satellite Award for Best Film – Drama.
</item 3>

<item 4>
Cash McCall
Cash McCall is a 1960 American romantic drama film in Technicolor from Warner Bros., produced by Henry Blanke, directed by Joseph Pevney, that stars James
Garner and Natalie Wood.

 It is based upon the novel of the same name by Cameron Hawley about a man who buys moribund businesses in order to refurbish them and then sell them on at
considerable profit.

 The film's screenplay is by Lenore J. Coffee and Marion Hargrove.
</item 4>

<item 5>
Milan (2004 film)
Milan is a 2004 romantic drama film released under Star Cinema, ABS-CBN Film Productions, Inc. in the Philippines.

 It stars Claudine Barretto and Piolo Pascual.

 It is a love story written by Raymond Lee, and under Olivia Lamasan's direction.

 The movie made more than P100 million in the box office.
</item 5>

<item 6>
Moonstruck
Moonstruck is a 1987 American romantic comedy film directed by Norman Jewison and written by John Patrick Shanley.

 It is about a widowed 37-year-old Italian-American woman (Cher) who falls in love with her fiancé's (Danny Aiello) estranged, hot-tempered younger brother
(Nicolas Cage).

 Vincent Gardenia and Olympia Dukakis play supporting roles.
</item 6>

<item 7>
Sleepless in Seattle
Sleepless in Seattle is a 1993 American romantic comedy-drama film directed and co-written by Nora Ephron, based on a story by Jeff Arch.

 It stars Tom Hanks and Meg Ryan, alongside a supporting cast featuring Bill Pullman, Ross Malinger, Rob Reiner, Rosie O'Donnell, Gaby Hoffmann, Victor Garber,
and Rita Wilson.

 The film was a critical and commercial success, grossing over $220 million worldwide.
</item 7>

<item 8>
Mister Buddwing
Mister Buddwing is a 1966 American film drama directed by Delbert Mann and starring James Garner.
</item 8>

<item 9>
The Notebook
The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.

 The film stars Ryan Gosling and Rachel McAdams as a young couple who fall in love in the 1940s.

 Their story is narrated from the present day by an elderly man (portrayed by James Garner) telling the tale to a fellow nursing home resident (played by Gena
Rowlands, who is Cassavetes's mother).
</item 9>

<item 10>
James Garner filmography
The following is the filmography for American actor James Garner.

 Garner was known for prominent roles in films such as "Sayonara" (1957) with Marlon Brando, "Cash McCall" (1960) with Natalie Wood, "The Great Escape" (1963)
with Steve McQueen, "The Thrill of It All" (1963) with Doris Day, "Move Over, Darling" (1963) with Doris Day, "The Wheeler Dealers" (1963) with Lee Remick,
Paddy Chayevsky's "The Americanization of Emily" (1964) with Julie Andrews, "Hour of the Gun" (1967), "Marlowe" (1969) with Bruce Lee, "Support Your Local
Sheriff!

"
</item 10>
=== END ITEM id=7e0b68143a18a037 ===

=== ITEM id=f01a07c7e1ffb3f4 ===
Gold sentences and facts:
Sentence 0: Akademisches Kunstmuseum (English:Academic Art Museum) is an art museum in Bonn, Germany.
  Fact 0.0: Akademisches Kunstmuseum is also known in English as the Academic Art Museum.
  Fact 0.1: Akademisches Kunstmuseum is an art museum.
  Fact 0.2: Akademisches Kunstmuseum is located in Bonn, Germany.
Sentence 1: The Federal City of Bonn (] ) is a city on the banks of the Rhine in the German state of North Rhine-Westphalia, with a population of over 300,000.
  Fact 1.0: The Federal City of Bonn is a city.
  Fact 1.1: The Federal City of Bonn is on the banks of the Rhine.
  Fact 1.2: The Federal City of Bonn is in the German state of North Rhine-Westphalia.
  Fact 1.3: The Federal City of Bonn has a population of over 300,000.

Retrieved items:
<item 1>
Croatian Natural History Museum
The Croatian Natural History Museum (Croatian: "Hrvatski prirodoslovni muzej" ) is the oldest and biggest natural history museum and the main body for natural
history research, preservation and collection in Croatia.

 Located on Dimitrije Demeter Street in Gornji Grad, one of the oldest neighbourhoods of the Croatian capital Zagreb, it owns one of the biggest museum
collections in Croatia, with over 2 million artefacts, including over 1.1 million animal specimens.

 It was founded in 1846 as the "National Museum".

 The National Museum was later split up into five museums, three of which were in 1986 merged as departments of the newly named Croatian Natural History Museum.

 The museum contains a large scientific library open to the public, and publishes the first Croatian natural history scientific journal, "Natura Croatica".
</item 1>

<item 2>
Bonn
The Federal City of Bonn (] ) is a city on the banks of the Rhine in the German state of North Rhine-Westphalia, with a population of over 300,000.

 About 24 km south-southeast of Cologne, Bonn is in the southernmost part of the Rhine-Ruhr region, Germany's largest metropolitan area, with over 11 million
inhabitants.
</item 2>

<item 3>
Harris Museum
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
</item 3>

<item 4>
National Museum of Poland
"National Museum of Poland" is the common name for several of the country's largest and most notable museums. Poland's National Museum comprises several
independent branches, each operating a number of smaller museums.

 The main branch is the National Museum in Kraków (Polish: "Muzeum Narodowe w Krakowie" ), established in 1879 with permanent collections consisting of several
hundred thousand items – kept in big part at the Main Building (along the 3 Maja St.), but also in the eight of its divisions around the city.
</item 4>

<item 5>
Akademisches Kunstmuseum
Akademisches Kunstmuseum (English:Academic Art Museum) is an art museum in Bonn, Germany.

 It is the oldest museum in Bonn and houses the antique collection of the University of Bonn with more than 500 antique statues and reliefs, and over 2,000
originals.

 It is located in a neoclassical building at the southern end of the Hofgarten, near the Electoral Palace.
</item 5>

<item 6>
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 6>

<item 7>
Animal Demography Unit
The Animal Demography Unit (ADU) is a formally recognized research unit of the University of Cape Town (UCT) located within the Department of Biological
Sciences of UCT.

 (The Department of Biological Sciences was formed from the merger of the Department of Botany and the Department of Zoology at the start of the 2013 academic
year).

 The Animal Demography Unit, popularly known as the ADU, was responsible for the management of the First and Second Southern African Bird Atlas Projects SABAP1
and SABAP2.

 The unit has submitted over eight million georeferenced biodiversity records to GBIF.
</item 7>

<item 8>
Alutiiq Museum
The Alutiiq Museum or Alutiiq Museum and Archaeological Repository is a non-profit museum and cultural center dedicated to preserving and sharing the cultural
traditions of the Koniag Alutiiq branch of Sugpiaq ~ Alutiiq of the Alaska Native people.

 The museum is a state-of-the-art facility and is located on the first floor of the "Alutiiq Center" of Kodiak, Alaska.

 Alutiiq Museum is one of three museums in Kodiak.

 The museum is the seventh museum in Alaska and the second "tribal museum" in the United States to be accredited.

 The museum provides tours of its exhibits, laboratory and collections storage facilities to educational groups.

 The museum will accept materials relevant to the prehistoric, historic, and contemporary cultural history of the Native peoples who settled the Koniag Alutiiq
Nation.

 Such materials include, but are not limited to, archaeological, ethnological, photographic, film, audio, archival, and natural history specimens.
</item 8>

<item 9>
National Museum, Kraków
The National Museum in Kraków (Polish: "Muzeum Narodowe w Krakowie" ), popularly abbreviated as MNK, established in 1879, is the main branch of Poland's
National Museum, which has several independent branches with permanent collections around the country.

 The Museum consists of 21 departments which are divided by art period; 11 galleries, 2 libraries, and 12 conservation workshops.

 It holds some 780,000 art objects, spanning from classical archeology to modern art, with special focus on Polish painting.

<ref name="culture.pl/muzea"> </ref><ref name="muzeum/Zbiory"> </ref>
</item 9>

<item 10>
Alutiiq Museum
This cultural center features a gallery, storage for more than 190,000 local artifacts, including faunal materials, ethnobotanical samples, sediment samples,
field notes, photographs, and maps and a research laboratory.

 The Alutiiq Museum is a small repository, but we care for a very large collection with nearly 250,000 items.

 As a newly founded institution, the Alutiiq Museum sought to develop its policies and practices in professional ways.

 The Alutiiq Museum also uses its collections for community-building among the Alutiiq.

 The Alutiiq Museum is supported and governed by the Alutiiq Heritage Foundation and is dedicated to preserving and sharing Alutiiq heritage.
</item 10>
=== END ITEM id=f01a07c7e1ffb3f4 ===

=== ITEM id=62bb3f9d70112e77 ===
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
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 1>

<item 2>
Harrimanella
Harrimanella is a genus of flowering plant in the heath family Ericaceae, with a single species, Harrimanella hypnoides, also known as moss bell heather.

 It was originally named Cassiope hypnoides by Linnaeus (1737) in his Flora Lapponica, but "Harrimanella hypnoides" is now the accepted name at ITIS.

 The species name "hypnoides" means 'like "Hypnum" ', which is a genus mosses.
</item 2>

<item 3>
Ericaceae
The Ericaceae are a family of flowering plants, commonly known as the heath or heather family, found most commonly in acid and infertile growing conditions.

 The family is large, with c. 4250 known species spread across 124 genera, making it the 14th most species-rich family of flowering plants.

 The many well-known and economically important members of the Ericaceae include the cranberry, blueberry, huckleberry, rhododendron (including azaleas), and
various common heaths and heathers ("Erica", "Cassiope", "Daboecia", and "Calluna" for example).
</item 3>

<item 4>
Deutzia silvestrii
Deutzia silvestrii is a plant species in the family Hydrangeaceae.
</item 4>

<item 5>
Deutzia
Deutzia ( or ) is a genus of about 60 species of flowering plants in the family Hydrangeaceae, native to eastern and central Asia (from the Himalayas east to
Japan and the Philippines), and Central America and also Europe.

 By far the highest species diversity is in China, where 50 species occur.
</item 5>

<item 6>
Echeveria
Echeveria is a large genus of flowering plants in the Crassulaceae family, native to semi-desert areas of Central America, Mexico and northwestern South
America.
</item 6>

<item 7>
Psychopsis
Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.

 It was formerly included in the massively paraphyletic "wastebin genus" "Oncidium".

 The genus as a whole is commonly called butterfly orchids, but some species of other orchid genera are also called thus.
</item 7>

<item 8>
Cassiope mertensiana
Cassiope mertensiana is a species of flowering plant known by the common names western moss heather and white mountain heather.
</item 8>

<item 9>
Cassiope
Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.

 They are native to the Arctic and north temperate montane regions.

 The genus is named after Cassiopeia of Greek mythology.

 Common names, shared with several other similar related genera, include heather and heath.

 They have scale-like leaves lying against the stems, and produce solitary bell-shaped flowers in late spring.

 Though hardy, flowers can be damaged by late frosts.
</item 9>

<item 10>
Deutzia ningpoensis
Deutzia ningpoensis is a shrub in the family Hydrangeaceae.

 The species is endemic to China.

 It grows to between 1 and 2.5 metres high and produces panicles of white flowers from May to July in its native range.
</item 10>
=== END ITEM id=62bb3f9d70112e77 ===

=== ITEM id=63412a3f56c6e730 ===
Gold sentences and facts:
Sentence 0:  The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
  Fact 0.0: She was the daughter of the American Consul in London.
  Fact 0.1: She was the first First Lady to be born outside the United States.
  Fact 0.2: She was the first First Lady to be born outside areas that were later to become part of the United States.
  Fact 0.3: Such areas include the Thirteen Colonies.
  Fact 0.4: The distinction of being the first First Lady born outside these areas was not replicated for 192 years.
  Fact 0.5: Melania Trump replicated this distinction 192 years later.
Sentence 1:  Before marrying future President Donald Trump in 2005, she worked as a fashion model; by 2016 she considered herself a "full-time mom."
  Fact 1.0: Before marrying future President Donald Trump in 2005, she worked as a fashion model.
  Fact 1.1: By 2016, she considered herself a 'full-time mom.'

Retrieved items:
<item 1>
List of current United States first spouses
In the United States, the first spouse (first lady for women, first gentleman for men) is the term used to refer to the spouse of a chief executive - that is,
of the spouse of the President of the United States (the First Lady of the United States and the First Gentleman of the United States) and the spouses of the
governors of the 50 U.S. states and U.S. territories (Guam, Northern Mariana Islands, Puerto Rico, American Samoa, and the United States Virgin Islands) and the
spouse of the mayor of the District of Columbia.

 (The spouses of many mayors are often called the "first lady" or "first gentleman" of the city as well, and the use of the terms sometimes extends even to the
spouses of college presidents).
</item 1>

<item 2>
Office of the First Lady of the United States
The Office of the First Lady of the United States is the staff accountable to the First Lady of the United States.

 The office and its responsibilities, while not mandated, have grown as the role of the First Lady has grown and formalized through the history of the United
States.

 The Office of the First Lady is an entity of the White House Office, part of the Executive Office of the President.
</item 2>

<item 3>
Louisa Adams
Louisa Catherine Johnson Adams (February 12, 1775 – May 15, 1852), wife of John Quincy Adams, was First Lady of the United States from 1825 to 1829.

 The daughter of the American Consul in London, she was the first First Lady to be born outside the United States, or outside areas that were later to become
part of the United States, such as the Thirteen Colonies – a distinction that would not be replicated until 192 years later by Melania Trump.
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
Vicki Risch
Vicki Risch is the former First Lady of Idaho and the wife of U.S. Sen. Jim Risch, who served as Governor of Idaho in 2006.

 She became First Lady on May 26, 2006, when her husband succeeded former Gov. Dirk Kempthorne, who resigned to become United States Secretary of the Interior.

 Mrs. Risch succeeded former First Lady Patricia Kempthorne who had held the post for over seven years.

 Mrs. Risch served as First Lady until January 2007, as her husband did not seek a full term as governor, but rather was reelected to his old post as lieutenant
governor.
</item 5>

<item 6>
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
</item 6>

<item 7>
Abigail Adams
Abigail Adams ("née" Smith; November 22  1744 – October 28, 1818) was the closest advisor and wife of John Adams, as well as the mother of John Quincy Adams.

 She is sometimes considered to have been a Founder of the United States, and is now designated as the first Second Lady and second First Lady of the United
States, although these titles were not in use at the time.
</item 7>

<item 8>
List of First Ladies of the United States
The First Lady of the United States is the hostess of the White House.

 The position is traditionally filled by the wife of the President of the United States, but, on occasion, the title has been applied to women who were not
presidents’ wives, such as when the president was a bachelor or widower, or when the wife of the president was unable to fulfill the duties of the First Lady
herself.

 The First Lady is not an elected position; it carries no official duties and receives no salary.

 Nonetheless, she attends many official ceremonies and functions of state either along with or in place of the president.

 Traditionally, the First Lady does not hold outside employment while occupying the office.

 She has her own staff, including the White House Social Secretary, the Chief of Staff, the Press Secretary, the Chief Floral Designer, and the Executive Chef.

 The Office of the First Lady is also in charge of all social and ceremonial events of the White House, and is a branch of the Executive Office of the
President.
</item 8>

<item 9>
Melania Trump
Melania Trump ( ; born Melanija Knavs; ] , Germanized to Melania Knauss; April 26, 1970) is the current First Lady of the United States.

 Before marrying future President Donald Trump in 2005, she worked as a fashion model; by 2016 she considered herself a "full-time mom."
</item 9>

<item 10>
Anna Harrison
Anna Tuthill Symmes Harrison (July 25, 1775 – February 25, 1864), wife of President William Henry Harrison and grandmother of President Benjamin Harrison, was
nominally First Lady of the United States during her husband's one-month term in 1841, but she never entered the White House.

 At the age of 65 years during her husband's presidential term, she is the oldest woman ever to become First Lady, as well as having the distinction of holding
the title for the shortest length of time, and the first person to be widowed while holding the title.

 She was the last First Lady to have been born in British America.
</item 10>
=== END ITEM id=63412a3f56c6e730 ===

=== ITEM id=2124e36c83dc35f0 ===
Gold sentences and facts:
Sentence 0: Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit single in 1986 with "Love Missile F1-11" – and six other EMI single releases.
  Fact 0.0: Martin Degville was born 27 January 1961 in Walsall.
  Fact 0.1: Martin Degville is the lead singer of the UK pop band Sigue Sigue Sputnik.
  Fact 0.2: Martin Degville is a co-songwriter of Sigue Sigue Sputnik.
  Fact 0.3: Sigue Sigue Sputnik had a worldwide hit single in 1986 with 'Love Missile F1-11'.
  Fact 0.4: Sigue Sigue Sputnik has six other EMI single releases.
Sentence 1: Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.
  Fact 1.0: Paul Richard 'Richie' Furay was born May 9, 1944.
  Fact 1.1: Richie Furay is an American singer.
  Fact 1.2: Richie Furay is an American songwriter.
  Fact 1.3: Richie Furay is a Rock & Roll Hall of Fame member.
  Fact 1.4: Richie Furay is best known for forming Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin.
  Fact 1.5: Richie Furay is best known for forming Poco with Jim Messina, Rusty Young, George Grantham, and Randy Meisner.

Retrieved items:
<item 1>
Sunshine Becker
Sunshine Becker (born Sunshine Flower Garcia on July 1, 1972) is an American singer who performed backing vocals for the band Furthur.

 Despite her maiden name, Garcia, she is not related to Jerry Garcia, an incorrect assumption made by some because of her involvement with Furthur, a
post-Garcia incarnation of the Grateful Dead.

 Similarly, despite her first name, Sunshine, she is not to be confused with Sunshine Kesey, daughter of Ken Kesey and Carolyn Adams (aka Mountain Girl or MG),
Jerry Garcia's second wife.
</item 1>

<item 2>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 2>

<item 3>
Martin Degville
Martin Degville is (born 27 January 1961 in Walsall) is the lead singer and co-songwriter of the UK pop band, Sigue Sigue Sputnik – which had a worldwide hit
single in 1986 with "Love Missile F1-11" – and six other EMI single releases.

 Sputnik was formed with ex-Generation X bassist Tony James.
</item 3>

<item 4>
Jerry Garcia
Jerome John Garcia (August 1, 1942 – August 9, 1995) was an American singer-songwriter and guitarist, best known for his work as the lead guitarist and as a
vocalist with the band the Grateful Dead, which came to prominence during the counterculture era in the 1960s.

 Although he disavowed the role, Garcia was viewed by many as the leader or "spokesman" of the group.
</item 4>

<item 5>
Poco: The Songs of Richie Furay
Poco: The Songs of Richie Furay is a compilation album consisting of songs by Richie Furay during his tenure with the band Poco, released in 1980.
</item 5>

<item 6>
Martin Flavin
Martin Archer Flavin (November 2, 1883 – December 27, 1967) was an American playwright and novelist.
</item 6>

<item 7>
Richie Furay
Paul Richard "Richie" Furay (born May 9, 1944) is an American singer, songwriter, and Rock & Roll Hall of Fame member who is best known for forming the bands
Buffalo Springfield with Stephen Stills, Neil Young, Bruce Palmer, and Dewey Martin, and Poco with Jim Messina, Rusty Young, George Grantham and Randy Meisner.

 His best known song (originally written during his tenure in Buffalo Springfield, but eventually performed by Poco, as well) was "Kind Woman", which he wrote
for his wife, Nancy.
</item 7>

<item 8>
Dean Martin
Dean Martin (born Dino Paul Crocetti; June 7, 1917 – December 25, 1995) was an American singer, actor, comedian, and film producer.

 One of the most popular and enduring American entertainers of the mid-20th century, Martin was nicknamed the "King of Cool" for his seemingly effortless
charisma and self-assurance.
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
Dean Martin Sings
Dean Martin Sings is the first studio album by Italian-American singer Dean Martin, released in 1953.

 It is the first long-play 10-inch album recorded by Martin for Capitol Records during two sessions recorded on the evening of November 20, 1952.

 The first session was recorded between 5 and 8 PM and it produced five songs featuring string arrangements.

 "There's My Lover" was recorded but not released.

 After a ninety-minute break, Martin was joined by a brass arrangement to record the remaining four songs.

 Seven of the eight songs on this album appeared in the Martin & Lewis film, "The Stooge".

 Two years later, the songs from this 10-inch album would be combined with four newly popular songs recorded between 1951 and 1953 to create a full-length
12-inch album.

 The 2005 Collectors' Choice reissue added four bonus songs recorded between 1949 and 1953 and was released with alternative cover artwork.
</item 10>
=== END ITEM id=2124e36c83dc35f0 ===

=== ITEM id=15a7d04ef2eb07b1 ===
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
Spencer Park, Queensland
Spencer Park (currently known as Corporate Travel Management Stadium) is a football stadium located in the suburb of Newmarket, north of Brisbane, Queensland,
Australia.

 It is home to National Premier Leagues Queensland team Brisbane City FC.

 The stadium, while officially holding 10,000, seats only 3,000 under the grandstand, the "Gino Merlo Stand".

 The stadium opened in 1963 on the site of an old Brisbane City Council waste disposal site, and underwent a major renovation in 1981 with the construction of
the Gino Merlo Stand.

 The stadium is home to La Rustica restaurant, where traditional Italian foods, including pizza, are served.
</item 2>

<item 3>
Brett McClure
Brett Dallas McClure (born February 19, 1981 in Yakima, Washington) is a retired American gymnast.

 He won a bronze medal in the pommel horse at the 2001 Goodwill Games in Brisbane, Australia, and later helped his U.S. gymnastics team earn a silver in the
team competition at the 2004 Summer Olympics in Athens.

 During his sporting career, McClure has collected two more silver medals in the same program at the World Artistic Gymnastics Championships (2001 and 2003).

 McClure is currently the High Performance Director with USA men's gymnastics.
</item 3>

<item 4>
2002 Commonwealth Games
The 2002 Commonwealth Games, officially known as the XVII Commonwealth Games and commonly known as Manchester 2002 were held in Manchester, England, from 25
July to 4 August 2002.

 The 2002 Games were to be hosted in the United Kingdom to coincide with the Golden Jubilee of Elizabeth II, head of the Commonwealth, and Manchester was
selected for the 2002 Games ahead of London.

 The XVII Commonwealth Games was, prior to the 2012 Summer Olympics, the largest multi-sport event ever to be held in the UK, eclipsing the London 1948 Summer
Olympics in numbers of teams and athletes participating.

 In terms of sports and events, the 2002 Games were the largest Commonwealth Games in history featuring 281 events across 17 sports.
</item 4>

<item 5>
Powerlifting at the 2015 Pacific Games
Powerlifting at the 2015 Pacific Games was held from 9–10 July 2015 in the Sir John Guise Indoor Power Dome at Port Moresby, Papua New Guinea.

 The host nation Papua New Guinea was the dominant team, particularly in the women's divisions where it claimed six of the seven possible gold medals.

 Nauru was the strongest team in the men's divisions, winning four of the eight weight categories.

 Telupe Iosefa received the first ever gold medal won by Tuvalu at the Pacific Games in the 120 kg male division.
</item 5>

<item 6>
Australia at the 2015 Pacific Games
Australia competed at the 2015 Pacific Games in Port Moresby, Papua New Guinea from 4 to 18 July 2015.

 Australia qualified 43 athletes.

 It was the first time that Australia has competed in the Pacific Games.
</item 6>

<item 7>
RAAF Base Amberley
RAAF Base Amberley (ICAO: YAMB) is a Royal Australian Air Force (RAAF) military airbase located 8 km southwest of Ipswich, Queensland in Australia and 50 km
southwest of Brisbane.

 It is currently home to No. 1 Squadron and No. 6 Squadron (operating the F/A-18F Super Hornet), No. 33 Squadron (taking delivery of the Airbus KC-30A) and No.
36 Squadron (operating the Boeing C-17 Globemaster III jet transport).

 Amberley is also home to Army units making up the 9th Force Support Battalion (9FSB).

 Located on 1600 ha , RAAF Amberley is the largest operational base in the RAAF, employing over 5,000 uniformed and civilian personnel.

 There are a variety of other formations on the base such as training colleges and maintenance areas.

 Amberley's largest squadron in terms of personnel is No. 382 Expeditionary Combat Support Squadron RAAF (ECSS) providing both garrison and deployed combat
support.
</item 7>

<item 8>
Anthony Biddle (Paralympian)
Anthony John Biddle, OAM (born 18 June 1975) is an Australian Paralympic tandem cyclist and athlete.

 He was born in the New South Wales city of Gosford.

 He competed in athletics without winning any medals at the 1996 Atlanta Games and the 2000 Sydney Games.

 At the 2004 Athens Games, he switched to cycling and won a gold medal in the Men's 1 km Time Trial Tandem B1–3 event, for which he received a Medal of the
Order of Australia, and a bronze medal in the Men's Sprint Tandem B1–3 event.

 Kial Stewart was his pilot for both events.
</item 8>

<item 9>
Klete Keller
Klete D. Keller (born March 21, 1982) is an American former competition swimmer who won medals at the 2000 Summer Olympics and the 2004 Summer Olympics in the
400-meter freestyle and the 4×200-meter freestyle relay.

 In the 4×200-meter freestyle relay, Keller held off a charging Ian Thorpe in the anchor leg to win the race by 0.13 seconds.

 This was the first time Australia had been beaten in the event in over seven years.

 The American relay of Michael Phelps, Ryan Lochte, Peter Vanderkaay, and Keller are undefeated since the Athens games.

 Vanderkaay, Larsen Jensen, Erik Vendt, and Keller make up the core of the premier American mid-distance/distance freestyle swimmers.
</item 9>

<item 10>
Offshore Festival
The Offshore Festival was a camp-out rock and alternative music festival held during Easter at a farm near Torquay, Victoria, Australia from the late 1990s to
2001.

 It was run by the same organisers as the Falls Festival, held at nearby Lorne on New Year's Eve.

 It interlinked with the Rip Curl Pro surfing event, held at nearby Bells Beach, and festival ticketholders had free admission to the surfing event.
</item 10>
=== END ITEM id=15a7d04ef2eb07b1 ===

=== ITEM id=2415430f922f0f58 ===
Gold sentences and facts:
Sentence 0:  It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
  Fact 0.0: The described act consists of a duet.
  Fact 0.1: Dave Hemingway is a vocalist featured in the duet.
  Fact 0.2: Briana Corrigan is a vocalist featured in the duet.
Sentence 1: Briana Corrigan (born 30 May 1965) is a Northern Irish singer.
  Fact 1.0: Briana Corrigan was born on 30 May 1965.
  Fact 1.1: Briana Corrigan is a Northern Irish singer.

Retrieved items:
<item 1>
Burn Your Playhouse Down – The Unreleased Duets
Burn Your Playhouse Down – The Unreleased Duets is the 60th and final studio album by American country music singer George Jones released on August 19, 2008 on
the Bandit Records label.

 It features duets never before released, including some that were cut from his 1994 duets album "The Bradley Barn Sessions".

 The only new recording in the collection is "You And Me And Time", a song Jones recorded with his daughter by Tammy Wynette, Georgette.

 A music video accompanied the song.

 The album features several duets with artist from outside the country music pantheon, including Mark Knopfler, Leon Russell, and Keith Richards.

 The album peaked at number 15 on the Billboard country albums chart.

 Of the album's title track, Andrew Meuller of "Uncut" opined in July 2013, "The segue from Richards trying to sing like Jones to Jones actually singing like
Jones is hilarious."
</item 1>

<item 2>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 2>

<item 3>
Just a Little Time
Just a Little Time is an album by Lynsey de Paul.

 It was originally released in 30 September 1994 on the Music Deluxe (catalogue number MSCD9) label but has since been released on the Tring International PLC
label (catalogue number JHD128) and on Arc Records/The Magic Collection (Catalogue number MEC 949080).

 All of the songs on the album are written or co-written by de Paul.

 The songs "Sugar Me", Getting a Drag", "Storm in a Teacup and "Dancing on a Saturday Night" are radical re-recordings of her earlier hits performed in an
updated and uptempo fashion.

 "Won't Somebody Dance with Me" is still performed as a ballad, but has been updated and extended with an extra refrain.

 Never before released songs on the album include the ballads "Words Don't Mean A Thing" and title track "Just a Little Time" as well as the more uptempo
"Instant Love" and "Now & Then".
</item 3>

<item 4>
Dave Hemingway
Dave Hemingway (born David Robert Hemingway, 20 September 1960) is an English musician and songwriter, and was a vocalist for the Hull-based band The Beautiful
South until they disbanded in 2007.

 Previously he had been a member of The Housemartins.
</item 4>

<item 5>
Takes a Little Time (Amy Grant song)
"Takes a Little Time" was a maxi-single released in 1997 (see 1997 in music) to promote Amy Grant's album "Behind the Eyes", which was also released that year.

 "Takes a Little Time" included two songs from "Behind the Eyes", as well as a new version of Grant's 1982 Christian radio hit, "El Shaddai".

 The maxi-single was also an enhanced CD (ECD), meaning it had a video that consumers could watch when they put it in their home computer.

 The ECD portion of the CD contained a live acoustic version of the song After the Fire, which Grant later released on her 2003 album "Simple Things"; Grant
stated on Oprah that she had written the song for her mother.
</item 5>

<item 6>
A Little Time
"A Little Time" is a song by The Beautiful South, and is the band's only single to reach number 1 in the UK Singles Chart.

 It consists of a duet featuring vocalists Dave Hemingway and Briana Corrigan.
</item 6>

<item 7>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 7>

<item 8>
Graph query row:
person: Briana Corrigan
nationality: Northern Irish
birthplace: None
birthplace_country: None
nationality_place: None
related_person: Dave Hemingway
person_relationship: None
creative_work: Just a Little Time
Briana_work_relationship: None
Dave_work_relationship: None
</item 8>

<item 9>
Dann Hume
Dann Hume (born Daniel Benjamin Cobbe, 1 September 1987, Whangaparaoa, New Zealand) is a singer-songwriter and record producer.

 Hume is the youngest of three brothers (with Peter Hume and Jon Hume) who make up the alternative rock band, Evermore since 1999.
</item 9>

<item 10>
Houston (Means I'm One Day Closer to You)
"Houston (Means I'm One Day Closer to You)" is a song written by Larry Gatlin and recorded by American country music group Larry Gatlin & the Gatlin Brothers
Band.

 It was released in September 1983 as the first single from the album ""Greatest Hits Vol.

 II"" then included to first track of ""Not Guilty"" (1984).

 "Houston (Means I'm One Day Closer to You)" was the group's third and last number one on the country chart.

 The single went to number one for two weeks and spent a total of fifteen weeks on the country chart.
</item 10>
=== END ITEM id=2415430f922f0f58 ===

=== ITEM id=26827d41e649307f ===
Gold sentences and facts:
Sentence 0: The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.
  Fact 0.0: The Shins are an American indie rock band.
  Fact 0.1: The Shins are from Albuquerque, New Mexico.
  Fact 0.2: The Shins were formed in 1996.
Sentence 1: Paramore is an American rock band from Franklin, Tennessee, formed in 2004.
  Fact 1.0: Paramore is an American rock band.
  Fact 1.1: Paramore is from Franklin, Tennessee.
  Fact 1.2: Paramore was formed in 2004.

Retrieved items:
<item 1>
List of songs recorded by Paramore
American rock band Paramore have recorded songs for five studio albums, a box set, an extended play and two soundtrack albums.

 In 2002, at age 13, vocalist Hayley Williams moved to Franklin, Tennessee, where she met brothers Josh Farro and Zac Farro.

 The band was officially formed by Josh Farro (lead guitar and backing vocals), Zac Farro (drums), Jeremy Davis (bass guitar) and Williams (lead vocals) in
2004, with the later addition of Williams' neighbor Jason Bynum (rhythm guitar).

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling" that year.

 Three singles were released to promote the album.
</item 1>

<item 2>
Souther–Hillman–Furay Band
The Souther Hillman Furay Band (SHF) was a country rock supergroup led by singer-songwriters Richie Furay (Buffalo Springfield, Poco), Chris Hillman (The Byrds,
The Flying Burrito Brothers, Manassas), and J.D. Souther (Longbranch Pennywhistle, noted songwriter for Linda Ronstadt and the Eagles).

 The band recorded two albums during the 1970s before breaking up due to disagreements between the members.
</item 2>

<item 3>
Hayley Williams
Hayley Nichole Williams (born December 27, 1988) is an American singer, songwriter and musician.

 She serves as the lead vocalist, primary songwriter and occasional keyboardist of the rock band Paramore.

 The band was formed in 2004 by Josh Farro, Zac Farro, Jeremy Davis and Williams.

 The band consists of Hayley Williams, Zac Farro and Taylor York.

 The band has five studio albums: "All We Know Is Falling" (2005), "Riot!

" (2007), "Brand New Eyes" (2009), "Paramore" (2013) and "After Laughter" (2017).
</item 3>

<item 4>
After Laughter
After Laughter is the fifth studio album by American rock band Paramore.

 It was released on May 12, 2017, through Fueled by Ramen as a follow-up to "Paramore", their 2013 self-titled album.

 The album was produced by guitarist Taylor York alongside previous collaborator, Justin Meldal-Johnsen.

 It is the band's first album since the return of drummer Zac Farro, who left the band with his brother Josh in 2010, and the departure of former bassist Jeremy
Davis, who left the band in 2015.

 "After Laughter" represents a complete departure from the usual pop punk and alternative rock sound of their previous releases.

 The album touches on themes of exhaustion, depression and anxiety, contrasting the upbeat and vibrant sound of the record.
</item 4>

<item 5>
Singles Club (EP)
Singles Club is an EP released by American rock band Paramore.

 The songs were released as promotional singles between October and December 2011, culminating in the release of a box set containing the three constituent
songs plus "Monster", which was recorded during the same sessions.

 The EP and box set were released on December 14, 2011 on Paramore's website.

 The EP is the first release of new material by Paramore not to include band members Josh and Zac Farro.
</item 5>

<item 6>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 6>

<item 7>
Paramore discography
The American rock band Paramore has released five studio albums, three extended plays, two live albums, eighteen singles, one video album, and nineteen music
videos.

 The band was formed in Franklin, Tennessee, in 2004 by lead vocalist Hayley Williams with guitarists Josh Farro and Taylor York, bassist Jeremy Davis, and
drummer Zac Farro.

 In 2005, Paramore signed with the New York City-based Fueled by Ramen and released their debut album entitled "All We Know Is Falling".

 Three singles were released from the album, but none of them charted.

 The album did not chart in the "Billboard" 200 either, although it peaked at number thirty in the "Billboard" Top Heatseekers.

 "All We Know Is Falling" received Gold certification in the United Kingdom and in July 2014 the RIAA certified the album Gold in the United States.
</item 7>

<item 8>
Paramore (album)
Paramore is the self-titled fourth studio album by American rock band Paramore.

 It was released on April 5, 2013, through Fueled by Ramen as a follow-up to "Brand New Eyes" (2009).

 It is their first full-length album released after the departure of co-founders Josh and Zac Farro in 2010.

 Recorded between April and November 1, 2012, the album was described by the band as being a "statement" and a reintroduction of the band to the world and to
themselves.

 It is the only album without Zac Farro on drums, and the final album recorded with bassist Jeremy Davis before he left the band in 2015.
</item 8>

<item 9>
Zac Farro
Zachary Wayne "Zac" Farro (born June 4, 1990) is an American musician and drummer of the rock band Paramore.

 He is also the younger brother of Josh Farro, who is Paramore's former lead guitarist and backing vocalist.

 After he and his brother exited Paramore in 2010, Josh formed a band named Novel American, which Zac was also a part of.

 Zac is currently the sole member of the band HalfNoise.

 Farro rejoined Paramore on February 2, 2017.
</item 9>

<item 10>
Paramore
Paramore is an American rock band from Franklin, Tennessee, formed in 2004.

 The band currently consists of lead vocalist Hayley Williams, guitarist Taylor York and drummer Zac Farro.
</item 10>
=== END ITEM id=26827d41e649307f ===

=== ITEM id=4490a357c886c1db ===
Gold sentences and facts:
Sentence 0: PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.
  Fact 0.0: PC Magazine is shortened as PCMag.
  Fact 0.1: PC Magazine is an American computer magazine.
  Fact 0.2: PC Magazine is published by Ziff Davis.
Sentence 1:  The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
  Fact 1.0: The magazine focuses on sculpture.
  Fact 1.1: The magazine focuses on sculpture collectors.
  Fact 1.2: The magazine focuses on sculpture enthusiasts.
  Fact 1.3: Painting is also frequently featured in the magazine.
  Fact 1.4: Poetry is also frequently featured in the magazine.

Retrieved items:
<item 1>
Computer Magazine
Computer Magazine (and their website www.ComputerMagazine.com) is a popular magazine and online news site on computing and technology, offering current news and
reviews of popular and new business and consumer technologies, software, hardware, mobile computing, tablets, PCs, Macs, Windows, Linux, telecom, cellular,
wireless, data, cloud and science news on digital technologies and everything in the "tech-sphere and digi-verse", especially focused on information technology,
devices, software and services and related subjects, such as networking, servers, data centers and corporate data infrastructure technologies, and the Internet.

 Their online site, since 1997, is located at ComputerMagazine.com.

 "Computer Magazine" produces industry instructional and a popular ongoing webcast/podcast talk show and performs evaluations and reviews of IT industry
technology products, hardware, software and services with objective reporting widely respected as independent and objective, and trusted in the industry.
</item 1>

<item 2>
Computer Magazine
as well as "Computer Magazine" staff writers, and is currently managed and edited by the industry veteran Christopher Swearingin an MCSE and former CIO and
regarded author as well as contributor/reporter for "Computer Magazine" and other publications.
</item 2>

<item 3>
Computer Press Association
Founded in 1983, the Computer Press Association (CPA) was established to promote excellence in the field of computer journalism.

 The association was composed of working editors, writers, producers, and freelancers who covered issues related to computers and technology.

 The CPA conducted the annual Computer Press Awards, which was the preeminent editorial awards of the computer and technology media.

 The CPA Awards honored outstanding examples in print, broadcast and electronic media.

 Awards were given for print publications, such as PC Magazine; online news media, such as Newsbytes News Network (both were multiple winners); individual
columns and features by well-known journalists such as Steven Levy (author of “”); broadcast awards such as “Best Radio Program”; as well as book awards in
categories such as Best Product Specific Book.

 CPA President Jeff Yablon (1994-1996) developed an updated code of ethics for technology journalists that was adopted by many major trade show groups, most
notably Bruno Blenheim.

 The Computer Press Association disbanded in 2000.
</item 3>

<item 4>
ST/Amiga Format
ST Amiga Format was a computer magazine that covered the Atari ST and Amiga computers.

 It was published by Future plc to cover the ever growing market for the, then-new, 16-bit home computers.

 Issues were equally balanced with coverage for both Commodore Amiga and Atari ST systems.

 Issue 1 included a main feature 'ST or Amiga?

 The Choice is Yours', where the pros and cons for each machine was examined.
</item 4>

<item 5>
Guitar World
Guitar World is a monthly music magazine devoted to guitarists, published since July 1980.

 It contains original interviews, album and gear reviews, and guitar and bass tablature of approximately five songs each month.

 The magazine is published 13 times per year (12 monthly issues and a holiday issue).

 Formerly owned by Harris Publications, Future US bought the magazine in 2003.

 In 2012, NewBay Media bought the Music division of Future US.

 The latter company also published a spin-off title, Guitar Legends, each issue of which typically combined past articles from "Guitar World" under a specific
theme.
</item 5>

<item 6>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 6>

<item 7>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 7>

<item 8>
Windows: The Official Magazine
Windows: The Official Magazine is a technology magazine produced by Future plc in association with Microsoft, and published worldwide.

 The title describes itself as "A PC magazine for real life", and contains news, features, guides and reviews designed to show readers how to make better use of
their Windows PC.
</item 8>

<item 9>
Sculptural Pursuit
Sculptural Pursuit was a quarterly art/literary magazine published by Hammer & Pen Productions, a Denver, Colorado publishing company.

 The magazine focuses on sculpture, its collectors and enthusiasts, but painting and poetry are also frequently featured.
</item 9>

<item 10>
Computer Magazine
"Computer Magazine" is a free publication (in addition to their webcasts and other resources) sponsored by the nonprofit UTCP (United Technology and Computing
Professionals) organization, and as such charges no fees for the publication nor is influenced by advertising, so their reviews are relied on in the industry
and considered unbiased and thorough.

 "Computer Magazine" is one of the early large technology publications and resources available on the web still existent and thriving today and that has
remained independent.

 ComputerMagazine.com is a tech news and resources consolidator that publishes part of the site in a semi-time line/blogging format that is popular among their
wide following of subscriber and non-subscriber readers, allowing readers to respond and comment on various articles.

 Site contributors include many of the well known technology authors, experts and publication sources, content and articles are provided by major technology
syndicators and by external expert technology sources (such as "Computer World", "Information Week", "Network World", "Wired," "Time", etc.)
</item 10>
=== END ITEM id=4490a357c886c1db ===

=== ITEM id=fa9e9bfd5d2e60a7 ===
Gold sentences and facts:
Sentence 0: Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.
  Fact 0.0: Aleksandr Ivanovsky lived from 1881 to 1968.
  Fact 0.1: Aleksandr Ivanovsky was a Russian screenwriter.
  Fact 0.2: Aleksandr Ivanovsky was a Russian film director.
  Fact 0.3: Aleksandr Ivanovsky worked during the Soviet era.
Sentence 1: Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera" (1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".
  Fact 1.0: Arthur Lubin was born on July 25, 1898.
  Fact 1.1: Arthur Lubin died on May 12, 1995.
  Fact 1.2: Arthur Lubin was an American film director.
  Fact 1.3: Arthur Lubin was an American producer.
  Fact 1.4: Arthur Lubin directed several 'Abbott & Costello' films.
  Fact 1.5: Arthur Lubin directed 'Phantom of the Opera', released in 1943.
  Fact 1.6: Arthur Lubin directed the 'Francis the Talking Mule' series.
  Fact 1.7: Arthur Lubin created the talking-horse TV series 'Mister Ed'.

Retrieved items:
<item 1>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 1>

<item 2>
Vladimir Ashkenazy
Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized
solo pianist, chamber music performer, and conductor.

 He is originally from Russia and has held Icelandic citizenship since 1972.

 He has lived in Switzerland since 1978.

 Ashkenazy has collaborated with well-known orchestras and soloists.

 In addition, he has recorded a large storehouse of classical and romantic works.

 His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
</item 2>

<item 3>
Aleksandr Ivanovsky
Aleksandr Ivanovsky (1881–1968) was a Russian screenwriter and film director who worked during the Soviet era.

 Ivanovsky was awarded the Stalin Prize in 1941, for his work on the 1940 film "Musical Story".

 His 1944 operetta film "Silva" was one of the most popular releases in the Soviet Union that year.
</item 3>

<item 4>
Aleksandra Sokolovskaya
Aleksandra Lvovna Sokolovskaya (Russian: Александра Львовна Соколовская ; 1872 – c. 1938) was a Russian Marxist revolutionary and Leon Trotsky's first wife.

 She perished in the Great Purges no earlier than 1938.
</item 4>

<item 5>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 5>

<item 6>
Vladimir Bazarov
Vladimir Alexandrovich Bazarov ( Влади́мир Алекса́ндрович База́ров; 1874–1939) was a Russian Marxist revolutionary, journalist, philosopher, and economist, born
Vladimir Alexandrovich Rudnev.

 Bazarov is best remembered as a pioneer in the development of economic planning in the Soviet Union.
</item 6>

<item 7>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 7>

<item 8>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 8>

<item 9>
Aleksei Petrenko
Aleksei Vasilyevich Petrenko (Russian: Алексей Васильевич Петренко ; 26 March 1938 – 22 February 2017) was a Soviet and Russian film and stage actor.

 He played Grigori Rasputin in the Elem Klimov's historical drama
</item 9>

<item 10>
Alexander Artemev
Alexander Vladimirovich "Sasha" Artemev (Russian: Александр Владимирович "Саша" Артемьев , Belarusian: Аляксандр Уладзіміравіч Арцем'еў ; born August 29, 1985)
is a retired American artistic gymnast.

 Artemev was a member of the bronze medal-winning U.S. team at the 2008 Olympic Games.

 He is the 2006 all-around U.S. national champion.

 Known for his ability on pommel horse, he is the 2007 and 2008 U.S. national champion on the pommel horse and won the bronze medal on the event at the 2006
World Championships.
</item 10>
=== END ITEM id=fa9e9bfd5d2e60a7 ===

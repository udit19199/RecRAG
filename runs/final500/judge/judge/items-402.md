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

This file has 10 items.

=== ITEM id=b865b9d7b8bc7c19 ===
Gold sentences and facts:
Sentence 0:  Akrofuom is located in the Obuasi Municipal district of the Ashanti Region of Ghana.
  Fact 0.0: Akrofuom is located in the Obuasi Municipal district.
  Fact 0.1: The Obuasi Municipal district is in the Ashanti Region of Ghana.
Sentence 1: The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.
  Fact 1.0: The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.

Retrieved items:
<item 1>
Ashanti Region
The Ashanti Region is located in south Ghana and is third largest of 10 administrative regions, occupying a total land surface of 24389 km2 or 10.2 per cent of
the total land area of Ghana.

 In terms of population, however, it is the most populated region with a population of 4,780,380 according to the 2010 census, accounting for 19.4% of Ghana’s
total population.

 The Ashanti Region is known for its major gold bar and cocoa production.

 The largest city and regional capital is Kumasi.
</item 1>

<item 2>
Tema
Tema is a city on the Bight of Benin and Atlantic coast of Ghana.

 It is located 25 km east of the capital city; Accra, in the region of Greater Accra, and is the capital of the Tema Metropolitan District.

 As of 2013, Tema is the eleventh most populous settlement in Ghana, with a population of approximately 161,612 people – a marked decrease from its 2005 figure
of 209,000.

 The Greenwich Meridian (00 Longitude) passes directly through the city.

 Tema is locally nicknamed the "Harbour Town" because of its status as Ghana's largest seaport.
</item 2>

<item 3>
Obuasi Municipal District
The Obuasi Municipality District is one of the twenty-seven (27) districts of the Ashanti Region of Ghana.

 The capital of the Obuasi Municipality is Obuasi.
</item 3>

<item 4>
Dodi Papase
Dodi Papase is a town in the Volta Region of Ghana.

 With a population of approximately 5,254 ,Dodi Papase is the second-most-populous city in the Volta region of Ghana.

 It is located in eastern Ghana on the banks of the Asukawkaw River and is known for the Dodi Papase Community Secondary School, a second cycle institution
(senior high school).
</item 4>

<item 5>
Techimantia
Techimantia is a city located in the Tano District of the Brong Ahafo Region of Ghana.

 It is one of the large cities in the region.

 It is believed to be originally inhabited by Hunters for the Ashanti King from Kumasi.

 These Hunters originally settled in the region after they found that there were lots of elephants in the region which could be hunted for the Ashanti King.The
name of the King is Oti Ampem.Currently,the chief of the town is no more(dead).

The occupation of the people in the town is large scale tomatoes farming.Techimantia is one of the largest towns in the region but there are few productive or
government work in the town.This was attributed to the curse from Okomfo Anokye of the Ashanti Kingdom.
</item 5>

<item 6>
Western Region (Ghana)
The Western Region is located in south Ghana, spreads from the Ivory Coast border in the west to the Central region in the east, includes the capital and large
twin city of Sekondi-Takoradi on the coast, coastal Axim, and a hilly inland area including Elubo.

 It includes Ghana's southernmost location, Cape Three Points, where crude oil was discovered in commercial quantities in June 2007.

 The Western Region enjoys a long coastline that stretches from South Ghana's border with Ivory Coast to the Western region's boundary with the Central Region
on the east.
</item 6>

<item 7>
Tafo
Tafo is a town in Kumasi Metropolitan District in the Ashanti Region of Ghana near the regional capital Kumasi.

 Tafo is the thirtieth most populous settlement in Ghana, in terms of population, with a population of 60,919 people.

 Because of the town's population and housing development in recent years, it is debatable whether Tafo is still regarded as a separate town, or already a
suburb of Kumasi, the capital of the Ashanti region.

 The town is near Kumasi, with a distance of approximately 3.3 kilometers to the center of a similar name sounding village named New Tafo and must be
distinguished from Tafo.

 Tarkwa is located just 4.6 km away from Tafo.

 The city center of Kumasi is located approximately 9.8 kilometers away.

 Tafo is one of the urban constituencies of the Kumasi Metropolitan Assembly, the town's parliamentary candidate shall have one direct seat to the Parliament of
Ghana.
</item 7>

<item 8>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 8>

<item 9>
Akrofuom (Ghana parliament constituency)
Akrofuom is one of the constituencies represented in the Parliament of Ghana.

 It elects one Member of Parliament (MP) by the first past the post system of election.

 Akrofuom is located in the Obuasi Municipal district of the Ashanti Region of Ghana.
</item 9>

<item 10>
Taifa, Accra
Taifa is a town in the Ga East Municipal District, a district in the Greater Accra Region of south-eastern Ghana near the capital Accra.

 Taifa is the twenty-sixth largest settlement in Ghana, in terms of population, with a population of 68,459 people.

 Taifa is located in the northwest suburbs area of Accra.

 It has a breakpoint on a railway line and a small park located on the northern edge of the location of the Taifa Ghana Atomic Energy Commission.

 At the Ghana 2000 census of 26 March 2000, the population was 26,145 inhabitants living in the city.

 Projections of 1 January 2007 estimated the population to be 48,927 inhabitants.

 In the census of 1984 there was only 1,009 inhabitants.

 The strong population growth of the Town is influenced by, among other things, a large number of illegal immigrants from west African countries who move to
towns and villages near the industrial town of Tema, just to find a job.
</item 10>
=== END ITEM id=b865b9d7b8bc7c19 ===

=== ITEM id=4d418bbfdf43383f ===
Gold sentences and facts:
Sentence 0: The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in 1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
  Fact 0.0: The Karlsschrein, English name Shrine of Charlemagne, is in Aachen Cathedral.
  Fact 0.1: The Karlsschrein was made in Aachen.
  Fact 0.2: The Karlsschrein was made at the command of Frederick II, Holy Roman Emperor.
  Fact 0.3: The Karlsschrein was completed in 1215.
  Fact 0.4: Frederick II's grandfather was Frederick Barbarossa.
  Fact 0.5: Frederick Barbarossa exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen.
  Fact 0.6: Frederick Barbarossa exhumed Charlemagne's bones in 1165.
  Fact 0.7: The exhumation of Charlemagne's bones happened before the Karlsschrein was completed.
Sentence 1: Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the Holy Roman Emperor from 1155 until his death.
  Fact 1.0: Frederick I's German name is "Friedrich I".
  Fact 1.1: Frederick I's Italian name is "Federico I".
  Fact 1.2: Frederick I lived from 1122 to 10 June 1190.
  Fact 1.3: Frederick I is also known as Frederick Barbarossa.
  Fact 1.4: Frederick Barbarossa's Italian name is "Federico Barbarossa".
  Fact 1.5: Frederick I was the Holy Roman Emperor from 1155 until his death.

Retrieved items:
<item 1>
Eric III of Denmark
Eric III Lamb (Danish: "Erik III Lam" ) (  1120 – 27 August 1146) was the King of Denmark from 1137 until 1146.

 He was the grandson of Eric I of Denmark and the nephew of Eric II of Denmark, whom he succeeded on the throne.

 He abdicated in 1146, as the first and only Danish monarch to do so.

 His succession led to a period of civil war between Sweyn III, Canute V, and Valdemar I.
</item 1>

<item 2>
Ernest Augustus, King of Hanover
Ernest Augustus (5 June 1771 – 18 November 1851) was King of Hanover from 20 June 1837 until his death.

 He was the fifth son and eighth child of King George III of the United Kingdom and Hanover.

 As a fifth son, Ernest seemed unlikely to become a monarch, but none of his four elder brothers had a legitimate son who survived infancy.

 The Salic Law, which barred succession to or through a female, prevailed in Hanover; therefore, when his elder brother King William IV died in 1837, Ernest
succeeded him as King of Hanover.

 In the United Kingdom the succession to the monarchy was determined by primogeniture and his niece Victoria succeeded to the throne, thus ending the personal
union between the British Isles and Hanover that had existed since 1714.
</item 2>

<item 3>
Karlsschrein
The Karlsschrein (English: Shrine of Charlemagne ) in Aachen Cathedral was made in Aachen at the command of Frederick II, Holy Roman Emperor and completed in
1215, after Frederick II's grandfather, Frederick Barbarossa had exhumed Charlemagne's bones from their resting place in the Palatine Chapel, Aachen in 1165.
</item 3>

<item 4>
Eric II of Denmark
Eric II the Memorable (Danish: "Erik II Emune" ) (  1090 – 18 September 1137) was king of Denmark between 1134 and 1137.

 Eric was an illegitimate son of Eric I of Denmark, who ruled Denmark from 1095 to 1103.

 Eric the Memorable rebelled against his uncle Niels of Denmark, and was declared king in 1134.

 He punished his adversaries severely, and rewarded his supporters handsomely.

 He was killed by a subject in 1137 and was promptly succeeded by his nephew Eric III of Denmark.
</item 4>

<item 5>
Harald II of Denmark
Harald II of Denmark (died 1018) was King of Denmark from 1014 until his death in 1018.

 He was the youngest son of Sweyn Forkbeard and Gunhild of Wenden, and was regent while his father was fighting Ethelred the Unready in England.

 He inherited the Danish throne in 1014, and held it while his brother, the later king Cnut the Great conquered England.

 After his death in 1018, he was succeeded by Cnut the Great.

 Little detail is known about Harald II.
</item 5>

<item 6>
List of Holy Roman Empresses
Holy Roman Empress or Empress of the Holy Roman Empire is the title given to the consort (wife) or regent of the Holy Roman Emperor.

 The elective dignity of Holy Roman Emperor was restricted to males only, therefore there was never a Holy Roman Empress regnant, though women such as Theophanu
or Maria Theresa of Austria, who controlled the power of rule, served as de facto Empresses regnant.
</item 6>

<item 7>
Frederick I, Holy Roman Emperor
Frederick I (German: "Friedrich I" , Italian: "Federico I" ; 1122 – 10 June 1190), also known as Frederick Barbarossa (Italian: "Federico Barbarossa" ), was the
Holy Roman Emperor from 1155 until his death.

 He was elected King of Germany at Frankfurt on 4 March 1152 and crowned in Aachen on 9 March 1152.

 He became King of Italy in 1155 and was crowned Roman Emperor by Pope Adrian IV on 18 June 1155.

 Two years later, the term "sacrum " ("holy") first appeared in a document in connection with his Empire.

 He was later formally crowned King of Burgundy, at Arles on 30 June 1178.

 He was named "Barbarossa " by the northern Italian cities which he attempted to rule: Barbarossa means "red beard" in Italian; in German, he was known as
"Kaiser Rotbart ", which has the same meaning.
</item 7>

<item 8>
Harald II ruled Denmark (1014–18), succeeding father Sweyn Forkbeard while brother Cnut conquered England; Cnut succeeded him. Christopher I (1252–59), son of
Valdemar II, succeeded Abel. Epirus’s Ptolemy briefly ruled after Pyrrhus II, dying on campaign (or assassinated); niece Deidamia followed. Chu’s King Kang
(559–545 BC) fought Jin and Wu; son Jia’ao succeeded
</item 8>

<item 9>
Karl Anselm, 4th Prince of Thurn and Taxis
Karl Anselm, 4th Prince of Thurn and Taxis, full German name: "Karl Anselm Fürst von Thurn und Taxis" (2 June 1733, Frankfurt am Main, Free Imperial City of
Frankfurt, Holy Roman Empire – 13 November 1805, Winzer bei Regensburg, Electorate of Bavaria, Holy Roman Empire) was the fourth Prince of Thurn and Taxis,
Postmaster General of the Imperial Reichspost, and Head of the Princely House of Thurn and Taxis from 17 March 1773 until his death on 13 November 1805.

 Karl Anselm served as "Prinzipalkommissar" at the Perpetual Imperial Diet in Regensburg for Joseph II, Holy Roman Emperor and Francis II, Holy Roman Emperor
from 1773 to 1797.
</item 9>

<item 10>
Prince Christian of Hanover (born 1985)
Prince Christian of Hanover (Christian Heinrich Clemens Paul Frank Peter Welf Wilhelm-Ernst Friedrich Franz Prince of Hanover, Duke of Brunswick and Lüneburg;
born 1 June 1985) is the younger son of Ernst August, Prince of Hanover, and his first wife Chantal Hochuli.

 He is the second in the line of succession to the former Hanoverian throne, after his elder brother Prince Ernst August.

 As a descendant of George III of the United Kingdom, Christian is also in the line of succession to the British throne.
</item 10>
=== END ITEM id=4d418bbfdf43383f ===

=== ITEM id=10d0c860dd598475 ===
Gold sentences and facts:
Sentence 0: Terri Kathleen Nunn (born June 26, 1961), is an American singer and actress.
  Fact 0.0: Terri Kathleen Nunn was born June 26, 1961.
  Fact 0.1: Terri Kathleen Nunn is an American singer.
  Fact 0.2: Terri Kathleen Nunn is an American actress.
Sentence 1: Sonya Scarlet (born 2 April 1980) is the singer and lyricist of the Italian extreme gothic metal band Theatres des Vampires.
  Fact 1.0: Sonya Scarlet was born 2 April 1980.
  Fact 1.1: Sonya Scarlet is the singer of the Italian extreme gothic metal band Theatres des Vampires.
  Fact 1.2: Sonya Scarlet is the lyricist of the Italian extreme gothic metal band Theatres des Vampires.

Retrieved items:
<item 1>
Terri Nunn
Terri Kathleen Nunn (born June 26, 1961), is an American singer and actress.

 She is best known as the lead vocalist of the new wave/synthpop band Berlin.
</item 1>

<item 2>
Cherie Currie
Cherie Ann Currie (born November 30, 1959) is an American musician (instrumentalist, singer, songwriter), actress and artist.

 Currie was the lead vocalist of the Runaways, a rock band from Los Angeles, in the mid-to-late 1970s.

 After the Runaways, she became a solo artist.

 Then she teamed up with her identical twin sister, Marie Currie, and released an album with her.

 They released a duet "Since You Been Gone" which charted number 95 on US charts.

 Their band was called Cherie and Marie Currie.

 She is also well known for her role in the movie "Foxes".
</item 2>

<item 3>
Park So-yeon (singer)
Park So-yeon (born October 5, 1987), better known mononymously as Soyeon, is a South Korean singer and actress.

 She is best known as the former main vocalist and member of South Korean girl group T-ara.
</item 3>

<item 4>
Sonya Scarlet
Sonya Scarlet (born 2 April 1980) is the singer and lyricist of the Italian extreme gothic metal band Theatres des Vampires.
</item 4>

<item 5>
Kim Tae-yeon
Kim Tae-yeon (born March 9, 1989), referred to as Taeyeon, is a South Korean singer.

 She had been a trainee at S.M. Entertainment's Starlight Academy during her middle school years before debuting as a member of the agency's girl group, Girls'
Generation, in 2007.

 Since then, she has risen to prominence due to the group's success on the Asian music scene and further participated in the agency's projects Girls'
Generation-TTS and SM the Ballad.

 Aside from group activities, she has also recorded songs for various television dramas and movies.
</item 5>

<item 6>
Saffron (singer)
Samantha Marie Sprackling (born 3 June 1968, Lagos, Nigeria) more commonly known as Saffron, is a singer, songwriter, actress and the lead singer of the
electronica band Republica.

 She also performed in London's "Starlight Express" for two years, appeared in the music video for the Chesney Hawkes hit, "The One and Only", and as a dancer
in the music video for N-Joi's single, "Mindflux".
</item 6>

<item 7>
Moment of Truth (Terri Nunn album)
Moment of Truth is the début and sole solo album from American singer and actress Terri Nunn, best known as lead singer of the American New Wave/Synthpop band
Berlin.
</item 7>

<item 8>
Sonya Yoncheva
Sonya Yoncheva (Bulgarian: Соня Йончева, born 25 December 1981) is a Bulgarian operatic soprano.
</item 8>

<item 9>
Briana Corrigan
Briana Corrigan (born 30 May 1965) is a Northern Irish singer.

 She was the first female singer for The Beautiful South, from 1988 to 1992.
</item 9>

<item 10>
Rachel Willis-Sørensen
Rachel Willis-Sørensen (born 1984) is an American operatic soprano.

 She was a winner of the Metropolitan Opera National Council Auditions in 2010.

 In 2014, she won first prize, the Birgit Nilsson Prize, and the Zarzuela Prize of the Operalia competition.

 Other awards include first place in the 2009 Eleanor McCollum Competition for Young Singers in Houston and the 2010 Sara Tucker Grant from the Richard Tucker
Music Foundation.
</item 10>
=== END ITEM id=10d0c860dd598475 ===

=== ITEM id=2c6c683c84a7839c ===
Gold sentences and facts:
Sentence 0:  He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.
  Fact 0.0: The person described in this sentence was the brother of Alfred Jodl.
  Fact 0.1: Alfred Jodl was Chief of the Operations Staff of the OKW.
Sentence 1: Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
  Fact 1.0: Alfred Josef Ferdinand Jodl lived from 10 May 1890 to 16 October 1946.
  Fact 1.1: Alfred Jodl was a German general.
  Fact 1.2: Alfred Jodl was a war criminal during World War II.
  Fact 1.3: Alfred Jodl served as Chief of the Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).

Retrieved items:
<item 1>
Hasso von Wedel (general)
Hasso von Wedel (20 November 1898 – 3 January 1961) was a German general who commanded the Wehrmacht Propaganda Troops during World War II.

 He was directly subordinate to the head of OKW Operations Staff ("Wehrmachtführungsstab, WFSt.")

, General Alfred Jodl.

 Wedel's Propaganda Department had control over the propaganda units and served to mediate between them and the Reich Propaganda Ministry of Joseph Goebbels.
</item 1>

<item 2>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 2>

<item 3>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 3>

<item 4>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 4>

<item 5>
Friedrich Jeckeln
Friedrich Jeckeln (2 February 1895 – 3 February 1946) was a high-ranking official in the SS of Nazi Germany who served as an SS and Police Leader in the
occupied Soviet Union during World War II.

 Jeckeln was the commander of one of the largest collection of "Einsatzgruppen" and was personally responsible for ordering and organizing the deaths of over
100,000 Jews, Slavs, Romani, and other "undesirables".

 After the end of World War II, Jeckeln was convicted for his crimes by a Soviet military tribunal in Riga, Latvia and executed in 1946.
</item 5>

<item 6>
Algernon Willis
Admiral of the Fleet Sir Algernon Usborne Willis (17 May 1889 – 12 April 1976) was a Royal Navy officer.

 He served in the First World War and saw action at the Battle of Jutland in May 1916.

 He also served in the Second World War as Commander-in-Chief, South Atlantic in which capacity he led actions against German and Japanese raiding ships.

 He continued his war service as Flag Officer commanding 3rd Battle Squadron and Second in command of the Eastern Fleet and then as Flag Officer commanding
Force H, the force which covered North African Operations, the Allied invasion of Sicily in July 1943 and then the Allied invasion of Italy in September 1943.

 He spent the final years of the war as Commander-in-Chief, Levant, in which capacity he conducted naval operations in support of the Dodecanese Campaign, and
then as Second Sea Lord, in which capacity he arranged the manpower for the campaign in the Pacific Ocean against the Imperial Japanese Navy.

 After the war he served as Commander-in-Chief, Mediterranean Fleet, in which role he was faced with unrest in Mandatory Palestine, before he became
Commander-in-Chief, Portsmouth.
</item 6>

<item 7>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 7>

<item 8>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 8>

<item 9>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 9>

<item 10>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 10>
=== END ITEM id=2c6c683c84a7839c ===

=== ITEM id=7f8ce6f8d8bcadea ===
Gold sentences and facts:
Sentence 0: Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris Institute or Art School, Harris Technical School and the Harris Orphanage.
  Fact 0.0: Edmund Robert Harris lived c. 1804 to 27 May 1877.
  Fact 0.1: Edmund Robert Harris was an English lawyer.
  Fact 0.2: Edmund Robert Harris was from Preston, Lancashire, UK.
  Fact 0.3: Edmund Robert Harris was the principal benefactor of the Harris Museum.
  Fact 0.4: Edmund Robert Harris was the principal benefactor of the Harris Institute or Art School.
  Fact 0.5: Edmund Robert Harris was the principal benefactor of the Harris Technical School.
  Fact 0.6: Edmund Robert Harris was the principal benefactor of the Harris Orphanage.
Sentence 1: The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
  Fact 1.0: The Harris Museum, Art Gallery & Preston Free Public Library is a museum building in Preston.
  Fact 1.1: The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed building.

Retrieved items:
<item 1>
Rice Northwest Museum of Rocks and Minerals
The Rice Northwest Museum of Rocks and Minerals is a non-profit museum in Hillsboro, Oregon, United States.

 Located just north of the Sunset Highway on the northern edge of Hillsboro, the earth science museum is in the Portland metropolitan area.

 Opened in 1997, the museum’s collections date to the 1930s with the museum housed in a home built to display the rock and mineral collections of the museum
founders.

 The ranch style home is listed on the National Register of Historic Places, the first of its kind listed in Oregon.

 In 2015 the museum became a Smithsonian Affiliate museum.
</item 1>

<item 2>
J. Robert Donnelly Husky Heritage Sports Museum
The J. Robert Donnelly Husky Heritage Sports Museum is a public museum located on the University of Connecticut's main campus at Storrs, Connecticut.

 The museum documents and celebrates UConn's intercollegiate athletics.

 Opening its doors on January 19, 2002, the 2,700-square-foot museum was named in honor of benefactor and 1940 Connecticut basketball and football captain J.
Robert (Bob) Donnelly (1971-2005).

 Exhibits include national and regional championship trophies, trading cards, photographs, and various sports memorabilia, as well as a six-screen video wall
replaying moments of triumph for the UConn Huskies.

 A life-size fiberglass sculpture of Jonathan the Husky, UConn's mascot, greets visitors at the museum entrance.

 The Connecticut Basketball Rotunda, featuring NCAA championship trophies and life-size cutouts of Ray Allen and Rebecca Lobo, is among the museum's permanent
exhibits.
</item 2>

<item 3>
Hanbys Corner, Delaware
The last Hanby to occupy the property, Albert T. Hanby (1881-1947), another son of Samuel, attended West Chester State College before getting his law degree
from Penn Law School.

 Albert became a Philadelphia lawyer and left his farm at Hanby's Corner to be used for the good of "all the children in Delaware".

 He and his wife created a foundation in 1945 to protect the property from further development.

 Today the YMCA operates their Hanby Camp there, and the trust provide scholarships for kids who might not otherwise be able to attend.

 In 2007 that support was over $70,000.
</item 3>

<item 4>
Edmund Robert Harris
Edmund Robert Harris (c. 1804 – 27 May 1877) was an English lawyer from Preston, Lancashire, UK who was the principal benefactor of the Harris Museum, Harris
Institute or Art School, Harris Technical School and the Harris Orphanage.
</item 4>

<item 5>
Harris Museum
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
</item 5>

<item 6>
HSBC Building (Hong Kong)
HSBC Main Building () is a headquarters building of The Hongkong and Shanghai Banking Corporation, which is today a wholly owned subsidiary of London-based HSBC
Holdings.

 It is located along the southern side of Statue Square near the location of the old City Hall, Hong Kong (built in 1869, demolished in 1933).

 The previous HSBC building was built in 1935 and pulled down to make way for the current building.

 The address remains as 1 Queen's Road Central (the north facing side of the building was served by Des Voeux Road, which was seashore, making Queen's Road was
the main entrance, in contrast to modern day that usually accessed the bank from Des Voeux Road).

 The building can be reached from Exit K of Central MTR Station and facing Statue Square.
</item 6>

<item 7>
The Sports Museum
The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.

 The museum was founded in 1977 in nearby Cambridge, Massachusetts before eventually moving to Boston to the Garden, which has been home to Celtics and Bruins
for twenty years.

 The museum's exhibits focus on the history of various sports in the Boston area, including the Boston Bruins, Boston Celtics, Boston Red Sox, New England
Patriots, Boston Marathon, New England Revolution, boxing, college sports, and others.

 The museum features diverse sports memorabilia, curiosities, and life-size sculptures of Bobby Orr, Larry Bird, Carl Yastrzemski and Ted Williams by Rhode
Island sculptor Armand LaMontagne.

 The museum is normally open daily between 10 AM and 4 PM, but may close when an event or game is scheduled in the Garden.

 It is located on a subway stop.
</item 7>

<item 8>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 8>

<item 9>
Edmund de Unger
Edmund Robert Anthony de Unger (Hungarian: "Odon Antal Robert de Unger" , b 6 August 1918, Budapest - d 25 January 2011, Ham, Surrey) was a Hungarian-born
property developer and art collector.

 In London he built up the Keir Collection, one of the greatest post-war collections of Islamic art, bequeathed in 2008 to the Pergamon Museum of Islamic Art in
Berlin.

 The arrangement for the museum to curate the collection came to an end in July 2012.

 The collection is now hosted by the Dallas Museum of Art as of May 2014 for a 15-year renewable loan.
</item 9>

<item 10>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 10>
=== END ITEM id=7f8ce6f8d8bcadea ===

=== ITEM id=88b44f2527910520 ===
Gold sentences and facts:
Sentence 0:  Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great America).
  Fact 0.0: The roller coaster was built in 1988 by the TOGO company.
  Fact 0.1: The roller coaster built by the TOGO company was the fastest roller coaster in the world when it was built.
  Fact 0.2: The roller coaster built by the TOGO company took the record for fastest roller coaster in the world from American Eagle at Six Flags Great America.
Sentence 1: TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky cycles and other amusement rides.
  Fact 1.0: TOGO (株式会社トーゴ, Kabushiki-gaisha Tōgo) was a Japanese amusement ride company.
  Fact 1.1: TOGO built roller coasters.
  Fact 1.2: TOGO built giant wheels.
  Fact 1.3: TOGO built carousels.
  Fact 1.4: TOGO built flumes.
  Fact 1.5: TOGO built dark rides.
  Fact 1.6: TOGO built sky cycles.
  Fact 1.7: TOGO built other amusement rides.

Retrieved items:
<item 1>
White Cyclone
White Cyclone (ホワイトサイクロン , Howaito Saikuron ) is a wooden roller coaster at Nagashima Spa Land in Mie Prefecture, Japan.

 At 1700 m in length, White Cyclone is the third longest wooden roller coaster in the world, and is the longest wooden roller coaster outside of the United
States.

 Despite its length, White Cyclone is still considerably shorter than the 2479 m Steel Dragon 2000, the world's longest steel roller coaster, which is also at
Nagashima Spa Land.

 In addition to being the third longest wooden roller coaster, White Cyclone is the seventh tallest wooden roller coaster in the world and the fourth tallest
wooden roller coaster outside the United States.

 A single ride on the White Cyclone costs ¥1,000 (approximately $9 USD), and the ride is restricted to those individuals above 1.3 m in height; and those
individuals under 54 years of age.
</item 1>

<item 2>
Bandit (Yomiuriland)
Bandit (バンデット , Bandetto ) is a steel roller coaster located at Yomiuriland in the city of Inagi, near Tokyo, Japan.

 Built in 1988 by the TOGO company, it was the fastest roller coaster in the world when it was built (taking the record from American Eagle at Six Flags Great
America).

 It lost the record to Magnum XL-200 at Cedar Point one year later.
</item 2>

<item 3>
List of roller coaster rankings
Roller coasters are amusement rides developed for amusement parks and modern theme parks.

 During the 16th and 17th centuries, rides consisting of wooden sleds that took riders down large slides made from ice were popular in Russia.

 The first roller coasters, where the train was attached to a wooden track, first appeared in France in the early 1800s.

 Although wooden roller coasters are still being produced, steel roller coasters are more common and can be found on every continent except Antarctica.
</item 3>

<item 4>
Anton Schwarzkopf
Anton Schwarzkopf (8 July 1924 – 30 July 2001) was a German engineer of amusement rides, and founder of the Schwarzkopf Industries Company, which built numerous
amusement rides and large roller coasters for both amusement parks and traveling funfairs.
</item 4>

<item 5>
Mack Rides
Mack Rides GmbH & Co KG, also known simply as Mack Rides, is a German company that designs and constructs amusement rides.

 Mack Rides in headquartered in Waldkirch, Baden-Württemberg, Germany.

 Mack Rides is one of the world's oldest amusement industry suppliers and builds all kinds of amusement devices including several types of flat rides, dark
rides, log flumes, tow boat rides and roller coasters.

 The family that owns Mack Rides also owns Europa-Park.
</item 5>

<item 6>
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
</item 6>

<item 7>
Pretzel Amusement Ride Company
The Pretzel Amusement Ride Company was a famous manufacturer of pretzel dark rides. Pretzel built over 1400 pretzel rides and sold them to carnivals and parks.

 The pretzel ride was invented by Marvin Rempfer.

 Leon Cassidy was Marvin's partner in the Pretzel Amusement Ride Company.

 Both names are on the patent.

 Leon Cassidy patented the single-rail dark ride in 1928 along with Marvin Rempfer.

 The company originated in Tumbling Dam Park on the banks of Sunset Lake in Bridgeton, New Jersey.

 A rider said that ""It felt like I was turned and twisted like a Pretzel"", so the name Pretzel was chosen.

 A large heavy pretzel design was originally affixed to the front of each car to prevent the car from flipping backwards.

 In 1929, a standard Pretzel ride had five cars, 350 feet of track, and was one and a half minutes per ride.

 A pretzel ride sold for $1,200.
</item 7>

<item 8>
TOGO
TOGO (株式会社トーゴ , Kabushiki-gaisha Tōgo ) was a Japanese amusement ride company that built roller coasters, giant wheels, carousels, flumes, dark rides, sky
cycles and other amusement rides.
</item 8>

<item 9>
Castle Park (amusement park)
Castle Park, formerly Castle Amusement Park, is a 25-acre amusement park and family amusement center located in Riverside, California.

 The park utilizes a medieval "castle" theme and includes attractions such as a miniature golf course, arcade, and 27 amusement rides including three roller
coasters such as "Merlin's Revenge", a junior rollercoaster, "Screamin' Demon" a spinning Wild Mouse rollercoaster, and "Little Dipper", a children's
rollercoaster.

 The main "castle" themed building, houses the arcade as well as its only dark ride; "Ghost Blasters", an interactive attraction, designed by Sally Corporation,
which can also be found at other amusement parks throughout North America.

 The park was designed, built and operated by Bud Hurlbut, who designed several rides at Knott's Berry Farm.

 Castle Park is currently owned and operated by Palace Entertainment.
</item 9>

<item 10>
Giovanola
Giovanola Freres SA was one of the prominent thrill ride manufacturers in the world.

 It was well known for thrill rides and also built electrical power stations, water storage tanks, pipelines, highway bridges, and many other steel products.

 The company started out as a small metal forging shop, founded by Joseph Giovanola in 1888.

 It served as a subcontractor to Intamin supplying rides and roller coasters.

 In 1998 Giovanola started marketing directly under the name Giovanola Amusement Rides Worldwide.

 The company was based in Monthey, Switzerland.

 During its last years in business, Giovanola fashioned steel behind the scenes for companies such as Intamin and Bolliger & Mabillard.

 The company also built its own thrill rides and roller coasters from 1998 to 2001.

 Examples of Giovanola roller coasters are Goliath at Six Flags Magic Mountain, Titan at Six Flags Over Texas and Anaconda at South Africa's Gold Reef City.
</item 10>
=== END ITEM id=88b44f2527910520 ===

=== ITEM id=16778c35b167b672 ===
Gold sentences and facts:
Sentence 0: Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from 1997-1998.
  Fact 0.0: Robert Suettinger was the national intelligence officer for East Asia at the National Intelligence Council (NIC).
  Fact 0.1: Robert Suettinger held the role of national intelligence officer for East Asia for United States President Bill Clinton.
  Fact 0.2: Robert Suettinger held the role from 1997 to 1998.
Sentence 1: William Jefferson Clinton (born William Jefferson Blythe III; August 19, 1946) is an American politician who served as the 42nd President of the United States from 1993 to 2001.
  Fact 1.0: William Jefferson Clinton was born William Jefferson Blythe III.
  Fact 1.1: William Jefferson Clinton was born on August 19, 1946.
  Fact 1.2: William Jefferson Clinton is an American politician.
  Fact 1.3: William Jefferson Clinton served as the 42nd President of the United States.
  Fact 1.4: William Jefferson Clinton served as President from 1993 to 2001.
Sentence 2:  Prior to the presidency, he was the Governor of Arkansas during two separate terms, from 1979 to 1981, and again from 1983 to 1992.
  Fact 2.0: Before the presidency, William Jefferson Clinton was Governor of Arkansas.
  Fact 2.1: William Jefferson Clinton was Governor of Arkansas during two separate terms.
  Fact 2.2: William Jefferson Clinton's first term as Governor of Arkansas was from 1979 to 1981.
  Fact 2.3: William Jefferson Clinton's second term as Governor of Arkansas was from 1983 to 1992.

Retrieved items:
<item 1>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 1>

<item 2>
Robert Suettinger served as Clinton’s East Asia intelligence officer (1997–98), specializing in China and North Korea; later he worked at Brookings, authored
*Beyond Tiananmen*, and advised the Stimson Center. Clinton was president, 1993–2001. Other profiles cover Reagan-era officials and administrations, including
Otto Reich, Lawrence Eagleburger, William Flynn Martin, and Elliott Abrams (pardoned after his Iran–Contra conviction); Reagan’s 1981 and
</item 2>

<item 3>
Susan Combs
Susan Combs (born February 26, 1945) is a Republican politician from the U.S. state of Texas, who served from 2007 to 2015 as the state's Comptroller of Public
Accounts.

 Prior to her tenure as Comptroller, Combs had served two terms as Commissioner of the Texas Department of Agriculture from 1999–2007, taking the reins as the
first woman elected to that office in 1998.

 Combs also served two terms in the Texas House of Representatives.

 On July 10, 2017, Combs was nominated by U.S. President Donald Trump to be the Assistant Secretary of the Interior for Policy, Management and Budget.
</item 3>

<item 4>
Lawrence Eagleburger
Lawrence Sidney Eagleburger (August 1, 1930 – June 4, 2011) was an American statesman and career diplomat, who served briefly as the Secretary of State under
President George H. W. Bush.

 Previously, he had served in lesser capacities under Presidents Richard Nixon, Jimmy Carter, and Ronald Reagan, and as Deputy Secretary of State under George
H. W. Bush.
</item 4>

<item 5>
The DIA Director, presidentially nominated and Senate-confirmed, is the senior military intelligence officer, advises Defense leaders, commands a USSTRATCOM
intelligence-surveillance-reconnaissance component, and chairs the Military Intelligence Board. The National Intelligence Board, led by the DNI, approves
intelligence estimates. NIEs are classified, coordinated assessments for policymakers. Intelligence awards include the highest ODNI honor, the National
Intelligence Cross; Superior Service and Reform medals (the latter retired in 2010); and
</item 5>

<item 6>
Otto Reich
Otto Juan Reich (born October 16, 1945), is an American former senior official in the administrations of Presidents Ronald Reagan, George H. W. Bush, and George
W. Bush.

 Among other positions held, he has been the President's Special Envoy for the Western Hemisphere; Assistant Secretary of State for Western Hemisphere Affairs;
United States Ambassador to Venezuela; and Assistant Administrator of the US Agency for International Development, a recess appointment.

 In 2003, Bush appointed him US Special Envoy to the Western Hemisphere.

 Since leaving the White House in 2004, he has headed his own international consulting firm, Otto Reich Associates, LLC, based in Washington, D.C..
</item 6>

<item 7>
Bill Clinton
William Jefferson Clinton (born William Jefferson Blythe III; August 19, 1946) is an American politician who served as the 42nd President of the United States
from 1993 to 2001.

 Prior to the presidency, he was the Governor of Arkansas during two separate terms, from 1979 to 1981, and again from 1983 to 1992.

 Before that, he served as Arkansas attorney general, from 1977 to 1979.

 A member of the Democratic Party, Clinton was ideologically a New Democrat and many of his policies reflected a centrist "Third Way" political philosophy.
</item 7>

<item 8>
James A. Winnefeld Jr.
James Alexander "Sandy" Winnefeld Jr. (born April 24, 1956) is a retired(On Special Mission in Afghanistan).

 United States Navy admiral who served as the ninth Vice Chairman of the Joint Chiefs of Staff from August 4, 2011 to July 31, 2015.

 He previously served as the fourth Commander, U.S. Northern Command (USNORTHCOM) and the 21st Commander, North American Aerospace Defense Command (NORAD) from
May 19, 2010 to August 3, 2011.

 Prior to that, Winnefeld served as Director for Strategic Plans and Policy, The Joint Staff which he concurrently served as the Senior Member, U.S. Delegation
to the U.N. Military Staff Committee.

 His other operational commands include serving as the Commander, U.S. Sixth Fleet and Commander, Allied Joint Command Lisbon.

 As the Vice Chairman of the Joint Chiefs of Staff, Winnefeld was the second highest-ranking officer in the United States Armed Forces, second only to the
Chairman of the Joint Chiefs of Staff.

 In 2015, he retired from the Navy after over 37 years of service.
</item 8>

<item 9>
Robert Suettinger
Robert Suettinger was United States President Bill Clinton's national intelligence officer for East Asia at the National Intelligence Council (NIC) from
1997-1998.

 While there, Suettinger oversaw the preparation of national intelligence estimates for the director of the Central Intelligence Agency.

 His areas of specialty included the People's Republic of China and the North Korean nuclear weapons program.

 After working in the Clinton administration, Suettinger has been a senior analyst for the Brookings Institution where he wrote the book "Beyond Tiananmen – The
Politics of U.S.-China Relations, 1989-2000".

 He also is a senior advisor at The Stimson Center.
</item 9>

<item 10>
Steven K. Galson
Steven Kenneth Galson (born 1956) is an American public health physician.

 He is currently Senior Vice President for Global Regulatory Affairs at Amgen, the S. California-based innovative global biopharmaceutical company.

 He is also Professor-at-Large at the Keck Graduate Institute for Applied Life Sciences in Clarmont, California.

 He is a retired rear admiral in the United States Public Health Service Commissioned Corps and public health administrator who served as the acting Surgeon
General of the United States from October 1, 2007 – October 1, 2009.

 He served concurrently as acting Assistant Secretary for Health from January 22, 2009 to June 25, 2009, and as the Deputy Director and Director of the Center
for Drug Evaluation and Research (CDER) at the Food and Drug Administration from 2001 to 2007.

 As the Acting Surgeon General, he was the commander of the United States Public Health Service Commissioned Corps and, while serving as the Assistant Secretary
for Health, was the operational head of the Public Health Service.
</item 10>
=== END ITEM id=16778c35b167b672 ===

=== ITEM id=5f354548a3c454c2 ===
Gold sentences and facts:
Sentence 0: Ronkonkoma is a major railroad station and transportation hub along the Main Line of the Long Island Rail Road in Ronkonkoma, New York.
  Fact 0.0: Ronkonkoma is a major railroad station.
  Fact 0.1: Ronkonkoma is a major transportation hub.
  Fact 0.2: Ronkonkoma is along the Main Line of the Long Island Rail Road.
  Fact 0.3: The Ronkonkoma railroad station is in Ronkonkoma, New York.
Sentence 1: Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.
  Fact 1.0: Ronkonkoma is a hamlet.
  Fact 1.1: Ronkonkoma is a census-designated place (CDP).
  Fact 1.2: Ronkonkoma is on Long Island.
  Fact 1.3: Ronkonkoma is in the Town of Islip.
  Fact 1.4: The Town of Islip is in New York, United States.

Retrieved items:
<item 1>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 1>

<item 2>
Long Island Electric Railway
The Long Island Electric Railway was a streetcar company operating in Brooklyn, Queens, and Nassau County, New York, United States between 1894 and 1926.

 The company was partially owned by the Long Island Consolidated Electric Companies, a holding company for the Long Island Rail Road and partially by August
Belmont and the Interborough Rapid Transit Company.

 It connected the east end of the Fulton Street El at Crescent Street station in City Line, Brooklyn with Jamaica, Queens, and ran from there to the Nassau
County line at Queens Village and to Far Rockaway, Queens via Nassau County.

 It also had a connection to Belmont Park.

 The New York and Long Island Traction Company used trackage rights over its line from Crescent Street to Queens Village.
</item 2>

<item 3>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 3>

<item 4>
Cambridge Springs (Erie Railroad station)
Cambridge Springs (formerly Cambridge) was a railroad station for the Erie Railroad in Cambridge Springs, Crawford County, Pennsylvania, United States.

 Cambridge Springs station was on the Main Line's Meadville Division, which was the section of the line between Salamanca, New York and Meadville, Pennsylvania.

 The station was located 501.2 mi from Manhattan and the Barclay Street Ferry, which connected to Pavonia Terminal in Jersey City, New Jersey and 480.8 mi from
Hoboken Terminal in Hoboken, New Jersey.

 For nearly three decades, the station had connections to the Northwestern Pennsylvania Railway, which was a trolley line that connected the city of Erie and
Meadville.

 Modern Erie Railroad station signage denoted the station as "Home of Alliance College," a local private university that closed in 1987.
</item 4>

<item 5>
Ronkonkoma, New York
Ronkonkoma ( ) is a hamlet and census-designated place (CDP) on Long Island in the Town of Islip, New York, United States.

 The population of the CDP was 19,082 at the 2010 census.
</item 5>

<item 6>
Ridgefield Park station
Ridgefield Park Station, also known as West Shore Station, was railroad station in Ridgefield Park, New Jersey at the foot of Mount Vernon Street served by the
New York, Susquehanna and Western Railroad (NYSW) and the West Shore Railroad, a division of New York Central (NYCRR) The New York, Ontario and Western Railway
(NYO&W) had running rights along the West Shore and sometimes stopped at Ridgefield Park.

 First opened in 1883 it was one of three passenger stations in the village, the others being the Little Ferry Station to the south and Westview Station to the
north.

 The station house, built at a cost $100,000 opened in 1927.

 Southbound service crossed Overpeck Creek and continued to terminals on the Hudson River waterfront where there was connecting ferry service across the Hudson
River to Manhattan.

 Northbound near Bogota the parallel NYSW and West Shore lines diverge and continue into northern New Jersey, Pennsylviania, and upstate New York.

 Passenger service ended in 1966.
</item 6>

<item 7>
Lacona Railroad Station and Depot
Lacona Railroad Station and Depot, also known as New York Central and Hudson River Railroad Station and Depot, is a historic railway depot located at Lacona in
Oswego County, New York.

 It was built in 1891 by the New York Central and Hudson River Railroad.

 It is a small rectangular, one story, gable ended structure.
</item 7>

<item 8>
Ronkonkoma (LIRR station)
Ronkonkoma is a major railroad station and transportation hub along the Main Line of the Long Island Rail Road in Ronkonkoma, New York.

 The station is the eastern terminus of the Ronkonkoma Branch and the western terminus of the out Branch.
</item 8>

<item 9>
Old Country Road
Old Country Road is a major east–west thoroughfare through central Nassau County and extending into western Suffolk County on Long Island, New York.

 It serves many of the major shopping centers in central Nassau County including Roosevelt Field Mall.

 The road also forms part of the border between the towns of Hempstead and North Hempstead.

 In Nassau County, Old Country Road is county-maintained as the unsigned County Route 25 (CR 25) west of Round Swamp Road.
</item 9>

<item 10>
North Creek Railroad Station
The North Creek Railroad Station is a historic railroad station complex located at North Creek, Warren County, New York.

 The complex consists of the railroad station, the freight house, round house, turntable, and horse barn.

 The station was built in 1874 and is a simple, rectangular, gable roofed building with a broad, overhanging strut-supported roof in the Stick-Eastlake style.

 Its exterior is covered with vertical boards.
</item 10>
=== END ITEM id=5f354548a3c454c2 ===

=== ITEM id=e150eb39fd90c6b6 ===
Gold sentences and facts:
Sentence 0: Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
  Fact 0.0: Alan Crosland was born on August 10, 1894.
  Fact 0.1: Alan Crosland died on July 16, 1936.
  Fact 0.2: Alan Crosland was an American stage actor.
  Fact 0.3: Alan Crosland was an American film director.
Sentence 1: Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.
  Fact 1.0: Kurt Voss was born Kurt Christopher Peter Wössner.
  Fact 1.1: Kurt Voss is an American film director.
  Fact 1.2: Kurt Voss is an American screenwriter.
  Fact 1.3: Kurt Voss is an American musician-songwriter.

Retrieved items:
<item 1>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 1>

<item 2>
Oswalt Kolle
Oswalt Kolle (2 October 1928, Kiel – 24 September 2010, Amsterdam) was a German sex educator, who became famous during the late 1960s and early 1970s for his
numerous pioneering books and films on human sexuality.

 His work was translated into all major languages, while his films found an audience of 140 million worldwide.

 In his 1997 book "Open to Both Sides" he came out as bisexual.

 He was awarded the Magnus Hirschfeld Medal in 2000.
</item 2>

<item 3>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 3>

<item 4>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
</item 4>

<item 5>
Doug Clifford
Douglas "Cosmo" Clifford (born April 24, 1945 in Palo Alto, California) is an American drummer, best known as a founding member of Creedence Clearwater Revival.

 After the group dissolved in the early 1970s, Clifford released a solo album and later joined CCR bassist Stu Cook in the Don Harrison Band.

 In 1995, Clifford and Cook formed the band Creedence Clearwater Revisited, performing live versions of Creedence Clearwater Revival songs.
</item 5>

<item 6>
Colby Keller
Colby Keller (born October 18, 1980) is an American visual artist, blogger, and pornographic film actor.

 His career in porn film started in 2004 at Sean Cody and has since expanded to include such studios as Cocksure Men, Randy Blue, Titan Men, Falcon, CockyBoys
and Men.com.

 Keller has amassed a large fan base with both his films and his long-standing blog, "The Big Shoe Diaries".

 He has been nominated for a number of awards and in 2013, World of Wonder awarded him the WOWie Award for Best Blog.

 Keller is nominated for 8 2015 Grabby Awards.

 Keller has also appeared in the short, Zolushka, a queer retelling of Cinderella and the popular series Capitol Hill both by the cult filmmaker Wes Hurley.
</item 6>

<item 7>
Richard Pleasance
Richard Arnold Pleasance is an Australian rock musician and producer.

 He was a founding member of Boom Crash Opera on guitar, bass guitar, vocals and as a songwriter in 1985; they released three albums before Pleasance left in
1992.

 Their hit Australian Recording Industry Association (ARIA) singles, "Great Wall" (No. 5, 1986) and "Onion Skin" (No. 11, 1989) were co-written by Pleasance,
who also co-produced their second album, "These Here Are Crazy Times" (No. 10, 1989).

 His debut solo release, "Galleon" received four nominations at the ARIA Music Awards for 1992.

 Pleasance composed the theme music for Australian television series, "SeaChange" (1998–2001), for 2006 feature film "Kenny" and more recently he composed the
theme music for the prison drama series "Wentworth".

 Pleasance is married to Michelle and, as from May 2009, he was living in Hepburn Springs, Victoria where he has a recording studio.
</item 7>

<item 8>
Alan Crosland
Alan Crosland (August 10, 1894 – July 16, 1936) was an American stage actor and film director.
</item 8>

<item 9>
Steve Alvord
Steven Lee Alvord (born October 2, 1964 in Bellingham, Washington) is a former professional American football defensive tackle and defensive end in the National
Football League and World League of American Football.

 In his four-year pro career he played for the St. Louis/Phoenix Cardinals of the NFL, and the Barcelona Dragons of the WLAF.

 Alvord played college football at Washington.

He has a cool daughter, happy birthday Katie.
</item 9>

<item 10>
Clinton Walker
Clinton Walker (born 1957) is an Australian writer, best known for his works on popular music but with a broader interest in social and cultural history and
theory.

 Sydney's "Sun-Herald" has called him "our best chronicler of Australian grass-roots culture."

 He has always been ahead of the curve.

 As "Rhythms" magazine said in 2015, "Like many of Walker’s projects, "Buried Country" was at least a decade ahead of its time," and as such he is remarkable as
a critic who has exerted a pro-active impact on Australian music and its development; groundbreaking books like "Inner City Sound" (1981) and "Buried Country"
(2000) especially have informed and inspired successive generations of musicians.

 Similarly, while he found best-selling success as Bon Scott's biographer, Walker's non-music books like "Football Life" (1998) and "Golden Miles" (2005) have
innovatively offered an appreciation of subjects hitherto hardly deemed worthy of serious consideration.
</item 10>
=== END ITEM id=e150eb39fd90c6b6 ===

=== ITEM id=084672c2632b9eb8 ===
Gold sentences and facts:
Sentence 0: The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.
  Fact 0.0: The 2012 United States Libertarian National Convention was a convention of the Libertarian Party (LP).
  Fact 0.1: Delegates of the Libertarian Party chose the party's nominees at the 2012 United States Libertarian National Convention.
  Fact 0.2: The nominees chosen at the 2012 United States Libertarian National Convention were for President of the United States.
  Fact 0.3: The nominees chosen at the 2012 United States Libertarian National Convention were for Vice President of the United States.
  Fact 0.4: The nominees were chosen for the 2012 general election.
  Fact 0.5: The 2012 United States Libertarian National Convention was held May 2-6, 2012.
  Fact 0.6: The 2012 United States Libertarian National Convention was held in Las Vegas, Nevada.
  Fact 0.7: The 2012 United States Libertarian National Convention was held at the Red Rock Resort Spa and Casino.
Sentence 1:  Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.
  Fact 1.0: Gary Johnson is a former Governor of New Mexico.
  Fact 1.1: Gary Johnson won the presidential nomination.
  Fact 1.2: Gary Johnson won the presidential nomination on the first ballot.
Sentence 2: Gary Earl Johnson (born January 1, 1953) is an American businessman, author, and politician.
  Fact 2.0: Gary Earl Johnson was born on January 1, 1953.
  Fact 2.1: Gary Earl Johnson is an American businessman.
  Fact 2.2: Gary Earl Johnson is an American author.
  Fact 2.3: Gary Earl Johnson is an American politician.
Sentence 3:  He was the 29th Governor of New Mexico from 1995 to 2003 as a member of the Republican Party.
  Fact 3.0: Gary Johnson was the 29th Governor of New Mexico.
  Fact 3.1: Gary Johnson was Governor of New Mexico from 1995 to 2003.
  Fact 3.2: Gary Johnson was a member of the Republican Party as Governor of New Mexico.

Retrieved items:
<item 1>
Libertarian National Committee
The Libertarian National Committee (LNC) controls and manages the affairs, properties, and funds of the United States Libertarian Party.

 It is composed of the party officers, five at-large representatives elected every two years at the national convention, and a theoretical maximum of 10
regional representatives.

 The current chair, elected in June 2014, is Nicholas Sarwark.

 The Executive Director is Wes Benedict.
</item 1>

<item 2>
Wayne W. Lambert
The Brigadier General retired from the US Air Force in 1989.

 Lambert also holds an MBA from Chapman College (’76).

 In retirement, Lambert publicly supported U.S. presidential candidate Mitt Romney and U.S. vice presidential candidate Paul Ryan in the 2012 general election.
</item 2>

<item 3>
2012 Libertarian National Convention
The 2012 United States Libertarian National Convention, in which delegates of the Libertarian Party (LP) chose the party's nominees for President of the United
States and Vice President of the United States in the 2012 general election, was held May 2–6, 2012 in Las Vegas, Nevada at the Red Rock Resort Spa and Casino.

 Former Governor of New Mexico Gary Johnson won the presidential nomination on the first ballot.

 Retired California state court judge Jim Gray won the vice-presidential nomination, also on the first ballot.

 The convention also chose to replace most of the Libertarian National Committee party officers and members-at-large.
</item 3>

<item 4>
Gary Johnson presidential campaign, 2012
The 2012 presidential campaign of Gary Johnson, the 29th Governor of New Mexico, was announced on April 21, 2011.

 He declared his candidacy for the 2012 Republican Party nomination for President of the United States.

 On December 28, 2011, Johnson withdrew his candidacy for the Republican nomination, and declared his candidacy for the 2012 presidential nomination of the
Libertarian Party.

 The 2012 Libertarian National Convention was held during the first weekend of May 2012.

 On May 5, 2012, after promoting his libertarian-oriented political positions to delegates, Johnson received the most votes at the convention and became the
official 2012 Libertarian presidential nominee.

 On November 6, 2012, Johnson received just under 1% of the popular vote in the general election, amounting to more than 1.2 million votes, more than double
what the Barr/Root ticket received in 2008.

 This was the most successful result for a third-party presidential candidacy since 2000, and the best in the Libertarian Party's history by vote number at the
time.

 (Johnson ran again in 2016 and received nearly four times his 2012 vote total.)
</item 4>

<item 5>
Mark Hinkle
Mark Hinkle (born January 28, 1951) is an American libertarian activist and businessperson.

 He was the National Chairman of the United States Libertarian Party until 2012.

 He was elected by the delegates of the 2010 Libertarian National Convention in St. Louis, Missouri on May 30, 2010.
</item 5>

<item 6>
Gary Johnson
Gary Earl Johnson (born January 1, 1953) is an American businessman, author, and politician.

 He was the 29th Governor of New Mexico from 1995 to 2003 as a member of the Republican Party.

 He was also the Libertarian Party nominee for President of the United States in the 2012 and 2016 elections.
</item 6>

<item 7>
Libertarian Party presidential primaries, 2016
The 2016 Libertarian Party presidential primaries and caucuses allowed electors to indicate non-binding preferences for the Libertarian Party's presidential
candidate.

 These differed from the Republican or Democratic presidential primaries and caucuses in that they did not appoint delegates to represent a candidate at the
party's convention to select the party's nominee for the United States presidential election.

 The party's nominee for the 2016 presidential election was chosen directly by registered delegates at the 2016 Libertarian National Convention, which ran from
May 26 to 30, 2016.

 The delegates nominated former New Mexico Governor Gary Johnson and former Massachusetts Governor Bill Weld.
</item 7>

<item 8>
David Bergland
David Peter Bergland (born June 4, 1935) is an American politician who was the United States Libertarian Party's nominee at the 1983 Libertarian National
Convention for President of the United States in the 1984 presidential election.
</item 8>

<item 9>
Political positions of Gary Johnson
Gary Johnson was the governor of New Mexico from 1995 to 2003 and ran for president in 2012 and 2016.

 In December 2011 he announced he would pursue the presidential nomination of the Libertarian Party for the 2012 general election.

 The Libertarian National Convention in May, 2012 chose Johnson as the party's candidate.

 In November 2014, Johnson announced he would pursue the presidential nomination of the Libertarian Party for the 2016 general election.

 Johnson has taken positions on many political issues as revealed through his public comments, his gubernatorial record, his Our America Initiative and his
campaigns to win the Republican and Libertarian nominations.
</item 9>

<item 10>
Libertarian National Convention
The Libertarian National Convention is held every two years by the Libertarian Party (United States) to choose members of the Libertarian National Committee
(LNC), and to conduct other party business.

 In presidential election years, the convention delegates enact a platform and nominate the Libertarian presidential and vice-presidential candidates who then
face the nominees of other parties in the November general election.
</item 10>
=== END ITEM id=084672c2632b9eb8 ===

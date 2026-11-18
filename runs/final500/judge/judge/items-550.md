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

=== ITEM id=ddd6eb777e000fcb ===
Gold sentences and facts:
Sentence 0:  It was released as the lead single from their fourth studio album, "Elephant", in March 2003, and reached number one on the Modern Rock Tracks—maintaining that position for three weeks.
  Fact 0.0: The song was released as the lead single from the fourth studio album of its act, "Elephant."
  Fact 0.1: The song was released in March 2003.
  Fact 0.2: The song reached number one on the Modern Rock Tracks.
  Fact 0.3: The song maintained its number one position on the Modern Rock Tracks for three weeks.
Sentence 1:  Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album charts.
  Fact 1.0: The album "Elephant" was released on April 1, 2003.
  Fact 1.1: The album "Elephant" was released on V2 Records.
  Fact 1.2: The album "Elephant" garnered near unanimous critical acclaim.
  Fact 1.3: The album "Elephant" garnered commercial success.
  Fact 1.4: The album "Elephant" received a nomination for Album of the Year at the 46th Grammy Awards in 2004.
  Fact 1.5: The album "Elephant" won Best Alternative Music Album at the 46th Grammy Awards in 2004.
  Fact 1.6: The album "Elephant" peaked at No. 6 on the US "Billboard" charts.
  Fact 1.7: The album "Elephant" topped the UK album charts.

Retrieved items:
<item 1>
Premio Lo Nuestro 2003
Premio Lo Nuestro 2003 was the 15th anniversary of the awards.

 the show was hosted by Mexican presenters Marco Antonio Regil and Adal Ramones.

 Juanes, Thalía, Marc Anthony, Pilar Montenegro, Sin Bandera, Banda el Recodo and other Latin music greats gave electrifying performances.

 In the show, there was 36 awards winners with 135 nominations.

 In Pop genre, Awards was given for : Album of the Year, Best Male Artist, Best Female Artist, Best Group or duo, Best New Artist and Song of the Year.

 In Rock Genre : Best Rock Album and Best rock Performer of the Year.

 In Tropical genre : Best Tropical Album of the Year, Best Tropical Male Artist, Best Tropical Female Artist, Best Tropical Group or Duo of the Year, Best
Tropical New Artist, Tropical Song of the Year, Best Merengue Performance, Best Salsa Performance and Best Traditional Performance.

 Juanes was the biggest winner of night, took home four awards Best Pop Male Artist, Best Music Video, Best Rock Performance, and Pop Song of the Year .
</item 1>

<item 2>
Grammy Award for Best Male Rap Solo Performance
The Grammy Award for Best Male Rap Solo Performance was an honor presented to male recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958, and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 2>

<item 3>
Elephant (album)
Elephant is the fourth album by the American alternative rock duo The White Stripes.

 Released on April 1, 2003 on V2 Records, its release garnered near unanimous critical acclaim and commercial success, garnering a nomination for Album of the
Year and a win for Best Alternative Music Album at the 46th Grammy Awards in 2004, peaking at No. 6 in the US "Billboard" charts and topping the UK album
charts.
</item 3>

<item 4>
Grammy Award for Best Female Rap Solo Performance
The Grammy Award for Best Female Rap Solo Performance was an honor presented to female recording artists at the 45th Grammy Awards in 2003 and the 46th Grammy
Awards in 2004 for quality rap solo performances.

 The Grammy Awards, an annual ceremony that was established in 1958 and originally called the Gramophone Awards, are presented by the National Academy of
Recording Arts and Sciences of the United States to "honor artistic achievement, technical proficiency and overall excellence in the recording industry, without
regard to album sales or chart position".
</item 4>

<item 5>
Seven (Poco album)
Seven is the sixth studio album (seventh overall) by American country rock band Poco.

 It is the first album they made after leader Richie Furay left the band.

 The front cover was designed by Phil Hartman.
</item 5>

<item 6>
Seven Nation Army
"Seven Nation Army" (also stylized as "7 Nation Army") is a song by American rock duo The White Stripes.

 It was released as the lead single from their fourth studio album, "Elephant", in March 2003, and reached number one on the Modern Rock Tracks—maintaining that
position for three weeks.

 It also became the third best-performing song of the decade on the same chart.

 It was well received commercially as well, and won the Grammy Award for Best Rock Song.
</item 6>

<item 7>
The White Stripes
The White Stripes were an American rock duo formed in 1997 in Detroit, Michigan.

 The group consisted of Jack White (songwriter, vocals, guitar, piano, and mandolin) and Meg White (drums and vocals).

 After releasing several singles and three albums within the Detroit music scene, The White Stripes rose to prominence in 2002, as part of the garage rock
revival scene.

 Their successful and critically acclaimed albums "White Blood Cells" and "Elephant" drew attention from a large variety of media outlets in the United States
and the United Kingdom, with the single "Seven Nation Army" and its guitar line becoming their signature song.

 The band recorded two more albums, "Get Behind Me Satan" in 2005 and "Icky Thump" in 2007, and dissolved in 2011 after a lengthy hiatus from performing and
recording.
</item 7>

<item 8>
Meg White
Megan Martha "Meg" White (born December 10, 1974) is an American drummer and occasional singer known for her work with Jack White in the Detroit rock duo The
White Stripes.

 On an impulse, she played on Jack's drums in 1997.

 The two decided to form a band and began performing two months later, calling themselves The White Stripes because of their last name and Meg's affinity for
peppermint candy.

 The band quickly became a Detroit underground favorite, before reaching national, then international fame.

 White has been nominated for various awards as a part of the White Stripes, and has received four Grammy Awards.
</item 8>

<item 9>
The Boys Are Back (The Oak Ridge Boys album)
The Boys Are Back is the thirtieth studio album of country music group The Oak Ridge Boys.

 It was released in 2009 under the Spring Hill Music Group label.

 The album marked the group's return to secular country music after releasing gospel albums since 1992.

 The track "Seven Nation Army," a cover of The White Stripes's 2003 song from their album "Elephant", was released as the album's first single.
</item 9>

<item 10>
Late Night Tales: The Flaming Lips
Late Night Tales: The Flaming Lips is a compilation album compiled by the members of The Flaming Lips, featuring songs by various artists.

 The album was released on March 7, 2005 and it features one new Flaming Lips recording, a cover of the White Stripes' "Seven Nation Army".
</item 10>
=== END ITEM id=ddd6eb777e000fcb ===

=== ITEM id=61e71009f8b3774e ===
Gold sentences and facts:
Sentence 0: Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.
  Fact 0.0: Miltoniopsis is a genus of orchids.
  Fact 0.1: Miltoniopsis is abbreviated Mltnps in horticultural trade.
  Fact 0.2: Miltoniopsis is native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.
Sentence 1: Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few species extending north into temperate regions of Asia.
  Fact 1.0: Strobilanthes is a genus of about 350 species of flowering plants.
  Fact 1.1: Strobilanthes is in the family Acanthaceae.
  Fact 1.2: Strobilanthes is mostly native to tropical Asia and Madagascar.
  Fact 1.3: A few Strobilanthes species extend north into temperate regions of Asia.

Retrieved items:
<item 1>
Strobilanthes
Strobilanthes is a genus of about 350 species of flowering plants in the family Acanthaceae, mostly native to tropical Asia and Madagascar, but with a few
species extending north into temperate regions of Asia.

 Many species are cultivated for their 2-lipped, hooded flowers in shades of blue, pink, white and purple.

 Most are frost-tender and require protection in frost-prone areas.
</item 1>

<item 2>
Tritonia (plant)
Tritonia (flame freesia) is a genus of flowering plants in the iris family first described as a genus in 1802.

 They are naturally distributed across southern Africa, with a high concentration of species in Cape Province of western South Africa.

 The genus is closely related to the genus "Ixia".
</item 2>

<item 3>
Tritoniopsis
Tritoniopsis is a genus of flowering plants in the family Iridaceae, first described as a genus in 1926.

 The entire genus is endemic to Cape Province in South Africa.

 The genus name refers to the African genus "Tritonia" and is combined with the Greek word "opsis", meaning "look-alike".
</item 3>

<item 4>
Hemigraphis
Hemigraphis is a genus of plants in the family Acanthaceae, consisting of about 30 species native to tropical Asia.

 "Hemigraphis" is sometimes included in the genus "Strobilanthes"
</item 4>

<item 5>
Miltoniopsis
Miltoniopsis, abbreviated Mltnps in horticultural trade, is a genus of orchids native to Costa Rica, Panama, Venezuela, Colombia, Ecuador, and Peru.

 This genus comprises 5 species.

 Miltoniopsis's common name is Pansy Orchid.
</item 5>

<item 6>
Miltoniopsis vexillaria
Miltoniopsis vexillaria ("the flag-like "Miltoniopsis"") is a species of epiphytic orchid in the genus Miltoniopsis.
</item 6>

<item 7>
Milium (plant)
Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
</item 7>

<item 8>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 8>

<item 9>
Zygotritonia
Zygotritonia is a genus of herbaceous, perennial and bulbous plants in the Iris family (Iridaceae).

 It contains four species distributed in Tropical Africa.

 The genus name is derived from the word zygomorphic (meaning bilabiate), and the apparent resemblance to some species in the genus "Tritonia".
</item 9>

<item 10>
Strobilanthes japonica
Strobilanthes japonica is a flowering herbaceous perennial plant from Asia, one of around 350 plants of the genus "Strobilanthes".

 The 20–50 cm ornamental plant is cultivated in Japan and China, and blooms in autumn with 1.5 cm purple to white funnel-shaped flowers.
</item 10>
=== END ITEM id=61e71009f8b3774e ===

=== ITEM id=f4897329cba2f82b ===
Gold sentences and facts:
Sentence 0:  One nunatak was first roughly mapped and named "Moltke Nunatak" by the Second German Antarctic Expedition of 1911–12 under Wilhelm Filchner.
  Fact 0.0: A nunatak was first roughly mapped by the Second German Antarctic Expedition of 1911–12
  Fact 0.1: The Second German Antarctic Expedition of 1911–12 was under Wilhelm Filchner
  Fact 0.2: The Second German Antarctic Expedition named the nunatak Moltke Nunatak
Sentence 1:  He named it for General Helmuth von Moltke, Chief of the German General Staff and Secretary of State for Home Affairs.
  Fact 1.0: The nunatak was named for General Helmuth von Moltke
  Fact 1.1: General Helmuth von Moltke was Chief of the German General Staff
  Fact 1.2: General Helmuth von Moltke was Secretary of State for Home Affairs
Sentence 2: Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal) Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.
  Fact 2.0: Helmuth Johann Ludwig von Moltke was born 23 May 1848
  Fact 2.1: Helmuth Johann Ludwig von Moltke died 18 June 1916
  Fact 2.2: Helmuth Johann Ludwig von Moltke is also known as Moltke the Younger
  Fact 2.3: Helmuth Johann Ludwig von Moltke was a nephew of Generalfeldmarschall (Field Marshal) Helmuth Karl Bernhard von Moltke
  Fact 2.4: Helmuth Johann Ludwig von Moltke served as Chief of the German General Staff from 1906 to 1914

Retrieved items:
<item 1>
Ferdinand Jodl
Ferdinand Alfred Friedrich Jodl (28 November 1896 – 9 June 1956) was a German general during World War II who commanded the Mountain Corps Norway during the
Petsamo–Kirkenes Offensive.

 He was the brother of Alfred Jodl, Chief of the Operations Staff of the OKW.

 Ferdinand Jodl was acquitted by a German court.
</item 1>

<item 2>
Helmuth von Moltke the Younger
Helmuth Johann Ludwig von Moltke (] ; 23 May 1848 – 18 June 1916), also known as Moltke the Younger, was a nephew of "Generalfeldmarschall" (Field Marshal)
Helmuth Karl Bernhard von Moltke and served as the Chief of the German General Staff from 1906 to 1914.

 The two are often differentiated as "Moltke the Elder" and "Moltke the Younger".
</item 2>

<item 3>
Herbert Büchs
Herbert Büchs (20 November 1913 – 19 May 1996) was a Lieutenant General of the German Air Force and a former Luftwaffe staff officer in Nazi Germany's
Oberkommando der Wehrmacht (OKW) during World War II.

 As second adjutant to General Alfred Jodl with the rank of Major he is notable for being present in the conference room when the 20 July plot bomb exploded in
1944.
</item 3>

<item 4>
Ludwig Beck
Ludwig August Theodor Beck (29 June 1880 – 21 July 1944) was a German general and Chief of the German General Staff during the early years of the Nazi regime in
Germany before World War II.

 Ludwig Beck was never a member of the Nazi Party, though in the early 1930s he supported Adolf Hitler's forceful denunciation of the Versailles Treaty and
belief in the need for Germany to rearm.

 Beck had grave misgivings regarding the Nazi demand that all German officers swear an oath of fealty to the person of Hitler in 1934, though he believed that
Germany needed strong government and that Hitler could successfully provide this so long as he was influenced by traditional elements within the military rather
than the SA and SS.
</item 4>

<item 5>
Alfred Jodl
Alfred Josef Ferdinand Jodl (   ; 10 May 1890 – 16 October 1946) was a German general and war criminal during World War II, who served as the Chief of the
Operations Staff of the Armed Forces High Command (Oberkommando der Wehrmacht).
</item 5>

<item 6>
German General Staff
The German General Staff, originally the Prussian General Staff and officially Great General Staff ("Großer Generalstab"), was a full-time body at the head of
the Prussian Army and later, the German Army, responsible for the continuous study of all aspects of war, and for drawing up and reviewing plans for
mobilization or campaign.

 It existed unofficially from 1806, and was formally established by law in 1814, the first general staff in existence.

 It was distinguished by the formal selection of its officers by intelligence and proven merit rather than patronage or wealth, and by the exhaustive and
rigorously structured training which its staff officers undertook.

 Its rise and development gave the German armed forces a decisive strategic advantage over their adversaries for nearly a century and a half.
</item 6>

<item 7>
Walter Dornberger
Major-General Dr. Walter Robert Dornberger (6 September 1895 – 27 June 1980) was a German Army artillery officer whose career spanned World War I and World War
II.

 He was a leader of Nazi Germany's V-2 rocket program and other projects at the Peenemünde Army Research Center.
</item 7>

<item 8>
Franz Halder
Franz Halder (30 June 1884 – 2 April 1972) was a German general and the chief of the "Oberkommando des Heeres" staff (OKH, Army High Command) from 1938 until
September 1942, when he was dismissed after frequent disagreements with Adolf Hitler.

 Until December 1941 Halder's military position corresponded to the old Chief of the General Staff position, which during World War I had been the highest
military office in the German Imperial Army.

 Halder's diary during his time as chief of OKH General Staff has been a source for authors that have written about such subjects as Hitler, World War II and
the Nazi Party.

 In William Shirer's "The Rise and Fall of the Third Reich", Halder's diary is cited hundreds of times.
</item 8>

<item 9>
Moltke Nunataks
The Moltke Nunataks ( ) are a chain of north–south trending nunataks close to the northeastern end of the Filchner Ice Shelf, Antarctica.

 One nunatak was first roughly mapped and named "Moltke Nunatak" by the Second German Antarctic Expedition of 1911–12 under Wilhelm Filchner.

 He named it for General Helmuth von Moltke, Chief of the German General Staff and Secretary of State for Home Affairs.

 Surveys during the mid-1950s by British, Argentine and United States expeditions indicate that a group of four or five nunataks exist in the area.
</item 9>

<item 10>
List of Generalobersts
Generaloberst, in English Colonel General, was, in Germany and Austria-Hungary—the German "Reichswehr" and "Wehrmacht", the Austro-Hungarian Common Army, and
the East German National People's Army, as well as the respective police services—the second highest general officer rank, ranking above full general but below
general field marshal.

 It was equivalent to "Generaladmiral" in the "Kriegsmarine" until 1945, or to Flottenadmiral in the "Volksmarine" until 1990.

 The rank was the highest ordinary military rank and the highest military rank awarded in peacetime; the higher rank of general field marshal was only awarded
in wartime by the head of state.

 In general, a Generaloberst had the same privileges as a general field marshal.
</item 10>
=== END ITEM id=f4897329cba2f82b ===

=== ITEM id=e038980920453a76 ===
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
Ethan of Athos
Ethan of Athos is a 1986 science fiction novel by American author Lois McMaster Bujold.

 The title character is Dr. Ethan Urquhart, Chief of Biology at the Severin District Reproduction Centre on the planet Athos, who is sent to find out what
happened to a shipment of vital ovarian tissue cultures.

 Set in the fictional universe of Bujold's Vorkosigan Saga, the novel mentions but does not feature her usual protagonist Miles Vorkosigan.

 To date, Bujold has never revisited the settings of Athos or Kline Station in her many subsequent novels, but the events of "Ethan of Athos" are later referred
to indirectly in the novels "Borders of Infinity" (1989) and "Cetaganda" (1995).
</item 1>

<item 2>
Passage (Bujold novel)
Passage is a novel by American writer Lois McMaster Bujold, published in 2008.

 It is the third in the tetralogy "The Sharing Knife".
</item 2>

<item 3>
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
</item 3>

<item 4>
Gentleman Jole and the Red Queen
Gentleman Jole and the Red Queen is a science fiction novel by Lois McMaster Bujold.

 It is an installment in Bujold's Vorkosigan Saga.

 Bujold has described it as "not a war story.

 It is about grownups."
</item 4>

<item 5>
Cryoburn
Cryoburn is a science fiction novel by Lois McMaster Bujold, first published in October 2010.

 Part of the Vorkosigan Saga, it was nominated for the Hugo Award for Best Novel in 2011, as Bujold's ninth Best Novel nomination.

 Also in 2011, it was one of the top five finishers in the poll for the Locus Award for Best Science Fiction Novel.
</item 5>

<item 6>
Horizon (novel)
Horizon is a fantasy novel by American writer Lois McMaster Bujold.

 It is the fourth in the tetralogy "The Sharing Knife".
</item 6>

<item 7>
Brothers in Arms (Bujold novel)
Brothers in Arms is a science fiction novel by American writer Lois McMaster Bujold, part of the Vorkosigan Saga.

 It was the fifth book published in the series, and is the twelfth story, including novellas, in the internal chronology of the series.

 "Brothers in Arms" was first published by Baen Books in January 1989, and is included in the 2002 omnibus "Miles Errant".
</item 7>

<item 8>
Bernard Malamud
Bernard Malamud (April 26, 1914 – March 18, 1986) was an American novelist and short story writer.

 Along with Saul Bellow and Philip Roth, he was one of the best known American Jewish authors of the 20th century.

 His baseball novel, "The Natural", was adapted into a 1984 film starring Robert Redford.

 His 1966 novel "The Fixer" (also filmed), about antisemitism in Tsarist Russia, won both the National Book Award and the Pulitzer Prize.
</item 8>

<item 9>
Shards of Honor
Shards of Honor is an English language science fiction novel by Lois McMaster Bujold, first published in June 1986.

 It is a part of the Vorkosigan Saga, and is the first full-length novel in publication order.

 "Shards of Honor" is paired with Bujold's 1991 "Barrayar" in the omnibus "Cordelia's Honor" (1996).
</item 9>

<item 10>
Barrayar
Barrayar is a science fiction novel by Lois McMaster Bujold.

 It was first published as four installments in "Analog" in July–October 1991, and then published in book form by Baen Books in October 1991.

 "Barrayar" won both the Hugo Award for Best Novel and the Locus Award for Best Science Fiction Novel in 1992.

 It is a part of the Vorkosigan Saga, and is the seventh full-length novel of the series, in publication order.

 "Barrayar" is a direct sequel to Bujold's first novel, "Shards of Honor" (1986), and the two are paired in the 1996 omnibus "Cordelia's Honor".
</item 10>
=== END ITEM id=e038980920453a76 ===

=== ITEM id=5638b968e3a80e8b ===
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
Random House Tower
The Random House Tower, also known as the Park Imperial Apartments, is a 52-story mixed-use tower in New York City, United States, that is used as the
headquarters of book publisher Random House and a luxury apartment complex.

 The book publisher entrance is on Broadway and goes up to 27 floors, while the apartment complex entrance is on West 56th Street.
</item 1>

<item 2>
270 Park Avenue
270 Park Avenue (also known as the JPMorgan Chase Tower and formerly the Union Carbide Building) is a high-rise office building located in Midtown Manhattan,
New York City.

 It was designed by Natalie de Blois for Skidmore, Owings & Merrill.
</item 2>

<item 3>
1 New York Plaza
1 New York Plaza is an office building in New York City's Financial District, built in 1969 at the intersection of South and Whitehall Streets.

 It is the southernmost of all Manhattan skyscrapers.
</item 3>

<item 4>
383 Madison Avenue
383 Madison Avenue is an office building owned and occupied by JP Morgan Chase in New York City on a full block bound by Madison Avenue and Vanderbilt Avenue
between East 46th and 47th Streets.

 Formerly known as the Bear Stearns Building, it housed the world headquarters of the now-defunct Bear Stearns from the building's completion until Bear's
collapse and sale to JPMorgan Chase in 2008.

 The building now houses the New York offices for J.P. Morgan's investment banking division, which formerly occupied 277 Park Avenue.

 Both 383 Madison and 277 Park are adjacent to JPMorgan Chase's world headquarters at 270 Park Avenue.
</item 4>

<item 5>
The entries cover notable U.S. buildings, districts, businesses, and people. New York features prominently: office towers such as 888 Seventh Avenue, 245 Park
Avenue, 383 and 270 Madison/Park Avenue, and Goldman Sachs’ 200 West Street; landmarks including the Woolworth Building and 90 West Street; and residential or
mixed-use towers such as MiMA and Random House Tower. Other topics include historic theaters, retail and media companies, transportation firms, and real-estate
developer
</item 5>

<item 6>
245 Park Avenue
245 Park Avenue (formerly American Tobacco Company Building, American Brands Building and Bear Stearns Building) is a 648-ft (198 m) tall skyscraper in New York
City, New York.

 It was completed in 1967 and has 48 floors (1.7 million square feet).

 Shreve, Lamb and Harmon designed the building.

 The Building Owners and Managers Association awarded the 2000/2001 Pinnacle Award to 245 Park Avenue.

 The site used to be occupied by the second Grand Central Palace exhibition hall, which was demolished in 1964 to make way for 245 Park Avenue.
</item 6>

<item 7>
MiMA (building)
MiMA, which stands for "Middle of Manhattan", is a mixed-use building located at 450 West 42nd Street between Dyer and 10th Avenues in the Hell's Kitchen
neighborhood of Manhattan, New York City.

 Ground was broken in 2007 and topping out occurred in early August 2010.

 It was designed by the Miami-based architecture firm of Arquitectonica, and has 43 floors of luxury rentals on floors 7 to 50, twelve floors of condominiums on
floors 51 to 63, and a Yotel hotel on the lower levels.

 The building's height is 204 meters (670 feet).
</item 7>

<item 8>
888 7th Avenue
888 7th Avenue is a 628 ft (191m) tall modern-style office skyscraper in Midtown Manhattan which was completed in 1969 and has 46 floors.

 Emery Roth & Sons designed the building, which is tied with Central Park Place for the 65th tallest building in New York City.

 It currently carries the Vornado Realty Trust corporate headquarters.

 Previously known as the Arlen Building, its namesake being the company responsible for its construction, Arlen Realty & Development Corporation.

 The Red Eye Grill is located in the building at street level.
</item 8>

<item 9>
200 West Street
200 West Street is the global headquarters of the Goldman Sachs investment banking firm.

 The building is a 749 ft , 44-story building located on West Street, between Vesey and Murray Streets in Lower Manhattan.

 It is adjacent to the World Financial Center and the Conrad Hotel, the Verizon Building to the east across West Street, and diagonally opposite the World Trade
Center.

 It is the only office building in Battery Park City north of the World Financial Center.
</item 9>

<item 10>
Park Avenue House
The Park Avenue House is a high rise residential building located at 2305 Park Avenue in the Park Avenue Historic District in Downtown Detroit, Michigan.

 It was listed on the National Register of Historic Places in 1996.

 It should not be confused with the nearby Park Avenue Hotel, which was demolished in 2015.
</item 10>
=== END ITEM id=5638b968e3a80e8b ===

=== ITEM id=c4be9847c5c8ac65 ===
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
Chad Gilbert
Chad Everett Gilbert (born March 9, 1981) is an American musician and record producer.

 He is a founding member of the rock band New Found Glory, for whom he plays lead guitar, composes music and provides backing vocals.

 He was also the lead vocalist for New Found Glory's now-defunct side-project, International Superheroes of Hardcore.

 Additionally, Gilbert was the vocalist for the hardcore punk band Shai Hulud between 1995 and 1998, and he rejoined the band in 2012.
</item 1>

<item 2>
Nothing Gold Can Stay (album)
Nothing Gold Can Stay is the debut studio album by American rock band New Found Glory, released on October 19, 1999 through independent record label Eulogy
Recordings.

 At the time, the band was then named "A New Found Glory", but later dropped the indefinite article "A" due to some fans struggling to find their records in
stores.

 The original pressings of the album contained samplings from several films including "The Outsiders" (1983), "Weird Science" (1985), and "That Thing You Do!

" (1996), as well as Robert Frost's poem "Nothing Gold Can Stay", after which the album is named.
</item 2>

<item 3>
Glassjaw
Glassjaw is an American post-hardcore band from Long Island, New York.

 The band is fronted by vocalist Daryl Palumbo and guitarist Justin Beck.

 The band has been influential in the progression of the underground music scene in the eastern United States and United Kingdom and on the post-hardcore genre,
and are known for their intense live shows and frequent line-up changes.
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
</item 5>

<item 6>
A Very New Found Glory Christmas
A Very New Found Glory Christmas is the third EP by American rock band New Found Glory.

 Self-produced by the band, it is a special edition Christmas-themed acoustic recording featuring two original songs and three covers.

 The EP was limited to 2,000 pressings and was released via cassette tape on December 1, 2012.

 1,000 white copies were sold on tour, and 1,000 red copies were sold in their online store.

 The band released "Nothing for Christmas" via Rdio and Spotify on December 20, after debuting songs off the EP during the tenth anniversary tour of their
"Sticks and Stones" album.

 The cover art is a parody of the 1990 blockbuster movie "Home Alone".
</item 6>

<item 7>
It's All About the Girls
It's All About the Girls is the debut EP by A New Found Glory (now New Found Glory) released on December 20, 1997 by Fiddler Records.

 It is the only release with their original drummer Joe Marino.

 A re-release, featuring revised cover artwork, was issued by Fiddler in June, 2000.
</item 7>

<item 8>
Clearview 77
Clearview 77 was a punk rock quartet hailing from Brooklyn, NY.

 Born in the summer of 1996, CV77 was the true definition of the "local" band.

 The band has played countless shows within the East Coast with acts like Face to Face, New Found Glory, Finch, Saves the Day, The Amazing Crowns, US Bombs, Big
Wig, Murphy's Law, Alkaline Trio, The Queers, GOB, and many more.

 In their 8-year existence the band managed to finish at the top of the charts and competitions for CMJ, Discmakers, Garageband.com, mp3.com, Jim Beam,
Loudenergy, had a song aired on CNN and Good Morning America, and added to college radio stations across the US.

 The band also finished in the top 10 in 92.3 FM K-ROCK' best unsigned band of 2000 contest.

 Clearview 77 have also been featured on multiple compilations with acts like NOFX, New Found Glory, Auto Pilot Off, Agnostic Front, Midtown, ALL, Saves the
Day, Down By Law, Circle Jerks, and many more.
</item 8>

<item 9>
New Found Glory
New Found Glory (formerly A New Found Glory) is an American rock band from Coral Springs, Florida, formed in 1997.

 The band currently consists of Jordan Pundik (lead vocals), Ian Grushka (bass guitar), Chad Gilbert (lead guitar, backing vocals), and Cyrus Bolooki (drums).

 Longtime rhythm guitarist and lyricist Steve Klein departed from the band in late 2013, following "personal differences."

 During their lengthy recording career, the band have released nine studio albums, one live album, two EPs, and three cover albums.
</item 9>

<item 10>
Hit or Miss (New Found Glory song)
"Hit or Miss (Waited Too Long)" is the first single by New Found Glory (formerly A New Found Glory), from their 1999 debut studio album, "Nothing Gold Can
Stay".
</item 10>
=== END ITEM id=c4be9847c5c8ac65 ===

=== ITEM id=8a87cd303087b0b2 ===
Gold sentences and facts:
Sentence 0: The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.
  Fact 0.0: The Appenzeller Sennenhund is a medium-size breed of dog.
  Fact 0.1: The Appenzeller Sennenhund is one of the four regional breeds of Sennenhund-type dogs.
  Fact 0.2: The Sennenhund-type dogs come from the Swiss Alps.
Sentence 1:  The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
  Fact 1.0: The name Sennenhund refers to people called "Senn".
  Fact 1.1: The Senn people are herders in the Appenzell region of Switzerland.
Sentence 2: The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.
  Fact 2.0: The Drentsche Patrijshond is a versatile spaniel-type hunting dog.
  Fact 2.1: The Drentsche Patrijshond is from the Dutch province of Drenthe.

Retrieved items:
<item 1>
Belgian Shepherd
The Belgian Shepherd (also known as the Belgian Sheepdog or Chien de Berger Belge) is a breed of medium-to-large-sized herding dog.

 It originated in Belgium and is similar to other sheep herding dogs from that region, including the Dutch Shepherd, the German Shepherd, the Briard, and
others.

 Four types have been identified by various registries as separate breeds or varieties: Groenendael, Laekenois, Tervuren, and Malinois.
</item 1>

<item 2>
Entlebucher Mountain Dog
The Entlebucher Sennenhund or Entlebucher Mountain Dog is a medium-sized herding dog, it is the smallest of the four Sennenhunds, a dog type that includes four
regional breeds.

 The name Sennenhund refers to people called "Senn", herders in the Swiss Alps.

 Entlebuch is a region in the canton of Lucerne in Switzerland.

 The breed is also known in English as the Entelbuch Mountain Dog, Entelbucher Cattle Dog, and similar combinations.
</item 2>

<item 3>
Dutch Shepherd
The Dutch Shepherd is a herding dog of Dutch origin.

 They were used by shepherds and farmers who needed a versatile dog, with few demands, and a dog that was able to adapt to a harsh and meager existence.
</item 3>

<item 4>
Appenzeller Sennenhund
The Appenzeller Sennenhund is a medium-size breed of dog, one of the four regional breeds of Sennenhund-type dogs from the Swiss Alps.

 The name Sennenhund refers to people called "Senn", herders in the Appenzell region of Switzerland.
</item 4>

<item 5>
Bernese Mountain Dog
The Bernese Mountain Dog (German: "Berner Sennenhund" ) is a large-sized breed of dog, one of the four breeds of Sennenhund-type dogs from the Swiss Alps.

 The name "Sennenhund" is derived from the German "Senne" ("alpine pasture") and "Hund" ("dog"), as they accompanied the alpine herders and dairymen called
"Senn".

 "Berner" (or "Bernese" in English) refers to the area of the breed’s origin, in the canton of Bern.

 This mountain dog was originally kept as a general farm dog.

 Large Sennenhunde in the past were also used as draft animals, pulling carts.

 The breed was officially established in 1907.

 In 1937, the American Kennel Club recognized it; today, the club classifies it as a member of the Working Group.
</item 5>

<item 6>
Drentse Patrijshond
The Drentsche Patrijshond is a versatile spaniel-type hunting dog from the Dutch province of Drenthe.

 Called the Dutch Partridge Dog (or "Drent" for Drenthe) in English, approximately 5,000 dogs are registered with the breed club in the Netherlands, and breed
clubs operate in Belgium, Denmark, Scandinavia and North America.

 The Drentsche Patrijshond bears some resemblance to both spaniel and setter types of dog.

 An excellent pointer and retriever, this dog is often used to hunt fowl and adapts equally well to the field or marshes.
</item 6>

<item 7>
Swiss mountain dog
Sennenhund, called Swiss mountain dogs or Swiss cattle dogs in English, are a type of dog originating in the Swiss Alps.

 The Sennenhund are farm dogs of the general molosser type.

 There are four breeds of Sennenhund, all sporting a unique tricolor coat.

 While the two larger ones share a heavy build and a calm temperament, the two smaller ones are more agile.

 The breeds range from medium in size to very large.

 The name Sennenhund refers to people called "Senn" or Senner, Swiss alpine herdsmen and dairymen, and does not translate as "mountain" or "cattle".
</item 7>

<item 8>
Greater Swiss Mountain Dog
The Greater Swiss Mountain Dog (German: "Grosser Schweizer Sennenhund" or French: "Grand Bouvier Suisse" ) is a dog breed which was developed in the Swiss Alps.

 The name "Sennenhund" refers to people called "Senn" or "Senner", dairymen and herders in the Swiss Alps.

 Greater Swiss Mountain Dogs are almost certainly the result of indigenous dogs mating with large mastiff types brought to Switzerland by foreign settlers.

 At one time, the breed was believed to have been among the most popular in Switzerland.

 It was assumed to have almost died out by the late 19th century, since its work was being done by other breeds or machines, but was rediscovered in the early
1900s.
</item 8>

<item 9>
Appenzeller (chicken)
The Appenzeller is a breed of chicken originating in Appenzell region of Switzerland.

 The Appenzeller comes in two varieties.

 The Spitzhauben variety, meaning "pointed hood", has a V-comb and feather crests in males and females.

 The word 'spitzhauben' derives from the frilly hat worn by the women in the Appenzeller region in Switzerland.

 The breed was imported into America by a doctor.

 The Barthuhner ("bearded hen") has a rose comb and no crest.

 Both types appear in either black, golden spangled and silver spangled plumage.

 They are mostly a show breed, but are decent egg layers.

 They lay small white eggs about 5 times per week.
</item 9>

<item 10>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 10>
=== END ITEM id=8a87cd303087b0b2 ===

=== ITEM id=333af83faff05edd ===
Gold sentences and facts:
Sentence 0:  She is a member of the "Game of Thrones" stunt teams that won two Screen Actors Guild Awards, and has done stunt performances for films including "Rogue One".
  Fact 0.0: The person is a member of the Game of Thrones stunt teams.
  Fact 0.1: The Game of Thrones stunt teams won two Screen Actors Guild Awards.
  Fact 0.2: The person has done stunt performances for films including Rogue One.
Sentence 1:  "Rogue One" follows a group of rebels on a mission to steal the plans for the Death Star, the Galactic Empire's superweapon.
  Fact 1.0: Rogue One follows a group of rebels.
  Fact 1.1: The rebels are on a mission to steal the plans for the Death Star.
  Fact 1.2: The Death Star is the Galactic Empire's superweapon.

Retrieved items:
<item 1>
Frenemies (film)
Frenemies is a 2012 teen comedy-drama television film and anthology based on the novel of the same name by Alexa Young which premiered on Disney Channel.

 It features an ensemble cast starring Bella Thorne, Zendaya, Stefanie Scott, Nick Robinson, Mary Mouser and features Connor Price, Jascha Washington and Dylan
Everett.

 The film follows three pairs of teenage friends that go from friends to enemies and back again.

 The film was directed by Daisy Mayer and written by Dava Savel, Wendy Weiner, and Jim Krieg.

 The Disney Channel Original Movie premiered on January 13, 2012 in the United States and Canada.
</item 1>

<item 2>
Innocence (2013 film)
Innocence is a 2013 American horror drama film directed by Hilary Brougher, who co-wrote the film with Tristine Skyler.

 The movie is based on the 2000 novel of the same name by Jane Mendelsohn.

 It had its world premiere on 26 October, 2013 at the Austin Film Festival and received a limited theatrical release in the United States on 5 September, 2014.

 The movie stars Sophie Curtis, Kelly Reilly, Graham Phillips, Linus Roache, Sarah Sutherland and Stephanie March.
</item 2>

<item 3>
The Runaways (film)
The Runaways is a 2010 American drama film about the 1970s rock band of the same name written and directed by Floria Sigismondi.

 It is based on the book "Neon Angel: A Memoir of a Runaway" by the band's original lead vocalist Cherie Currie.

 The film stars Dakota Fanning as Currie, Kristen Stewart as rhythm guitarist and vocalist Joan Jett, and Michael Shannon as record producer Kim Fowley.

 "The Runaways" depicts the formation of the band in 1975 and focuses on the relationship between Currie and Jett until Currie's departure from the band.

 The film grossed about $4.7 million worldwide and received generally favorable reviews from critics.
</item 3>

<item 4>
Race to Nowhere
Race to Nowhere is a 2009 documentary film written by Maimone Attia and directed by Vicki Abeles and Jessica Congdon.
</item 4>

<item 5>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 5>

<item 6>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 6>

<item 7>
Rachelle Beinart
Rachelle Amy Beinart (born 29 September 1984) is a British actress and stunt performer.

 She is a member of the "Game of Thrones" stunt teams that won two Screen Actors Guild Awards, and has done stunt performances for films including "Rogue One".

 She has also acted in preschool children's shows as a suit actor, including the reboot of "Teletubbies".
</item 7>

<item 8>
Rogue One
Rogue One: A Star Wars Story, or simply Rogue One, is a 2016 American epic space opera film directed by Gareth Edwards.

 The screenplay by Chris Weitz and Tony Gilroy is from a story by John Knoll and Gary Whitta.

 It was produced by Lucasfilm and distributed by Walt Disney Studios Motion Pictures.

 It is the first installment of the "Star Wars" "Anthology" series, set immediately before the events of the original "Star Wars" film.

 The cast includes Felicity Jones, Diego Luna, Riz Ahmed, Ben Mendelsohn, Donnie Yen, Mads Mikkelsen, Alan Tudyk, Jiang Wen and Forest Whitaker.

 "Rogue One" follows a group of rebels on a mission to steal the plans for the Death Star, the Galactic Empire's superweapon.
</item 8>

<item 9>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 9>

<item 10>
Rachelle Henry
Rachelle Henry (born December 16, 2000) is an American teen actress and filmmaker.

 She played the role of Sandy Hobbs in the TLC (TV network) Series "Escaping the Prophet"and Lissa Golaski in "Depth", the film prequel to the Soma (video game)
by Frictional Games.

 She is also known for directing and producing short films containing messages of social influence and coming of age themes including "MISSING" and "Almost
Boyfriends".
</item 10>
=== END ITEM id=333af83faff05edd ===

=== ITEM id=6aec4a68cb7ea4eb ===
Gold sentences and facts:
Sentence 0:  The first edition was written by J. Eric Holmes based on Gary Gygax and Dave Arneson's original work.
  Fact 0.0: The first edition was written by J. Eric Holmes.
  Fact 0.1: The first edition was based on the original work of Gary Gygax and Dave Arneson.
Sentence 1: John Eric Holmes M.D. (February 16, 1930 – March 20, 2010) was an associate professor of neurology at the University of Southern California School of Medicine, an author and promoter of fantasy role-playing games, a noted fan and enthusiast of Edgar Rice Burroughs, and an American writer of non-fiction, fantasy and science fiction.
  Fact 1.0: John Eric Holmes M.D. was born on February 16, 1930.
  Fact 1.1: John Eric Holmes M.D. died on March 20, 2010.
  Fact 1.2: John Eric Holmes M.D. was an associate professor of neurology at the University of Southern California School of Medicine.
  Fact 1.3: John Eric Holmes M.D. was an author and promoter of fantasy role-playing games.
  Fact 1.4: John Eric Holmes M.D. was a noted fan and enthusiast of Edgar Rice Burroughs.
  Fact 1.5: John Eric Holmes M.D. was an American writer of non-fiction, fantasy and science fiction.

Retrieved items:
<item 1>
Dungeons &amp; Dragons Basic Set
The Dungeons & Dragons Basic Set is a set of rulebooks for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 First published in 1977, it saw a handful of revisions and reprintings.

 The first edition was written by J. Eric Holmes based on Gary Gygax and Dave Arneson's original work.

 Later editions were edited by Tom Moldvay, Dave Cook, and Frank Mentzer.
</item 1>

<item 2>
Dungeons &amp; Dragons Master Rules
Dungeons & Dragons Master Rules is an expansion boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 It was first published in 1985 as an expansion to the "Basic Set".
</item 2>

<item 3>
Dungeons &amp; Dragons Companion Set
The Dungeons & Dragons Companion Set is an expansion boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game.

 It was first published in 1984 as an expansion to the "Dungeons & Dragons Basic Set".
</item 3>

<item 4>
Graph query row:
person: John Eric Holmes
university: University of Southern California School of Medicine
role: None
work_relationship: None
creative_work: None
</item 4>

<item 5>
The Keep on the Borderlands
The Keep on the Borderlands is a "Dungeons & Dragons" module by Gary Gygax, first printed in December 1979.

 In it, player characters are based at a keep and investigate a nearby series of caves that are filled with a variety of monsters.

 It was designed to be used with the "Dungeons & Dragons Basic Set", and was included in the 1979–1982 editions of the "Basic Set".

 It was designed for people new to "Dungeons & Dragons".
</item 5>

<item 6>
GURPS Basic Set
GURPS Basic Set is a hard-bound two volume set written by Steve Jackson, Sean M. Punch, and David L. Pulver.

 The first edition "GURPS Basic Set" was published in 1986.
</item 6>

<item 7>
Dungeons &amp; Dragons Immortals Rules
Dungeons & Dragons Immortals Rules, written by Frank Mentzer, is a boxed set for the "Dungeons & Dragons" ("D&D") fantasy role-playing game first published by
TSR in 1986 as an expansion to the "Basic Set".
</item 7>

<item 8>
Dungeons &amp; Dragons Game (1991 boxed set)
The New Easy-to-Master Dungeons & Dragons Game is an introductory set for the "Dungeons & Dragons" fantasy role-playing game, published by TSR, Inc. in 1991.

 It was a replacement for the previous "Dungeons & Dragons Basic Set", serving to introduce new players to the game, using the rule set previously established.
</item 8>

<item 9>
Dungeons &amp; Dragons Basic Game
The Dungeons & Dragons Basic Game is an introductory version of "Dungeons & Dragons" ("D&D") role-playing game packaged in the form of a board game.

 The original game was released in 2004 by Wizards of the Coast and was designed by Jonathan Tweet, one of the "D&D" 3rd edition designers.

 A new version of this game was released in September 2006.
</item 9>

<item 10>
The Lost City (Dungeons &amp; Dragons)
The Lost City (B4) is a "Dungeons & Dragons" adventure module by Tom Moldvay.

 It was first published by TSR in 1982 and was designed as a stand-alone adventure for use with the "Dungeons & Dragons Basic Set".

 The working title for the module was "The Lost City of Cynidecia".

 Moldvay designed the module to give novice Dungeon Masters experience fleshing out adventures and is only partially complete.

 The module is described as a low-level scenario, in which the only hope of the player characters' survival can be found in a ruined city slowly rising out of
the sands.

 The adventure is set inside a huge step pyramid, with the lower pyramid only sketched out and the city itself described with a list of the major areas and a
map.

 The adventure’s main villain is Zargon, a giant one-eyed monster and his minions.

 The entire double pyramid, not including the city, contains over 100 rooms.
</item 10>
=== END ITEM id=6aec4a68cb7ea4eb ===

=== ITEM id=8dfe66710e82ab96 ===
Gold sentences and facts:
Sentence 0:  His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed according to its experimental approach.
  Fact 0.0: His experimental works are well known in the Malayalam Cinema industry.
  Fact 0.1: His experimental works are appropriately discussed in the Malayalam Cinema industry.
  Fact 0.2: His debut movie is Jalachhayam.
  Fact 0.3: Jalachhayam was released in 2010.
  Fact 0.4: Jalachhayam was well discussed according to its experimental approach.
Sentence 1: Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA, Britain and Germany.
  Fact 1.0: Jacques Feyder was born on 21 July 1885.
  Fact 1.1: Jacques Feyder died on 24 May 1948.
  Fact 1.2: Jacques Feyder was a Belgian actor.
  Fact 1.3: Jacques Feyder was a Belgian screenwriter.
  Fact 1.4: Jacques Feyder was a Belgian film director.
  Fact 1.5: Jacques Feyder worked principally in France.
  Fact 1.6: Jacques Feyder also worked in the USA.
  Fact 1.7: Jacques Feyder also worked in Britain.
  Fact 1.8: Jacques Feyder also worked in Germany.

Retrieved items:
<item 1>
Jacques Feyder
Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA,
Britain and Germany.

 He was a leading director of silent films during the 1920s, and in the 1930s he became associated with the style of poetic realism in French cinema.

 He adopted French nationality in 1928.
</item 1>

<item 2>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 2>

<item 3>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 3>

<item 4>
Alain Jessua
Alain Jessua (born 16 January 1932) is a French film director and screenwriter.

 He directed ten films between 1956 and 1997.

 He worked as assistant director for Jacques Becker on the set of "Casque d'or", with Max Ophüls for "Madame de..." and "Lola Montès" and with Marcel Carné on
"Wasteland". "

Léon la lune" his first short film won the influential Prix Jean Vigo in 1957.

 He directed first feature film in 1963 "La vie à l'envers" that won Best First Film at Venice Film Festival, in 1964.
</item 4>

<item 5>
Shafiq Syed
Shafiq Syed (born 1976) is a former Indian child actor turned auto rickshaw driver.

 At the age of twelve, he starred in a lead role in the Mira Nair's Academy Award-nominated film "Salaam Bombay!

" (1988), for which he also received the 1989 National Film Award for Best Child Artist at the 36th National Film Awards.

 He drives an auto rickshaw in Bangalore and also works as an assistant in television production units.
</item 5>

<item 6>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 6>

<item 7>
Sathish Kalathil
Sathish Kalathil (മലയാളം: ) is an Indian film and documentary Director and Producer in malayalam.

 He is also Story Writer, and Lyricist.

 His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed
according to its experimental approach.

 Actually, his first venture was in 2008 named Veena Vaadanam, a documentary film about Art (Painting) and that was his first experimental entry in movie.

 In 2012, he directed Laloorinu Parayanullathu, a social committed documentary film about Municipal solid waste.
</item 7>

<item 8>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 8>

<item 9>
L'Atlantide (1921 film)
L'Atlantide is a 1921 French-Belgian silent film directed by Jacques Feyder, and the first of several adaptations of the best-selling novel "L'Atlantide" by
Pierre Benoit.
</item 9>

<item 10>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 10>
=== END ITEM id=8dfe66710e82ab96 ===

=== ITEM id=9d03712d3613358c ===
Gold sentences and facts:
Sentence 0: Sasson Gabai (Hebrew: ששון גבאי‎ ‎ ; born 24 November 1947) is an Israeli actor.
  Fact 0.0: Sasson Gabai's Hebrew name is ששון גבאי.
  Fact 0.1: Sasson Gabai was born on 24 November 1947.
  Fact 0.2: Sasson Gabai is an Israeli actor.
Sentence 1:  Winner of an Ophir Award, Asia Pacific Screen Award (nominated), European Film Awards and Jerusalem Film Festival in 2007 as Best Actor in a highly acclaimed Israeli film "The Band's Visit".
  Fact 1.0: Sasson Gabai won an Ophir Award.
  Fact 1.1: Sasson Gabai was nominated for an Asia Pacific Screen Award.
  Fact 1.2: Sasson Gabai won a European Film Awards award.
  Fact 1.3: Sasson Gabai won a Jerusalem Film Festival award.
  Fact 1.4: These awards were won in 2007.
  Fact 1.5: The awards were for Best Actor.
  Fact 1.6: The award-winning role was in the highly acclaimed Israeli film The Band's Visit.
Sentence 2: The Band's Visit (Hebrew: ביקור התזמורת - "Bikur Ha-Tizmoret") is a 2007 Israeli film directed by Eran Kolirin.
  Fact 2.0: The Band's Visit is a 2007 Israeli film.
  Fact 2.1: The Band's Visit is written in Hebrew as ביקור התזמורת.
  Fact 2.2: The Hebrew title of The Band's Visit is Bikur Ha-Tizmoret.
  Fact 2.3: The Band's Visit was directed by Eran Kolirin.

Retrieved items:
<item 1>
Beyond the Sea (1991 film)
Beyond the Sea (translit.

 Me'ever Layam) is a 1991 Israeli drama film directed by Jacob Goldwasser.

 It stars Aryeh Moskona, Dafna Rechter, Moti Giladi.

 The film, a family drama set in the 1960s, was critically acclaimed, and won the Ophir Award for Best Film.

 The film was selected as the Israeli entry for the Best Foreign Language Film at the 64th Academy Awards, but was not accepted as a nominee.
</item 1>

<item 2>
Sh'Chur
Sh'Chur is a 1994 Israeli drama film starring Gila Almagor, Ronit Elkabetz and Hanna Azoulay-Hasfari.

 It was written by Hanna Azoulay Hasfari and directed by her partner Shmuel Hasfari.

 Sh'Chur received critical acclaim and was the 1994 official Israeli submission for the Academy Award for Best Foreign Language Film.

 It was also awarded the Ophir Award for best film by the Israeli Film and Television Academy.

 After its release, the film garnered various discussions in the Israeli press over its representation of the Moroccan community in Israel.
</item 2>

<item 3>
The Band's Visit
The Band's Visit (Hebrew: ביקור התזמורת - "Bikur Ha-Tizmoret") is a 2007 Israeli film directed by Eran Kolirin.
</item 3>

<item 4>
Yana's Friends
Yana's Friends (Hebrew: החברים של יאנה, "HaHaverim shel Yana"‎ ‎ ) is a 1999 Israeli film directed by Arik Kaplun.

 script editor: Savi Gabizon.

 Critically acclaimed, it won 10 Israeli Academy Awards including the Ophir Award for Best Picture.

 It also won the Crystal Globe at the Karlovy Vary International Film Festival in 1999.

 The film has a very rare 100% rating on the film website Rotten Tomatoes based on 30 reviews.
</item 4>

<item 5>
Sasson Gabai
Sasson Gabai (Hebrew: ששון גבאי‎ ‎ ; born 24 November 1947) is an Israeli actor.

 Winner of an Ophir Award, Asia Pacific Screen Award (nominated), European Film Awards and Jerusalem Film Festival in 2007 as Best Actor in a highly acclaimed
Israeli film "The Band's Visit".
</item 5>

<item 6>
The Farewell Party
The Farewell Party (Hebrew: מיתה טובה‎ ‎ Mita Tova) is a 2014 Israeli drama film about the use of a euthanasia device, directed by Tal Granit and Sharon Maymon.

 It was nominated for the Ophir Award for Best Film.

 The film was screened in the Venice Days section of the 71st Venice International Film Festival and has been selected to be screened in the Contemporary World
Cinema section at the 2014 Toronto International Film Festival.

 The film was also screened in the 44th edition of the International Film Festival Rotterdam, reaching the second place in the IFFR audience award.
</item 6>

<item 7>
Hunting Elephants
Hunting Elephants (Hebrew: לצוד פילים‎ ‎ , "Latzud Pilim", "lit."

 To Hunt Elephants) is a 2013 Israeli crime comedy film directed by Reshef Levi.

 It stars, among others, Sasson Gabai, Moni Moshonov and Patrick Stewart.

 It was released to theaters in Israel on July 4, 2013.
</item 7>

<item 8>
Joseph Cedar
Yossef (Joseph) Cedar (Hebrew: יוסף סידר; born August 31, 1968) is an Israeli film director and screenwriter.

 He has won a Silver Bear and an Ophir Award for Best Director, and an Ophir Award for writing a Best Screenplay.

 He also won the best screenplay award at the 2011 Cannes Film Festival for his film "Footnote" (2011).
</item 8>

<item 9>
Foxtrot (2017 film)
Foxtrot (Hebrew: פוֹקְסטְרוֹט‎ ‎ ) is a 2017 Israeli drama film directed by Samuel Maoz.

 It was screened in the competition section of the 74th Venice International Film Festival where it won the Grand Jury Prize Silver Lion.

 It was also screened in the Special Presentations section at the 2017 Toronto International Film Festival.

 It won the Ophir Award for Best Film, therefore becoming the Israeli entry for the Best Foreign Language Film at the 90th Academy Awards.
</item 9>

<item 10>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 10>
=== END ITEM id=9d03712d3613358c ===

=== ITEM id=c1cd69fdd3596cda ===
Gold sentences and facts:
Sentence 0: Wilmington International Airport (IATA: ILM, ICAO: KILM, FAA LID: ILM) is a public airport located just north of Wilmington, North Carolina, in unincorporated Wrightsboro, Cape Fear Township, New Hanover County.
  Fact 0.0: Wilmington International Airport has the IATA code ILM.
  Fact 0.1: Wilmington International Airport has the ICAO code KILM.
  Fact 0.2: Wilmington International Airport has the FAA LID ILM.
  Fact 0.3: Wilmington International Airport is a public airport.
  Fact 0.4: Wilmington International Airport is located just north of Wilmington, North Carolina.
  Fact 0.5: Wilmington International Airport is located in unincorporated Wrightsboro, Cape Fear Township, New Hanover County.
Sentence 1: Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km) northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.
  Fact 1.0: Texarkana Regional Airport has the IATA code TXK.
  Fact 1.1: Texarkana Regional Airport has the ICAO code KTXK.
  Fact 1.2: Texarkana Regional Airport has the FAA LID TXK.
  Fact 1.3: Texarkana Regional Airport is also known as Webb Field.
  Fact 1.4: Texarkana Regional Airport is a public use airport.
  Fact 1.5: Texarkana Regional Airport is located three nautical miles (6 km) northeast of the central business district of Texarkana.
  Fact 1.6: Texarkana is a city in Miller County, Arkansas, United States.

Retrieved items:
<item 1>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 1>

<item 2>
Wilmington International Airport
Wilmington International Airport (IATA: ILM, ICAO: KILM, FAA LID: ILM) is a public airport located just north of Wilmington, North Carolina, in unincorporated
Wrightsboro, Cape Fear Township, New Hanover County.
</item 2>

<item 3>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 3>

<item 4>
Sharp County Regional Airport
Sharp County Regional Airport (IATA: CKK, ICAO: KCVK, FAA LID: CVK) is a public-use airport in Sharp County, Arkansas, United States.

 It is located three nautical miles (4 mi, 6 km) northeast of the central business district of Ash Flat, Arkansas.

 The airport is owned by the Sharp County Regional Airport Authority.
</item 4>

<item 5>
The entries describe U.S. and Saudi airports and authorities: Atlanta South (formerly Tara Field); remote Yakutat, served through Essential Air Service;
Wilmington, NC; Pennsylvania’s SARAA, overseeing four airports; Ta’if, a regional airport with international flights; Raleigh Exec, a reliever; Texarkana,
serving general aviation and American Eagle; Pensacola, despite its name, without direct international flights; Wilmington, Delaware; Columbus (formerly Port
Columbus); Amarillo, renamed for
</item 5>

<item 6>
Atlantic Southeast Airlines Flight 2311
Atlantic Southeast Airlines Flight 2311 was a regularly scheduled commuter flight from Hartsfield–Jackson Atlanta International Airport to Glynco Jetport (since
renamed Brunswick Golden Isles Airport) in Brunswick, Georgia on April 5, 1991.

 The flight, operated using a twin-turboprop Embraer EMB 120 Brasilia, crashed just north of Brunswick while approaching the airport for landing.

 All 23 people aboard the plane were killed, including passengers Sonny Carter and John Tower.

 Four years later, another Embraer Brasilia of ASA crashed in the Georgia countryside in similar circumstances, with nine fatalities.
</item 6>

<item 7>
Raleigh Executive Jetport
Raleigh Exec: The Raleigh Executive Jetport @ Sanford-Lee County or Raleigh Exec Jetport at Sanford-Lee CountyFAA Airport Master Record for TTA (Form 5010 )
(ICAO: KTTA, FAA LID: TTA) is a public use airport located seven nautical miles (8 mi, 13 km) northeast of the central business district of Sanford, a city in
Lee County, North Carolina, United States.

 It is owned by the Sanford-Lee County Regional Airport Authority and was previously known as Sanford-Lee County Regional Airport.

 This airport is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "reliever airport" for Raleigh-Durham
International Airport.
</item 7>

<item 8>
Dane County Regional Airport
Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown
Madison, the capital of Wisconsin.

 In the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, it is categorized as a small-hub primary commercial
service facility.
</item 8>

<item 9>
Wilmington Airport (Delaware)
Wilmington Airport (IATA: ILG) (also known as Wilmington/Philadelphia Regional Airport, New Castle County Airport, or New Castle Airport) is an airport located
in unincorporated New Castle County, Delaware near Wilmington, Delaware.

 Owned by the Delaware River and Bay Authority, it is five miles (8 km) south of Wilmington and about 30 miles (50 km) from Philadelphia.

 It is included in the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, in which it is categorized as a non-hub
primary commercial service facility.
</item 9>

<item 10>
Texarkana Regional Airport
Texarkana Regional Airport (IATA: TXK, ICAO: KTXK, FAA LID: TXK) , also known as Webb Field, is a public use airport located three nautical miles (6 km)
northeast of the central business district of Texarkana, a city in Miller County, Arkansas, United States.

 It is owned by the Texarkana Airport Authority.

 The airport is located within the city limits of Texarkana, roughly 3 miles east of State Line Avenue.

 The front gate opens to the northwest, at the intersection of Arkansas Boulevard and U.S. Route 67.

 A Union Pacific Railroad line runs parallel to US 67 on the side of the highway facing the airport.

 It is mostly used for general aviation, but is also served by American Eagle.
</item 10>
=== END ITEM id=c1cd69fdd3596cda ===

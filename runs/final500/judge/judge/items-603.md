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

=== ITEM id=169a85a3f1e840b3 ===
Gold sentences and facts:
Sentence 0:  In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
  Fact 0.0: In Crash, there is no betting, unlike in Brag.
  Fact 0.1: In Crash, players aim to reach a total of 11 points.
  Fact 0.2: In Crash, the 11 points are gained over successive deals.
Sentence 1: Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.
  Fact 1.0: Three-card brag is a 16th-century British card game.
  Fact 1.1: Three-card brag is the British national representative of the vying or 'bluffing' family of gambling games.

Retrieved items:
<item 1>
Triomphe
Triomphe (French for triumph) is a card game dating from the late 15th century.

 It most likely originated in France or Spain (as triunfo) and later spread to the rest of Europe.

 When the game arrived in Italy, it shared a similar name with the pre-existing game and deck known as "trionfi" (tarot).

 While trionfi has a fifth suit that acts as permanent trumps, triomphe randomly selects one of the existing four suits as trumps.

 Another common feature of this game is the robbing of the stock.

 Triomphe became so popular that during the 16th century the earlier game of trionfi was gradually renamed tarocchi, tarot, or tarock.

 This game is the origin of the English word "trump" and is the ancestor of many trick-taking games like Euchre (via Écarté) and Whist (via Ruff and Honours).
</item 1>

<item 2>
Egyptian Ratscrew
Egyptian Ratscrew (also known as Egyptian Ratkiller, Egyptian War, and by other names) is a card game of the matching family of games.

 The game is similar to the 19th century British card game Beggar-My-Neighbour, with the added concept of "slapping" cards when certain combinations are played,
similar to and perhaps borrowed from Slapjack.
</item 2>

<item 3>
Crash (card game)
Crash is a British card game extension of Nine-card Brag.

 In Crash, there is no betting, as in Brag, but rather players aim to reach a total of 11 points, gained over successive deals.
</item 3>

<item 4>
TriBond (1990) is a knowledge-and-reasoning board game sold in 14 countries. Tapp-Tarock, an Austrian three-player tarot game recorded in 1821, was once popular
in Viennese cafés. Three-card brag, a 16th-century British bluffing game descended from Primero, helped inspire poker. Triomphe, a late-15th-century
trick-taking game, introduced randomly chosen trumps and inspired many later games. Ombre
</item 4>

<item 5>
Three card brag
Three-card brag is a 16th-century British card game, and the British national representative of the vying or "bluffing" family of gambling games.

 Brag is a direct descendant of the Elizabethan game of Primero and one of the several ancestors to poker, just varying in betting style and hand rankings.
</item 5>

<item 6>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 6>

<item 7>
Écarté
Écarté is a two-player card game originating from France, the word literally meaning "discarded".

 It is a trick-taking game, similar to whist, but with a special and eponymous discarding phase.

 It is closely related to Euchre, a card game played mainly in the United States.

 Écarté was popular in the 19th century, but is now rarely played.
</item 7>

<item 8>
Put (card game)
Put is an English tavern trick-taking card game first recorded in the 16th century and later castigated by 17th century moralists as one of ill repute.

 It belongs to a very ancient family of card games and clearly relates to a group known as Trut, "Truque", also "Tru", and the South American game Truco.

 Its more elaborate version is the Spanish game of Truc, which is still much played in many parts of Southern France and Spain.
</item 8>

<item 9>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 9>

<item 10>
Ombre
The historical importance of Ombre in the field of playing cards is the fact that it was the first card game in which a trump suit was established by bidding
rather than by the random process of turning the first card of the stock.

 This game developed from Triunfo, though it was from L'Hombre that the idea of bidding was adopted into other card games such as Skat, and Tarot, which owes
Hombre a good portion of its betting system as well.

 The game continued to be in vogue almost in every corner of Europe from the late 17th through the 18th centuries.
</item 10>
=== END ITEM id=169a85a3f1e840b3 ===

=== ITEM id=bb9911e36a7ffa10 ===
Gold sentences and facts:
Sentence 0: Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.
  Fact 0.0: Buried Country is a highly regarded documentary film
  Fact 0.1: Buried Country is a highly regarded book
  Fact 0.2: Buried Country is a highly regarded soundtrack album
  Fact 0.3: Buried Country is now also a stageshow
Sentence 1: South of the Border is a 2009 American documentary film directed by Oliver Stone.
  Fact 1.0: South of the Border is a documentary film
  Fact 1.1: South of the Border is American
  Fact 1.2: South of the Border is from 2009
  Fact 1.3: South of the Border was directed by Oliver Stone

Retrieved items:
<item 1>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 1>

<item 2>
The entries survey theatre, film and literary figures/works: Boucicault’s 1860 Irish melodrama *The Colleen Bawn*; director Jules Wright and actress-filmmaker
Laura Bayley; films including *Summer Holiday*, *Trio*, *Julia*, *The King and I*, and *Anna and the King of Siam*; and plays by Lillian Hellman, Steven Dietz,
Aphra Behn, John Burgoyne, and Thornton
</item 2>

<item 3>
The Colleen Bawn
The Colleen Bawn, or The Brides of Garryowen is a melodramatic play written by Irish playwright Dion Boucicault.

 It was first performed at Miss Laura Keene's Theatre, New York, on 27 March 1860 with Laura Keene playing Anne Chute and Boucicault playing Myles na Coppaleen.

 It was most recently performed in Dublin at the Project Arts Centre in July and August 2010.

 Several film versions have also been made.
</item 3>

<item 4>
Buried Country
Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.

 It is a prosopography, created by noted writer Clinton Walker, that tells the story of Australian country music in the Aboriginal community by focussing on the
genre's most important stars.
</item 4>

<item 5>
Cinema of Uruguay
The cinema of Uruguay has a role in the culture of Uruguay and is a part of Latin American cinema.

 Since the late 1990s, the Uruguayan cinema undergoes a process of evolution, during which its films have received positive reviews and been internationally
recognized.

 Over 120 films, fictions and non-fictions, have been produced since then.
</item 5>

<item 6>
The entries span film, art, and music: *Cool and the Crazy* (1994) is Ralph Bakshi’s first live-action feature, about an unfaithful 1950s couple; *No eres tú,
soy yo* (2010) is a Mexican remake of an Argentine comedy; and *The Straight Story* (1999) follows Alvin Straight’s 240-mile tractor journey to visit his
estranged, ill brother. Other films include *Lilly the Witch
</item 6>

<item 7>
Simon Bent
Simon Bent is a British screenwriter and playwright, notable for work including BBC TV drama "" (2006), the screenplay for the feature film "Christie Malry's
Own Double-Entry" (2000), and the Joe Orton biographical play "Prick Up Your Ears" based on John Lahr's book.
</item 7>

<item 8>
South of the Border (2009 film)
South of the Border is a 2009 American documentary film directed by Oliver Stone.

 The documentary premiered at the 2009 Venice Film Festival.

 Writer for the project Tariq Ali calls the documentary "a political road movie".

 Stone stated that he hopes the film will help people better understand a leader who is wrongly ridiculed "as a strongman, as a buffoon, as a clown."
</item 8>

<item 9>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 9>

<item 10>
Plain Clothes Theatre Productions
Plain Clothes Theatre Productions is a Bristol-based theatre company producing contemporary plays from around the globe.

 Formed in 2003 by artistic director Sam Berger, the company has produced work in London, Bristol, Cheltenham, Toronto and Vancouver.

 Their work has included plays by Joe Orton, Laura Wade and David Mamet, and their style is based around the teachings of American acting coach Sanford Meisner.
</item 10>
=== END ITEM id=bb9911e36a7ffa10 ===

=== ITEM id=1a1a6514d9c07348 ===
Gold sentences and facts:
Sentence 0: Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central business district of Brainerd, a city in Crow Wing County, Minnesota, United States.
  Fact 0.0: Brainerd Lakes Regional Airport has IATA code BRD.
  Fact 0.1: Brainerd Lakes Regional Airport has ICAO code KBRD.
  Fact 0.2: Brainerd Lakes Regional Airport has FAA LID BRD.
  Fact 0.3: Brainerd Lakes Regional Airport is a public use airport.
  Fact 0.4: Brainerd Lakes Regional Airport is located three nautical miles (6 km) northeast of the central business district of Brainerd.
  Fact 0.5: Brainerd is a city in Crow Wing County, Minnesota, United States.
Sentence 1: Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City, Utah in the United States.
  Fact 1.0: Salt Lake City International Airport has IATA code SLC.
  Fact 1.1: Salt Lake City International Airport has ICAO code KSLC.
  Fact 1.2: Salt Lake City International Airport has FAA LID SLC.
  Fact 1.3: Salt Lake City International Airport is a civil-military airport.
  Fact 1.4: Salt Lake City International Airport is located about 4 miles west of Downtown Salt Lake City.
  Fact 1.5: Downtown Salt Lake City is in Utah, United States.

Retrieved items:
<item 1>
Salt Lake City International Airport
Salt Lake City International Airport (IATA: SLC, ICAO: KSLC, FAA LID: SLC) is a civil-military airport located about 4 miles west of Downtown Salt Lake City,
Utah in the United States.

 The airport is the closest commercial airport for more than 2.5 million people and is within a 30-minute drive of nearly 1.3 million jobs.
</item 1>

<item 2>
City Center (UTA station)
City Center is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit
Authority's (UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 2>

<item 3>
900 South (UTA station)
900 South is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by all three lines of Utah Transit Authority's TRAX light rail
system.

 The Blue Line provides service from Downtown Salt Lake City to Draper.

 The Red Line provides service from the University of Utah to the Daybreak Community of South Jordan.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).

 The station opened on 19 September 2005 and is operated by the Utah Transit Authority.

 900 South is notable for being the first infill station constructed along an existing line of the TRAX system.
</item 3>

<item 4>
Salt Lake City Air Route Traffic Control Center
Salt Lake City Air Route Traffic Control Center (ZLC) (radio communications "Salt Lake Center") is one of 22 FAA Area Control Centers in the United States.

 It is located in Salt Lake City, Utah, adjacent to Salt Lake City International Airport.

 It was opened in 1939 and was originally located on the third floor of the old Salt Lake City International Airport terminal.

 The Salt Lake Center (ZLC) covers one of the largest geographical areas of any other control center, totaling approximately 350,000 squares miles.
</item 4>

<item 5>
North Temple Bridge/Guadalupe (UTA station)
North Temple Bridge/Guadalupe is a light rail and commuter rail station in Salt Lake City, Utah, United States served by the Green Line of Utah Transit
Authority's (UTA) TRAX light rail system and by the "FrontRunner", UTA's commuter rail train that operates along the Wasatch Front with service from Pleasant
View in northern Weber County through Ogden, Davis County, Salt Lake City, and Salt Lake County to Provo in central Utah County.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City) and connects with the rest of
the TRAX system and UTA's S Line streetcar.

 Although the official name of the station is "North Temple Bridge/Guadalupe", it often referenced by UTA as simply North Temple.
</item 5>

<item 6>
San Antonio International Airport
San Antonio International Airport (IATA: SAT, ICAO: KSAT, FAA LID: SAT) is an international airport located in San Antonio, Texas and serving the Greater San
Antonio metropolitan area.

 It is located in Uptown Central San Antonio, about 8 miles north of Downtown.

 Its has three runways and covers 2,305 acres (933 ha).

 Its elevation is 809 ft above sea level.

 SAT is a Class C airport.
</item 6>

<item 7>
Arena (UTA station)
Arena is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit Authority's
(UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line has service from the Salt Lake City International Airport and to West Valley City via Downtown Salt Lake City.
</item 7>

<item 8>
Temple Square (UTA station)
Temple Square is a light rail station in Downtown Salt Lake City, Utah, in the United States, served by the Blue Line and Green Line of the Utah Transit
Authority's (UTA) TRAX system.

 The Blue Line has service from the Salt Lake Intermodal Hub in Downtown Salt Lake City to Draper.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 8>

<item 9>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
</item 9>

<item 10>
Courthouse (UTA station)
Courthouse is a light rail station in Downtown Salt Lake City, Utah, United States serviced by all three lines of Utah Transit Authority's TRAX light rail
system.

 The Blue Line provides service from Downtown Salt Lake City to Draper.

 The Red Line provides service from the University of Utah to the Daybreak Community of South Jordan.

 The Green Line provides service from the Salt Lake City International Airport to West Valley City (via Downtown Salt Lake City).
</item 10>
=== END ITEM id=1a1a6514d9c07348 ===

=== ITEM id=2e346ddcf328dc0d ===
Gold sentences and facts:
Sentence 0: Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil War.
  Fact 0.0: Am Rong was born in 1929.
  Fact 0.1: Am Rong died in May 1975.
  Fact 0.2: Am Rong was Cambodian.
  Fact 0.3: Am Rong was a soldier.
  Fact 0.4: Am Rong was a filmmaker.
  Fact 0.5: Am Rong acted as a spokesman on military matters for the Khmer Republic.
  Fact 0.6: Am Rong acted as a spokesman during the Cambodian Civil War.
Sentence 1: Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.
  Fact 1.0: Ava Marie DuVernay was born August 24, 1972.
  Fact 1.1: Ava Marie DuVernay is American.
  Fact 1.2: Ava Marie DuVernay is a director.
  Fact 1.3: Ava Marie DuVernay is a screenwriter.
  Fact 1.4: Ava Marie DuVernay is a film marketer.
  Fact 1.5: Ava Marie DuVernay is a film distributor.

Retrieved items:
<item 1>
Ava DuVernay
Ava Marie DuVernay ( ; born August 24, 1972) is an American director, screenwriter, film marketer, and film distributor.

 At the 2012 Sundance Film Festival, DuVernay won the Best Director Prize for her second feature film "Middle of Nowhere", becoming the first African-American
woman to win the award.

 For her work in "Selma" (2014), DuVernay was the first black female director to be nominated for a Golden Globe Award.

 With "Selma", she was also the first black female director to have her film nominated for the Academy Award for Best Picture.

 In 2017, she was nominated for the Academy Award for Best Documentary Feature for her film "13th" (2016).
</item 1>

<item 2>
Annie Belle
Annie Brilland (born 10 December 1956) is a French actress and social worker.

 Her acting career began in 1974 and throughout the seventies, has had a series of varied roles in both French and Italian cinema, working with such directors
as Jean Rollin, Ruggero Deodato and Joe D'Amato.
</item 2>

<item 3>
Spencer Averick
Spencer Averick is an American film editor and producer.

 Best known for his work an editor on critically acclaimed films "Middle of Nowhere" (2012), "Selma" (2014) and for producing 2016 acclaimed documentary "13th"
for which he received Academy Award for Best Documentary Feature nominations at 89th Academy Awards, that he shared with director Ava DuVernay and co-producer
Howard Barish.
</item 3>

<item 4>
Graph query row:
person: Ava DuVernay
birth_date: August 24, 1972
</item 4>

<item 5>
Am Rong
Am Rong (1929 – May 1975) was a Cambodian soldier and filmmaker, who acted as a spokesman on military matters for the Khmer Republic during the Cambodian Civil
War.

 Western journalists commented on the irony of his name as he gave briefings which "painted a rosy picture of the increasingly desperate situation on the
ground" during the war.
</item 5>

<item 6>
Selma (film)
Selma is a 2014 American historical drama film directed by Ava DuVernay and written by Paul Webb.

 It is based on the 1965 Selma to Montgomery voting rights marches led by James Bevel, Hosea Williams, Martin Luther King, Jr. and John Lewis.

 The film stars actors David Oyelowo as King, Tom Wilkinson as President Lyndon B. Johnson, Tim Roth as George Wallace, Carmen Ejogo as Coretta Scott King and
Common as Bevel.
</item 6>

<item 7>
Betty Ann Davies
Betty Ann Davies (24 December 1910 – 14 May 1955) was a British stage and film actress active from the 1920s to the 1950s.

 She died young in 1955 aged 44 following an operation.

 Miss Davies made her first stage appearance at the Palladium in a revue in 1924.

 The following year she joined Cochran's Young Ladies in revues such as "One Dam Thing After Another" and "This Year of Grace".

 Betty Ann Davies enjoyed a long and distinguished West End career which included "The Good Companions" 1934, "Morning Star" 1942, and "Blithe Spirit" 1943.

 Her outstanding stage triumph was in the role of Blanche du Bois, which she took over from Vivien Leigh, in the original West End production of " A Streetcar
Named Desire".

 Miss Davis appeared in 38 films, and was active in TV at the time of her death.

 She left one son Brook Blackford.
</item 7>

<item 8>
Emayatzy Corinealdi
Emayatzy Evett Corinealdi ( ; born January 14, 1980) is an American actress.

 She is best known for her leading role in the Ava DuVernay film "Middle of Nowhere" (2012).
</item 8>

<item 9>
Howard Barish
Howard Barish is an Oscar nominated, Emmy award winning producer and director known for his producing partnership with acclaimed director, writer Ava DuVernay.

 Their most recognized project to date, "13TH", is a 2016 American documentary from Netflix directed by DuVernay.

 Centered on race in the United States criminal justice system, the critically lauded film is titled after the Thirteenth Amendment to the United States
Constitution, which outlawed slavery (unless as punishment for a crime).

 It argues that slavery is being effectively perpetuated through mass incarceration.
</item 9>

<item 10>
Graph query row:
person: Am Rong
birth_date: 1929
</item 10>
=== END ITEM id=2e346ddcf328dc0d ===

=== ITEM id=a55f032766a61b8a ===
Gold sentences and facts:
Sentence 0: Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
  Fact 0.0: Milium's vernacular name is milletgrass.
  Fact 0.1: Milium is a genus of plants in the grass family.
  Fact 0.2: Milium plants are Eurasian, North American, and North African.
Sentence 1: Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.
  Fact 1.0: Eucommia is a genus of small trees.
  Fact 1.1: Eucommia trees are now native to China.
  Fact 1.2: The fossil record of Eucommia shows a much wider distribution.
Sentence 2:  The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in herbology such as traditional Chinese medicine.
  Fact 2.0: Eucommia has a single living species, Eucommia ulmoides.
  Fact 2.1: Eucommia ulmoides is near threatened in the wild.
  Fact 2.2: Eucommia ulmoides is widely cultivated in China for its bark.
  Fact 2.3: Eucommia ulmoides is highly valued in herbology, such as traditional Chinese medicine.

Retrieved items:
<item 1>
Hydrangea
Hydrangea ( ; common names hydrangea or hortensia) is a genus of 70–75 species of flowering plants native to southern and eastern Asia (China, Japan, Korea, the
Himalayas, and Indonesia) and the Americas.

 By far the greatest species diversity is in eastern Asia, notably China, Japan, and Korea.

 Most are shrubs 1 to 3 meters tall, but some are small trees, and others lianas reaching up to 30 m by climbing up trees.

 They can be either deciduous or evergreen, though the widely cultivated temperate species are all deciduous.
</item 1>

<item 2>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 2>

<item 3>
Eucryphia
Eucryphia is a small genus of trees and large shrubs native to the south temperate regions of South America and coastal eastern Australia.

 Sometimes placed in a family of their own, the Eucryphiaceae, more recent classifications place them in the Cunoniaceae.

 There are seven species, two in South America and five in Australia, and several named hybrids.

 They are mostly evergreen though one species ("E. glutinosa") is usually deciduous.
</item 3>

<item 4>
Eucommia
Eucommia is a genus of small trees now native to China, with a fossil record that shows a much wider distribution.

 The single living species, "Eucommia ulmoides", is near threatened in the wild, but is widely cultivated in China for its bark, and is highly valued in
herbology such as traditional Chinese medicine.
</item 4>

<item 5>
Oak
An oak is a tree or shrub in the genus Quercus ( ; Latin "oak tree") of the beech family, Fagaceae.

 There are approximately 600 extant species of oaks.

 The common name "oak" also appears in the names of species in related genera, notably "Lithocarpus" (stone oaks), as well as in those of unrelated species such
as "Grevillea robusta" (silky oaks) and the Casuarinaceae (she-oaks).

 The genus "Quercus" is native to the Northern Hemisphere, and includes deciduous and evergreen species extending from cool temperate to tropical latitudes in
the Americas, Asia, Europe, and North Africa.

 North America contains the largest number of oak species, with approximately 90 occurring in the United States, while Mexico has 160 species of which 109 are
endemic.

 The second greatest center of oak diversity is China, which contains approximately 100 species.
</item 5>

<item 6>
Milium (plant)
Milium (vernacular name milletgrass) is a genus of Eurasian, North American, and North African plants in the grass family.
</item 6>

<item 7>
Visayan spotted deer
The Visayan spotted deer ("Rusa alfredi"), also known as the Philippine spotted deer, is a nocturnal and endangered species of deer located primarily in the
rainforests of the Visayan islands of Panay and Negros though it once roamed other islands such as Cebu, Guimaras, Leyte, Masbate, and Samar.

 It is one of three endemic deer species in the Philippines, although it was not recognized as a separate species until 1983.

 An estimated 2,500 mature individuals survived worldwide as of 1996, according to the IUCN, although it is uncertain of how many of them still survive in the
wild.

 The diet of the deer, which consists of a variety of different types of grasses, leaves, and buds within the forest, is the primary indicator of its habitat.

 Since 1991 the range of the species has severely decreased and is now almost co-extensive with that of the Visayan warty pig.
</item 7>

<item 8>
Ocimum
Ocimum is a genus of aromatic annual and perennial herbs and shrubs in the family Lamiaceae, native to the tropical and warm temperate regions of all 6
inhabited continents, with the greatest number of species in Africa.

 Its best known species are the cooking herb Cooking basil, "O. basilicum" and the medicinal herb Tulsi (holy basil), "O. tenuiflorum".
</item 8>

<item 9>
Myrsine
Myrsine is a genus of flowering plants in the family Primulaceae.

 It was formerly placed in the family Myrsinaceae before this was merged into the Primulaceae.

 It is found nearly worldwide, primarily in tropical and subtropical areas.

 It contains about 200 species, including several notable radiations, such as the matipo of New Zealand and the kōlea of Hawaiʻ i (the New Zealand "black
matipo", "Pittosporum tenuifolium", is not related to "Myrsine").

 In the United States, members of this genus are known as colicwood.

 Some species, especially "M. africana", are grown as ornamental shrubs.
</item 9>

<item 10>
Wild boar
The wild boar ("Sus scrofa"), also known as the wild swine or Eurasian wild pig, is a suid native to much of Eurasia, North Africa, and the Greater Sunda
Islands.

 Human intervention has spread its distribution further, making the species one of the widest-ranging mammals in the world, as well as the most widely spread
suiform.

 Its wide range, high numbers, and adaptability mean that it is classed as least concern by the IUCN and it has become an invasive species in part of its
introduced range.

 The animal probably originated in Southeast Asia during the Early Pleistocene, and outcompeted other suid species as it spread throughout the Old World.
</item 10>
=== END ITEM id=a55f032766a61b8a ===

=== ITEM id=2f4e9939350634e7 ===
Gold sentences and facts:
Sentence 0:  He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".
  Fact 0.0: The person described has written a column for Fortune magazine for more than twenty years.
  Fact 0.1: The person described spent a decade at Esquire before writing for Fortune.
Sentence 1:  Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
  Fact 1.0: The described publication was founded in 1933.
  Fact 1.1: The described publication flourished during the Great Depression.
  Fact 1.2: The described publication flourished under the guidance of founders Arnold Gingrich, David A. Smart, and Henry L. Jackson.

Retrieved items:
<item 1>
This Is Bing Crosby
This Is Bing Crosby was a fifteen-minute five times a week daytime radio program featuring Bing Crosby acting as a disc jockey.

 Minute Maid quick frozen concentrated orange juice was promoted on the shows.
</item 1>

<item 2>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 2>

<item 3>
Dispatches (book)
Dispatches is a New Journalism book by Michael Herr that describes the author's experiences in Vietnam as a war correspondent for "Esquire" magazine.

 First published in 1977, "Dispatches" was one of the first pieces of American literature that portrayed the experiences of soldiers in the Vietnam War for
American readers.
</item 3>

<item 4>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
</item 4>

<item 5>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 5>

<item 6>
Jerry Beck
Jerry Beck (born February 9, 1955 in New York City) is an American animation historian, author, blogger, and video producer.

 The author and/or editor of several books on classic American animation and classic characters, including "The 50 Greatest Cartoons" (1994), "The Animated
Movie Guide" (2005), "Not Just Cartoons: Nicktoons!"

 (2007), "The Flintstones: The Official Guide to the Cartoon Classic" (2011), "The Hanna-Barbera Treasury: Rare Art Mementos from Your Favorite Cartoon
Classics" (2007), "The SpongeBob SquarePants Experience: A Deep Dive into the World of Bikini Bottom" (2013), "Pink Panther: The Ultimate Guide" (2005), and
"Looney Tunes and Merrie Melodies: A Complete Illustrated Guide to the Warner Bros.

 Cartoons" (with Will Friedwald, 1989) alongside "The 100 Greatest Looney Tunes Cartoons" (2010).

 He is also an authority on the making of modern films, with his books detailing the art of "Mr.
</item 6>

<item 7>
Esquire (magazine)
Esquire is an American men's magazine, published by the Hearst Corporation in the United States.

 Founded in 1933, it flourished during the Great Depression under the guidance of founders Arnold Gingrich, David A. Smart and Henry L. Jackson.
</item 7>

<item 8>
André Laguerre
Marc André Laguerre (February 21, 1915 – January 18, 1979) was a journalist and magazine editor, best known as the managing editor of "Sports Illustrated" from
1960 to 1974, during which time he oversaw the growth in the magazine from a niche publication to become the industry leader in weekly sports magazines.

 It was under his leadership that the annual "Sports Illustrated Swimsuit Issue" was first published.

 When he retired in 1974, he had been managing editor of the magazine for 704 issues, then a record among magazines published by Time, Inc., SI's parent
company.
</item 8>

<item 9>
Stanley Bing
Stanley Bing is the pen name of Gil Schwartz (born May 20, 1951 in New York, NY), a business humorist and novelist.

 He has written a column for "Fortune" magazine for more than twenty years, after having spent a decade at "Esquire".

 He is the author of thirteen books including "What Would Machiavelli Do?"

 and "The Curriculum", a satirical textbook for a business school that also offers lessons on the Web.

 Schwartz is the senior executive vice president of corporate communications and Chief Communications Officer for CBS.
</item 9>

<item 10>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 10>
=== END ITEM id=2f4e9939350634e7 ===

=== ITEM id=eacd543247604ec4 ===
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
Volvo S90
The Volvo S90 is an executive sedan manufactured and marketed by Swedish automaker Volvo Cars since 2016.

 Its estate variant is called the Volvo V90.
</item 1>

<item 2>
Volvo S40
The Volvo S40 is a compact luxury car marketed and produced by the Swedish manufacturer Volvo Cars from 1995 to 2012 across two generations.
</item 2>

<item 3>
Volvo C30
The Volvo C30 is a near-luxury compact hatchback, manufactured and marketed by Volvo Cars for model years 2006-2013, available across its single generation as a
three-door hatchback.

 Powered by inline-four and straight-five engines, the C30 is variant of the Volvo S40/V50/C70 range, sharing the same Ford C1/Volvo P1 platform.

 Volvo marketed the C30 as a premium hatchback / sports coupe.
</item 3>

<item 4>
Volvo P80 platform
The Volvo P80 platform was a Swedish mid-size unibody automobile platform developed and produced by Volvo Cars.

 It was in use from 1991 to 2005.

 It is designed for different wheelbases in front-wheel drive configurations and was adapted to all wheel drive.

 It debuted with the 1991 Volvo 850 and again in autumn 1996 with the Volvo 850 AWD.

 Although heavily modified by TWR, the same basic chassis was used as the underpinnings for the C70.

 After the model year 2000 most P80 models were replaced by their P2 successors, with the exception of the C70 convertible which remained in production until
2005.

 A total of 1,360,522 cars based on this platform were built.
</item 4>

<item 5>
Volvo V50
"For the sedan variant of the car, see Volvo S40."
</item 5>

<item 6>
Volvo 480
The Volvo 480 is a compact car that was produced in Born, Netherlands, by Volvo from 1986 to 1995.

 It was the first front-wheel drive car made by the automaker.

 The 480 was available in only one body style on a automobile platform related to the Volvo 440/460 five door hatchback and four door sedan models.
</item 6>

<item 7>
Volvo 300 Series
The Volvo 300 Series is a rear wheel drive small family car sold as both a hatchback and (later) a conventional saloon from 1976 to 1991.

 It was launched in the Netherlands shortly after Volvo acquired a major stake in the passenger car division of DAF in 1973.

 The series consisted of the Volvo 340 (previously 343/345) and the later Volvo 360.
</item 7>

<item 8>
Volvo 850
The Volvo 850 is a compact executive car that was produced by the Swedish manufacturer Volvo Cars from 1991 to 1996.

 Designed by Jan Wilsgaard it was available in saloon and estate body styles.

 The 850 was the first front-wheel drive vehicle from Volvo to be sold in North America and also the first all-wheel drive Volvo.

 The range was replaced for 1997 by the Volvo S70 and Volvo V70.
</item 8>

<item 9>
Volvo’s 480 (1986–95), its first front-wheel-drive car, shared components with the Dutch-built 440 hatchback and 460 sedan (1987–96). The rear-wheel-drive 300
Series (1976–91) comprised 340/343/345 and 360 models. The luxury 164 (1968–75) was a six-cylinder sedan; 46,008 were built. Volvo’s compact transverse SI6
straight-six debuted in 200
</item 9>

<item 10>
Volvo S70
The Volvo S70 was a compact executive car produced by Volvo Cars from 1996 to 2000.

 The S70 was essentially a facelifted 850 saloon.

 The S70 was replaced with the Volvo S60.
</item 10>
=== END ITEM id=eacd543247604ec4 ===

=== ITEM id=c4bacde655a5112e ===
Gold sentences and facts:
Sentence 0: The Battle of Stones River or Second Battle of Murfreesboro, was fought from December 31, 1862, to January 2, 1863, in Middle Tennessee, as the culmination of the Stones River Campaign in the Western Theater of the American Civil War.
  Fact 0.0: The Battle of Stones River is also called the Second Battle of Murfreesboro.
  Fact 0.1: The Battle of Stones River was fought from December 31, 1862, to January 2, 1863.
  Fact 0.2: The Battle of Stones River was fought in Middle Tennessee.
  Fact 0.3: The Battle of Stones River was the culmination of the Stones River Campaign.
  Fact 0.4: The Stones River Campaign took place in the Western Theater of the American Civil War.
Sentence 1: The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.
  Fact 1.0: The Battle of the Ch'ongch'on River is also known as the Battle of the Ch'ongch'on.
  Fact 1.1: The Battle of the Ch'ongch'on River is also known as the Second Phase Campaign Western Sector.
  Fact 1.2: The Battle of the Ch'ongch'on River was a decisive battle in the Korean War.
  Fact 1.3: The Battle of the Ch'ongch'on River took place from November 25 to December 2, 1950.
  Fact 1.4: The Battle of the Ch'ongch'on River took place along the Ch'ongch'on River Valley.
  Fact 1.5: The Ch'ongch'on River Valley is in the northwestern part of North Korea.

Retrieved items:
<item 1>
Battle of Malvern Hill
The Battle of Malvern Hill, also known as the Battle of Poindexter's Farm, was fought on July 1, 1862 between the Confederate Army of Northern Virginia, led by
Gen. Robert E. Lee, and the Union Army of the Potomac under Maj. Gen. George B. McClellan.

 It was the final battle of the Seven Days Battles during the American Civil War, taking place on a 130 ft elevation of land known as Malvern Hill, near the
Confederate capital of Richmond, Virginia and just one mile (1.6 km) from the James River.

 Including inactive reserves, more than fifty thousand soldiers from each side took part, using more than two hundred pieces of artillery and three warships.
</item 1>

<item 2>
Battle of Blue Licks
The Battle of Blue Licks, fought on August 19, 1782, was one of the last battles of the American Revolutionary War.

 The battle occurred ten months after Lord Cornwallis's famous surrender at Yorktown, which had effectively ended the war in the east.

 On a hill next to the Licking River in what is now Robertson County, Kentucky (but was then in Kentucky County, Virginia), a force of about 50 American and
Canadian Loyalists along with 300 American Indians ambushed and routed 182 Kentucky militiamen.

 It was the last victory for the Loyalists and Natives during the frontier war.
</item 2>

<item 3>
Second Battle of Fort Wagner
The Second Battle of Fort Wagner, also known as the Second Assault on Morris Island or the Battle of Fort Wagner, Morris Island, was fought on July 18, 1863,
during the American Civil War.

 Union Army troops commanded by Brig. Gen. Quincy Gillmore, launched an unsuccessful assault on the Confederate fortress of Fort Wagner, which protected Morris
Island, south of Charleston Harbor.

 The battle came one week after the First Battle of Fort Wagner.
</item 3>

<item 4>
Battle of Cross Keys
The Battle of Cross Keys was fought on June 8, 1862, in Rockingham County, Virginia, as part of Confederate Army Maj. Gen. Thomas J. "Stonewall" Jackson's
campaign through the Shenandoah Valley during the American Civil War.

 Together, the battles of Cross Keys and Port Republic the following day were the decisive victories in Jackson's Valley Campaign, forcing the Union armies to
retreat and leaving Jackson free to reinforce Gen. Robert E. Lee for the Seven Days Battles outside Richmond, Virginia.
</item 4>

<item 5>
Korean War
The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR:
"Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet
Union) and South Korea (with the principal support of the United States).

 The war began on 25 June 1950 when North Korea invaded South Korea.

 The United Nations, with the United States as the principal force, came to the aid of South Korea.

 China came to the aid of North Korea, and the Soviet Union also gave some assistance to the North.
</item 5>

<item 6>
Battle of Gaines's Mill
The Battle of Gaines's Mill, sometimes known as the First Battle of Cold Harbor or the Battle of Chickahominy River, took place on June 27, 1862, in Hanover
County, Virginia, as the third of the Seven Days Battles (Peninsula Campaign) of the American Civil War.

 Following the inconclusive Battle of Beaver Dam Creek (Mechanicsville) the previous day, Confederate General Robert E. Lee renewed his attacks against the
right flank of the Union Army, relatively isolated on the northern side of the Chickahominy River.

 There, Brig. Gen. Fitz John Porter's V Corps had established a strong defensive line behind Boatswain's Swamp.

 Lee's force was destined to launch the largest Confederate attack of the war, about 57,000 men in six divisions.

 Porter's reinforced V Corps held fast for the afternoon as the Confederates attacked in a disjointed manner, first with the division of Maj. Gen. A.P. Hill,
then Maj. Gen. Richard S. Ewell, suffering heavy casualties.

 The arrival of Maj. Gen. Stonewall Jackson's command was delayed, preventing the full concentration of Confederate force before Porter received some
reinforcements from the VI Corps.
</item 6>

<item 7>
Battle of Stones River
The Battle of Stones River or Second Battle of Murfreesboro, was fought from December 31, 1862, to January 2, 1863, in Middle Tennessee, as the culmination of
the Stones River Campaign in the Western Theater of the American Civil War.

 Of the major battles of the Civil War, Stones River had the highest percentage of casualties on both sides.

 Although the battle itself was inconclusive, the Union Army's repulse of two Confederate attacks and the subsequent Confederate withdrawal were a much-needed
boost to Union morale after the defeat at the Battle of Fredericksburg, and it dashed Confederate aspirations for control of Middle Tennessee.
</item 7>

<item 8>
Battle of the Ch'ongch'on River
The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the
Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.

 In response to the successful Chinese First Phase Campaign against the United Nations forces, General Douglas MacArthur launched the Home-by-Christmas
Offensive to evict the Chinese forces from Korea and to end the war.

 Anticipating this reaction, the Chinese People's Volunteer Army Commander Peng Dehuai planned a counteroffensive, dubbed the "Second Phase Campaign", against
the advancing UN forces.
</item 8>

<item 9>
Battle of Piqua
The Battle of Piqua, also known as the Battle of Pekowee or Pekowi, was part of the western campaign during the American Revolutionary War.

 Led by General George Rogers Clark, over 1,000 soldiers (among them Daniel Boone and Simon Kenton) crossed the Ohio River near present-day Cincinnati and
burned five Shawnee villages, including Old Chillicothe, along the Little Miami River.

 Peter Loramie's Store, a British trading post-located in what was later Fort Loramie, Ohio in Shelby County, Ohio-, was also burned by Clark's men.

 The Shawnee gradually withdrew during the first few days before finally engaging American forces 7 miles west of Springfield, Ohio on August 8, 1780.

 Joseph Rogers, a cousin of George Rogers Clark, had previously accompanied him to Kentucky and was later captured by the Shawnee near Maysville.

 Despite having been adopted by the tribe, he was killed during the battle while trying to join American forces.
</item 9>

<item 10>
Battle of Port Republic
The Battle of Port Republic was fought on June 9, 1862, in Rockingham County, Virginia, as part of Confederate Army Maj. Gen. Thomas J. "Stonewall" Jackson's
campaign through the Shenandoah Valley during the American Civil War.

 Port Republic was a fierce contest between two equally determined foes and was the most costly battle fought by Jackson's Army of the Valley during its
campaign.

 Together, the battles of Cross Keys (fought the previous day) and Port Republic were the decisive victories in Jackson's Valley Campaign, forcing the Union
armies to retreat and leaving Jackson free to reinforce Gen. Robert E. Lee for the Seven Days Battles outside Richmond, Virginia.
</item 10>
=== END ITEM id=c4bacde655a5112e ===

=== ITEM id=b0476ec4ad09d194 ===
Gold sentences and facts:
Sentence 0: Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.
  Fact 0.0: Lowell J. Sherman was born on October 11, 1885.
  Fact 0.1: Lowell J. Sherman died on December 28, 1934.
  Fact 0.2: Lowell J. Sherman was an American actor.
  Fact 0.3: Lowell J. Sherman was an American film director.
Sentence 1: Jonathan Kaplan (born November 25, 1947) is an American film producer and director.
  Fact 1.0: Jonathan Kaplan was born on November 25, 1947.
  Fact 1.1: Jonathan Kaplan is an American film producer.
  Fact 1.2: Jonathan Kaplan is an American film director.

Retrieved items:
<item 1>
Jonathan Kaplan
Jonathan Kaplan (born November 25, 1947) is an American film producer and director.

 His film "The Accused" (1988) earned actress Jodie Foster her first Oscar for Best Actress and was nominated for the Golden Bear at the 39th Berlin
International Film Festival.

 Kaplan received five Emmy nominations for his roles directing and producing "ER".
</item 1>

<item 2>
He Knew Women
He Knew Women is a 1930 American comedy film, directed by Hugh Herbert, from a screenplay by him and William B. Jutte, which was adapted from S. N. Behrman's
1927 play "The Second Man".

 It starred Lowell Sherman and Alice Joyce, in her second to last film role.

 The film just broke even.
</item 2>

<item 3>
Tommy Atkins (director)
Tommy Atkins was an American director of the silent and early sound film eras.

 Born on July 18, 1887 in Springfield, Massachusetts, he would make his entrance into the film industry as the assistant director to Ralph Ince on the 1920
silent film, "Out of the Snows".

 It would be another eight years before he would make another film, again as assistant director, this time for FBO Pictures, on another silent film, "Crooks
Can't Win".

 He'd work as the assistant director on another sixteen films between 1928 and 1934, the most notable of which would be 1933's "Morning Glory", directed by
Lowell Sherman and starring Katharine Hepburn and Douglas Fairbanks Jr..

 In 1934 he would be given the chance to helm his first picture, "The Silver Streak", which was one of the top money-makers for RKO Pictures that year.

 He would only direct two more films, the second of which, "Hi, Gaucho!

", he would also write the story for.
</item 3>

<item 4>
Born to Be Bad (1934 film)
Born to Be Bad is a 1934 American Pre-Code drama film directed by Lowell Sherman, and starring Loretta Young and Cary Grant.
</item 4>

<item 5>
High Stakes (1931 film)
High Stakes is a 1931 American Pre-Code comedy drama produced and released by RKO Pictures.

 The picture was directed by Lowell Sherman who also stars and marks the last starring screen appearance of silent screen diva Mae Murray.

 It is based on a 1924 Broadway play that starred Sherman playing the same role he plays in this film.
</item 5>

<item 6>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 6>

<item 7>
Lawful Larceny
Lawful Larceny is a 1930 American melodramatic film, directed by Lowell Sherman from Jane Murfin's screenplay.

 The screenplay, a melodrama, was based on the play of the same name by Samuel Shipman, which originally was a comedy.

 It starred a staple of the early RKO stable, Bebe Daniels, along with Kenneth Thomson, Olive Tell and Lowell Sherman, who reprised the role he had created in
the original Broadway play.

 This film was a remake of the 1923 silent film version of the same name, produced by Famous Players-Lasky Corporation
</item 7>

<item 8>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 8>

<item 9>
Convoy (1927 film)
Convoy is a lost 1927 silent World War I drama starring Lowell Sherman and Dorothy Mackaill and released through First National Pictures.

 The film is an early producing credit for the Halperin Brothers, Victor and Edward, later of "White Zombie" fame, and is the final screen appearance of
Broadway stars Gail Kane and Vincent Serrano.
</item 9>

<item 10>
Lowell Sherman
Lowell J. Sherman (October 11, 1885 – December 28, 1934) was an American actor and film director.

 At a time when it was highly unusual, he was both the actor and director on several films in the early 1930s, before completely transitioning to the role of
director.

 At the height of his career, after scoring huge successes with his direction of the films "She Done Him Wrong" and "Morning Glory" (which starred Mae West and
won her first Academy Award for Katharine Hepburn respectively), he succumbed to pneumonia after a brief illness.
</item 10>
=== END ITEM id=b0476ec4ad09d194 ===

=== ITEM id=042820ac3fefcefa ===
Gold sentences and facts:
Sentence 0:  She was also a member of a secret society composed exclusively of Cheyenne women.
  Fact 0.0: The woman in question was also a member of a secret society.
  Fact 0.1: The secret society was composed exclusively of Cheyenne women.
Sentence 1: Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the Council of Forty-four.
  Fact 1.0: Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance.
  Fact 1.1: The other central institution of traditional Cheyenne Indian tribal governance is the Council of Forty-four.

Retrieved items:
<item 1>
Concho Indian Boarding School
Concho Indian Boarding School (also known as the Cheyenne-Arapaho Boarding School at Concho or Concho Indian School and home to the Concho Demonstration School)
was a boarding school for members of the Cheyenne and Arapaho Tribes and later opened to other Native American students.

 It existed from 1909 to 1983.

 It was located in central Oklahoma, approximately 1 mile south of Concho, Oklahoma and 4 miles north of El Reno, Oklahoma.

 The name of the town and school is the Spanish word for "shell" and was named for the Indian agent, Charles E. Shell.
</item 1>

<item 2>
Lame Deer, Montana
Lame Deer (Meaveʼhoʼeno in Cheyenne) is a census-designated place (CDP) in Rosebud County, Montana.

 The community is named after Miniconjou Lakota chief Lame Deer, who was killed by the U.S. Army in 1877 under a flag of truce south of the town.

 It is the tribal and government agency headquarters of the Northern Cheyenne Indian Reservation.

 This is the location of the Chief Dull Knife College and the annual Northern Cheyenne Powwow.
</item 2>

<item 3>
Ehyophsta
Ehyophsta (Cheyenne for "Yellow-Haired Woman") was a Cheyenne woman.

 She was the daughter of Stands-in-the-Timber who died in 1849, and she was the niece of Bad Faced Bull.

 She fought in the Battle of Beecher Island in 1868, and also fought the Shoshone that same year, where she counted coup against one enemy and killed another.

 She fought the Shoshone again in 1869.

 She was also a member of a secret society composed exclusively of Cheyenne women.

 She died in 1915.
</item 3>

<item 4>
Northern Cheyenne Indian Reservation
The Northern Cheyenne Indian Reservation (Tsėhéstáno in Cheyenne, formerly named the Tongue River Indian Reservation) is home of the federally recognized
Northern Cheyenne Tribe.

 Located in southeastern Montana, the reservation is approximately 444,000 acres in size and home to approximately 5,000 Cheyenne people.

 The tribal and government headquarters are in Lame Deer, which is also the home of the annual Northern Cheyenne Pow wow.

 The reservation is bounded on the east by the Tongue River and on the west by the Crow Reservation.

 There are small parcels of non-contiguous off-reservation trust lands in Meade County, South Dakota, northeast of the city of Sturgis.

 Its timbered ridges that extend into northwestern South Dakota are part of Custer National Forest and it is approximately 40 mi east of the site of the 1876
Battle of the Greasy Grass (known to most Americans as the Battle of the Little Bighorn).
</item 4>

<item 5>
Chief Dull Knife College
Chief Dull Knife College (originally Dull Knife Memorial College) is a small, open-admission, Native American tribal community college and land grant
institution.

 Located on the Northern Cheyenne Indian Reservation, in the city of Lame Deer, it has a current enrollment of about 141 students.

 On average, more than half of its graduates move on to four-year colleges.
</item 5>

<item 6>
Cheyenne military societies
Cheyenne military societies are one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the Council of Forty-four.

 While council chiefs are responsible for overall governance of individual bands and the tribe as a whole, the headmen of military societies are in charge of
maintaining discipline within the tribe, overseeing tribal hunts and ceremonies, and providing military leadership.

 Historically, council chiefs selected which of the six military societies would assume these duties; after a period of time on-duty, the chiefs would select a
different society to take up the duties.
</item 6>

<item 7>
Lean Bear
Lean Bear (Cheyenne name Awoninahku, 1813–1864), alternatively translated as Starving Bear, was a Cheyenne peace chief.

 He was a member of the Council of Forty-four, a tribal governance devoted to maintaining peace with encroaching United States settlers.

 Lean Bear's most notable peace deals include the Treaty of Fort Wise and a meeting with US President Abraham Lincoln.

 His work towards peace between his people and the American settlers in the Southern Plains were cut short when he was murdered by the 1st Colorado Cavalry
Regiment and violent retaliations ensued.
</item 7>

<item 8>
Cheyenne
The Cheyenne ( ) are one of the Indigenous peoples of the Great Plains and their language is of the Algonquian language family.

 The Cheyenne comprise two Native American tribes, the Só'taeo'o or Só'taétaneo'o (more commonly spelled as Suhtai or Sutaio) and the Tsétsêhéstâhese (also
spelled Tsitsistas).

 These tribes merged in the early 19th century.

 Today, the Cheyenne people are split into two federally recognized Nations: the Southern Cheyenne, who are enrolled in the Cheyenne and Arapaho Tribes in
Oklahoma, and the Northern Cheyenne, who are enrolled in the Northern Cheyenne Tribe of the Northern Cheyenne Indian Reservation in Montana.
</item 8>

<item 9>
Eugene Little Coyote
Eugene Little Coyote was the president of the Northern Cheyenne Indian Reservation from 2004 to 2007.

 He was elected in November 2004, defeating the incumbent president, Geri Small.

 However, after a conflict with reservation vice president Rick Wolfname that began in July 2007 escalated, the Northern Cheyenne tribal council declared that
it would oust Little Coyote as the tribe's president.

 Little Coyote was forcibly removed from office and arrested for trespassing in the tribal office building on December 28, 2007.

 It is worth noting that the move by the tribal council to remove Little Coyote from office was deemed unconstitutional and void by the Northern Cheyenne
Constitutional Court.

 The Constitutional Court was however overruled by Bureau of Indian Affairs' regional director Ed Parisian, and later the Assistant Secretary of Indian Affairs
Carl J. Artman.

 Their involvement in this matter set a historical precedence of the BIA directly meddling with tribal governments.
</item 9>

<item 10>
Council of Forty-four
The Council of Forty-four was one of the two central institutions of traditional Cheyenne Indian tribal governance, the other being the military societies such
as the Dog Soldiers.

 The influence of the Council of Forty-four waned in the face of internal conflict among the Cheyenne about Cheyenne policy toward encroaching white settlers on
the Great Plains, and was dealt a severe blow by the Sand Creek Massacre.
</item 10>
=== END ITEM id=042820ac3fefcefa ===

=== ITEM id=561eb68e63895087 ===
Gold sentences and facts:
Sentence 0: Samsung Galaxy S7 and Samsung Galaxy S7 Edge are Android smartphones manufactured and marketed by Samsung Electronics.
  Fact 0.0: Samsung Galaxy S7 is an Android smartphone.
  Fact 0.1: Samsung Galaxy S7 Edge is an Android smartphone.
  Fact 0.2: Samsung Galaxy S7 and S7 Edge are manufactured by Samsung Electronics.
  Fact 0.3: Samsung Galaxy S7 and S7 Edge are marketed by Samsung Electronics.
Sentence 1:  The S7 series serves as the successor to the Galaxy S6, S6 Edge and S6 Edge+ released in 2015.
  Fact 1.0: The S7 series is the successor to the Galaxy S6.
  Fact 1.1: The S7 series is the successor to the Galaxy S6 Edge.
  Fact 1.2: The S7 series is the successor to the Galaxy S6 Edge+.
  Fact 1.3: The Galaxy S6, S6 Edge and S6 Edge+ were released in 2015.
Sentence 2: Samsung Galaxy S6, Samsung Galaxy S6 Edge, and Samsung Galaxy S6 Edge+ are Android smartphones manufactured and marketed by Samsung Electronics.
  Fact 2.0: Samsung Galaxy S6 is an Android smartphone.
  Fact 2.1: Samsung Galaxy S6 Edge is an Android smartphone.
  Fact 2.2: Samsung Galaxy S6 Edge+ is an Android smartphone.
  Fact 2.3: Samsung Galaxy S6, S6 Edge and S6 Edge+ are manufactured by Samsung Electronics.
  Fact 2.4: Samsung Galaxy S6, S6 Edge and S6 Edge+ are marketed by Samsung Electronics.
Sentence 3:  The S6 line serves as a successor to the Galaxy S5.
  Fact 3.0: The S6 line is a successor to the Galaxy S5.
Sentence 4:  The S6 and S6 Edge smartphones were officially unveiled in the first "Samsung Unpacked 2015" event at Mobile World Congress on 1 March 2015, while the bigger S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5 in the second "Samsung Unpacked 2015" event at New York on 13 August 2015.
  Fact 4.0: The S6 was officially unveiled in the first Samsung Unpacked 2015 event.
  Fact 4.1: The S6 Edge was officially unveiled in the first Samsung Unpacked 2015 event.
  Fact 4.2: The first Samsung Unpacked 2015 event was held at Mobile World Congress on 1 March 2015.
  Fact 4.3: The S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5.
  Fact 4.4: The S6 Edge+ was officially unveiled in the second Samsung Unpacked 2015 event.
  Fact 4.5: The second Samsung Unpacked 2015 event was held in New York on 13 August 2015.

Retrieved items:
<item 1>
Samsung SGH-D500
The SGH-D500 has been succeeded by the Samsung SGH-D600.

 There is a very similar variant handset, the SGH-D500E.
</item 1>

<item 2>
Samsung SGH-E900
The Samsung E900, introduced in 2006, is a high-end mobile phone and is derived from Samsung's D500-D600-D800 series of slide phones.

 It is of a slide-up design and has touch-sensitive keys, similar to the LG Chocolate (KG800).

 It was one of the most popular phones at the time but was discontinued late 2007 when buyers then turned their attention to the Samsung D900i, followed by the
Samsung U600, which was the slimmest phone by Samsung with a thickness of 1.09 cm until the E840 was released.
</item 2>

<item 3>
Samsung Galaxy S6
Samsung Galaxy S6, Samsung Galaxy S6 Edge, and Samsung Galaxy S6 Edge+ are Android smartphones manufactured and marketed by Samsung Electronics.

 The S6 line serves as a successor to the Galaxy S5.

 The S6 and S6 Edge smartphones were officially unveiled in the first "Samsung Unpacked 2015" event at Mobile World Congress on 1 March 2015, while the bigger
S6 Edge+ was officially unveiled together with the Samsung Galaxy Note 5 in the second "Samsung Unpacked 2015" event at New York on 13 August 2015.

 Alongside the S6, Samsung also unveiled the S6 Edge (and later on the bigger S6 Edge+), a variant whose screen is wrapped along the sides of the device; the
curvature is usable for several additional features.

 The Galaxy S6 and S6 Edge were released on 10 April 2015 in 20 countries while the S6 Edge+ was released on 21 August 2015 in 20 countries.
</item 3>

<item 4>
Samsung Galaxy Gear
The Samsung Galaxy Gear is a smartwatch produced by Samsung Electronics in the Samsung Gear family of devices.

 Unveiled during a "Samsung Unpacked" event in Berlin, Germany on September 4, 2013, the device serves as a companion for all Samsung Galaxy smartphones and
tablets which run Android 4.3 or newer; it was released on September 25, 2013.

 Originally released as an Android-based device, Samsung replaced the operating system with Tizen through the May 2014 software update.
</item 4>

<item 5>
Samsung SGH-T919
The Samsung Behold SGH-T919 is a touch-screen, 3G candybar-style mobile phone introduced by Samsung late in 2008.

 The Behold is one of the first Samsung mobile phones released to have a touch-screen along with the Samsung Omnia, the Samsung Instinct, the Samsung Eternity,
and the Samsung Tocco.

 The Behold is the "American" release of Samsung Tocco, which is only sold in Europe, Asia and Australia so the features and design of both of these phones are
similar.

 Some of the Behold's biggest competitors were Apple's iPhone, T-Mobile G1 (HTC Dream), LG Dare, Samsung Instinct, and Samsung Eternity.

 It was only available only to customers of T-Mobile.
</item 5>

<item 6>
Samsung Galaxy Tab 8.9
The Samsung Galaxy Tab 8.9 is an Android-based tablet computer designed and manufactured by Samsung, introduced on 22 March 2011 at CTIA wireless convention in
its Samsung Unpacked event in Orlando.

 It is part of the Samsung Galaxy Tab series, and features an 8.9-inch display and a 1 GHz dual-core Nvidia Tegra 2 processor.
</item 6>

<item 7>
Samsung Galaxy S Duos 2
Samsung Galaxy S Duos 2 is a version of Samsung Galaxy Trend Plus with dual SIM Android smartphone, produced and marketed by Samsung Electronics, which serves
as an immediate successor to the original Galaxy S Duos, which is made in second quarter of 2012.

 Unveiled on 30 November 2013 at some Asian markets, it was initially released on December 5, 2013 in several countries.

 In contrast with other dual sim Samsung models, this phone is a part of the high-end "S" series, this is why it is marketed as a part of the "Galaxy S" family.

 Despite being called as a direct successor to the original S Duos phone, external and physical design of the model is identical to the original model, placing
emphasis on internal upgrades such as upgraded processor and updated Operating System software.
</item 7>

<item 8>
Samsung Galaxy S7
Samsung Galaxy S7 and Samsung Galaxy S7 Edge are Android smartphones manufactured and marketed by Samsung Electronics.

 The S7 series serves as the successor to the Galaxy S6, S6 Edge and S6 Edge+ released in 2015.

 The phones were officially unveiled on 21 February 2016, during a Samsung press conference at Mobile World Congress, with a European and North American release
on 11 March 2016.
</item 8>

<item 9>
Samsung REX
The Samsung REX series is a range of Samsung smartphones and feature phones, ranging from low to mid-end phones, running a Java-based operating system with
Samsung's TouchWiz user interface.

 It is currently one of the Samsung line of devices that are aimed at the budget phone market alongside lower end Samsung Galaxy devices.

 Its main competition includes the Nokia Asha range and Firefox OS.
</item 9>

<item 10>
Samsung Ativ S
The Samsung ATIV S is a touchscreen, slate smartphone manufactured by Samsung Electronics running the Windows Phone 8 operating system, upgradeable to Windows
Phone 8.1.

 The ATIV S is Samsung's first Windows Phone 8 device, and one of the first devices under its ATIV series of Microsoft Windows-based products.

 It was shown at the IFA 2012 in Berlin and was the first Windows Phone 8 device to be officially unveiled, ahead of the Nokia Lumia 920.
</item 10>
=== END ITEM id=561eb68e63895087 ===

=== ITEM id=5106141afeb86196 ===
Gold sentences and facts:
Sentence 0: Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.
  Fact 0.0: Jelena Janković was born on 28 February 1985.
  Fact 0.1: Jelena Janković's name is written in Serbian Cyrillic as Јелена Јанковић.
  Fact 0.2: Jelena Janković is a Serbian professional tennis player.
Sentence 1: Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian politician and retired professional tennis player.
  Fact 1.0: Marat Mubinovich Safin's name is written in Russian as Мара́т Муби́нович Са́фин.
  Fact 1.1: Marat Mubinovich Safin's name is written in Tatar as Марат Мөбин улы Сафин.
  Fact 1.2: Marat Mubinovich Safin's Tatar name is romanized as Marat Möbin uğlı Safin.
  Fact 1.3: Marat Mubinovich Safin was born on 27 January 1980.
  Fact 1.4: Marat Mubinovich Safin is a Russian politician.
  Fact 1.5: Marat Mubinovich Safin is a retired professional tennis player.

Retrieved items:
<item 1>
Dinara Safina career statistics
This is a list of the main career statistics of retired, Russian professional tennis player, Dinara Safina.

 Throughout her career, Safina won twelve WTA singles titles including three Tier I singles titles at the 2008 Qatar Telecom German Open, Rogers Cup and Toray
Pan Pacific Open respectively; one Premier Mandatory singles title at the 2009 Mutua Madrileña Madrid Open and one Premier 5 singles title at the 2009
Internazionali BNL d'Italia.

 She was also the runner-up at the 2008 French Open and the 2009 Australian Open and French Open as well as a silver medalist in singles at the 2008 Beijing
Olympics.
</item 1>

<item 2>
Jelena Janković
Jelena Janković (Serbian Cyrillic: Јелена Јанковић, ] , born 28 February 1985) is a Serbian professional tennis player.

 Janković is a former World No. 1 in singles, a ranking achieved preceding her finals appearance at the 2008 US Open.

 Janković's career highlights include winning the 2007 Wimbledon Mixed Doubles title with Jamie Murray, the 2010 Indian Wells Masters, the Internazionali BNL
d'Italia twice, in 2007 and 2008, and the aforementioned appearance in the finals of the 2008 US Open.

 She is coached by her brother Marko.
</item 2>

<item 3>
Ana Ivanovic career statistics
This is a list of the main career statistics of Serbian professional tennis player, Ana Ivanovic.

 Ivanovic won fifteen WTA singles titles including one grand slam singles title at the 2008 French Open and three WTA Tier I singles titles.

 She was also the runner-up at the 2007 French Open and 2008 Australian Open and a semi-finalist at the 2007 Wimbledon Championships and 2007 WTA Tour
Championships.

 On June 9, 2008 Ivanovic became the world No. 1 for the first time in her career.
</item 3>

<item 4>
Ivanovic–Janković rivalry
Serbian tennis players Ana Ivanovic and Jelena Janković are both former world No. 1 and have met 12 times so far.

 Their overall head-to-head series is 9–3, in Ivanovic's favor.

 Janković is two years older than Ivanovic, so their paths connect them to play each other.

 They are the first female players to achieve major successes for independent Serbia.
</item 4>

<item 5>
2008 Wimbledon Championships – Mixed Doubles
Jamie Murray and Jelena Janković were the defending champions, but Janković chose not to participate, and only Murray competed that year.
</item 5>

<item 6>
Jelena Janković career statistics
This is a list of the main career statistics of Serbian professional tennis player and former World No. 1, Jelena Janković.

 To date, Janković has won 15 WTA singles titles including four WTA Tier I singles titles, one WTA Premier Mandatory singles title and one WTA Premier 5 singles
title.

 Janković was also the runner-up in singles at the 2008 US Open, a semi-finalist at the year-ending WTA Tour Championships in 2008, 2009 and 2013 and a
quarterfinalist at the 2008 Beijing Olympics.
</item 6>

<item 7>
2014 Rogers Cup – Women's Doubles
Jelena Janković and Katarina Srebotnik were the defending champions, but they chose not to compete together.

 Janković played alongside Klára Koukalová, but lost in the second round to Michaëlla Krajicek and Barbora Záhlavová-Strýcová.

 Srebotnik teamed up with Květa Peschke, but lost in the second round to Chan Hao-ching and Chan Yung-jan.

<br>
</item 7>

<item 8>
At the 1976 Italian Open, Adriano Panatta and Mima Jaušovec won singles; Panatta survived 11 match points in round one. Jan Kodeš and Margaret Court won the
1970 French Open. Vitas Gerulaitis won Wimbledon doubles (1975), the 1977 Australian Open, Italian Opens (1977, 1979), and 1978 WCT Finals; he also won Brussels
(1982), Ocean City (1977
</item 8>

<item 9>
Marat Safin
Marat Mubinovich Safin (Russian: Мара́т Муби́нович Са́фин ; ] , Tatar: Марат Мөбин улы Сафин "Marat Möbin uğlı Safin" ; born 27 January 1980) is a Russian
politician and retired professional tennis player.

 He achieved the Association of Tennis Professionals (ATP) world No. 1 singles ranking on November 20, 2000.

 Nobody taller than Safin has ever reached his world number-one ranking.

 Safin is the older brother of former world No. 1 WTA player, Dinara Safina.

 They are the first and only brother–sister tandem in tennis history who have both achieved No. 1 rankings.

 On court, Safin was famous for his emotional outbursts and sometimes fiery temper.
</item 9>

<item 10>
Kavita Tungar holds India’s 10-km and half-marathon records and won 2010 Commonwealth bronze and Asian Games silver in the 10,000m. Tennis profiles cover Kim
Warwick’s Australian Open final and doubles titles; Evonne Goolagong’s seven Grand Slam singles titles and former No. 1 ranking; and achievements by Ivanovic,
Janković, Stosur, Safina, Azarenka, Watson and Hlaváčková
</item 10>
=== END ITEM id=5106141afeb86196 ===

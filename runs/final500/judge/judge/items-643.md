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

=== ITEM id=edf048a04fe0fa61 ===
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
Charles J. Biddle (aviator)
Major Charles John Biddle (13 May 1890 – 22 March 1972) was an American aviator, attorney, and author.

 He was a flying ace during World War I.

 Postwar, he launched a career in law and wrote his memoirs.
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
Cecil Frederick King
Captain Cecil Frederick King (19 February 1899 – 24 January 1919), Military Cross, Distinguished Flying Cross, French Croix de Guerre was a World War I fighter
ace credited with 22 aerial victories.
</item 7>

<item 8>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 8>

<item 9>
Witmer Stone
Witmer Stone (September 22, 1866 – May 24, 1939) was an American ornithologist, botanist, and mammalogist, and was considered one of the last of the “great
naturalists.”

 Stone is remembered principally as an ornithologist.

 He was president of the American Ornithologists’ Union (AOU) 1920–23, and was editor of the AOU’s periodical "The Auk" 1912–1936.

 He spearheaded the production of the 4th edition of the AOU checklist, published in 1931.

 He worked for over 50 years in the Ornithology Department at the Academy of Natural Sciences of Philadelphia, eventually serving as Director of the
institution.

 Stone was one of the founding members of the Delaware Valley Ornithological Club (DVOC) in 1890 and was actively involved in the organization for the remainder
of his life.

 Stone was one of only two scientists (Joseph Grinnell was the other) to serve as president of both the AOU and the American Society of Mammalogists, and he
co-authored two popular books about mammals.
</item 9>

<item 10>
Bird strike
A bird strike—sometimes called birdstrike, bird ingestion (for an engine), bird hit, or bird aircraft strike hazard (BASH)—is a collision between an airborne
animal (usually a bird or bat) and a manmade vehicle, especially an aircraft.

 The term is also used for bird deaths resulting from collisions with structures such as power lines, towers and wind turbines (see Bird-skyscraper collisions
and Towerkill).
</item 10>
=== END ITEM id=edf048a04fe0fa61 ===

=== ITEM id=d668654190f345d2 ===
Gold sentences and facts:
Sentence 0: Yucca is a genus of perennial shrubs and trees in the family Asparagaceae, subfamily Agavoideae.
  Fact 0.0: Yucca is a genus of perennial shrubs and trees.
  Fact 0.1: Yucca is in the family Asparagaceae.
  Fact 0.2: Yucca is in the subfamily Agavoideae.
Sentence 1: Rodgersia is a genus of flowering plants in the Saxifragaceae family.
  Fact 1.0: Rodgersia is a genus of flowering plants.
  Fact 1.1: Rodgersia is in the Saxifragaceae family.
Sentence 2:  "Rodgersia" are herbaceous perennials originating from east Asia.
  Fact 2.0: Rodgersia are herbaceous perennials.
  Fact 2.1: Rodgersia originate from east Asia.

Retrieved items:
<item 1>
Yucca
Yucca is a genus of perennial shrubs and trees in the family Asparagaceae, subfamily Agavoideae.

 Its 40-50 species are notable for their rosettes of evergreen, tough, sword-shaped leaves and large terminal panicles of white or whitish flowers.

 They are native to the hot and dry (arid) parts of the Americas and the Caribbean.

 Early reports of the species were confused with the cassava ("Manihot esculenta").

 Consequently, Linnaeus mistakenly derived the generic name from the Taíno word for the latter, "yuca" (spelled with a single "c").

 It is commonly found growing in rural graveyards and when in bloom the cluster of (usually pale) flowers on a thin stalk appear as floating apparitions.
</item 1>

<item 2>
Yucca gloriosa var. tristis
Yucca gloriosa" var.

 "tristis (syn.

 Yucca recurvifolia, "Yucca gloriosa" var.

 "recurvifolia"), known as curve-leaf yucca, curved-leaved Spanish-dagger or pendulous yucca, is a variety of "Yucca gloriosa".

 It is often grown as an ornamental plant, but is native to the southeastern United States.

 In contrast to "Y. gloriosa" var.

 "tristis", the leaves of "Y. gloriosa" var.

 "gloriosa" are hard stiff, erect and narrower.
</item 2>

<item 3>
Hesperoyucca whipplei
Hesperoyucca whipplei (syn.

 "Yucca whipplei" ) (chaparral yucca, our Lord's candle, Spanish bayonet, Quixote yucca or foothill yucca is a species of flowering plant closely related to,
and formerly usually included in, the genus "Yucca".

 It is native to southern California, United States and Baja California, Mexico, where it occurs mainly in chaparral, coastal sage scrub, and oak woodland plant
communities at altitudes of 0–2500 m.
</item 3>

<item 4>
Yucca glauca
Yucca glauca (syn.

 "Yucca angustifolia") is a species of perennial evergreen plant, adapted to xeric (dry)growth conditions.

 It is also known as small soapweed, soapweed yucca, Spanish bayonet, Great Plains yucca and beargrass.
</item 4>

<item 5>
Yucca × schottii
Yucca" × "schottii is a plant species in the genus "Yucca", native to southern Arizona, southwestern New Mexico, and the northern parts of Sonora and Chihuahua.

.

 The common names are Schott's yucca, hoary yucca, and mountain yucca.

 The "×" in the name indicates that this is a nothospecies, regarded as being a natural hybrid between two other species.

 In this case, "Yucca" × "schottii" is believed to have originated as a hybrid between "Y. baccata" and "Y. madrensis".

 "Yucca" × "schottii" is firmly established and does reproduce freely in the wild.
</item 5>

<item 6>
Yucca rupicola
Yucca rupicola is a plant in the family Asparagaceae, known as the twistleaf yucca, twisted-leaf yucca, Texas yucca or twisted-leaf Spanish-dagger.

 The species was described by George Heinrich Adolf Scheele in 1850.

 This is a small, acaulescent plant with distinctive twisted leaves.

 It is native to the Edwards Plateau region of Texas and also to northeastern Mexico (Coahuila, Nuevo León).
</item 6>

<item 7>
Yucca aloifolia
Yucca aloifolia is the type species for the genus Yucca.

 Common names include aloe yucca, dagger plant, and Spanish bayonet.

 It grows in sandy soils, especially on sand dunes along the coast.

 "Yucca aloifolia" is native to the Atlantic and Gulf Coasts of the United States from southern Virginia south to Florida and west to the Texas Gulf Coast, to
Mexico along the Yucatán coast, and to Bermuda, and parts of the Caribbean.

 Normally Yucca aloifolia is grown in USDA zones 8 through 11.

 Yucca aloifolia is a popular landscape plant in beach areas along the lower East Coast from Virginia to Florida.
</item 7>

<item 8>
Yucca jaliscensis
Yucca jaliscensis is a "Yucca" species native to the highlands of southwestern Mexico.

 Common names for this species include Jalisco Yucca, Jalisco Soapwort, Izote Yucca.

 It is native to mountainous areas at about 5000 feet in the States of Jalisco, Colima and Guanajuato.
</item 8>

<item 9>
Yucca brevifolia
Yucca brevifolia is a plant species belonging to the genus "Yucca".

 It is tree-like in habit, which is reflected in its common names: Joshua tree, yucca palm, tree yucca, and palm tree yucca.
</item 9>

<item 10>
Rodgersia
Rodgersia is a genus of flowering plants in the Saxifragaceae family.

 "Rodgersia" are herbaceous perennials originating from east Asia.
</item 10>
=== END ITEM id=d668654190f345d2 ===

=== ITEM id=f41536f81ff5d7e2 ===
Gold sentences and facts:
Sentence 0: Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock, and jazz.
  Fact 0.0: Stephen Lawrence 'Steve' Winwood was born 12 May 1948.
  Fact 0.1: Steve Winwood is an English musician.
  Fact 0.2: Steve Winwood's genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock, and jazz.
Sentence 1: Per "Pelle" Almqvist (born 29 May 1978), also known as Howlin' Pelle Almqvist, is a Swedish singer and songwriter.
  Fact 1.0: Per 'Pelle' Almqvist was born 29 May 1978.
  Fact 1.1: Per 'Pelle' Almqvist is also known as Howlin' Pelle Almqvist.
  Fact 1.2: Per 'Pelle' Almqvist is a Swedish singer and songwriter.

Retrieved items:
<item 1>
Pelle Almqvist
Per "Pelle" Almqvist (born 29 May 1978), also known as Howlin' Pelle Almqvist, is a Swedish singer and songwriter.

 He is the lead singer of garage rock band The Hives.
</item 1>

<item 2>
Revolutions – The Very Best of Steve Winwood
Revolutions – The Very Best of Steve Winwood is the sixth compilation album by Steve Winwood.

 The album includes music from Winwood's solo career, as well as groups with which he has performed, including The Spencer Davis Group, Traffic, and Blind
Faith.

 The CD was released as a box set and a single disc.

 As of October 2014, the box set is out of print (but can be purchased digitally), while the single disc is still available.

 The songs "The Finer Things" and "Roll With It" are exclusive to the single disc version and cannot be found on the box set.
</item 2>

<item 3>
Winwood (album)
Winwood is the first compilation album of music featuring Steve Winwood.

 This two-record set was issued in 1971 by United Artists Records and features music which Winwood performed with The Spencer Davis Group, Powerhouse, Traffic
and Blind Faith.

 UA Records issued this album after Winwood's band Traffic left UA when their home label Island Records set up their own American operation.

 Issued without Winwood's authorization as catalogue number UAS-9950, it was taken off the market after legal action by Winwood and Island Records.

 It was then reissued with minor changes as catalogue number UAS-9964.

 Currently out of print, it was issued on CD by Universal Music of Japan for the Japanese market.
</item 3>

<item 4>
Steve Winwood (album)
Steve Winwood is the debut solo studio album by blue-eyed soulster Steve Winwood.

 It was released three years after the break-up of Traffic.

 Though the album sold moderately well in the USA, it was a commercial disappointment compared to Traffic's recent albums, peaking at number 22 on the
"Billboard" 200 album chart.

 In Winwood's home country the situation was reversed; while Traffic's recent albums had only been moderately successful in the UK, "Steve Winwood" reached
number 12 on The Official Charts.

 Island Records launched two singles from the album, "Hold On" and "Time Is Running Out", both of which failed to make the charts.
</item 4>

<item 5>
Go Too
Go Too (1977) is Go's third album.

 Go was founded by Stomu Yamashta, Steve Winwood and Michael Shrieve.

 For this album, Jess Roden replaced Steve Winwood (who had left the group).

 The style of the music became modified accordingly.

 In addition, Linda Lewis was hired as a singer.

 Together with Paul Jackson, Jr. and the orchestra of Martyn Ford the album unified various soundscapes.
</item 5>

<item 6>
Robert Holdstock
Robert Paul Holdstock (2 August 1948 – 29 November 2009) was an English novelist and author best known for his works of Celtic, Nordic, Gothic and Pictish
fantasy literature, predominantly in the fantasy subgenre of mythic fiction.
</item 6>

<item 7>
Back in the High Life Again
"Back in the High Life Again" is a 1986 single written by Steve Winwood and Will Jennings and performed by Winwood.

 The single was included on his album "Back in the High Life" and included backing vocals by James Taylor as well as a prominent mandolin played by Winwood.

 "Back in the High Life Again" was Winwood's second number one on the Adult Contemporary chart.

 The single stayed at number one for three weeks and went to number 13 on the "Billboard" Hot 100.
</item 7>

<item 8>
Peter Wichers
Peter Wichers is best known as one of the founding members of Swedish melodic death metal band, Soilwork.

 From 1995 until late 2005, when he announced his departure from the band, he was known as the lead guitarist and also a songwriter, especially on their
"Stabbing the Drama" album., He rejoined Soilwork in 2008 and played on 2010s The Panic Broadcast and in June 2012 he quit Soilwork for the second time.

 After leaving Soilwork, he filled in for Adam Dutkiewicz when he was injured on Killswitch Engage's European tour.

 More recently, he has collaborated with singers from across the melodic death metal genre including Anders Fridén, former band mate Björn "Speed" Strid and
John Bush (vocalist of Anthrax) to make a CD much in the same style as "Roadrunner United", entitled "Nuclear Blast All-Stars".

 He also co-wrote Warrel Dane's solo record "Praises to the War Machine".
</item 8>

<item 9>
Refugees of the Heart
Refugees of the Heart is the sixth solo studio album by Steve Winwood, released in 1990.

 The album contained the hit single, "One and Only Man", which topped the Mainstream Rock Tracks chart and saw the return of former Traffic bandmate Jim Capaldi
to Winwood's songwriting team.

 By coincidence, a Traffic reunion followed in 1994.

 Because of that collaboration, Winwood would not record another solo album until late 1997.

 Also "I Will Be Here" and "Another Deal Goes Down" were released as singles.

 Winwood stated about the closing track, “In The Light Of Day”: 'When Will and I wrote the song (.

.)

 it was our idea of what Nelson Mandela's dream was, while he was in prison.

 It was really just a fantasy of ours, but that’s what we based the song on.'
</item 9>

<item 10>
Steve Winwood
Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock,
and jazz.

 Though primarily a vocalist and keyboardist, Winwood also plays bass guitar, drums, acoustic and electric guitar, mandolin, violin, and other strings.
</item 10>
=== END ITEM id=f41536f81ff5d7e2 ===

=== ITEM id=a17ab7a383b04687 ===
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
Dudley Zoo
Dudley Zoological Gardens is a 40 acre zoo located within the grounds of Dudley Castle in the town of Dudley, in the Black Country region of the West Midlands,
England.

 The Zoo opened to the public on 18 May 1937.

 Dudley Zoo is owned and operated by Dudley and West Midlands Zoological Society, founded in 1935 and a registered charity.
</item 1>

<item 2>
Mr. Wilson's Cabinet of Wonder
Mr. Wilson's Cabinet of Wonder: Pronged Ants, Horned Humans, Mice on Toast, and Other Marvels of Jurassic Technology is a book by Lawrence Weschler primarily
about the Museum of Jurassic Technology and, more broadly, the history and role of museums.
</item 2>

<item 3>
Rice Northwest Museum of Rocks and Minerals
The Rice Northwest Museum of Rocks and Minerals is a non-profit museum in Hillsboro, Oregon, United States.

 Located just north of the Sunset Highway on the northern edge of Hillsboro, the earth science museum is in the Portland metropolitan area.

 Opened in 1997, the museum’s collections date to the 1930s with the museum housed in a home built to display the rock and mineral collections of the museum
founders.

 The ranch style home is listed on the National Register of Historic Places, the first of its kind listed in Oregon.

 In 2015 the museum became a Smithsonian Affiliate museum.
</item 3>

<item 4>
Graph query row:
city: Los Angeles
connection: ['FROM_CHUNK', 'FROM_CHUNK']
</item 4>

<item 5>
Tudor's Biscuit World
Tudor's Biscuit World is a restaurant chain based in Huntington, West Virginia, most commonly found in West Virginia.

 Many West Virginia locations share a building with Gino's Pizza and Spaghetti, although the chain is more extensive than Gino's (which is exclusive to West
Virginia), having locations in southern Ohio, eastern Kentucky, and southwestern Virginia.

 In 2016 a franchise was opened in Panama City, Florida.

 Tudor's serves biscuits, biscuit sandwiches, homestyle breakfasts and dinners, muffins, and several side dishes.

 The chain was originally based in Charleston, West Virginia and many of the biscuit sandwiches are named for sports teams of interest in that area, including
teams at Marshall University, West Virginia University, and The University of Charleston.
</item 5>

<item 6>
Museum of Jurassic Technology
The Museum of Jurassic Technology is a museum located at 9341 Venice Boulevard in the Palms district of Los Angeles, California.

 It was founded by David Hildebrand Wilson and Diana Drake Wilson (husband and wife) in 1988.
</item 6>

<item 7>
Imperial War Museum
Imperial War Museums (IWM) is a British national museum organisation with branches at five locations in England, three of which are in London.

 Founded as the Imperial War Museum in 1917, the museum was intended to record the civil and military war effort and sacrifice of Britain and its Empire during
the First World War.

 The museum's remit has since expanded to include all conflicts in which British or Commonwealth forces have been involved since 1914.

 As of 2012, the museum aims "to provide for, and to encourage, the study and understanding of the history of modern war and 'wartime experience'".
</item 7>

<item 8>
David Hildebrand Wilson
David Hildebrand Wilson is the co-founder, along with his wife, Diana Wilson, of the enigmatic Museum of Jurassic Technology in Los Angeles, California.

 After high school, Wilson enrolled at Kalamazoo College where he majored in urban entomology with a minor in art.

 He received an MFA in Experimental Animation from the California Institute of the Arts in 1976.

 He was awarded the MacArthur Fellowship and the Creative Capital Moving Image Award in 2001.

 He and his museum are the subject of the book, "Mr. Wilson's Cabinet of Wonder", published in 1995.
</item 8>

<item 9>
Primeval Whirl
Primeval Whirl is a steel wild mouse roller coaster at Disney's Animal Kingdom at the Walt Disney World Resort.

 The ride is a spinning roller coaster purchased from Reverchon Industries.

 The ride is part of Chester and Hester's Dino-Rama, which is itself part of DinoLand U.S.A. It is a roller coaster in the "Mild But Wild Thrills" category.

 It has cars that spin in circles while traveling on tracks, permitting the ride experience to vary greatly each time it is ridden.

 The ride features 13 cars, each seating up to 4 riders.
</item 9>

<item 10>
Imperial War Museum (disambiguation)
The Imperial War Museum is a British national museum organisation with branches at five locations in England.
</item 10>
=== END ITEM id=a17ab7a383b04687 ===

=== ITEM id=1c7d72a5a213df5a ===
Gold sentences and facts:
Sentence 0: The Wire Fox Terrier is a breed of dog, one of many terrier breeds.
  Fact 0.0: The Wire Fox Terrier is a breed of dog.
  Fact 0.1: The Wire Fox Terrier is one of many terrier breeds.
Sentence 1: The West Siberian Laika or "WSL", is a breed of hunting dog and a breed of spitz type.
  Fact 1.0: The West Siberian Laika, also called WSL, is a breed of hunting dog.
  Fact 1.1: The West Siberian Laika, also called WSL, is a breed of spitz type.

Retrieved items:
<item 1>
The Bulldog is a muscular, wrinkled English breed, ranked fourth among US purebreds in 2007. West Siberian Laikas are versatile barking hunting dogs whose
formal breeding resumed in 1946. Mediterranean warren hounds, including Cretan and Canary Island hounds, typically hunt using smell, sight, and hearing. Segugio
Italiano has short- and wire-haired varieties. Guatemala’s national Dogo is locally recognized; Cuba’s Bull Mastiff-type Dog
</item 1>

<item 2>
A wide-ranging reference collection spans animals, food, forests, film, television, animation, games, music, theatre, and literature. It notes Wire Fox Terrier
Signal Circuit of Halleston’s 1926 Westminster win, the show’s 50th, and the breed’s fourth Best in Show; covers terriers, hounds, dogs, cheese, and wine; and
profiles many works and creators, from Batman, Superman, Disney, Pixar, and *Adventure Time* to classic films
</item 2>

<item 3>
West Siberian Laika
The West Siberian Laika or "WSL", is a breed of hunting dog and a breed of spitz type.

 Russian publications indicate that the term West Siberian Laika loosely applied to hunting dogs originating with the Mansi and Khanty people in Ural and West
Siberia, but there were no standards or registrations of WSL as such until 1930.

 Then WWll disrupted it for a while, but "systematic breeding with registrations" resumed after the war ended, in 1946.

 This was the time the breed began taking modern shape.

 Before that hunters only knew of Mansi Laika and Khanty Laika.

 In early 1960 many hunters in Ural still preferred the term Mansi Laika, when speaking of West Siberian Laika.

 In Russian language, the term Laika originated from the word "layat" that means to bark.

 The word Laika simply means "barker".

 Any hunting Laika is a bark pointer (pointing at animal of interest by barking and staying with the animal ).

 It is a versatile dog depending on use and environment, but in certain parts of the country they have become more specialized.
</item 3>

<item 4>
Wire Fox Terrier
The Wire Fox Terrier is a breed of dog, one of many terrier breeds.

 It is a fox terrier, and although it bears a resemblance to the smooth fox terrier, they are believed to have been developed separately.
</item 4>

<item 5>
Fox Terrier
Fox Terriers are two different breeds of the terrier dog type: the Smooth Fox Terrier and the Wire Fox Terrier.

 Both of these breeds originated in the 19th century from a handful of dogs who are descended from earlier varieties of British terriers, and are related to
other modern white terrier breeds.

 In addition, a number of breeds have diverged from these two main types of fox terrier and have been recognised separately, including the Jack Russell Terrier,
Miniature Fox Terrier and Rat Terrier.

 The Wire and Smooth Fox Terriers share similar characteristics, the main differences being in the coat and markings.

 They have been successful in conformation shows, more prominently in America than their homeland.
</item 5>

<item 6>
Signal Circuit of Halleston
Signal Circuit of Halleston was a Wire Fox Terrier and winner of the 1926 Westminster Kennel Club Dog Show.

 This was the 50th Westminster show, and the fourth Fox Terrier to win best in show.
</item 6>

<item 7>
Teddy Roosevelt Terrier
The Teddy Roosevelt Terrier is a small to medium-sized American hunting terrier.

 Lower-set with shorter legs, more muscular, and heavier bone density than its cousin the American Rat Terrier.

 There is much diversity in the history of the Teddy Roosevelt Terrier breed and it shares a common early history with the American Rat Terrier, Fox Paulistinha
and Tenterfield Terrier.

 It is said the Rat Terrier background stems from the terriers or other dogs that were brought over by early English and other working class immigrants.

 Since the breed was a farm, hunting and utility dog there was little to no planned breeding other than breeding dogs with agreeable traits to each other in
order to produce the desired work ethic in the dog.

 It is assumed that the Feist (dog), Bull Terrier, Smooth Fox Terrier, Manchester Terrier, Whippet, Italian Greyhound, the now extinct English White Terrier,
Turnspit dog and or Wry Legged Terrier all share in the Teddy Roosevelt Terrier's ancestry.

 These early Ratting Terriers were then most likely bred to the Beagle or Beagle cross bred dogs (for increased scenting ability) and other dogs.
</item 7>

<item 8>
Signal Circuit of Halleston, a Wire Fox Terrier, won the 1926 Westminster show—the event’s 50th and the fourth Fox Terrier Best in Show winner. Fox Terriers
comprise Smooth and Wire breeds, developed in 19th-century Britain; related breeds include the Jack Russell, Miniature Fox and Rat Terriers. Teddy Roosevelt
Terriers, descended from working terriers and other breeds, are intelligent, alert hunting companions. Caesar, King Edward VII’s Wire Fox Terrier
</item 8>

<item 9>
Signal Circuit of Halleston, a Wire Fox Terrier, won the 1926 Westminster show, its 50th, marking the fourth Fox Terrier Best in Show victory. Notes cover Fox
Terrier varieties and related terriers; Teddy Roosevelt Terriers; Swiss Appenzeller cheese, Sennenhunde and chicken varieties; and Australia’s native forests.
They also describe French scenthounds and pointing dogs, Gascony wine, Bulldogs, West Siberian Laikas, Mediterranean h
</item 9>

<item 10>
Old Jock
Old Jock (1859–1871), was a Fox Terrier famous during the late 19th and early 20th centuries.

 A mostly white dog, he ran briefly with a hunting kennel before becoming a show dog, most notably with a victory at the show which popularised the Fox Terrier.

 His main show rivalry was with a dog named Tartar, and along with a dog named Trap, the three were popular sires of the Fox Terrier breed.

 He was also involved in the early formation of the Jack Russell Terrier and the Dandie Dinmont Terrier breeds.
</item 10>
=== END ITEM id=1c7d72a5a213df5a ===

=== ITEM id=34b920c73fa0f6c4 ===
Gold sentences and facts:
Sentence 0: Fastbacks were a Seattle, Washington, punk rock band.
  Fact 0.0: Fastbacks were a punk rock band.
  Fact 0.1: Fastbacks were from Seattle, Washington.
Sentence 1: Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.
  Fact 1.0: Local Natives is an American indie rock band.
  Fact 1.1: Local Natives was previously known as Cavil at Rest.
  Fact 1.2: Local Natives is based in Silver Lake, Los Angeles, California, United States.

Retrieved items:
<item 1>
American Hi-Fi
American Hi-Fi is an American rock band formed in Boston in 1998.

 The band consists of lead vocalist/rhythm guitarist Stacy Jones, lead guitarist Jamie Arentzen, bassist/backing vocalist Drew Parsons, and drummer Brian Nolan.

 Prior to the group's formation, Stacy Jones was well known for being a drummer in the successful alternative rock bands Veruca Salt and Letters to Cleo.

 American Hi-Fi has a close relationship with Miley Cyrus, whose band shares two members with American Hi-Fi.

 The group has a mixed musical style that includes influences from pop punk, alternative rock, and power pop.
</item 1>

<item 2>
Local Natives
Local Natives (previously known as Cavil at Rest) is an American indie rock band based in Silver Lake, Los Angeles, California, United States.

 Their debut album, "Gorilla Manor", was first released in the UK in November 2009, and later released in the US on February 16, 2010.

 The album received mostly positive reviews and debuted in the Billboard 200 and at No. 3 in the New Artist Chart.

 Their second album, "Hummingbird", was released in January 2013.

 Their third album, "Sunlit Youth", was released in September 2016.
</item 2>

<item 3>
Back to the Beat (EP)
Back to the Beat was the first significant EP release by the Minneapolis based pop punk band Motion City Soundtrack.

 The EP was recorded in Sound In Motion Recordings and released by Modern Radio Records and includes tracks that were later re-released.

 "Throw Down", "Back to the Beat" and "Capital H" were re-released on a split EP with Schatzi.

 "Capital H" was re-released on Motion City Soundtrack's first album "I Am the Movie".

 At the time of recording, the band members differed from the Motion City Soundtrack line-up formed in 2002.

 They were Joshua Cain on guitar and vocals, Austin Lindstrom on bass guitar, Joel Habedank on drums and Justin Pierre on lead vocals, guitar and keyboard.
</item 3>

<item 4>
The Shins
The Shins are an American indie rock band from Albuquerque, New Mexico, formed in 1996.

 The band's current lineup consists of James Mercer (vocals, guitar, songwriter), Jon Sortland (drums), Mark Watrous (guitar), Casey Foubert (guitar), Yuuki
Matthews (bass), and Patti King (keyboards).

 The band is based in Portland, Oregon.
</item 4>

<item 5>
The Lumineers
The Lumineers are an American folk rock/Americana band based in Denver, Colorado.

 The founding members are Wesley Schultz (lead vocals, guitar) and Jeremiah Fraites (drums, percussion).

 Schultz and Fraites began writing and performing together in Ramsey, New Jersey in 2005.

 Cellist and vocalist Neyla Pekarek joined the band in 2010.

 The Lumineers emerged as one of the most popular folk rock/Americana artists during the revival of those genres and their growing popularity in the 2010s.

 The band's stripped back raw sound draws heavily from artists that influenced Schultz and Fraites such as Bruce Springsteen, Bob Dylan and Tom Petty.

 They are known for their energetic live shows and several international hit singles including "Ho Hey”, “Stubborn Love”, “Ophelia” and “Cleopatra”.

 The band has become one of the top touring bands in the U.S. and also sells out shows around the world.
</item 5>

<item 6>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 6>

<item 7>
Hummingbird (Local Natives album)
Hummingbird is the second studio album by American indie rock band Local Natives, released on January 29, 2013 on Frenchkiss Records in the US, and on
Infectious Records in Europe.

 Produced by The National guitarist Aaron Dessner, the album was preceded by the singles, "Breakers" and "Heavy Feet."

 "Breakers" reached #12 on the Billboard Top 200.

 The song "Mt. Washington" was also featured in the 2015 video game "Life Is Strange".
</item 7>

<item 8>
The material covers country, country-rock, folk-rock and bluegrass artists and releases. Oak Ridge Boys’ 2009 *The Boys Are Back* marked a return to secular
country, led by a White Stripes cover. Richie Furay helped form Buffalo Springfield and Poco, later joining the Souther–Hillman–Furay Band; Poco’s debut
featured his “Pickin’ Up the Pieces.” Other subjects include Buffalo Springfield’s final album, the Lonesome Pine F
</item 8>

<item 9>
Fastbacks
Fastbacks were a Seattle, Washington, punk rock band.

 Formed in 1979 by songwriter/guitarist Kurt Bloch (born August 28, 1960), and friends Lulu Gargiulo (guitar and vocals, born October 12, 1960) and Kim Warnick
(bass and vocals, born April 7, 1959), they disbanded in 2001.

 Their sound mixed a generally punk rock approach to vocals and sound textures with poppy tunes and strong musicianship.
</item 9>

<item 10>
OK Go
OK Go is an American rock band originally from Chicago, Illinois, now based in Los Angeles, California.

 The band is composed of Damian Kulash (lead vocals, guitar), Tim Nordwind (bass guitar and vocals), Dan Konopka (drums and percussion) and Andy Ross (guitar,
keyboards and vocals), who joined them in 2005, replacing Andy Duncan.

 The band is known for its often quirky and elaborate one-take music videos.
</item 10>
=== END ITEM id=34b920c73fa0f6c4 ===

=== ITEM id=54900031cad8e876 ===
Gold sentences and facts:
Sentence 0:  The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
  Fact 0.0: The first of the books was optioned by Universal Pictures.
  Fact 0.1: The option followed a bidding war over the manuscript.
Sentence 1:  The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film, and the oldest in terms of the overall film market .
  Fact 1.0: The company was founded in 1912.
  Fact 1.1: The company was founded by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H. Cochrane, and Jules Brulatour.
  Fact 1.2: The company is the oldest surviving film studio in the United States.
  Fact 1.3: The company is the world's fourth oldest film studio, after Gaumont, Pathé and Nordisk Film.
  Fact 1.4: The company is the oldest film studio in terms of the overall film market.

Retrieved items:
<item 1>
Universal Pictures
Universal Pictures (also referred to as Universal Studios or simply Universal) is an American film studio owned by Comcast through the Universal Filmed
Entertainment Group division of its wholly owned subsidiary NBCUniversal.

 The company was founded in 1912 by Carl Laemmle, Mark Dintenfass, Charles O. Baumann, Adam Kessel, Pat Powers, William Swanson, David Horsley, Robert H.
Cochrane, and Jules Brulatour, and is the oldest surviving film studio in the United States, the world's fourth oldest after Gaumont, Pathé and Nordisk Film,
and the oldest in terms of the overall film market .

 Its studios are located in Universal City, California, and its corporate offices are located in New York City.
</item 1>

<item 2>
Joshua Bloom
Joshua Simon Bloom (born June 8, 1974 in Washington, D.C.) is an American astrophysicist, full professor of astronomy at the University of California, Berkeley,
and was the CTO and co-founder of the machine-learning company wise.io (acquired by General Electric, 2016).

 He received a Bachelor of Arts in astronomy and astrophysics and physics from the Harvard College in 1996, an M.Phil from Cambridge University in 1997, and a
PhD in astronomy from the California Institute of Technology in 2002.

 He was a Junior Fellow of the Harvard Society of Fellows from 2002 to 2005.

 His astronomy research focuses on gamma-ray bursts and other astrophysical transients such as supernovae and tidal disruption events.

 He is author of the book "What Are Gamma-Ray Bursts?"

 published by Princeton University Press in 2011.
</item 2>

<item 3>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 3>

<item 4>
Bogle-L'Ouverture Publications
Bogle-L’Ouverture Publications (BLP) is a radical London-based publishing company founded in 1968 by Guyanese activists Jessica Huntley (23 February 1927 – 13
October 2013) and Eric Huntley (born 25 September 1929).

 Named in honour of two outstanding liberation fighters in Caribbean history, Toussaint L'Ouverture and Paul Bogle, the company began operating during a period
in the UK when "books by Black authors or written with a sympathetic view of Black people’s history and culture were rare in mainstream bookshops in the UK."

 Alongside New Beacon Books (founded in 1966) and Allison & Busby (founded in 1967), BLP was one of the first black-owned independent publishing companies in
the UK.

 BLP has been described as "a small, unorthodox, self-financing venture that brought a radical perspective to non-fiction, fiction, poetry and children's
books."
</item 4>

<item 5>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 5>

<item 6>
Walt Disney Studios Motion Pictures
Walt Disney Studios Motion Pictures (formerly Buena Vista Pictures Distribution and Buena Vista Film Distribution Company) is an American film distributor owned
by The Walt Disney Company.

 Established in 1953 as Buena Vista Film Distribution Company, the company handles theatrical distribution, marketing and promotion for films produced and
released by the Walt Disney Studios, including Walt Disney Pictures, Walt Disney Animation Studios, Pixar Animation Studios, DisneyToon Studios, Marvel Studios,
Lucasfilm, Disneynature, and Touchstone Pictures.

 The division took on its current name in late 2007, which before that had been Buena Vista Pictures Distribution since 1987.
</item 6>

<item 7>
Simon Bloom
Simon Bloom is a series of young adult novels by Michael Reisman.

 Set in the fictional town of Lawnville, New Jersey, it is about a boy who finds a book that lets him control the laws of physics.

 The first of the books was optioned by Universal Pictures after a bidding war over the manuscript.
</item 7>

<item 8>
Simon &amp; Schuster
Simon & Schuster, Inc. ( ), a subsidiary of CBS Corporation, is an American publishing company founded in New York City in 1924 by Richard Simon and Max
Schuster.

 As of 2016, Simon & Schuster publishes 2,000 titles annually under 35 different imprints.
</item 8>

<item 9>
Arthur Lubin
Arthur Lubin (July 25, 1898 – May 12, 1995) was an American film director and producer who directed several "Abbott & Costello" films, "Phantom of the Opera"
(1943), the "Francis the Talking Mule" series and created the talking-horse TV series "Mister Ed".

 A prominent director for Universal Pictures in the 1940s and 1950s, he is perhaps best known today as the man who gave Clint Eastwood his first contract in
film.
</item 9>

<item 10>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 10>
=== END ITEM id=54900031cad8e876 ===

=== ITEM id=1486b8ad5bb1426f ===
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
Teide National Park
Teide National Park (Spanish: "Parque nacional del Teide" , ] ) is a national park located in Tenerife (Canary Islands, Spain).
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
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 3>

<item 4>
Guadarrama National Park
Sierra de Guadarrama National Park (in Spanish: "Parque Nacional de la Sierra de Guadarrama") is a national park in Spain, covering nearly 34,000 hectares, the
fifth largest in Spain's national parks system.

 The Guadarrama mountain range ("Sistema Central") contains some ecologically valuable areas, located in the Community of Madrid and Castile and León (provinces
of Segovia and Ávila).

 The law that regulates the recently approved national park was published in the BOE in  26, 2013 (2013--) .
</item 4>

<item 5>
2011 Pan American Games
The 2011 Pan American Games, officially the XVI Pan American Games, was an international multi-sport event that was held from October 14–30, 2011, in
Guadalajara, Jalisco, Mexico.

 Some events were held in the nearby cities of Ciudad Guzmán, Puerto Vallarta, Lagos de Moreno and Tapalpa.

 It was the largest multi-sport event of 2011, with approximately 6,000 athletes from 42 nations participating in 36 sports.

 Both the Pan American and Parapan American Games were organized by the Guadalajara 2011 Organizing Committee (COPAG).

 The 2011 Pan American Games were the third Pan American Games hosted by Mexico (the first country to do so) and the first held in the state of Jalisco.

 Previously, Mexico hosted the 1955 Pan American Games and the 1975 Pan American Games, both in Mexico City.

 The 2011 Parapan American Games were held 20 days after the Pan American Games have ended.
</item 5>

<item 6>
Golf La Moraleja
It has 6,000 shareholder members.

 In 2013 it was voted the best golf course in Spain by Deporte & Business and in 2014, La Moraleja 3 was voted the second best golf course in Spain by the
American magazine Golf Digest.

 The Club has hosted competitions such as the World Cup of Golf, the Spanish Open or the Paddle Tennis World Championship.
</item 6>

<item 7>
Badiar National Park
Badiar National Park (French: "Parc National du Badiar" ) is a national park in Guinea, on the border with Senegal and contiguous with Senegal's much larger
Niokolo-Koba National Park.

 It was established on 30 May 1985 (by ordonnance N°124/PRG/85), partly in response to Senegal's concern about poaching in Niokolo-Koba National Park.

 Badiar is an International Union for Conservation of Nature Category II park.
</item 7>

<item 8>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 8>

<item 9>
Pirin National Park
Pirin National Park (Bulgarian: Национален парк "Пирин" ) is a national park that encompasses the larger part of the Pirin Mountains in south-west Bulgaria,
spanning an area of 403.56 km2 .

 It is one of the three national parks in the country, the others being Rila National Park and Central Balkan National Park.

 The park was established in 1962 and its territory was expanded several times since then.

 Pirin National Park was declared a UNESCO World Heritage Site in 1983.

 The altitude varies from 950 m to 2,914 m at Vihren, Bulgaria's second highest summit and the Balkans' third.
</item 9>

<item 10>
Landes de Gascogne Regional Natural Park
Landes de Gascogne Regional Natural Park (French: "Parc naturel régional des Landes de Gascogne") is a protected area of pine forest, wetland and oceanic
coastline located in the Aquitaine region of southwestern France.
</item 10>
=== END ITEM id=1486b8ad5bb1426f ===

=== ITEM id=14909a52ad178a4f ===
Gold sentences and facts:
Sentence 0: Dr. Seuss' The Cat in the Hat is a 2003 American family comedy film directed by Bo Welch.
  Fact 0.0: The Cat in the Hat is a Dr. Seuss film.
  Fact 0.1: The Cat in the Hat is a 2003 film.
  Fact 0.2: The Cat in the Hat is an American film.
  Fact 0.3: The Cat in the Hat is a family comedy film.
  Fact 0.4: The Cat in the Hat was directed by Bo Welch.
Sentence 1: Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
  Fact 1.0: Robert W. 'Bo' Welch III was born November 30, 1951.
  Fact 1.1: Robert W. 'Bo' Welch III is an American production designer.
  Fact 1.2: Robert W. 'Bo' Welch III is an American director.

Retrieved items:
<item 1>
Rick Carter
Rick Carter (born 1950) is an American production designer and art director.

 He is known for his work in the film "Forrest Gump", which earned him an Oscar nomination, as well as numerous nominations of other awards for his work in
"Amistad" and "A.I. Artificial Intelligence".

 Other films include "Cast Away", "War of the Worlds", "What Lies Beneath", "Jurassic Park", "Avatar", and "Back to the Future Part II" and "Part III".

 Many of the films that he has worked on are directed by Steven Spielberg or Robert Zemeckis.

 For his part in the Art Direction of "Avatar", he was awarded the Academy Award for Best Production Design alongside Robert Stromberg and Kim Sinclair.

 In 2013, Carter won his second Academy Award, for production design on Steven Spielberg's biopic, "Lincoln".
</item 1>

<item 2>
Bo Welch
Robert W. "Bo" Welch III (born November 30, 1951) is an American production designer and director.
</item 2>

<item 3>
The Cat in the Hat (film)
Dr. Seuss' The Cat in the Hat is a 2003 American family comedy film directed by Bo Welch.

 It is based on the 1957 Dr. Seuss book "of the same name".

 The film stars Mike Myers in the title role of the Cat in the Hat, and Dakota Fanning as Sally.

 Sally's brother (who is unnamed in the book and the 1971 TV special), Conrad, is portrayed by Spencer Breslin.

 The film is the second feature-length Dr. Seuss adaptation after the 2000 holiday film "How the Grinch Stole Christmas".
</item 3>

<item 4>
Winnie the Pooh (2011 film)
The film is directed by Stephen Anderson and Don Hall, adapted from Milne's books by a story team led by Burny Mattinson, produced by Peter Del Vecho, Clark
Spencer, John Lasseter, and Craig Sost, and narrated by John Cleese.
</item 4>

<item 5>
Nitish Roy
Nitish Roy is an Indian film art director, production designer, and costume designer in Hindi cinema and a Bengali Film Director, who is also known for his work
with art cinema directors, Shyam Benegal, Mrinal Sen and Govind Nihlani, Hindi mainstream cinema, directors like Rajkumar Santoshi, as well as international
directors like Mira Nair and Gurinder Chadha.
</item 5>

<item 6>
That Darn Cat (1997 film)
That Darn Cat is a 1997 American mystery comedy film starring Christina Ricci and Doug E. Doug.

 It is a remake of the 1965 film "That Darn Cat!

", which in turn was based on the book "Undercover Cat" by Gordon and Mildred Gordon.

 It is directed by British TV veteran Bob Spiers (most famous for "Fawlty Towers", as well as "Spice World") and written by Scott Alexander and Larry
Karaszewski, best known for "Ed Wood" and the first two "Problem Child" films.
</item 6>

<item 7>
Chris Gorak
Chris Gorak is an American film director who started as an art director and production designer.

 He directed the 2006 film "Right at Your Door" and the 2011 film "The Darkest Hour".

 Gorak began working in the film industry in the 1990s as an art director.

 In the 2000s, he worked on several films as production designer before becoming a film director.
</item 7>

<item 8>
Fritz the Cat (film)
Fritz the Cat is a 1972 American adult animated comedy film written and directed by Ralph Bakshi.

 It was Bakshi's feature film debut and is loosely based on the Fritz the Cat comic strips by Robert Crumb.

 It was the first animated feature film to receive an X rating in the United States.
</item 8>

<item 9>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 9>

<item 10>
That Darn Cat!
That Darn Cat!

 is a 1965 American Walt Disney Productions thriller comedy film starring Hayley Mills (in her last of the six films she made for the Walt Disney Studios) and
Dean Jones (starring in his first film for Disney) in a story about bank robbers, a kidnapping and a mischievous cat.

 The film was based on the 1963 novel "Undercover Cat" by Gordon and Mildred Gordon and was directed by Robert Stevenson.

 The title song was written by the Sherman Brothers and sung by Bobby Darin.

 The 1997 remake includes a cameo appearance by Dean Jones.
</item 10>
=== END ITEM id=14909a52ad178a4f ===

=== ITEM id=f008a6d5e5ae835c ===
Gold sentences and facts:
Sentence 0: Robert Harry Philibosian (born 1940) is an American politician.
  Fact 0.0: Robert Harry Philibosian was born in 1940.
  Fact 0.1: Robert Harry Philibosian is an American politician.
Sentence 1: David R. Ignatius (May 26, 1950), is an American journalist and novelist.
  Fact 1.0: David R. Ignatius was born on May 26, 1950.
  Fact 1.1: David R. Ignatius is an American journalist.
  Fact 1.2: David R. Ignatius is an American novelist.

Retrieved items:
<item 1>
David I. Walsh
David Ignatius Walsh (November 11, 1872June 11, 1947) was a United States politician from Massachusetts.

 A member of the Democratic Party, he served as the 46th Governor of Massachusetts before serving several terms in the United States Senate.
</item 1>

<item 2>
Jim Geraghty
Jim Geraghty is a conservative blogger and regular contributor to "National Review Online" and "National Review".

 In addition to writing columns for "National Review", Geraghty also blogs for National Review Online and is a former reporter for States News Service.
</item 2>

<item 3>
Lewis Nicola
Lewis Nicola (1717 – August 9, 1807) was an Irish-born American military officer, merchant, and writer who held various military and civilian positions
throughout his career.

 Nicola is most notable for authoring the Newburgh letter, which urged George Washington to assume a royal title.

 Born in Dublin, Ireland, Nicola had been an officer in the British Army, serving in Europe before immigrating to the Thirteen Colonies.

 Establishing a residence in Philadelphia with his family, Nicola opened a library in 1767 and was active in colonial philosophical organizations.

 As a result of his work to establish the American Philosophical Society, he was elected as one of its curators.

 When the American Revolution broke out, Nicola offered his services to the colonial government, which eventually appointed him to various positions with local
forces.
</item 3>

<item 4>
Robert Philibosian
Robert Harry Philibosian (born 1940) is an American politician.

 He served as Los Angeles County District Attorney from 1982 to 1984.

 He received his B.A. in history from Stanford University and his law degree from Southwestern Law School, and was admitted to the California State Bar in 1968.

 He is now Of counsel at the law firm of Sheppard, Mullin, Richter & Hampton.
</item 4>

<item 5>
Body of Lies (novel)
Body of Lies is an American spy thriller novel by David Ignatius, a columnist for "The Washington Post".

 It was published by W. W. Norton in 2007.

 It was originally titled "Penetration" but was renamed after Warner Bros. bought the rights in 2006.

 It was made into a film of the same name in 2008.
</item 5>

<item 6>
Phil Driscoll
Phil Driscoll (born November 9, 1947) is a trumpeter, singer, composer, and producer.

 He performs in varying music genres and styles which include rock and roll, rhythm and blues, and patriotic music, and is best known for his work in Christian
music and his longterm Christian ministry.

 In 1985, Driscoll won the Grammy Award for Best Gospel Performance – Duo/Group for a duet with Debby Boone on "Keep the Flame Burning", and he has been
nominated for three additional Grammys, two for Best Gospel Performance – Male and one for Best Gospel/Pop Album.

 He has also won three Dove Awards for his music, and the 1999 Christian Country Music Association Award for Best Musician.

 In 2006, Driscoll was found guilty on 2 counts of tax evasion and one count of conspiracy, and was sentenced to serve one year in Federal prison, beginning on
March 14, 2007.
</item 6>

<item 7>
Mark Krikorian
Mark Krikorian has been the executive director of the Center for Immigration Studies, a think-tank in Washington, D. C., since 1995.

 The Center describes itself as an "independent, non-partisan research organization" in Washington, D. C., that examines and critiques the impact of immigration
on the United States.

 Animated by a "pro-immigrant, low-immigration vision which seeks fewer immigrants, but a warmer welcome for those admitted", the Center was established in 1985
to provide immigration research.

 Krikorian is a regular contributor to the conservative publication "National Review", and is a regular participant at "National Review Online's" "The Corner."
</item 7>

<item 8>
Philip K. Dick
Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
</item 8>

<item 9>
David Smick
David M. Smick is a macroeconomic adviser to a select group of prominent global investment funds and a non-fiction author.

 He is the chairman and CEO of Johnson Smick International, a financial advisory firm in Washington, D.C. where he is in partnership with former Federal Reserve
Vice Chairman Manuel H. Johnson.

 Smick wrote an acclaimed book "The World Is Curved" on the financial perils of globalization, in response to "The World Is Flat" bestseller by Thomas Friedman.

 The book received positive reviews by the "USA Today" and "Forbes", was called "astonishingly prescient" by David Brooks and "clairvoyant" by David Ignatius.
</item 9>

<item 10>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 10>
=== END ITEM id=f008a6d5e5ae835c ===

=== ITEM id=4d33d8e812472f6e ===
Gold sentences and facts:
Sentence 0: Cars 2 is a 2011 American computer-animated action comedy spy film produced by Pixar Animation Studios for Walt Disney Pictures.
  Fact 0.0: Cars 2 is a film.
  Fact 0.1: Cars 2 was released in 2011.
  Fact 0.2: Cars 2 is American.
  Fact 0.3: Cars 2 is computer-animated.
  Fact 0.4: Cars 2 is an action comedy spy film.
  Fact 0.5: Cars 2 was produced by Pixar Animation Studios.
  Fact 0.6: Cars 2 was produced for Walt Disney Pictures.
Sentence 1: Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway musicals "Hamilton" and "In the Heights".
  Fact 1.0: Lin-Manuel Miranda was born January 16, 1980.
  Fact 1.1: Lin-Manuel Miranda is American.
  Fact 1.2: Lin-Manuel Miranda is a composer, lyricist, playwright, and actor.
  Fact 1.3: Lin-Manuel Miranda is best known for creating and starring in the Broadway musical Hamilton.
  Fact 1.4: Lin-Manuel Miranda is best known for creating and starring in the Broadway musical In the Heights.
Sentence 2:  He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".
  Fact 2.0: Lin-Manuel Miranda co-wrote the songs for Disney's Moana soundtrack.
  Fact 2.1: The Moana soundtrack is from 2016.
  Fact 2.2: Lin-Manuel Miranda is set to star in Disney's upcoming film Mary Poppins Returns.

Retrieved items:
<item 1>
Lin-Manuel Miranda
Lin-Manuel Miranda ( ; born January 16, 1980) is an American composer, lyricist, playwright, and actor best known for creating and starring in the Broadway
musicals "Hamilton" and "In the Heights".

 He co-wrote the songs for Disney's "Moana" soundtrack (2016) and is set to star in their upcoming film "Mary Poppins Returns".

 Miranda's awards include a Pulitzer Prize, two Grammy Awards, an Emmy Award, a MacArthur Fellowship, and three Tony Awards.
</item 1>

<item 2>
Cars 2 (video game)
Cars 2 (also known as Cars 2: The Video Game) is a racing game developed by Avalanche Software.

 It is based on the film "Cars 2", the sequel to the 2006 film "Cars".

 Originally announced at E3 2011, the game was released by Disney Interactive on major platforms in North America on June 21, 2011, and in Australia 2 days
later.

 The game was released in Europe on July 22, 2011.

 The PlayStation 3 version of the game was reported to feature stereoscopic 3D gameplay.

 A Nintendo 3DS version was released on November 1, 2011 (the same day the film was released on DVD and Blu-ray).

 A PlayStation Portable version was released in North America on November 8, 2011.

 It was also released in Europe on November 4, 2011.

 It was the last Pixar game to be released for the PlayStation Portable.
</item 2>

<item 3>
Moana (soundtrack)
Moana: Original Motion Picture Soundtrack is the soundtrack to the 2016 Disney animated film "Moana".

 The soundtrack was released by Walt Disney Records on November 19, 2016.

 It features songs written by Lin-Manuel Miranda, Mark Mancina and Opetaia Foa'i, with lyrics in English, Samoan, Tokelauan, and Tuvalu.

 The two-disc deluxe edition includes the score, which was composed by Mancina, as well as demos, outtakes and instrumental karaoke tracks.
</item 3>

<item 4>
Wreck-It Ralph
Wreck-It Ralph is a 2012 American 3D computer-animated fantasy-comedy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 52nd Disney animated feature film.

 The film was directed by Rich Moore, who has directed episodes of "The Simpsons" and "Futurama", and the screenplay was written by Phil Johnston and Jennifer
Lee from a story by Moore, Johnston, and Jim Reardon.

 John Lasseter served as the executive producer.

 The film features the voices of John C. Reilly, Sarah Silverman, Jack McBrayer, and Jane Lynch.

 The film tells the story of the eponymous arcade game villain who rebels against his role and dreams of becoming a hero.

 He travels between games in the arcade and ultimately must eliminate a dire threat that could affect the entire arcade and one that Ralph himself
unintentionally started.
</item 4>

<item 5>
Frozen (2013 film)
Frozen is a 2013 American 3D computer-animated musical fantasy film produced by Walt Disney Animation Studios and released by Walt Disney Pictures.

 It is the 53rd Disney animated feature film.

 Inspired by Hans Christian Andersen's fairy tale "The Snow Queen", the film tells the story of a fearless princess who sets off on a journey alongside a rugged
iceman, his loyal pet reindeer, and a naïve snowman to find her estranged sister, whose icy powers have inadvertently trapped the kingdom in eternal winter.
</item 5>

<item 6>
Cars 2
Cars 2 is a 2011 American computer-animated action comedy spy film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the sequel to 2006's "Cars", and features the voices of Owen Wilson, Larry the Cable Guy, Michael Caine, Emily Mortimer, John Turturro, and Eddie Izzard.

 In the film, race car Lightning McQueen and tow truck Mater head to Japan and Europe to compete in the World Grand Prix, but Mater becomes sidetracked with
international espionage.

 The film was directed by John Lasseter, written by Ben Queen, and produced by Denise Ream.
</item 6>

<item 7>
Walt Disney Pictures
Walt Disney Pictures, Inc. is an American film production company and a subsidiary of Walt Disney Studios, owned by The Walt Disney Company.

 The division is based at the Walt Disney Studios in Burbank, California, and is the main producer of live-action feature films within the Walt Disney Studios
unit.

 It took on its current name in 1983.

 Today, in conjunction with the other units of Walt Disney Studios, Walt Disney Pictures is classified as one of Hollywood's "Big Six" film studios.

 Films produced by Walt Disney Animation Studios and Pixar Animation Studios are also released under this brand.
</item 7>

<item 8>
Toy Story
Toy Story is a 1995 American computer-animated buddy comedy adventure film produced by Pixar Animation Studios for Walt Disney Pictures.

 The directorial debut of John Lasseter, "Toy Story" was the first feature-length computer-animated film and the first feature film produced by Pixar.

 Taking place in a world where anthropomorphic toys pretend to be lifeless whenever humans are present, the film's plot focuses on the relationship between
Woody, an old-fashioned pullstring cowboy doll (voiced by Tom Hanks), and Buzz Lightyear, an astronaut action figure (voiced by Tim Allen), as they evolve from
rivals competing for the affections of Andy, their owner, to friends who work together to be reunited with Andy as his family prepares to move to a new home.

 The screenplay was written by Joss Whedon, Andrew Stanton, Joel Cohen and Alec Sokolow, based on a story by Lasseter, Pete Docter, Stanton and Joe Ranft.

 The film features music by Randy Newman, and was executive-produced by Steve Jobs and Edwin Catmull.
</item 8>

<item 9>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 9>

<item 10>
Toy Story 3
Toy Story 3 is a 2010 American 3D computer-animated comedy-drama film produced by Pixar Animation Studios for Walt Disney Pictures.

 It is the third installment in Pixar's "Toy Story" series, and the sequel to 1999's "Toy Story 2".

 It was directed by Lee Unkrich, the editor of the first two films and the co-director of "Toy Story 2", written by Michael Arndt, while Unkrich wrote the story
along with John Lasseter and Andrew Stanton, respectively director and co-writer of the first two films.

 The plot focuses on the toys Woody, Buzz Lightyear, and their friends dealing with an uncertain future as their owner, Andy, prepares to leave for college.
</item 10>
=== END ITEM id=4d33d8e812472f6e ===

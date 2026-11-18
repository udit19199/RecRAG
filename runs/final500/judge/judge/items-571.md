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

=== ITEM id=46137afe8095f23b ===
Gold sentences and facts:
Sentence 0: The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna (Tenerife, Canary Islands, Spain).
  Fact 0.0: The Church of the Guanche People's Spanish name is 'Iglesia del Pueblo Guanche'.
  Fact 0.1: The Church of the Guanche People is a religious organisation.
  Fact 0.2: The Church of the Guanche People was founded in 2001.
  Fact 0.3: The Church of the Guanche People was founded in the city of San Cristobal de La Laguna.
  Fact 0.4: San Cristobal de La Laguna is in Tenerife, Canary Islands, Spain.
Sentence 1:  Tenerife is the largest and most populous island of Macaronesia.
  Fact 1.0: Tenerife is the largest island of Macaronesia.
  Fact 1.1: Tenerife is the most populous island of Macaronesia.

Retrieved items:
<item 1>
Garajonay National Park
Garajonay National Park (Spanish: "Parque nacional de Garajonay" , ] ) is located in the center and north of the island of La Gomera, one of the Canary Islands
(Spain).

 It was declared a national park in 1981 and a World Heritage Site by UNESCO in 1986.

 It occupies 40 km (15 sq mi) and it extends into each of the six municipalities on the island.
</item 1>

<item 2>
Gran i General Consell
The Gran i General Consell (Catalan: "Great and General Council") was the supreme political, administrative, and representative organ of the Kingdom of Majorca.

 Since the Kingdom of Majorca did not have courts, the Gran i General Consell took over most of the functions they would otherwise have had, including the role
of a representative body.

 The Gran i General Consell evolved from the Catalan municipal councils, especially that of the City of Majorca (present-day Palma de Mallorca.

 The body that was to become the Gran i General Consell was founded in 1249, and the Consell was abolished on 22 July 1718 by Philip V, empowered by the Nova
Planta Decree of Majorca and Ibiza (28 November 1715).

 At the same time, all other separate Majorcan institutions were dissolved.
</item 2>

<item 3>
Chiguata District
Cherubs also decorate the circular cornice, along with four saints in full-length reliefs that occupy the "pechinas" of the columns that sustain the temple's
dome, which is considered one of the most original works of the colony and now visited by occasional tourists.
</item 3>

<item 4>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 4>

<item 5>
Graph query row:
organization: Church of the Guanche People
church_connection: LOCATED_IN
city: San Cristóbal de La Laguna
city_connection: LOCATED_IN
island: Tenerife
area_connection: LOCATED_IN
area: Canary Islands
</item 5>

<item 6>
Tenerife
Tenerife ( ; ] ) is the largest and most populated island of the seven Canary Islands.

 It is also the most populated island of Spain, with a land area of 2034.38 km2 and 898,680 inhabitants, 43 percent of the total population of the Canary
Islands.

 Tenerife is the largest and most populous island of Macaronesia.
</item 6>

<item 7>
Roque Cinchado
The Roque Cinchado is a rock formation, regarded as emblematic of the island of Tenerife (Canary Islands, Spain).

 It lies within the Teide National Park (a World Heritage Site) in the municipality of La Orotava, near the volcano of the same name, in the heart of the
island.

 The Roque Cinchado is one of the largest in the world by altitude, for the entire park totals more than 2000 metres.
</item 7>

<item 8>
Hermigua
Hermigua is a town and a municipality in the northeastern part of La Gomera in the province of Santa Cruz de Tenerife of the Canary Islands, Spain.

 It is located 12 km northwest of the island's capital, San Sebastián de la Gomera.

 The Garajonay National Park covers the southern part of the municipality.
</item 8>

<item 9>
Guanche mummies
Guanche mummies are the intentionally desiccated remains of members of the indigenous Guanche people of the Canary Islands.

 The majority of Guanche mummies were made during the eras prior to Spanish settlement of the area in the 15th century.

 The methods of embalming are similar to those that were used by the Ancient Egyptians, though fewer mummies remain from the Guanche due to looting and
desecration.
</item 9>

<item 10>
Chiguata District
Chiguata is a district located in the Arequipa province, 30 km away from the city of Arequipa, Peru, on the slopes of the volcano Pikchu Pikchu (where at its
summit pre-Inca worshipped remains and artifacts have been found).

 The name "chiguata" comes from the Quechua words "chiri" meaning "cold" and "wata" meaning "year" (cold all year).

 Ancient Chiguata was inhabited by the local natives until the arrival of Spanish influence in charge of the emissary Diego Hernández de Córdova in 1540.

 Soon after, the construction of a small plaza was initiated, and according to the Dominican religious order and local sources, the plaza's "Templo del Espíritu
Santo" ("Sacred Spirit temple") along with its domes and lateral cover was not begun until 1739.

 The design of the temple is quite interesting, with two angels in feathered nightwear and their arms lifted vertically, separated by twelve fringes of
rectangular flowers which add to the whole decoration of the temple.
</item 10>
=== END ITEM id=46137afe8095f23b ===

=== ITEM id=a945e3c94a694fb8 ===
Gold sentences and facts:
Sentence 0:  Levitt & Sons developed the Monmouth Heights section from 1965-1969.
  Fact 0.0: Levitt & Sons developed the Monmouth Heights section from 1965 to 1969.
Sentence 1:  The company is most famous for having built the town of Levittown, New York.
  Fact 1.0: Levitt & Sons is most famous for having built the town of Levittown, New York.

Retrieved items:
<item 1>
Monmouth Heights at Manalapan, New Jersey
Monmouth Heights at Manalapan is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 Levitt & Sons developed the Monmouth Heights section from 1965-1969.

 Residents of the 685 homes in the development are required to join the Monmouth Heights Community Association, which provides pools and other amenities.
</item 1>

<item 2>
Riverton, New York
Riverton, New York was a planned community built in 1973 in the town of Henrietta, New York, along the Genesee River.

 Riverton was seventh of the thirteen communities receiving Federal guarantee assistance from HUD as part of their New Communities Program created by the Urban
Growth and New Community Development Act of 1970.

 $12 million was earmarked for the community.

 Riverton was supposed to encompass a 2335 acre land area bordered on the north by the New York State Thruway, on the west by the Genesee River, on the east by
East River Rd., and on the south by the Rush-Henrietta town line, with some development continuing east along Erie Station Rd.

 Original plans called for construction to occur for a 16-year period and eventually house over 25,000 people, with the community including an 18-hole golf
course, 12 swimming pools, 3 artificial lakes, a marina on the Genesee, and commercial and industrial centers.

 However, only a small portion of this was built along Scottsville-West Henrietta Rd.
</item 2>

<item 3>
Province of New Jersey
The Province of New Jersey was one of the Middle Colonies of Colonial America and became the U.S. state of New Jersey in 1776.

 The province had originally been settled by Europeans as part of New Netherland, but came under English rule after the surrender of Fort Amsterdam in 1664,
becoming a proprietary colony.

 The English then renamed the province after the Isle of Jersey in the English Channel.

 The Dutch Republic reasserted control for a brief period in 1673–1674.

 After that it consisted of two political divisions, East Jersey and West Jersey, until they were united as a royal colony in 1702.

 The original boundaries of the province were slightly larger than the current state, extending into a part of the present state of New York, until the border
was finalized in 1773.
</item 3>

<item 4>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 4>

<item 5>
Oakland Mills, New Jersey
Oakland Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The area is named for Oakland Mill, a mill along the Manalapan Brook operated by the Boo family.

 Some areas within the Manalapan Brook valley and surround forestland are preserved while the remainder of the area surrounding Oakland Mills has since been
developed into housing developments.
</item 5>

<item 6>
Riverview, New Brunswick
Riverview is a Canadian town in Albert County, New Brunswick, Canada.

 Riverview is located on the south side of the Petitcodiac River, across from the larger cities of Moncton and Dieppe.

 Riverview has an area of 34 km2 , and a population density of 564.6 PD/km2 .

 Riverview's slogan is "A Great Place To Grow".

 With a population of 19,667, Riverview is the fifth largest municipality in New Brunswick, having a larger population than the cities of Edmundston, Bathurst,
Campbellton, and Miramichi, despite its designation of "town".
</item 6>

<item 7>
Taylors Mills, New Jersey
Taylors Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The main road that runs through the area is named for the community, Taylors Mills Road.
</item 7>

<item 8>
HMS Monmouth
Seven ships of the Royal Navy have been named HMS "Monmouth".

 Monmouth was the name of a castle and is now the name of a town in Wales; the name also recognises James Scott, 1st Duke of Monmouth, the "Black Duke".

 The first HMS "Monmouth" was named after the town but whilst the second one was being used the Monmouth Rebellion took place and this caused some concern to
the crew.

 Since that time, all of "Monmouth"s have had their names painted in black and the name and the crest are covered because of the shame of the Duke of Monmouth's
execution.

 Today the officers of the latest HMS "Monmouth" carry black handkerchiefs and the ship is known as the Black Duke.

 However the town of Monmouth still gives the freedom of the town to the ship's crew and talks of the "enduring link" between the town and the ship.
</item 8>

<item 9>
Hanbys Corner, Delaware
Hanbys Corner is an unincorporated community in New Castle County, Delaware, United States.

 It is located at the intersection of Delaware Route 3 (Marsh Road) and Delaware Route 92 (Naamans Road), in Brandywine Hundred.

 The area is named for Richard G. Hanby, who first purchased the 125 acre parcel from the descendants of William Penn in 1753.

 His descendants included several notable figures in the local political scene including Samuel Winfield Hanby (1817-1892) who was elected as a State
Representative in 1874, Jacob Klose Hanby (1839-1932) who was Samuel's son and was elected State Representative in 1904, Robert Johnson Hanby (1834-1898), who
served in the 124th PA infantry during the Civil War and was elected State Senator in 1896, and Florence Wood Hanby (1870-1963), who was Robert's
daughter-in-law and was the first woman elected to the Delaware House of Representatives in 1924.

 In addition Wayne Hanby and James Hanby have both served as Justices of the Peace for New Castle County.
</item 9>

<item 10>
Monmouth Heights at Marlboro, New Jersey
Monmouth Heights at Marlboro is an unincorporated community located within Marlboro Township in Monmouth County, New Jersey, United States.

 The housing development was created in the 1960s by Levitt & Sons though some protests occurred by long-time residents of Marlboro Township at the time of its
creation.

 The community is located along New Jersey Route 79 across the highway from Marlboro High School.
</item 10>
=== END ITEM id=a945e3c94a694fb8 ===

=== ITEM id=d26acd5ce4883636 ===
Gold sentences and facts:
Sentence 0: The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.
  Fact 0.0: The Long–Allen Bridge is a truss bridge.
  Fact 0.1: The Long–Allen Bridge is in Shreveport, Louisiana.
  Fact 0.2: The Long–Allen Bridge is named for Louisiana governor Huey Long.
  Fact 0.3: The Long–Allen Bridge is named for Louisiana governor Oscar K. Allen.
Sentence 1:  Opened in 1933, it carries U.S. Route 80 across the Red River.
  Fact 1.0: The Long–Allen Bridge opened in 1933.
  Fact 1.1: The Long–Allen Bridge carries U.S. Route 80 across the Red River.
Sentence 2: U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.
  Fact 2.0: U.S. Route 80 (US 80) is an east–west United States Numbered Highway.
  Fact 2.1: Much of U.S. Route 80 was once part of the early auto trail known as the Dixie Overland Highway.
Sentence 3:  The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
  Fact 3.0: The eastern terminus of U.S. Route 80 is in Tybee Island, Georgia.
  Fact 3.1: The eastern terminus of U.S. Route 80 is at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street.
  Fact 3.2: The eastern terminus of U.S. Route 80 is near the Atlantic Ocean.

Retrieved items:
<item 1>
Rocky Creek Bridge No. 01089
Rocky Creek Bridge No. 01089, also known as Ben Jones Bridge, is a concrete highway bridge spanning Rocky Creek along the Pacific Ocean coast of the U.S. state
of Oregon.

 The bridge crosses a gorge near the creek's mouth, about 2 mi south of Depoe Bay in Lincoln County.

 Built in 1927, the bridge originally carried U.S. Route 101 (the Oregon Coast Highway) over the creek.

 After a stretch of the highway was relocated to make it straighter, the abandoned piece of the old highway became Otter Crest Loop Road, a bypass west of the
new highway.
</item 1>

<item 2>
Long–Allen Bridge (Shreveport)
The Long–Allen Bridge is a truss bridge in Shreveport, Louisiana, named for Louisiana governors Huey Long and Oscar K. Allen.

 Opened in 1933, it carries U.S. Route 80 across the Red River.

 It is also known as the Texas Street Bridge.
</item 2>

<item 3>
Outer barrier
The outer barrier, also known as the Long Island and New York City barrier islands, refers to the string of barrier islands that divide the lagoons south of
Long Island, New York from the Atlantic Ocean.

 These islands include Coney Island, Long Beach Barrier Island, Island Park, Jones Beach Island, Fire Island and Westhampton Island.

 The outer barrier extends 75 mi along the South Shore of Long Island, from Rockaway Beach on the New York City/Nassau County border from Long Beach Barrier
Islands' western edge, to Suffolk County's east end of Shinnecock Bay.
</item 3>

<item 4>
U.S. Route 80
U.S. Route 80 (US 80) is an east–west United States Numbered Highway, much of which was once part of the early auto trail known as the Dixie Overland Highway.

 As the "0" in the route number indicates, it was originally a cross-country route, from the Pacific Ocean to the Atlantic Ocean.

 However, the entire segment west of Dallas, Texas, has been decommissioned in favor of various Interstate Highways and state highways.

 Currently, the highway's western terminus is at an interchange with Interstate 30 (I-30) on the Dallas–Mesquite, Texas city line.

 The highway's eastern terminus is in Tybee Island, Georgia, at the intersection of Butler Avenue, Inlet Avenue, and Tybrisa Street, near the Atlantic Ocean.
</item 4>

<item 5>
Ocean To Ocean Bridge
The Ocean To Ocean Bridge is a through truss bridge spanning the Colorado River in Yuma, Arizona.

 Built in 1915, it was the first highway crossing of the lower Colorado and is the earliest example of a through truss bridge in Arizona.

 It is also the only example of a Pennsylvania truss within Arizona.

 Originally the bridge carried the transcontinental Ocean-to-Ocean Highway and later carried its successor, US 80 until a new bridge was built to the west in
1956.

 Between 1988 and 2001, the bridge was closed to vehicular traffic and only traversable by pedestrians and bicyclists.

 After a major restoration, the bridge was rehabilitated and reopened to vehicular traffic in 2002, with a re-dedication by the Quechan nation and Yuma Crossing
National Heritage Area.
</item 5>

<item 6>
Broad Street Bridge (Rochester, New York)
Erie Canal: Second Genesee Aqueduct, also known as the Broad Street Aqueduct or Broad Street Bridge, is a historic stone aqueduct located at Rochester in Monroe
County, New York.

 It was constructed in 1836-1842 and originally carried the Erie Canal over the Genesee River.

 The overall length of the aqueduct including the wings and abutments is 800 ft .

 The aqueduct is 70 ft wide and has massive parapets on either side.

 It is one of four major aqueducts in the mid-19th century Erie Canal system.

 In 1927, a roadbed was added to carry automobile traffic and named Broad Street.

 It also carried a part of the Rochester Subway.
</item 6>

<item 7>
New York State Route 108
New York State Route 108 (NY 108) is a 1.72 mi north–south state highway located on the Suffolk County side of the Suffolk–Nassau county line on Long Island,
New York, in the United States.

 It is a spur route connecting NY 25A in Cold Spring Harbor to the Cold Spring Harbor station on the Long Island Rail Road's Port Jefferson Branch via Harbor
Road.

 Harbor Road terminates at an intersection with Woodbury Road, on the Nassau County line, which carries County Route 11 to the east and unsigned County Route 12
to the west.

 NY 108, assigned in the early 1930s, is the shortest state highway on Long Island.
</item 7>

<item 8>
Pennsylvania Route 145
Pennsylvania Route 145 (PA 145) is a 20.89 mi long north–south state highway in the Lehigh Valley area of eastern Pennsylvania.

 It connects Interstate 78 (I-78) and PA 309 in Lanark, Lehigh County to PA 248 in Weiders Crossing, Northampton County.

 PA 145 is the main north-south arterial into Allentown, the third-largest city in the state.

 The route enters the city on South 4th Street and follows multiple streets to downtown, where it follows the one-way pair of 6th Street northbound and 7th
Street southbound.

 North of Allentown in Whitehall Township, a seven-mile (11.2 km) portion of PA 145 is known as MacArthur Road, named in honor of General Douglas MacArthur.

 MacArthur Road is a divided highway; between U.S. Route 22 (US 22) and Eberhart Road, it is six lanes wide with a Jersey barrier and jughandles while the
remainder of the road a four-lane divided highway.

 MacArthur Road is the location of the main commercial center of the Lehigh Valley.
</item 8>

<item 9>
Grand Central Parkway
The Grand Central Parkway (GCP) is a 14.61-mile (23.51 km) long parkway that stretches from the Triborough Bridge in New York City to Nassau County on Long
Island.

 At the Queens–Nassau border, it becomes the Northern State Parkway, which runs across the northern part of Long Island through Nassau County and into Suffolk
County, where it ends in Hauppauge.

 The westernmost stretch (from the Triborough Bridge to exit 4) also carries a short stretch of Interstate 278 (I-278).

 The parkway runs through Queens and passes the Cross Island Parkway, Long Island Expressway, LaGuardia Airport and Citi Field, home of the New York Mets.

 The parkway is designated New York State Route 907M (NY 907M), an unsigned reference route.

 Despite its name, the Grand Central Parkway was not named after Grand Central Terminal.
</item 9>

<item 10>
Low Bridge (song)
The popular song "Low Bridge, Everybody Down" was written in 1905 by Thomas S. Allen after Erie Canal barge traffic was converted from mule power to engine
power, raising the speed of traffic.

 Also known as "Fifteen Years on the Erie Canal", "Fifteen Miles on the Erie Canal", "Erie Canal Song", and "Mule Named Sal", the song memorializes the years
from 1825 to 1880 when the mule barges made boomtowns out of Utica, Rome, Syracuse, Rochester, and Buffalo, and transformed New York into the Empire State.

 The tune is sadly nostalgic.
</item 10>
=== END ITEM id=d26acd5ce4883636 ===

=== ITEM id=a8420e62608ea58d ===
Gold sentences and facts:
Sentence 0:  The tour accompanied the release of the piano-based album "In the Running" and featured pared-back versions of his hits, album tracks, and The Beatles cover "Come Together".
  Fact 0.0: The tour accompanied the release of the piano-based album 'In the Running'.
  Fact 0.1: The tour featured pared-back versions of his hits.
  Fact 0.2: The tour featured pared-back versions of his album tracks.
  Fact 0.3: The tour featured a pared-back version of The Beatles cover 'Come Together'.
Sentence 1:  The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
  Fact 1.0: The song reached the top of the charts in the United States.
  Fact 1.1: The song peaked at No. 4 in the United Kingdom.

Retrieved items:
<item 1>
Howard Jones’s 1996 *Live Acoustic America* captures a sold-out Los Angeles show from a worldwide tour, featuring pared-back hits, album tracks, “Come
Together,” and percussionist Carol Steele. Jerry Garcia (1942–1995), Grateful Dead guitarist and vocalist, composed “Sugaree” with Robert Hunter for his 1972
solo album. *Garcia Plays Dylan* compiles 1973–95 live Dylan covers. Bluegrass musician Sandy Rothman
</item 1>

<item 2>
The entries cover songs and recordings spanning R&B, rock, pop, and folk. Highlights include Johnny Otis’s 1958 “Willie and the Hand Jive” (US #9), later
covered by Eric Clapton (#26) and George Thorogood; Van Halen’s “Finish What Ya Started,” co-written overnight by Eddie Van Halen and Sammy Hagar; and Johnny
Otis’s “Every Beat of My Heart,” later a major hit for the Pips.
</item 2>

<item 3>
Beatles (song)
"Beatles", written by Sven Olov Bagge and Claes Bure, was the song that, performed by Swedish dansband Forbes, won the Swedish Melodifestivalen 1977.

 The song's lyrics are about the popular 1960s British rock group the Beatles and include the phrase "Yeah, yeah, yeah" as an homage to the Beatles song "She
Loves You".

 Forbes took the song to the United Kingdom for the Eurovision Song Contest 1977, with Anders Berglund as conductor.

 The audience cheered at the song, not unexpectedly since the contest took place in the United Kingdom, but the jury had another opinion.

 The song finished last, with only two points, from West Germany.
</item 3>

<item 4>
Live Acoustic America
Live Acoustic America is an acoustic live recording of classic Howard Jones songs, released in 1996.

 It was recorded in Los Angeles to a sell out crowd and features Carol Steele on percussion.

 The acoustic tour travelled the world, with Jones playing venues in Europe, the USA and Jamaica.

 The tour accompanied the release of the piano-based album "In the Running" and featured pared-back versions of his hits, album tracks, and The Beatles cover
"Come Together".
</item 4>

<item 5>
Come Together
"Come Together" is a song by the Beatles written primarily by John Lennon and credited to Lennon–McCartney.

 The song is the opening track on the album "Abbey Road" and was also released as a single coupled with "Something".

 The song reached the top of the charts in the United States and peaked at No. 4 in the United Kingdom.
</item 5>

<item 6>
On Top of Old Smoky
"On Top of Old Smokey" is a traditional folk song of the United States.

 As recorded by The Weavers, the song reached the pop music charts in 1951.

 It is catalogued as Roud Folk Song Index No. 414.

 In one version the first verse is the following; for more on the words see below.
</item 6>

<item 7>
Profiles cover pop, R&B, country, and rock releases: Tevin Campbell’s Grammy-nominated “Can We Talk” topped the US R&B chart; Steve Winwood’s “Back in the High
Life Again” reached No. 1 Adult Contemporary, while *Refugees of the Heart* featured a chart-topping rock single. Other highlights include LeAnn Rimes’s
“Looking Through Your Eyes,” Westlife’s UK No. 1 debut “Swear It Again,” and Mar
</item 7>

<item 8>
Robson & Jerome’s 1995 debut topped the UK chart, selling 2.04 million copies; its 42-day record for reaching two million lasted until Adele’s *25*. Forbes’
Beatles tribute won Sweden’s 1977 selection but finished last at Eurovision. Dylan’s 1978 “Baby, Stop Crying” reached UK #13. Jobim’s “Corcovado” became “Quiet
Nights of Quiet Stars”; Andy Williams and Kitty Kallen recorded it. “
</item 8>

<item 9>
McLemore Avenue
McLemore Avenue is a 1970 album by Booker T. & the M.G.s, consisting entirely of mostly instrumental covers of songs from the Beatles' album "Abbey Road"
(released only months earlier, in September 1969).

 The title and cover are an homage to the Beatles album, 926 East McLemore Avenue being the address of the Stax Studios in Memphis, as Abbey Road was for EMI
Studios in London, which was soon renamed Abbey Road Studios.
</item 9>

<item 10>
Baby, Stop Crying
"Baby, Stop Crying" is a song written by Bob Dylan, released in the summer of 1978 as a single and in a longer album version on "Street Legal".

 It was performed live only during his big-band worldwide tour of 1978.

 The song charted at #13 in the UK and was a top-ten song in much of Europe, although it failed to chart in the United States.
</item 10>
=== END ITEM id=a8420e62608ea58d ===

=== ITEM id=13425aa4cf2d6dc7 ===
Gold sentences and facts:
Sentence 0:  They have produced two studio albums and an EP.
  Fact 0.0: They have produced two studio albums.
  Fact 0.1: They have produced an EP.
Sentence 1: Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.
  Fact 1.0: Motion City Soundtrack was an American rock band.
  Fact 1.1: Motion City Soundtrack formed in Minneapolis, Minnesota.
  Fact 1.2: Motion City Soundtrack formed in 1997.
Sentence 2:  Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.
  Fact 2.0: Motion City Soundtrack had a nearly twenty-year career.
  Fact 2.1: Motion City Soundtrack toured heavily over the course of its career.
  Fact 2.2: Motion City Soundtrack released six studio albums.
  Fact 2.3: The majority of Motion City Soundtrack's six studio albums were released on the independent label Epitaph Records.

Retrieved items:
<item 1>
Back to the Beat (EP)
Back to the Beat was the first significant EP release by the Minneapolis based pop punk band Motion City Soundtrack.

 The EP was recorded in Sound In Motion Recordings and released by Modern Radio Records and includes tracks that were later re-released.

 "Throw Down", "Back to the Beat" and "Capital H" were re-released on a split EP with Schatzi.

 "Capital H" was re-released on Motion City Soundtrack's first album "I Am the Movie".

 At the time of recording, the band members differed from the Motion City Soundtrack line-up formed in 2002.

 They were Joshua Cain on guitar and vocals, Austin Lindstrom on bass guitar, Joel Habedank on drums and Justin Pierre on lead vocals, guitar and keyboard.
</item 1>

<item 2>
Motion City Soundtrack
Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 2>

<item 3>
List of songs recorded by Motion City Soundtrack
American rock band Motion City Soundtrack recorded songs for six studio albums, various soundtracks, compilations, and non-album singles.

 The majority of the group's original material was written by vocalist and guitarist Justin Pierre.

 The band also recorded cover versions of other artists' songs at various points throughout their career, including by the Beatles, Limbeck, the Police, R.E.M.,
the Rentals, Rilo Kiley, Trampled by Turtles, and the Weakerthans.

 In all, the group recorded 107 songs, nine of which were covers.
</item 3>

<item 4>
My Dinosaur Life
My Dinosaur Life is the fourth studio album by the American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on January 19, 2010 by Columbia Records.

 After many years on independent label Epitaph Records, Motion City Soundtrack signed to major label Columbia in 2006, prior to the release of their third
album, "Even If It Kills Me" (2007).

 Blink-182 bassist Mark Hoppus returned to produce the album; he had previously worked with the band on their 2005 breakthrough "Commit This to Memory".
</item 4>

<item 5>
Even If It Kills Me
Even if It Kills Me is the third studio album by American rock band Motion City Soundtrack.

 Produced by Ric Ocasek, Adam Schlesinger and Eli Janney, the album was released on September 18, 2007, in the United States by Epitaph Records.

 Motion City Soundtrack, based in Minneapolis, Minnesota, made a breakthrough with their second album, "Commit This to Memory", garnering praise and independent
buzz upon its 2005 release.

 Following the release, the band toured relentlessly over the next two years, during which time frontman Justin Pierre struggled with alcohol and substance
abuse.

 These addictions were infused into the writing process of "Even If Kills Me", which was recorded in early 2007 Stratosphere Studios in Chelsea and the
legendary Electric Lady Studios in Greenwich Village.
</item 5>

<item 6>
Commit This to Memory
Commit This to Memory is the second studio album by American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on June 7, 2005, in the United States by Epitaph Records.

 Motion City Soundtrack, formed in 1997, had first found success with their debut album, "I Am the Movie" (2002).

 The band toured in the interim years, creating positive word-of-mouth.

 In 2004, the band joined Blink-182 on the road for a string of shows, which led to their bassist, Mark Hoppus, joining the band in the studio for his first
producing effort.
</item 6>

<item 7>
From the Screen to Your Stereo
From the Screen to Your Stereo is an EP by A New Found Glory (now New Found Glory) released on 28 March 2000 by Drive-Thru Records.

 All the tracks are covers of songs from motion picture soundtracks.

 The album was pressed on 10" vinyl and released in three different colors: light marble blue, marble red, and white.

 The vinyl pressing featured a bonus track.

 In 2002, the album was released on a non-US split with the Japanese band Nicotine entitled Movie Addiction, where on this split, Nicotine also covered songs
from movies.

 In 2007 the band released a follow up, this time a full-length album titled "From the Screen to Your Stereo Part II".
</item 7>

<item 8>
Motion City Soundtrack discography
The following is the discography of Motion City Soundtrack, an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 8>

<item 9>
Candlelight Red
Candlelight Red is a rock band from Williamsport, Pennsylvania.

 They have produced two studio albums and an EP.

 Their EP "Demons" and album "Reclamation" was produced by Morgan Rose of Sevendust.
</item 9>

<item 10>
Panic Stations (album)
Panic Stations is the sixth and final studio album by American rock band Motion City Soundtrack.

 Produced by John Agnello, the album was released on September 18, 2015 by Epitaph Records.

 Following an extended touring cycle in which the group performed several past albums in full, Motion City Soundtrack lost one of its longest-serving members:
drummer Tony Thaxton, who resigned from the group in 2013.

 The band continued to tour with new drummer Claudio Rivera over the ensuing year while writing new material for a sixth album.
</item 10>
=== END ITEM id=13425aa4cf2d6dc7 ===

=== ITEM id=329c006eda769187 ===
Gold sentences and facts:
Sentence 0: Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca O'Brien and Lauren Dark.
  Fact 0.0: Mediastan is a documentary film.
  Fact 0.1: Mediastan was released in 2013.
  Fact 0.2: Mediastan is about the 2010 United States diplomatic cables leak.
  Fact 0.3: Mediastan was directed by Johannes Wahlström.
  Fact 0.4: Mediastan was produced by Julian Assange.
  Fact 0.5: Mediastan was produced by Rebecca O'Brien.
  Fact 0.6: Mediastan was produced by Lauren Dark.
Sentence 1: Julian Paul Assange ( ; born Julian Paul Hawkins, 3 July 1971) is an Australian computer programmer and the founder of WikiLeaks, an organisation which he founded in 2006.
  Fact 1.0: Julian Paul Assange was born Julian Paul Hawkins.
  Fact 1.1: Julian Paul Assange was born on 3 July 1971.
  Fact 1.2: Julian Paul Assange is Australian.
  Fact 1.3: Julian Paul Assange is a computer programmer.
  Fact 1.4: Julian Paul Assange is the founder of WikiLeaks.
  Fact 1.5: WikiLeaks is an organisation.
  Fact 1.6: Julian Paul Assange founded WikiLeaks in 2006.

Retrieved items:
<item 1>
Micom Film and TV Productions
MICOM Film and TV Productions was established in 1989 by Sreco Mihelcic in a Former Yugoslavian republic Slovenia.

 In the early 90' company joined two sons, Matjaz Mihelcic - Program director, Producer and Vasja Mihelcic - technical director and Executive Producer.

 MICOM was in 1990s the leading Slovenian creative house, producing TV shows for youth, popular entertainment TV shows and discovered many new TV concepts.

 Micom discovered many talents, different creative people and gave them possibility to express themselves.

 The most popular TV shows were: Lahkih nog naokrog (educational youth TV series), Cari zacimb (entertaining cooking TV series), Razpoke v casu (fantasy TV
series).

 They also produced documentary movies.

 One of them got in 1996 internationally release - Fight for the river and it's distributed by International Historic Films.

.

 In the beginning of 2000 company joined Marjana Mihelcic.

 It was a family based company with long tradition in show and entertainment business.
</item 1>

<item 2>
Profiles span journalists, artists, politicians, scholars and public figures: Nikola Mladenov founded Macedonia’s respected Fokus and inspired an
investigative-journalism award; Johannes Grenzfurthner leads art group monochrom; François Genoud financed Nazi networks and Middle Eastern militants; Edmund de
Unger collected Islamic art; Adel Ferdosipour hosts Iran’s Navad; Hans Aschenborn painted African wildlife; Katarina Ivanovska models and acts. Nicolae
Titulescu
</item 2>

<item 3>
The text surveys diverse religious, political, civic, and governmental groups. The Atman Foundation, a Brahma Kumaris offshoot, faced a false collective-suicide
scare in Tenerife. Montenegro’s 2006 “Eagle’s Flight” operation arrested 17 alleged militants. Greece’s anarchist Conspiracy of Fire Cells carried out bombings;
the Indian National Council supported independence efforts. Other entries cover Sahara-Sahel environmental work, a declining Indian Trotskyist group,
Arab-American civil rights, diplomacy,
</item 3>

<item 4>
Micom Film and TV Productions
It was a family based company with long tradition in show and entertainment business.

 Micom produced around 30.000 minutes of program for National TV Slovenia and other media.

 In 2002 company stopped producing because of the tragedy in the family.

 Both sons went separate way and build their own Film and TV production house.
</item 4>

<item 5>
Profiles span companies, organizations, people, and technologies: Pharma Medica conducts pharmaceutical trials in Ontario and Missouri; Genpact is a global
BPM/IT firm, formerly led by Pramod Bhasin; Summit BHC operates U.S. addiction-treatment centers. Triple Canopy merged with Academi in 2014 to form Constellis.
Other subjects include geothermal binary-cycle power, FuelCell Energy, packaging maker Krones, Polish firms, media and investment groups, retailers,
</item 5>

<item 6>
Music West Records
Music West Records was an independent record company founded by Allan Kaplan on December 1985 in San Rafael, California.

 The company was initially formed to promote Ray Lynch.

 During its run, artists released under the record company included Jim Chappell, Kenneth Nash, Chris Spheeris, and Øystein Sevåg.

 According to Gary Chappell, the manufacturer for Music West, the artists originated independently, claiming that the company's idea "has a statement that comes
directly from the
</item 6>

<item 7>
Low Transit Industries
Low Transit Industries was an independent record label founded in Melbourne, Australia in 1999 by Darren Smallman (BATTLE WORLDWIDE) and Simon Baird.

 The label toured and released bands across the world until 2010 when the partners closed operations due to Darren Smallman moving to the UK where he now runs
BATTLE WORLDWIDE which is a record label, book publisher, music publisher and PR services company.
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
Mediastan
Mediastan is a 2013 documentary film about the 2010 United States diplomatic cables leak, directed by Johannes Wahlström and produced by Julian Assange, Rebecca
O'Brien and Lauren Dark.
</item 9>

<item 10>
Pharma Medica
Pharma Medica Research Inc. is an American research and development company that works with pharmaceutical companies and performs clinical trials in the
pharmaceutical and biotechnology industry.

 The company is based in Toronto, Canada with headquarters in Mississauga, Ontario.

 Pharma Medica has locations in Scarborough, Ontario and St. Charles, Missouri.
</item 10>
=== END ITEM id=329c006eda769187 ===

=== ITEM id=7c8f3a09943946b4 ===
Gold sentences and facts:
Sentence 0: Rivermaya is a Filipino rock band.
  Fact 0.0: Rivermaya is a Filipino rock band.
Sentence 1: Kodaline ( ) are an Irish band.
  Fact 1.0: Kodaline is an Irish band.

Retrieved items:
<item 1>
Raindancer
Raindancer was an electroacoustic rock band formed in Newport, Wales in the early 1990s.

 The band featured Grant Nicholas and Jon Lee, who later became founding members of Feeder.
</item 1>

<item 2>
Kodaline
Kodaline ( ) are an Irish band.

 Originally known as 21 Demands, in 2012 the band changed their name to Kodaline.
</item 2>

<item 3>
Honest (Kodaline song)
"Honest" is a song by Dublin-based alternative rock quartet Kodaline.

 The song was released on 1 February 2015 as the lead single from the band's second studio album, "Coming Up for Air" (2015).

 "Honest" became the band's third top 10 single in their home country, Ireland, following "High Hopes" (2013) and "Love Like This" (2013).
</item 3>

<item 4>
Inkwell (band)
Inkwell is an indie rock band from Winter Park, Florida.

 They have released three full length albums and one EP, as well as a collaboration with former techno artist Floorboard.

 They are currently signed with One Eleven Records.

 The band consists of two members, Travis Adams, former frontman from My Hotel Year and Davey Pierce who has worked with of Montreal.

 The band recently performed a small tour with of Montreal along the east coast.

 Their most recent album "Rivers of Blood and Sadness, or Maybe Happy" was released on iTunes music store April 21, 2009.
</item 4>

<item 5>
The material profiles albums, singles and bands across rock, folk, electronic and punk. Highlights include Parlor Hawk’s 2010 debut; Freestylers’ 2006
“Painkiller” with Pendulum and SirReal; James Ferraro’s 2011 conceptually retro “Far Side Virtual”; and Polish punk band Fliper. It covers Radford, Kodaline,
Richard Ashcroft’s RPA, and the Lumineers, plus Richard Hawley’s Sheffield-inspired albums
</item 5>

<item 6>
Rivermaya
Rivermaya is a Filipino rock band.

 Formed in 1994, it is one of several bands who spearheaded the 1990s Philippine alternative rock explosion.

 Rivermaya is currently composed of original members Mark Escueta and Nathan Azarcon, together with Mike Elgar and Ryan Peralta.

 Former original members include Rico Blanco, who had been the original songwriter of the band and vocalist Bamboo Mañalac, who later formed the band Bamboo and
later went on his solo career.

 Rivermaya is listed as the twentieth biggest-selling artists/act in the Philippines as of present.
</item 6>

<item 7>
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 7>

<item 8>
Leeds band Vib Gyor took its name from rainbow colors ordered by wavelength. Its debut single “Fallen” (July 2006) gained acclaim and radio play. The Secret EP
(March 2007), recorded in Bath by Paul Corkett, includes a hidden instrumental; The White EP followed in November, recorded at Rockfield by Ken Thomas. Leeds
alternative-rock band Distortion Mirrors formed in 2011. Other releases include Edward Sharpe and the Magnetic Zeros’ 201
</item 8>

<item 9>
Native Tour
The Native Tour (also known as the Native Summer Tour in North America or Native Fall Tour in Europe) is the third headlining concert tour by American pop-rock
band, OneRepublic in support of their third studio album, "Native" (2013).

 OneRepublic were joined on the Native Summer leg of the tour by, The Script and American Authors, and on the European "Native Fall Tour" leg by Kongos.

 The tour has traveled across five continents; Asia, Europe, Oceania, North America, and Africa.

 The tour began on April 2, 2013, in Milan, Italy and finished on September 20, 2015, in São Paulo, Brazil.
</item 9>

<item 10>
Vib Gyor
Vib Gyor was a band that was formed in Leeds, England.

 The name was derived from the first letters of all the colours in the rainbow, in order from the shortest to longest wavelengths.
</item 10>
=== END ITEM id=7c8f3a09943946b4 ===

=== ITEM id=fb15d26e66d62bb2 ===
Gold sentences and facts:
Sentence 0: Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.
  Fact 0.0: Margaret Landon was born September 7, 1903.
  Fact 0.1: Margaret Landon died December 4, 1993.
  Fact 0.2: Margaret Landon was an American writer.
  Fact 0.3: Margaret Landon is best remembered for 'Anna and the King of Siam'.
  Fact 0.4: 'Anna and the King of Siam' is Margaret Landon's best-selling novel, published in 1944.
  Fact 0.5: 'Anna and the King of Siam' is a novel about the life of Anna Leonowens.
  Fact 0.6: 'Anna and the King of Siam' eventually sold over a million copies.
  Fact 0.7: 'Anna and the King of Siam' was translated into more than twenty languages.
Sentence 1:  In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.
  Fact 1.0: In 1950, Margaret Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II.
  Fact 1.1: Richard Rodgers and Oscar Hammerstein II created the musical 'The King and I' from Margaret Landon's book.
Sentence 2:  A later work, "Never Dies the Dream", appeared in 1949.
  Fact 2.0: 'Never Dies the Dream' is a later work.
  Fact 2.1: 'Never Dies the Dream' appeared in 1949.
Sentence 3: Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.
  Fact 3.0: Dame Elizabeth Rosemond Taylor, DBE, was born February 27, 1932.
  Fact 3.1: Dame Elizabeth Rosemond Taylor, DBE, died March 23, 2011.
  Fact 3.2: Dame Elizabeth Rosemond Taylor was a British-American actress.
  Fact 3.3: Dame Elizabeth Rosemond Taylor was a businesswoman.
  Fact 3.4: Dame Elizabeth Rosemond Taylor was a humanitarian.
Sentence 4:  She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.
  Fact 4.0: Elizabeth Taylor began her career as a child actress in the early 1940s.
  Fact 4.1: Elizabeth Taylor was one of the most popular stars of classical Hollywood cinema in the 1950s.
Sentence 5:  She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.
  Fact 5.0: Elizabeth Taylor continued her career successfully into the 1960s.
  Fact 5.1: Elizabeth Taylor remained a well known public figure for the rest of her life.
Sentence 6:  In 1999, the American Film Institute named her the seventh-greatest female screen legend.
  Fact 6.0: In 1999, the American Film Institute named Elizabeth Taylor the seventh-greatest female screen legend.

Retrieved items:
<item 1>
These Old Broads
These Old Broads is a 2001 television film written by Carrie Fisher and starring her mother Debbie Reynolds, as well as Shirley MacLaine, Joan Collins, and
Elizabeth Taylor in her final film role.

 In a 2001 BBC "Omnibus" documentary about Elizabeth Taylor, Shirley MacLaine says that Julie Andrews and Lauren Bacall were originally planned to be in the
movie.

 The role of Miriam Hodges was originally offered to June Allyson.
</item 1>

<item 2>
Elizabeth Taylor
Dame Elizabeth Rosemond Taylor, DBE (February 27, 1932 – March 23, 2011) was a British-American actress, businesswoman, and humanitarian.

 She began her career as a child actress in the early 1940s, and was one of the most popular stars of classical Hollywood cinema in the 1950s.

 She continued her career successfully into the 1960s, and remained a well known public figure for the rest of her life.

 In 1999, the American Film Institute named her the seventh-greatest female screen legend.
</item 2>

<item 3>
Laura Bayley
Laura Eugenia Bayley (1864 – 1938) was a British actress and filmmaker, active in the Brighton School of early cinema pioneers.

 Born in Ramsgate, Bayley performed onstage in Victorian burlesques, revues, and pantomimes, often with her three sisters.

 After marrying the showman George Albert Smith, she entered the world of early experiments with motion picture film; she played main roles in many of the most
important films Smith made between 1897 and 1903, including "The Kiss in the Tunnel" (1899) and "Mary Jane's Mishap" (1903).
</item 3>

<item 4>
At Land
At Land (1944) is a 15-minute silent experimental film written, directed by, and starring Maya Deren.

 It has a dream-like narrative in which a woman, played by Deren, is washed up on a beach and goes on a strange journey encountering other people and other
versions of herself.

 Deren once said that the film is about the struggle to maintain one's personal identity.
</item 4>

<item 5>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 5>

<item 6>
Cate Blanchett
Catherine Elise Blanchett, {'1': ", '2': ", '3': ", '4': "} ( ; born 14 May 1969) is an Australian actress and theatre director.

 She has received international acclaim and many accolades, including two Academy Awards, three Golden Globe Awards, three BAFTA Awards, six AACTA Awards, and
three Screen Actors Guild Awards. Blanchett came to international attention for her role as Elizabeth I of England in Shekhar Kapur's 1998 film "Elizabeth", for
which she won the BAFTA Award for Best Actress, the Golden Globe Award, and earned her first Academy Award for Best Actress nomination.

 Her portrayal of Katharine Hepburn in Martin Scorsese's 2004 film "The Aviator" brought her critical acclaim and many accolades, including the Academy Award
for Best Actress in a Supporting Role, making her the only actor to win an Oscar for portraying another Oscar-winning actor.

 In 2013, she starred as Jasmine Francis in Woody Allen's "Blue Jasmine", for which she won numerous accolades including the Academy Award for Best Actress.
</item 6>

<item 7>
Margaret Landon
Margaret Landon (September 7, 1903 – December 4, 1993) was an American writer best remembered for "Anna and the King of Siam", her best-selling 1944 novel of
the life of Anna Leonowens which eventually sold over a million copies and was translated into more than twenty languages.

 In 1950, Landon sold the musical play rights to Richard Rodgers and Oscar Hammerstein II, who created the musical "The King and I" from her book.

 A later work, "Never Dies the Dream", appeared in 1949.
</item 7>

<item 8>
Anna and the King of Siam (novel)
Anna and the King is a 1944 semi-fictionalized biographical novel by Margaret Landon.
</item 8>

<item 9>
Cher filmography
Throughout her acting career, Cher has mainly in comedy, drama, and romance films.

 She has appeared in thirteen films, including two as a cameo.

 She has also appeared in one starring theater role, numerous television commercials and directed a piece of the motion picture "If These Walls Could Talk" in
1996 and some of her music videos of the Geffen-era in late 1980s and in early 1990s.

 Cher has starred in various international television commercials, as well as high-profile print advertising for Lori Davis (1992).

 Before she started her film career, she had a couple of hits in the 1960s, as a solo artist, and with her ex-husband Sonny Bono as the couple Sonny & Cher.
</item 9>

<item 10>
Sam Taylor (director)
Sam Taylor (August 13, 1895, New York City – March 6, 1958, Santa Monica) was a film director, screenwriter, and producer, most active in the silent film era.

 Taylor is best known for his comedic directorial work with Harold Lloyd and Mary Pickford, and also later worked with Laurel and Hardy.
</item 10>
=== END ITEM id=fb15d26e66d62bb2 ===

=== ITEM id=59483bd24bb751f6 ===
Gold sentences and facts:
Sentence 0: Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.
  Fact 0.0: Like many of Marivaux's plays, La Surprise de l'amour makes use of stock characters from the Commedia dell'arte.
  Fact 0.1: Many of Marivaux's plays make use of stock characters from the Commedia dell'arte.
Sentence 1:  In this play, Arlequin and Columbine are featured.
  Fact 1.0: In La Surprise de l'amour, Arlequin is featured.
  Fact 1.1: In La Surprise de l'amour, Columbine is featured.
Sentence 2: Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the Italian "Commedia dell'arte".
  Fact 2.0: Harlequin is the best-known of the zanni, the comic servant characters from the Italian Commedia dell'arte.
  Fact 2.1: Harlequin is called Arlecchino in Italian.
  Fact 2.2: Harlequin is called Arlequin in French.
  Fact 2.3: Harlequin is called Harlequin in Old French.

Retrieved items:
<item 1>
Gilles (stock character)
Gilles (] )—sometimes Gille—is a stock character of French farce and Commedia dell'Arte.

 He enjoyed his greatest vogue in 18th-century France, in entertainments both at the fairgrounds of the capital and in private and public theaters, though his
origins can be traced back to the 17th century and, possibly, the century previous.

 A "zanni", or comic servant, he is a type of bungling clown, stupid, credulous, and lewd—a character that shares little, problematically, with the sensitive
figure in Watteau's famous portrait that, until the latter half of the 20th century, bore his name alone.

 Gilles fades from view in the 19th century, to persist in the 20th and 21st as the Belgian Gilles of Binche Carnival.
</item 1>

<item 2>
Commedia dell'arte
Commedia dell'arte (] , comedy of the profession) was an early form of professional theatre, originating from Italy, that was popular in Europe from the 16th
through the 18th century.

 "Commedia dell'arte" also is known as "commedia alla maschera", "commedia improvviso", and "commedia dell'arte all'improvviso".

 "Commedia" is a form of theatre characterized by masked "types" which began in Italy in the 16th century and was responsible for the advent of actresses
(Isabella Andreini) and improvised performances based on sketches or scenarios.

 A "commedia", such as "The Tooth Puller", is both scripted and improvised.

 Characters' entrances and exits are scripted.

 A special characteristic of "commedia dell'arte" are the "lazzi".

 A "lazzo" is a joke or "something foolish" or "witty".

 Another characteristic of "commedia dell'arte" is pantomime, which is mostly used by the character Arlecchino (Harlequin).
</item 2>

<item 3>
Armand Castelmary
Armand Castelmary, real name Comte Armand de Castan, born Toulouse 16 August 1834, died New York City 10 February 1897, was a French operatic bass.

 He created roles in three major premieres at the Paris Opera – Don Diego in "L'Africaine" by Meyerbeer (1865), the Monk in Verdi's "Don Carlos" (1867), and
Horatio in Ambroise Thomas's "Hamlet" (1868).

 Castelmary also appeared at opera houses in England and the United States, and died onstage at the Metropolitan Opera House, New York, during a performance of
"Martha" by Friedrich von Flotow.
</item 3>

<item 4>
Niccolò Barbieri
Niccolò (or Nicolò) Barbieri (Vercelli, 1586 - 1641) was an Italian writer and actor of the commedia dell'arte theatrical genre.

 He was also known as Beltrame di Milano ("Beltrame of Milan") in reference to one of his most popular characters, Beltrame; this was the main character of one
of Barbieri's best known plays, "L'inavertito", which is known to have inspired Moliere's "L'Étourdi ou Les Contretemps" ("The Blunderer").

 Besides popularizing Beltrame, Barbieri is also credited with creating another "commedia dell'arte" mask, Scapino.
</item 4>

<item 5>
Harlequin
Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the
Italian "Commedia dell'arte".

 The role is traditionally believed to have been introduced by Zan Ganassa in the late 16th century, and was definitively popularized by the Italian actor
Tristano Martinelli in Paris in 1584–1585 and became a stock character after Martinelli's death in 1630.
</item 5>

<item 6>
Pedrolino
Pedrolino is a "primo zanni", or comic servant, of the "Commedia dell'Arte"; the name is a hypocorism of "Pedro" (Peter), via the suffix "-lino".

 The character made its first appearance in the last quarter of the 16th century, apparently as the invention of the actor with whom the role was to be long
identified, Giovanni Pellesini.

 Contemporary illustrations suggest that his white blouse and trousers constituted "a variant of the typical "zanni" suit", and his Bergamasque dialect marked
him as a member of the "low" rustic class.

 But if his costume and social station were without distinction, his dramatic role was certainly not: as a multifaceted "first" "zanni", his character was—and
still is—rich in comic incongruities.
</item 6>

<item 7>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 7>

<item 8>
Tartaglia (commedia dell'arte)
Tartaglia is a dainty character in the Commedia dell'arte.

 He is farsighted and with a minor stutter (hence his name; cf. Spanish tartamudear), he is usually classed as one of the group of old characters (vecchio) who
appears in many scenarios as one of the lovers (innamorati).

 His social status varies; he is sometimes a bailiff, lawyer, notary or chemist.

 Dramatist Carlo Gozzi turned him into a statesman, and so he remained thereafter.

 Tartaglia wears a large felt hat, an enormous cloak, oversized boots, a long sword, a giant moustache and a cardboard nose.

 He usually represents the lower working class but at times the middle or upper class in the commedia dell'arte.

 In the opera "Le maschere" by Mascagni, one of the characters is Tartaglia, a stuttering servant.
</item 8>

<item 9>
La Surprise de l'amour
Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.

 In this play, Arlequin and Columbine are featured.
</item 9>

<item 10>
Columbina
Columbina (in Italian Colombina, meaning "little dove"; in French and English Colombine) is a stock character in the "Commedia dell'Arte".

 She is Harlequin's mistress, a comic servant playing the tricky slave type, and wife of Pierrot.

 Rudlin and Crick use the Italian spelling Colombina in "Commedia dell'arte: A Handbook for Troupes".
</item 10>
=== END ITEM id=59483bd24bb751f6 ===

=== ITEM id=56f84b7a5cc42c7b ===
Gold sentences and facts:
Sentence 0: Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.
  Fact 0.0: Henry Louis Mencken lived from September 12, 1880 to January 29, 1956.
  Fact 0.1: Henry Louis Mencken was an American journalist.
  Fact 0.2: Henry Louis Mencken was an American satirist.
  Fact 0.3: Henry Louis Mencken was an American cultural critic.
  Fact 0.4: Henry Louis Mencken was a scholar of American English.
Sentence 1: Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.
  Fact 1.0: Albert Camus lived from 7 November 1913 to 4 January 1960.
  Fact 1.1: Albert Camus was a French philosopher.
  Fact 1.2: Albert Camus was an author.
  Fact 1.3: Albert Camus was a journalist.

Retrieved items:
<item 1>
The entries profile notable figures across philanthropy, arts, journalism, religion, academia, business, and politics: Edmund R. Harris funded Preston
institutions; Edmund de Unger built the Keir Islamic-art collection, now loaned to Dallas; Mencken was a major American satirist; and H. Donald Wilson helped
create LexisNexis. Others include leaders, scholars, artists, and advocates such as George Albert Smith, Andrew Benjamin, Inderpal Grewal, Emeka Es
</item 1>

<item 2>
Albert Camus
Albert Camus (] ; 7 November 1913 – 4 January 1960) was a French philosopher, author, and journalist.

 His views contributed to the rise of the philosophy known as absurdism.

 He wrote in his essay "The Rebel" that his whole life was devoted to opposing the philosophy of nihilism while still delving deeply into individual freedom.

 He won the Nobel Prize in Literature in 1957.
</item 2>

<item 3>
The American Mercury
The American Mercury was an American magazine published from 1924 to 1981.

 It was founded as the brainchild of H. L. Mencken and drama critic George Jean Nathan.

 The magazine featured writing by some of the most important writers in the United States through the 1920s and 1930s.

 After a change in ownership in the 1940s, the magazine attracted conservative writers.

 A second change in ownership a decade later turned the magazine into a virulently anti-Semitic publication.

 It was published monthly in New York City.

 The magazine went out of business in 1981, having spent the last 25 years of its existence in decline and controversy.
</item 3>

<item 4>
Reflections on the Guillotine
"Reflections on the Guillotine" is an extended essay written in 1957 by Albert Camus.

 In the essay Camus takes an uncompromising position for the abolition of the death penalty.

 Camus's view is similar to that of Cesare Beccaria and the Marquis de Sade, the latter having also argued that murder premeditated and carried out by the state
was the worst kind.

 Camus states that he does not base his argument on sympathy for the convicted but on logical grounds and on proven statistics.

 Camus also argues that capital punishment is an easy option for the government where remedy and reform may be possible.
</item 4>

<item 5>
Marcel J. Melançon
Marcel J. Mélançon is a Canadian philosopher and scientist.

 He was born in Saint-Barnabé, Quebec, in 1938, and is mostly known for his book on the French philosopher Albert Camus "Albert Camus, An Analysis of his
thought" (Albert Camus.

 Analyse de sa pensée).

 He has mostly worked together with the biologist Richard D. Lambert.
</item 5>

<item 6>
The entries profile writers, artists, scholars, and public figures across literature, journalism, film, and politics. Highlights include Thomas Merton, the
Catholic monk and writer honored by a peace award and studied by an international society; W. Somerset Maugham, whose namesake prize supports young writers’
travel; and notable authors such as Mencken, Orwell, Updike, Malamud, Chatwin, and Gass. Other topics include literary awards, bibliographies, influential
editors, Camb
</item 6>

<item 7>
The Rebel (book)
The Rebel (French: "L'Homme révolté" ) is a 1951 book-length essay by Albert Camus, which treats both the metaphysical and the historical development of
rebellion and revolution in societies, especially Western Europe.

 Camus relates writers and artists as diverse as Epicurus and Lucretius, Marquis de Sade, Georg Wilhelm Friedrich Hegel, Fyodor Dostoyevsky, Friedrich
Nietzsche, Max Stirner, André Breton, and others in an integrated, historical portrait of man in revolt.

 Examining both rebellion and revolt, which may be seen as the same phenomenon in personal and social frames, Camus examines several 'countercultural' figures
and movements from the history of Western thought and art, noting the importance of each in the overall development of revolutionary thought and philosophy.
</item 7>

<item 8>
The entries cover pornographic publisher Auguste Brancart; Albert Camus, his absurdist philosophy, Nobel Prize, fiction and essays opposing nihilism and capital
punishment; and his biographers. Other topics include the disputed Smiley Face murder theory; Joe McGinniss, Jeffrey MacDonald and the Fatal Vision controversy;
the Guantánamo memoir Enemy Combatant; New Zealand’s Parker–Hulme case and Mazengarb Report; the Lindbergh kidnapping; Nazi collaborator Christian de
</item 8>

<item 9>
H. L. Mencken
Henry Louis Mencken (September 12, 1880 – January 29, 1956) was an American journalist, satirist, cultural critic and scholar of American English.

 Known as the "Sage of Baltimore", he is regarded as one of the most influential American writers and prose stylists of the first half of the twentieth century.

 He commented widely on the social scene, literature, music, prominent politicians and contemporary movements.

 His satirical reporting on the Scopes trial, which he dubbed the "Monkey Trial", also gained him attention.
</item 9>

<item 10>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 10>
=== END ITEM id=56f84b7a5cc42c7b ===

=== ITEM id=468f35adb379e00f ===
Gold sentences and facts:
Sentence 0: The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
  Fact 0.0: The Consul is an opera.
  Fact 0.1: The Consul is an opera in three acts.
  Fact 0.2: The Consul has music by Gian Carlo Menotti.
  Fact 0.3: The Consul has a libretto by Gian Carlo Menotti.
  Fact 0.4: The Consul is Gian Carlo Menotti's first full-length opera.
Sentence 1:  He completed the music for the opera while living in Zurich in 1916.
  Fact 1.0: Gian Carlo Menotti completed the music for The Consul.
  Fact 1.1: Gian Carlo Menotti completed the music for The Consul in 1916.
  Fact 1.2: Gian Carlo Menotti was living in Zurich in 1916.
  Fact 1.3: Gian Carlo Menotti completed the music for The Consul while living in Zurich.

Retrieved items:
<item 1>
Commedia dell'arte
Commedia dell'arte (] , comedy of the profession) was an early form of professional theatre, originating from Italy, that was popular in Europe from the 16th
through the 18th century.

 "Commedia dell'arte" also is known as "commedia alla maschera", "commedia improvviso", and "commedia dell'arte all'improvviso".

 "Commedia" is a form of theatre characterized by masked "types" which began in Italy in the 16th century and was responsible for the advent of actresses
(Isabella Andreini) and improvised performances based on sketches or scenarios.

 A "commedia", such as "The Tooth Puller", is both scripted and improvised.

 Characters' entrances and exits are scripted.

 A special characteristic of "commedia dell'arte" are the "lazzi".

 A "lazzo" is a joke or "something foolish" or "witty".

 Another characteristic of "commedia dell'arte" is pantomime, which is mostly used by the character Arlecchino (Harlequin).
</item 1>

<item 2>
Pedrolino
Pedrolino is a "primo zanni", or comic servant, of the "Commedia dell'Arte"; the name is a hypocorism of "Pedro" (Peter), via the suffix "-lino".

 The character made its first appearance in the last quarter of the 16th century, apparently as the invention of the actor with whom the role was to be long
identified, Giovanni Pellesini.

 Contemporary illustrations suggest that his white blouse and trousers constituted "a variant of the typical "zanni" suit", and his Bergamasque dialect marked
him as a member of the "low" rustic class.

 But if his costume and social station were without distinction, his dramatic role was certainly not: as a multifaceted "first" "zanni", his character was—and
still is—rich in comic incongruities.
</item 2>

<item 3>
Arlecchino (opera)
Arlecchino, "oder Die Fenster" ("Harlequin, or The Windows", is a one-act opera with spoken dialog by Ferruccio Busoni, with a libretto in German, composed in
1913.

 He completed the music for the opera while living in Zurich in 1916.

 It is a number opera written in neo-classical style and includes ironic allusions to operatic conventions and situations typical of the late 18th and early
19th centuries.

 It even includes a parody of a duel.
</item 3>

<item 4>
Graph query row:
work: The Consul
kind: opera
relationship: CREATED_BY
person: Gian Carlo Menotti
</item 4>

<item 5>
Tartaglia (commedia dell'arte)
Tartaglia is a dainty character in the Commedia dell'arte.

 He is farsighted and with a minor stutter (hence his name; cf. Spanish tartamudear), he is usually classed as one of the group of old characters (vecchio) who
appears in many scenarios as one of the lovers (innamorati).

 His social status varies; he is sometimes a bailiff, lawyer, notary or chemist.

 Dramatist Carlo Gozzi turned him into a statesman, and so he remained thereafter.

 Tartaglia wears a large felt hat, an enormous cloak, oversized boots, a long sword, a giant moustache and a cardboard nose.

 He usually represents the lower working class but at times the middle or upper class in the commedia dell'arte.

 In the opera "Le maschere" by Mascagni, one of the characters is Tartaglia, a stuttering servant.
</item 5>

<item 6>
The Consul
The Consul is an opera in three acts with music and libretto by Gian Carlo Menotti, his first full-length opera.
</item 6>

<item 7>
Graph query row:
work: Arlecchino, oder Die Fenster
kind: one-act opera
relationship: CREATED_BY
person: Ferruccio Busoni
</item 7>

<item 8>
The Carp (opera)
The Carp is a one-act comic opera (styled "a whimsicality") with a libretto by Frank Desprez and music by Alfred Cellier.

 It was first produced at the Savoy Theatre from 13 February 1886 to 19 January 1887, as a companion piece to "The Mikado".

 It was then revived as companion to "Ruddigore" from 21 February 1887 to 5 November 1887.

 The piece also toured throughout 1888.
</item 8>

<item 9>
La Surprise de l'amour
Like many of Marivaux's , "La Surprise de l'amour" makes use of stock characters from the Commedia dell'arte.

 In this play, Arlequin and Columbine are featured.
</item 9>

<item 10>
Harlequin
Harlequin ( ; Italian: "Arlecchino" ] , French: "Arlequin" ] , Old French "Harlequin") is the best-known of the "zanni" or comic servant characters from the
Italian "Commedia dell'arte".

 The role is traditionally believed to have been introduced by Zan Ganassa in the late 16th century, and was definitively popularized by the Italian actor
Tristano Martinelli in Paris in 1584–1585 and became a stock character after Martinelli's death in 1630.
</item 10>
=== END ITEM id=468f35adb379e00f ===

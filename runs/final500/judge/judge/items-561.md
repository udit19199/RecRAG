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

=== ITEM id=300ab13ca5acc250 ===
Gold sentences and facts:
Sentence 0:  The trains were built between 1994 and 2009 – for the most part by Talbot, part of Bombardier Inc., with some railcars built by De Dietrich.
  Fact 0.0: The trains were built between 1994 and 2009.
  Fact 0.1: The trains were built for the most part by Talbot.
  Fact 0.2: Talbot is part of Bombardier Inc.
  Fact 0.3: Some railcars were built by De Dietrich.
Sentence 1: Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.
  Fact 1.0: Bombardier Inc. is a multinational aerospace and transportation company.
  Fact 1.1: Bombardier Inc. is based in Montreal, Quebec, Canada.

Retrieved items:
<item 1>
Transportation Building
The Transportation Building is a 44-story office building located at 225 Broadway on the corner of Barclay Street in the Civic Center neighborhood of Lower
Manhattan in New York City.

 It also carries the address 2-4 Barclay Street.

 It was built in 1927 and was designed by the architecture firm of York & Sawyer, in the Renaissance Revival style, using setbacks common to skyscrapers built
after the adoption of the 1916 Zoning Resolution.

 It sits across Barclay Street from the Woolworth Building.
</item 1>

<item 2>
Shaver Transportation Company
The Shaver Transportation Company is an inland water freight transportation company based in Portland, Oregon, United States.

 The company was founded in 1880 and played a major role in the development of freight transport in the Portland area and along the Columbia.
</item 2>

<item 3>
Bombardier Inc.
Bombardier Inc. (] ) is a multinational aerospace and transportation company based in Montreal, Quebec, Canada.

 Starting as a maker of snow machines or snowmobiles, over the years it has grown into a large manufacturer of regional airliners, business jets, mass
transportation equipment, and recreational equipment, and a provider of financial services.
</item 3>

<item 4>
Transportation in Jacksonville, Florida
The Jacksonville transportation network includes ground, air, and sea options for passenger and freight transit.

 The Jacksonville Port Authority (Jaxport) operates the Port of Jacksonville, which includes container shipping facilities at "Blount Island Marine Terminal",
the "Talleyrand Marine Terminal" and the "Dames Point Marine Terminal".

 Jacksonville Aviation Authority managers Jacksonville International Airport in Northside, as well as several smaller airports.

 The Jacksonville Transportation Authority (JTA) operates bus, people mover, and park-n-ride services throughout the city and region.

 A major bus terminal at the intermodal Rosa Parks Transit Station serves as JTA's main transit hub.

 Various intercity bus companies terminate near Central Station.

 Amtrak operates passenger rail service to and from major cities throughout North America.

 The city is bisected by major highways, I-95 and I-10, I-295 creates a full beltway around the city.
</item 4>

<item 5>
NS VIRM
VIRM trains, full name Verlengd InterRegio Materieel ("Lengthened interregional rolling stock"), are a series of electric multiple unit (EMU) double-deck trains
operated by Nederlandse Spoorwegen or NS ("Dutch Railways"), the principal railway operator in the Netherlands.

 NS has 178 of these double-deckers – 98 four-carriage sets, and 80 six-carriage sets.

 The trains were built between 1994 and 2009 – for the most part by Talbot, part of Bombardier Inc., with some railcars built by De Dietrich.
</item 5>

<item 6>
Merchants Transportation Company
The Merchants Transportation Company was a shipping firm that operated on Puget Sound from 1905 to 1929.

 This company should not be confused with the similarly named Merchants Transportation Company of Olympia, formed in 1874 and a completely separate firm.
</item 6>

<item 7>
New York Transportation Company
The New York Transportation Company (originally New York Electrical Vehicle Transportation Company) was a company incorporated in New Jersey in 1899 as the 'New
York Electrical Vehicle Transportation Company' which changed its name to the "New York Transportation Company" in 1902 and was wound up in 1936.

 From 1922 it was controlled by the 'Fifth Avenue Bus Securities Corporation' which was incorporated in Delaware in 1922 which was controlled by The Omnibus
Corporation also incorporated in Delaware.
</item 7>

<item 8>
Toland, Pennsylvania
Toland is less than 1/4 of a mile long, and it is located less than half a mile from the Appalachian trail crossing.

 The closest town is Mount Holly Springs, Pennsylvania located (3 miles north of Toland).

 The village is roughly 10 miles south of Carlisle, Pennsylvania, 7 miles south east of Boiling Springs, Pennsylvania, and 15 miles South of Gettysburg,
Pennsylvania.

 The only business in Toland is the Cherokee Campground formerly known as the Tagg Run Campground.

 The campground restaurant is now closed.

 Toland Mission is a small non-denominational church that can hold up to 74 persons.

 It was originally built by the owner of Beetem Lumber Company in Carlisle for the families of the community of Toland.

 A one-room school house near the church originally served the community, but it was converted to a home when residents' children were transported to a township
school.

 Local people in Toland usually travel to the town of Mount Holly Springs for necessities such as gas, food, and toiletries.
</item 8>

<item 9>
Toland, Pennsylvania
Toland is an unincorporated community in Cumberland County, Pennsylvania, United States.

 The very small village is located on Pine Grove Road, east of Mountain Creek Campground.

 Michaux State Forest, the site of Laurel Lake, Fuller Lake and Pine Grove Furnace State Park lie a few miles to the west near the intersection of Pine Grove
Road and PA route 233.

 Toland has a mailing address of Gardners Pennsylvania, because the size of the community doesn't warrant a post-office, nor an official incorporated name.

 What is now a sand pit operation was originally a clay bank mining operation that was the reason for the location of the village of Toland.

 Less than 50 people reside in the close-knit community, which was built for the clay bank company workers in the first quarter of the 20th century.

 The original community of Toland, Pennsylvania consisted of 11 duplex houses, built side by side along Pine Grove Road, with less than 0.17 of an acre of land
to each.

 Additional homes have been built since.

 When the community was originally built there was only a common well with a hand pump for all of the families to draw from.
</item 9>

<item 10>
Northern Transportation Company
Northern Transportation Company Limited (NTCL) is a marine transportation company in the Canadian and American Arctic owned by Norterra, a holding company that
is owned by the Inuvialuit of the Northwest Territories.

 While primarily a marine freight hauler on Canadian rivers, like the Mackenzie and the Hay and along the Arctic coast of Canada, they are also a petroleum
wholesaler.

 Its head office is now in the town of Hay River, Northwest Territories.

 Today, it primarily utilizes tugs and barges.
</item 10>
=== END ITEM id=300ab13ca5acc250 ===

=== ITEM id=8ecf0c11486b309f ===
Gold sentences and facts:
Sentence 0:  There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.
  Fact 0.0: There is no optimal strategy involved in the game.
  Fact 0.1: Play in the game entirely rests upon a random drawing of the cards.
Sentence 1: Diplomacy is a strategic board game created by Allan B. Calhamer in 1954 and released commercially in 1959.
  Fact 1.0: Diplomacy is a strategic board game.
  Fact 1.1: Diplomacy was created by Allan B. Calhamer.
  Fact 1.2: Diplomacy was created in 1954.
  Fact 1.3: Diplomacy was released commercially in 1959.
Sentence 2:  Its main distinctions from most board wargames are its negotiation phases (players spend much of their time forming and betraying alliances with other players and forming beneficial strategies) and the absence of dice and other game elements that produce random effects.
  Fact 2.0: Diplomacy's main distinctions from most board wargames include its negotiation phases.
  Fact 2.1: In Diplomacy's negotiation phases, players spend much of their time forming and betraying alliances with other players.
  Fact 2.2: In Diplomacy's negotiation phases, players spend much of their time forming beneficial strategies.
  Fact 2.3: Diplomacy's main distinctions from most board wargames include the absence of dice and other game elements that produce random effects.

Retrieved items:
<item 1>
Dungeons &amp; Dragons Basic Game
The Dungeons & Dragons Basic Game is an introductory version of "Dungeons & Dragons" ("D&D") role-playing game packaged in the form of a board game.

 The original game was released in 2004 by Wizards of the Coast and was designed by Jonathan Tweet, one of the "D&D" 3rd edition designers.

 A new version of this game was released in September 2006.
</item 1>

<item 2>
Simultaneous action selection
Simultaneous action selection, or SAS, is a game mechanic that occurs when players of a game take action (such as moving their pieces) at the same time.

 An example of a game that uses this type of movement is the game Diplomacy.

 Typically, a "secret yet binding" method of committing to one's move is necessary, so that as players' moves are revealed and implemented, others do not change
their moves in light of the new information.

 Thus, in Diplomacy, players write down their moves and then reveal them simultaneously.

 Because no player gets the first move, this potentially arbitrary source of advantage is not present.

 It is also possible for simultaneous movement games to proceed relatively quickly, because players are acting at the same time, rather than waiting for their
turn.

 Simultaneous action selection is easily implemented in card games such as Apples to Apples in which players simply select cards and throw them face-down into
the center.
</item 2>

<item 3>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 3>

<item 4>
Attack! (board game)
Attack!

 is a board game created by Glenn Drover and published by Eagle Games in 2003.

 It is a light war game that is midway between "Risk" and "Axis and Allies" in complexity.

 The game is loosely set in the 1930s and includes plastic pieces featuring tanks, planes, infantry, and artillery.

 "Attack!"

 won the Origins Award for "Best Historical Game 2003".
</item 4>

<item 5>
Diplomacy (game)
Diplomacy is a strategic board game created by Allan B. Calhamer in 1954 and released commercially in 1959.

 Its main distinctions from most board wargames are its negotiation phases (players spend much of their time forming and betraying alliances with other players
and forming beneficial strategies) and the absence of dice and other game elements that produce random effects.

 Set in Europe before the beginning of World War I, "Diplomacy" is played by two to seven players, each controlling the armed forces of a major European power
(or, with fewer players, multiple powers).

 Each player aims to move his or her few starting units and defeat those of others to win possession of a majority of strategic cities and provinces marked as
"supply centers" on the map; these supply centers allow players who control them to produce more units.
</item 5>

<item 6>
Uncle Wiggily
Uncle Wiggily Longears is the main character of a series of children's stories by American author Howard R. Garis.

 He began writing the stories for the "Newark News" in 1910.

 Garis penned an Uncle Wiggily story every day (except Sundays) for more than 30 years, and published 79 books within the author's lifetime.

 According to his obituary in the "Chicago Tribune", a walk in the woods in Verona, New Jersey was his inspiration.

 The books featured work by several illustrators, notably Lansing Campbell.

 Other illustrators of the series included George L. Carlson, Louis Wisa, Elmer Rache, Edward Bloomfield, Lang Campbell and Mary and Wallace Stover.
</item 6>

<item 7>
Can't Stop (board game)
Can't Stop is a board game designed by Sid Sackson originally published by Parker Brothers in 1980, and was long out of print in the United States.

 It was reprinted by Face 2 Face Games in 2007.

 An iOS version was developed by Playdek and released in 2012.

 The goal of the game is to "claim" (get to the top of) three of the columns before any of the other players can.

 But the more that the player risks rolling the dice during a turn, the greater the risk of losing the advances made during that turn.
</item 7>

<item 8>
Uncle Wiggily (board game)
Uncle Wiggily Game is a track board game based on a character in a series of children's books by American writer Howard Roger Garis.

 The game is of the "racing" variety in the style of the European "Goose Game".

 Players advance along the track from Uncle Wiggily's Bungalow to Dr. Possum's House.

 There is no optimal strategy involved as play entirely rests upon a random drawing of the cards.

 The game was first published by Milton Bradley in 1916 and has seen several editions with minor modifications over the years.

 "Uncle Wiggily" remains one of the first and favorite games of childhood, and, with "Candy Land", is considered a classic juvenile American board game.
</item 8>

<item 9>
Treehouse (game)
Treehouse is a game in which players try to get their configuration of Icehouse pieces to match the central configuration, shared by all players.

 The rolling of the special "Treehouse Die" tells the player what kind of move to make to change his own or the central configuration, and then he does so to
best move towards the goal.
</item 9>

<item 10>
Alhambra (board game)
Alhambra (German: Der Palast von Alhambra , literally "The Palace of the Alhambra") is a 2003 tile-based German-style board game designed by Dirk Henn.

 It was originally published in Germany by Queen Games in a language-interdependent version; an English-specific version was released in North America by the
now-defunct Überplay.

 The game is an Arabian-themed update, set during the construction of the Alhambra palace in 14th century Granada, of the 1998 stock trading board game "Stimmt
So!"

, which in turn was an update of the 1992 mafia influence board game "Al Capone"; the original version was subsequently released as "".
</item 10>
=== END ITEM id=8ecf0c11486b309f ===

=== ITEM id=a4a6ae7d6559eba1 ===
Gold sentences and facts:
Sentence 0: Doris May Lessing, CH ("née" Tayler; 22 October 1919  – 17 November 2013) was a British novelist, poet, playwright, librettist, biographer and short story writer.
  Fact 0.0: Doris May Lessing, CH, was born Doris May Tayler (née Tayler).
  Fact 0.1: Doris May Lessing was born on 22 October 1919.
  Fact 0.2: Doris May Lessing died on 17 November 2013.
  Fact 0.3: Doris May Lessing was a British novelist.
  Fact 0.4: Doris May Lessing was a British poet.
  Fact 0.5: Doris May Lessing was a British playwright.
  Fact 0.6: Doris May Lessing was a British librettist.
  Fact 0.7: Doris May Lessing was a British biographer.
  Fact 0.8: Doris May Lessing was a British short story writer.
Sentence 1: Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
  Fact 1.0: Philip Kindred Dick was born on December 16, 1928.
  Fact 1.1: Philip Kindred Dick died on March 2, 1982.
  Fact 1.2: Philip Kindred Dick was an American writer.
  Fact 1.3: Philip Kindred Dick is notable for publishing works of science fiction.

Retrieved items:
<item 1>
Adjustment Team
"Adjustment Team" is a science fiction short story by American writer Philip K. Dick.

 It was first published in "Orbit Science Fiction" (September–October 1954, No. 4) with illustration by Faragasso.

 It was later reprinted in "The Sands of Mars and Other Stories" (Australian) in 1958, "The Book of Philip K. Dick" in 1973, "The Turning Wheel and Other
Stories" (United Kingdom) in 1977, "The Collected Stories of Philip K. Dick" in 1987 (Underwood–Miller), 1988 (Gollancz, United Kingdom), 1990 (Citadel
Twilight, United States), "Selected Stories of Philip K. Dick" in 2002 and in "The Early Work of Philip K. Dick, Volume One: The Variable Man & Other Stories"
in 2009.
</item 1>

<item 2>
The Days of Perky Pat (collection)
The Days of Perky Pat is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Gollancz in 1990 and reprints Volume IV of "The Collected Stories of Philip K. Dick".

 It had not previously been published as a stand-alone volume.

 The stories had originally appeared in the magazines "Galaxy Science Fiction", "Science Fiction Stories", "If", "Fantastic Universe", "Fantasy and Science
Fiction", "Fantastic", "Worlds of Tomorrow", "Escapade" and "Amazing Stories".
</item 2>

<item 3>
Doris Lessing
Doris May Lessing, CH ("née" Tayler; 22 October 1919  – 17 November 2013) was a British novelist, poet, playwright, librettist, biographer and short story
writer.

 Her novels include "The Grass Is Singing" (1950), the sequence of five novels collectively called "Children of Violence" (1952–69), "The Golden Notebook"
(1962), "The Good Terrorist" (1985), and five novels collectively known as "Canopus in Argos: Archives" (1979–1983).
</item 3>

<item 4>
Philip K. Dick
Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
</item 4>

<item 5>
The Good Terrorist
The Good Terrorist is a 1985 political novel written by the British novelist Doris Lessing.

 It was first published in September that year by Jonathan Cape in the United Kingdom and by Alfred A. Knopf in the United States.

 The book's protagonist is the naïve drifter Alice, who squats with a group of radicals in London and is drawn into their terrorist activities.
</item 5>

<item 6>
Philip K. Dick Award
The Philip K. Dick Award is a science fiction award given annually at Norwescon sponsored by the Philadelphia Science Fiction Society and (since 2005) supported
by the Philip K. Dick Trust, and named after science fiction and fantasy writer Philip K. Dick.

 It has been awarded since 1983, the year after Dick's death.

 Works that have received the award are identified on their covers as "Best Original SF Paperback".

 They are awarded to the best original paperback published each year in the US.
</item 6>

<item 7>
The Philip K. Dick Reader
The Philip K. Dick Reader is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Citadel Twilight in 1997.

 Many of the stories had originally appeared in the magazines "If", "Science Fiction Adventures", "Science Fiction Stories", "Orbit", "Fantasy and Science
Fiction", "Imagination", "Future", "Galaxy Science Fiction", "Beyond Fantasy Fiction", "Satellite", "Imaginative Tales", "Fantastic Universe" and "Space Science
Fiction".

 It is identical in content and order to the edition of volume 3 of the Collected Stories of Philip K. Dick produced by the same publisher apart from the
substitution of three stories in positions 21-23 of 24 and the omission of the end notes in the Collected Stories edition.

 At press time, stories 21 and 24 had already been made into successful movie adaptations and stories 22 and 23 had been optioned.
</item 7>

<item 8>
Jenny Diski
Jenny Diski FRSL (née Simmonds; 8 July 1947 – 28 April 2016) was an English writer.

 She had a troubled childhood, but was rescued by the older novelist Doris Lessing; she lived in Lessing's house for four years.

 Diski was educated at University College London, and worked as a teacher during the 1970s and early 1980s.
</item 8>

<item 9>
Penelope Fitzgerald
Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.

 In 2008, "The Times" included her in a list of "the 50 greatest British writers since 1945".

 In 2012, "The Observer" named her final novel, "The Blue Flower" one of "the ten best historical novels".
</item 9>

<item 10>
Roger Lancelyn Green
Roger (Gilbert) Lancelyn Green (2 November 1918 – 8 October 1987) was a British biographer and children's writer.

 He was an Oxford academic who formed part of the Inklings literary discussion group along with C. S. Lewis and J. R. R. Tolkien.
</item 10>
=== END ITEM id=a4a6ae7d6559eba1 ===

=== ITEM id=45e593479beba778 ===
Gold sentences and facts:
Sentence 0: Brunswick Golden Isles Airport (IATA: BQK, ICAO: KBQK, FAA LID: BQK) , previously known as Glynco Jetport, is a county-owned public-use airport located five nautical miles (9 km) north of the central business district of Brunswick, a city in Glynn County, Georgia, United States.
  Fact 0.0: Brunswick Golden Isles Airport has IATA code BQK.
  Fact 0.1: Brunswick Golden Isles Airport has ICAO code KBQK.
  Fact 0.2: Brunswick Golden Isles Airport has FAA LID BQK.
  Fact 0.3: Brunswick Golden Isles Airport was previously known as Glynco Jetport.
  Fact 0.4: Brunswick Golden Isles Airport is a county-owned public-use airport.
  Fact 0.5: Brunswick Golden Isles Airport is located five nautical miles (9 km) north of the central business district of Brunswick.
  Fact 0.6: Brunswick is a city in Glynn County, Georgia, United States.
Sentence 1: Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana, United States.
  Fact 1.0: Lafayette Regional Airport has IATA code LFT.
  Fact 1.1: Lafayette Regional Airport has ICAO code KLFT.
  Fact 1.2: Lafayette Regional Airport has FAA LID LFT.
  Fact 1.3: Lafayette Regional Airport is a public airport.
  Fact 1.4: Lafayette Regional Airport is two miles (4 km) southeast of Lafayette.
  Fact 1.5: Lafayette Regional Airport is in Lafayette Parish, Louisiana, United States.

Retrieved items:
<item 1>
Atlanta South Regional Airport
Henry County Airport (FAA LID: KHMP) , is a public-use county airport located three nautical miles (6 km) west of the central business district of Hampton, a
city in Henry County, Georgia, United States.

 It was known as Clayton County Airport – Tara Field, which was the name still used by the Federal Aviation Administration (FAA) and the Georgia Department of
Transportation (GDOT) until approval of the Atlanta South Regional Airport name.

 The airport was renamed Atlanta South Regional Airport, which was approved by the airport board of commissioners in December 2011, and was approved by the GDOT
and the FAA before it went into effect.
</item 1>

<item 2>
Golden Isles of Georgia
The Golden Isles of Georgia are a group of four barrier islands and the mainland port city of Brunswick on the 100-mile-long coast of the U.S. state of Georgia
on the Atlantic Ocean.

 They include St. Simons Island, Sea Island, Jekyll Island, Little St. Simons Island, and Historic Brunswick.
</item 2>

<item 3>
Brunswick Golden Isles Airport (BQK/KBQK), formerly Glynco Jetport, is a county-owned airport north of Brunswick, serving mainly general aviation and one
airline. Brunswick’s Old Town Historic District entered the National Register in 1979. The Golden Isles comprise St. Simons, Sea, Jekyll, and Little St. Simons
islands, plus Brunswick. Jekyll Island Airport supports general aviation and scenic tours, with no commercial service. The Colonel’s Island Railroad operated
</item 3>

<item 4>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 4>

<item 5>
Atlantic Southeast Airlines Flight 2311
Atlantic Southeast Airlines Flight 2311 was a regularly scheduled commuter flight from Hartsfield–Jackson Atlanta International Airport to Glynco Jetport (since
renamed Brunswick Golden Isles Airport) in Brunswick, Georgia on April 5, 1991.

 The flight, operated using a twin-turboprop Embraer EMB 120 Brasilia, crashed just north of Brunswick while approaching the airport for landing.

 All 23 people aboard the plane were killed, including passengers Sonny Carter and John Tower.

 Four years later, another Embraer Brasilia of ASA crashed in the Georgia countryside in similar circumstances, with nine fatalities.
</item 5>

<item 6>
Brunswick Golden Isles Airport
Brunswick Golden Isles Airport (IATA: BQK, ICAO: KBQK, FAA LID: BQK) , previously known as Glynco Jetport, is a county-owned public-use airport located five
nautical miles (9 km) north of the central business district of Brunswick, a city in Glynn County, Georgia, United States.

 It is mostly used for general aviation, but is also served by one commercial airline.
</item 6>

<item 7>
College of Coastal Georgia
The College of Coastal Georgia (also known as CCGA) is a public college located at historical Brunswick, Georgia, United States and the surrounding historical
Golden Isles.

 The College was established in 1961 and opened in 1964, making it one of Georgia's newest state colleges.

 The college transitioned from a two-year community college into a four-year comprehensive institution and conferred its first baccalaureate degrees on May 7,
2011.
</item 7>

<item 8>
Jekyll Island Airport
Jekyll Island Airport is a small general aviation airport located in Jekyll Island, GA.

 It is around 2 miles away from central Jekyll Island, GA.

 Due to its small runway size and the nearby Brunswick Golden Isles Airport, no commercial airlines fly there, but scenic tours of the island fly from there.

 The airport has 3 based aircraft, and around 74 aircraft operations a day.
</item 8>

<item 9>
Colonel's Island Railroad
In the late 1960s, the State of Georgia started an improvement project for the port at Brunswick, Georgia.

 In this project, the Colonel's Island Railroad (reporting mark CISD) was created as a part of the Georgia Ports Authority.

 The CISD operated 33 miles of track in and around the Brunswick area.

 In 1998, the railroad sold most of its assets and the Golden Isles Terminal Railroad began operating on the line.

 The Ports Authority retained ownership of the track, but the CISD was disbanded.
</item 9>

<item 10>
Brunswick Old Town Historic District
Brunswick Old Town Historic District is a historic district in Brunswick, Georgia.

 It was added to the National Register of Historic Places on April 26, 1979 and includes an area bounded by 1st Street, Bay Street, New Bay Street, H Street,
and Cochran Street (4th Ave. and G St., according to one source).

 Brunswick is one of Georgia's two deep-water ports and is the mainland city associated with the Golden Isles of Georgia, at the junction of I-95 and US 82.
</item 10>
=== END ITEM id=45e593479beba778 ===

=== ITEM id=4d14144b494385d4 ===
Gold sentences and facts:
Sentence 0: Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate, representing the 13th District since 1996.
  Fact 0.0: Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate.
  Fact 0.1: Suzanne N.J. 'Susie' Chun Oakland represents the 13th District.
  Fact 0.2: Suzanne N.J. 'Susie' Chun Oakland has represented the 13th District since 1996.
Sentence 1:  Previously she was a member of the Hawaii House of Representatives from 1990-96.
  Fact 1.0: Suzanne N.J. 'Susie' Chun Oakland was previously a member of the Hawaii House of Representatives.
  Fact 1.1: Suzanne N.J. 'Susie' Chun Oakland was a member of the Hawaii House of Representatives from 1990 to 1996.
Sentence 2:  The lower chamber of the legislature is the Hawaiʻi House of Representatives.
  Fact 2.0: The lower chamber of the Hawaii legislature is the Hawaiʻi House of Representatives.

Retrieved items:
<item 1>
Suzanne Chun Oakland
Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate, representing the 13th District since 1996.

 Previously she was a member of the Hawaii House of Representatives from 1990-96.

 She is generally considered to have liberal political views.
</item 1>

<item 2>
Hawaii State Capitol
The Hawaii State Capitol is the official statehouse or capitol building of the U.S. state of Hawaii.

 From its chambers, the executive and legislative branches perform the duties involved in governing the state.

 The Hawaii State Legislature—composed of the twenty-five member Hawaii State Senate led by the President of the Senate and the fifty-one member Hawaii State
House of Representatives led by the Speaker of the House—convenes in the building.

 Its principal tenants are the Governor of Hawaii and Lieutenant Governor of Hawaii, as well as all legislative offices and the Legislative Reference Bureau.
</item 2>

<item 3>
Hawaii State Legislature
The Hawaii State Legislature is the state legislature of the U.S. state of Hawaii.

 The state legislature is a bicameral body consisting of a lower house, the Hawaii State House of Representatives, with 51 representatives, and an upper house,
the 25-member Hawaii State Senate.

 There are a total of 76 representatives in the legislature, each representing single member districts across the islands.

 The powers of the legislature are granted under Article III of the Constitution of Hawaii.
</item 3>

<item 4>
Suzanne Chun Oakland: liberal Hawaii senator since 1996; state representative, 1990–96. John Márquez: first Latino Richmond councilman; served 18 years,
including vice mayor. Suresh Haware: BJP politician, former nuclear scientist and affordable-housing developer. Hajime Hirota: Japanese national legislator,
2004–16. Frank Jackson: Alabama legislator and Opp civic leader. Paul Clyne: Albany DA, 2001–04
</item 4>

<item 5>
Aloha Dalire
Fabianne Pomaialoha Wong Dalire, known professionally as Aloha Dalire, (June 22, 1950 – August 6, 2014) was an American Hawaiian kumu hula, or master hula
teacher.

 She won the first Miss Aloha Hula as Aloha Wong, in 1971, the same year that the Merrie Monarch Festival was established.

 The Miss Aloha Hula title is hula's top solo wahine (women's) honor.
</item 5>

<item 6>
The material surveys legislatures, institutions, and historical sites. The West Indies Federation’s territories became nine independent states and four British
territories; Guyana and Belize were observers. Hawaii, Alaska, Maryland, and Quebec have bicameral or historically bicameral legislatures; Alaska’s is the
smallest bicameral U.S. legislature. Oregon’s territorial legislature preceded statehood. Hawaii’s Capitol houses state government. Quebec’s legislature became
unicameral in 1968; Majorca’s Gran i General
</item 6>

<item 7>
Oregon Territorial Legislature
Oregon’s Territorial Legislature was a bicameral legislative body created by the United States Congress in 1848 as the legislative branch of the government of
the Oregon Territory.

 The upper chamber Council and lower chamber House of Representatives first met in July 1849; they served as the region's legislative body until Oregon became a
state in February 1859, when they were replaced by the bicameral Oregon State Legislature.
</item 7>

<item 8>
Maryland General Assembly
The Maryland General Assembly is the state legislature of the U.S. state of Maryland that convenes within the State House in Annapolis.

 It is a bicameral body: the upper chamber, the Maryland State Senate, has 47 representatives and the lower chamber, the Maryland House of Delegates, has 141
representatives.

 Members of both houses serve four-year terms.

 Each house elects its own officers, judges the qualifications and election of its own members, establishes rules for the conduct of its business, and may
punish or expel its own members.
</item 8>

<item 9>
Alaska Legislature
The Alaska Legislature is the state legislature of the U.S. state of Alaska.

 It is a bicameral institution, consisting of the lower Alaska House of Representatives, with 40 members, and the upper house Alaska Senate, with 20 members.

 There are 40 House Districts (1-40) and 20 Senate Districts (A-T).

 With a total of 60 lawmakers, the Alaska Legislature is the smallest bicameral state legislature in the United States and the second-smallest of all state
legislatures (only the 49-member unicameral Nebraska Legislature is smaller).

 There are no term limits for either chamber.
</item 9>

<item 10>
Hawaii Senate
The Hawaiʻi State Senate is the upper chamber of the Hawaii State Legislature.

 The senate consists of twenty-five members elected from an equal number of constituent districts across the islands.

 The senate is led by the President of the Senate, elected from the membership of the body, currently Ron Kouchi.

 The forerunner of the Hawaii State Senate during the government of the Kingdom of Hawaiʻ i was the House of Nobles originated in 1840.

 In 1894 the Constitution of the Republic of Hawaii renamed the upper house the present senate.

 Senators are elected to four-year terms and are not subject to term limits.

 Like most state legislatures in the United States, the Hawaii State Senate is a part-time body and senators often have active careers outside government.

 The lower chamber of the legislature is the Hawaiʻi House of Representatives.

 The membership of the Senate also elects additional officers to include the Senate Vice President, Senate Chief Clerk, Assistant Chief Clerk, Senate Sergeant
at Arms and Assistant Sergeant at Arms.
</item 10>
=== END ITEM id=4d14144b494385d4 ===

=== ITEM id=bba98ca9951c28ab ===
Gold sentences and facts:
Sentence 0: The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista Distribution.
  Fact 0.0: The Rescuers is a 1977 film.
  Fact 0.1: The Rescuers is an American animated film.
  Fact 0.2: The Rescuers is a comedy-drama adventure film.
  Fact 0.3: The Rescuers was produced by Walt Disney Productions.
  Fact 0.4: The Rescuers was first released on June 22, 1977.
  Fact 0.5: The Rescuers was first released by Buena Vista Distribution.
Sentence 1: The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.
  Fact 1.0: The Muppets is a 2011 film.
  Fact 1.1: The Muppets is an American film.
  Fact 1.2: The Muppets is a musical comedy film.
  Fact 1.3: The Muppets is the seventh theatrical film featuring the Muppets.

Retrieved items:
<item 1>
The Muppets (soundtrack)
The Muppets: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on November 22, 2011 for the musical comedy film
"The Muppets".

 The soundtrack features five original songs, four re-recordings and remasterings of popular Muppet songs ("The Muppet Show Theme", "Rainbow Connection", and
"Mah Nà Mah Nà"), two cover versions of existing songs (Cee Lo Green's "Forget You" and Nirvana's "Smells Like Teen Spirit"), two standalone songs (Paul Simon's
"Me and Julio Down by the Schoolyard" and Starship's "We Built This City"), and fifteen dialogue tracks.

 It also features the song "Man or Muppet", which won the Academy Award for Best Original Song.

 The soundtrack was also nominated for Best Compilation Soundtrack for Visual Media at the 55th Grammy Awards.
</item 1>

<item 2>
List of Walt Disney Pictures films
This is a list of films released theatrically under the Walt Disney Pictures banner (known as that since 1983, with "Never Cry Wolf" as its first release) and
films released before that under the former name of the parent company, Walt Disney Productions (1929–1983).

 Most films listed here were distributed in the United States by the company's distribution division, Walt Disney Studios Motion Pictures (formerly known as
Buena Vista Distribution Company [1953–1987] and Buena Vista Pictures Distribution [1987–2007]).

 The Disney features produced before "Peter Pan" (1953) were originally distributed by RKO Radio Pictures, and are now distributed by Walt Disney Studios Motion
Pictures.
</item 2>

<item 3>
Muppets Most Wanted (soundtrack)
Muppets Most Wanted: An Original Walt Disney Records Soundtrack is a soundtrack album released by Walt Disney Records on March 18, 2014 for the musical comedy
film "Muppets Most Wanted".

 The soundtrack features six original songs, two re-recordings of popular Muppet songs ("The Muppet Show Theme" and "Together Again"), three cover versions of
existing songs (Allen Toussaint's "Working in the Coal Mine", Maroon 5's "Moves Like Jagger", and Los del Río's "Macarena (Bayside Boys Mix)"), an orchestral
suite by Christophe Beck, five demos by Bret McKenzie, and eight dialogue tracks.
</item 3>

<item 4>
Muppet Treasure Island
Muppet Treasure Island is a 1996 American musical adventure comedy film based on Robert Louis Stevenson's "Treasure Island".

 It is the fifth feature film to star The Muppets, and was directed by Brian Henson.
</item 4>

<item 5>
The Muppets at Walt Disney World
The Muppets at Walt Disney World is a television special starring Jim Henson's Muppets at Walt Disney World in Lake Buena Vista, Florida.

 The special aired on NBC as part of "The Magical World of Disney" on May 6, 1990, and turned out to be the last Muppets special that Henson would work on (he
would die 10 days following its airing, on May 16).
</item 5>

<item 6>
The Muppets (film)
The Muppets is a 2011 American musical comedy film and the seventh theatrical film featuring the Muppets.

 The film is directed by James Bobin, written by Jason Segel and Nicholas Stoller, produced by David Hoberman and Todd Lieberman, and stars Segel, Amy Adams,
Chris Cooper and Rashida Jones, as well as Muppet performers Steve Whitmire, Eric Jacobson, Dave Goelz, Bill Barretta, David Rudman, Matt Vogel, and Peter Linz.

 Bret McKenzie served as music supervisor, writing four of the film's five original songs, and Christophe Beck composed the film's score.

 In "The Muppets", devoted fan Walter, his brother Gary, and Gary's girlfriend Mary help Kermit the Frog reunite the disbanded Muppets, as they must raise $10
million to save the Muppet Theater from Tex Richman, a businessman who plans to demolish the studio to drill for oil.
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
The Rescuers Down Under
The Rescuers Down Under is a 1990 American animated comedy-drama adventure film produced by Walt Disney Feature Animation and released by Walt Disney Pictures
on November 16, 1990.

 The 29th Disney animated feature film, the film is the sequel to the 1977 animated film "The Rescuers", which was based on the novels of Margery Sharp.

 Set in the Australian Outback, the film centers on Bernard and Bianca traveling to Australia to save a boy named Cody from a villainous poacher in pursuit of
an endangered bird of prey.
</item 8>

<item 9>
The Rescuers
The Rescuers is a 1977 American animated comedy-drama adventure film produced by Walt Disney Productions and first released on June 22, 1977, by Buena Vista
Distribution.

 The 23rd Disney animated feature film, the film is about the Rescue Aid Society, an international mouse organization headquartered in New York City and
shadowing the United Nations, dedicated to helping abduction victims around the world at large.

 Two of these mice, jittery janitor Bernard (Bob Newhart) and his co-agent, the elegant Miss Bianca (Eva Gabor), set out to rescue Penny (Michelle Stacy), an
orphan girl being held prisoner in the Devil's Bayou by treasure huntress Madame Medusa (Geraldine Page).
</item 9>

<item 10>
List of The Muppets productions
This is a list of productions based on The Muppets characters and franchise, including films, television series and specials, and other media.

 The franchise's main work is "The Muppet Show", a syndicated television series which ran from 1976 to 1981.

 The franchise includes eight feature films; "The Muppet Movie", "The Great Muppet Caper", "The Muppets Take Manhattan", "The Muppet Christmas Carol", "Muppet
Treasure Island", "Muppets from Space", "The Muppets", and "Muppets Most Wanted".

 The franchise also includes other series such as "Muppets Tonight" and "The Muppets".
</item 10>
=== END ITEM id=bba98ca9951c28ab ===

=== ITEM id=6282ccbc4578c03e ===
Gold sentences and facts:
Sentence 0: Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
  Fact 0.0: Javier Alberto Frana was born on 25 December 1966.
  Fact 0.1: Javier Alberto Frana is a former tennis player.
  Fact 0.2: Javier Alberto Frana is from Argentina.
  Fact 0.3: Javier Alberto Frana is a current tennis commentator.
  Fact 0.4: Javier Alberto Frana is a tennis commentator for ESPN Latin America.
Sentence 1: Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and the U.S. National Championships.
  Fact 1.0: Thomaz Koch was born on May 11, 1945.
  Fact 1.1: Thomaz Koch was born in Porto Alegre.
  Fact 1.2: Thomaz Koch is a left-handed former tennis player.
  Fact 1.3: Thomaz Koch is from Brazil.
  Fact 1.4: Thomaz Koch was a quarter-finalist at the French Open.
  Fact 1.5: Thomaz Koch was a quarter-finalist at Wimbledon.
  Fact 1.6: Thomaz Koch was a quarter-finalist at the U.S. National Championships.

Retrieved items:
<item 1>
1996 IPB Czech Indoor – Doubles
Jonas Björkman and Javier Frana was the defending champions but did not compete that year.
</item 1>

<item 2>
1996 Nottingham Open – Singles
Javier Frana was the defending champion but lost in the first round to Marc-Kevin Goellner.
</item 2>

<item 3>
1993 ATP Buenos Aires – Doubles
Pablo Albano and Javier Frana were the defending champions, but lost in the quarterfinals this year.
</item 3>

<item 4>
1996 Abierto Mexicano de Tenis – Doubles
Javier Frana and Leonardo Lavalle were the defending champions but lost in the first round to Luis Lobo and Javier Sánchez.
</item 4>

<item 5>
1994 Movistar Open – Singles
Javier Frana was the defending champion but lost in the quarterfinals to Alberto Berasategui.
</item 5>

<item 6>
Franz Josef Och
Franz Josef Och (November 2, 1971) is a German computer scientist.

 He is currently working at Grail (an Illumina company) as Head of Data Science.

 Prior to this he was Chief Data Scientist at Human Longevity Inc., and earlier worked for Google as a Distinguished Research Scientist and head of machine
translation based at Google's Mountain View, California, headquarters south of San Francisco.
</item 6>

<item 7>
Thomaz Koch
Thomaz Koch (born May 11, 1945 in Porto Alegre), is a left-handed former tennis player from Brazil, who was a quarter-finalist at the French Open, Wimbledon and
the U.S. National Championships.
</item 7>

<item 8>
Javier Frana
Javier Alberto Frana (born 25 December 1966) is a former tennis player from Argentina and current tennis commentator for ESPN Latin America.
</item 8>

<item 9>
Paulo Borrachinha
Paulo Henrique Costa (born April 21, 1991), better known as Paulo Borrachinha, is a Brazilian mixed martial artist currently competing in the Middleweight
division of the Ultimate Fighting Championship.

 In the day August 3 of 2017, Borracinha and the Clube Atlético Mineiro officialized institutional partnership.

 The agreement did not create a bond between the club and the athlete.

 The agreement allows the use of the athletic shield of Atlético in the training uniforms of the fighter, within the possibilities of his contract with the UFC.
</item 9>

<item 10>
1995 Nottingham Open – Singles
Javier Frana defeated Todd Woodbridge 7–6, 6–3 in the final to secure the title.
</item 10>
=== END ITEM id=6282ccbc4578c03e ===

=== ITEM id=744fd83cf2cfb308 ===
Gold sentences and facts:
Sentence 0: Ex parte Young, 209 U.S. 123 (1908) , is a United States Supreme Court case that allows suits in federal courts against officials acting on behalf of states of the union to proceed despite the State's Sovereign immunity, when the State acted unconstitutionally.
  Fact 0.0: Ex parte Young is a United States Supreme Court case.
  Fact 0.1: Ex parte Young is cited as 209 U.S. 123.
  Fact 0.2: Ex parte Young is dated 1908.
  Fact 0.3: Ex parte Young allows suits in federal courts against officials acting on behalf of states of the union.
  Fact 0.4: Ex parte Young allows such suits to proceed despite the state's sovereign immunity.
  Fact 0.5: Ex parte Young allows such suits when the state acted unconstitutionally.
Sentence 1:  The case was a test of the separation of church and state with respect to education.
  Fact 1.0: The Ex parte Young case was a test of the separation of church and state.
  Fact 1.1: The test of church and state separation in the Ex parte Young case concerned education.

Retrieved items:
<item 1>
Criminal law in the Chase Court
The Chase Court (1864–1873) issued thirty-five opinions in criminal cases over nine years, at a significantly higher rate than the Marshall Court or Taney Court
before it.

 Notable such cases include "Ex parte Milligan" (1866), "Pervear v. Massachusetts" (1866), "Ex parte McCardle" (1867, 1869), "Ex parte Yerger" (1868), and
"United States v. Kirby" (1868).
</item 1>

<item 2>
Seminole Tribe of Florida v. Florida
Seminole Tribe of Florida v. Florida, 517 U.S. 44 (1996) , was a United States Supreme Court case which held that Article One of the U.S. Constitution did not
give the United States Congress the power to abrogate the sovereign immunity of the states that is further protected under the Eleventh Amendment.

 Such abrogation is permitted where it is necessary to enforce the rights of citizens guaranteed under the Fourteenth Amendment as per "Fitzpatrick v. Bitzer"
427 U.S. 445 (1976).

 The case also held that the doctrine of "Ex parte Young", 209 U.S. 123 (1908), which allows state officials to be sued in their official capacity for
prospective injunctive relief, was inapplicable under these circumstances, because any remedy was limited to the one that Congress had provided.
</item 2>

<item 3>
Duncan v. Kahanamoku
Duncan v. Kahanamoku, 327 U.S. 304 (1946), was a decision by the United States Supreme Court.

 It is often associated with the Japanese exclusion cases ("Hirabayashi v. United States", "Korematsu v. United States" and "Ex parte Endo") because it involved
wartime curtailment of fundamental civil liberties under the aegis of military authority, though in this case neither the plaintiff nor the nominal defendant
were Japanese.
</item 3>

<item 4>
McCollum v. Board of Education
McCollum v. Board of Education, 333 U.S. 203 (1948), was a landmark 1948 United States Supreme Court case related to the power of a state to use its
tax-supported public school system to aid religious instruction.

 The case was a test of the separation of church and state with respect to education.
</item 4>

<item 5>
Criminal law in the Waite Court
Notable cases include "United States v. Cruikshank" (1875), "United States v. Reese" (1875), "Reynolds v. United States" (1878), "Wilkerson v. Utah" (1879), the
"Trade-Mark Cases" (1879), "Strauder v. West Virginia" (1880), "Pace v. Alabama" (1883), "United States v. Harris" (1883), "Ex parte Crow Dog" (1883), "Hurtado
v. California" (1884), "Clawson v. United States" (1885), "Yick Wo v. Hopkins" (1886), "United States v. Kagama" (1886), "Ker v. Illinois" (1886), and "Mugler
v. Kansas" (1887).
</item 5>

<item 6>
Abington School District v. Schempp
Abington School District v. Schempp, 374 U.S. 203 (1963) , was a United States Supreme Court case in which the Court decided 8–1 in favor of the respondent,
Edward Schempp, and declared school-sponsored Bible reading in public schools in the United States to be unconstitutional.

 The Chief Justice of the Supreme Court during this case was Earl Warren.
</item 6>

<item 7>
Ex parte Endo
Ex parte Endo, or Ex parte Mitsuye Endo, 323 U.S. 283 (1944), was a United States Supreme Court ex parte decision handed down on December 18, 1944, in which the
Justices unanimously ruled that the U.S. government could not continue to detain a citizen who was "concededly loyal" to the United States.

 Although the Court did not touch on the constitutionality of the exclusion of people of Japanese ancestry from the West Coast—which they had found not to
violate citizen rights in their "Korematsu v. United States" decision on the same date—the Endo ruling nonetheless led to the reopening of the West Coast to
Japanese Americans after their incarceration in camps across the U.S. interior during World War II.
</item 7>

<item 8>
Ex parte Young
Ex parte Young, 209 U.S. 123 (1908) , is a United States Supreme Court case that allows suits in federal courts against officials acting on behalf of states of
the union to proceed despite the State's Sovereign immunity, when the State acted unconstitutionally.
</item 8>

<item 9>
Chy Lung v. Freeman
Chy Lung v. Freeman, 92 U.S. 275 (1876) was a United States Supreme Court case where the Supreme Court ruled that the power to set rules surrounding
immigration, and to manage foreign relations, rested with the United States Federal Government, rather than with the states.

 The case has been cited in other Supreme Court cases related to government authority on matters relating to immigration policy and immigration enforcement,
most recently in "Arizona v. United States" (2012).
</item 9>

<item 10>
Owasso Independent School District v. Falvo
Owasso Independent School District v. Falvo, 534 U.S. 426 (2002) , was a case in which the United States Supreme Court held in favor of the school district that
students scoring each other's tests and calling out the grades do not violate the Family Educational Rights and Privacy Act of 1974 (FERPA).

 Justice Kennedy wrote the opinion for the unanimous court.

 Justice Scalia wrote a concurring opinion in which he agreed with the ruling, but took issue with parts of Kennedy's opinion.
</item 10>
=== END ITEM id=744fd83cf2cfb308 ===

=== ITEM id=5cf920869470315b ===
Gold sentences and facts:
Sentence 0:  Jan Kodeš and Margaret Court won the singles titles.
  Fact 0.0: Jan Kodeš won the singles titles.
  Fact 0.1: Margaret Court won the singles titles.
Sentence 1:  She is currently a Christian minister in Perth, Western Australia.
  Fact 1.0: A woman referred to in this sentence is currently a Christian minister.
  Fact 1.1: The Christian minister referred to in this sentence is based in Perth, Western Australia.

Retrieved items:
<item 1>
Marc Rosset
Marc Rosset (born 7 November 1970) is a former professional tennis player from Switzerland who is best remembered for winning the men's singles gold medal at
the 1992 Olympic Games.

 He also won one Grand Slam doubles title at the French Open in 1992 partnering compatriot Jakob Hlasek.
</item 1>

<item 2>
Carlos Fernandes (tennis)
Carlos Fernandes (1936-) is a retired Brazilian tennis player.

 He had a good all-round game with excellent passing shots and moved fast around the court.

 Fernandes had a reputation for being a charismatic ladies man.

 He later became a coach.

 He made his Grand Slam debut at Wimbledon 1957, losing in the opening round to Jorgen Ulrich.

 At Roland Garros in 1958, Fernandes lost in round one to Pierre Darmon.

 At Wimbledon he lost in round one to Butch Buchholz.

 At 1959 French Open, Fernandes lost in round three to Jacques Brichant.At Wimbledon he lost in round two to Neale Fraser.

 At French Open 1960, Fernandes lost in round three to Bobby Wilson and at Wimbledon lost in round two to Wolfgang Stuck.

 At 1961 French Open, Fernandes achieved his best Grand Slam singles result by beating Pierre Darmon and Bob Hewitt before losing to Jan-Erik Lundquist in the
quarter finals.

 He lost in round two of Wimbledon to Wilson.

 At Roland Garros 1962, Fernandes lost to Lundquist in round two.
</item 2>

<item 3>
Margaret Court
Margaret Court {'1': ", '2': ", '3': ", '4': "} (née Smith; born 16 July 1942), also known as Margaret Smith Court, is a retired Australian tennis player and
former world No. 1.

 She is currently a Christian minister in Perth, Western Australia.

 In tennis, she amassed more major titles than any other player in history.
</item 3>

<item 4>
Carlos Fernandes (tennis)
At Roland Garros 1962, Fernandes lost to Lundquist in round two.

 At Wimbledon he lost in round one to Neale Fraser's brother John.

 At 1963 French Open he lost in round one to Christian Duxin.

 In round one of Wimbledon he led a young Arthur Ashe by two sets to 0, but lost in five sets.

 At Wimbledon 1964 he lost in round two to Gene Scott.

 An ankle injury in 1964 took its' toll and Fernandes played less after that.

 As Open tennis arrived, Fernandes was reaching the end of his career.

 He lost in round one of 1968 French Open to Bernard Montrenaud.

 At Wimbledon he lost in round one to Mike Sangster.
</item 4>

<item 5>
Jorge Lozano
Jorge Lozano (born 17 May 1963 in San Luis Potosí) is a retired professional tennis player from Mexico.

 He reached his highest doubles ranking of World No. 4 in August 1988.

 His highest singles ranking was World No. 51, achieved the following month.

 During his career, he won two mixed doubles titles at the French Open: in 1990 with Arantxa Sánchez, and in 1988 with Lori McNeil.

 He reached the round of 16 in singles at the 1988 US Open and also the semifinal in doubles that same year.

 Qualified twice for the Doubles Masters at the Royal Albert Hall in London and reached the semifinals in 1988.

 In his career, he won nine doubles titles, but no singles titles.

 He turned professional in 1986, and in his career, he won $U.S.739,424 in prize money.

 He was the first player to be beaten by Pete Sampras in the main draw of a Grand Slam tournament, at the French Open in 1989.
</item 5>

<item 6>
Martin Legner
Martin Legner (born December 17, 1961 in Tyrol) is a professional Austrian wheelchair tennis player who has been ranked number one for wheelchair doubles.

 Legner has won the Australian Open doubles title four times with Robin Ammerlaan since 2000, and has won eight doubles titles.

 Six of them are Australian Open Titles, the other two being French Open titles.

 He has won one French Open and one Australian Open singles title.

 He has represented his country at every Summer Paralympics since 1992, and has competed in both singles and doubles at all of those Games.

 His favorite surfaces are clay and hard court.

 As of April 7, 2007 his highest singles rank was number three.

 Also of that date, he was ranked number six for singles and doubles.

 His wheelchair is manufactured by Kronbickler.

 Legner is a fan of Bob Marley, and listens to his music aside from tennis.
</item 6>

<item 7>
Kim Warwick
Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour from 1970–1987 reaching the final of the
singles Australian Open in 1980.

 He defeated over 35 players ranked in the top 10 including Guillermo Vilas, Raul Ramerez, Vitas Gerulaitis, Jan Kodeš, Bob Lutz and Arthur Ashe.

 Warwick's career-high singles ranking was World No. 15, achieved in 1981.

 He won three singles titles and 26 doubles, including Australian Open 1978 (with Wojtek Fibak) and Australian Open 1980 and 1981, Roland Garros 1986 and also a
runner-up in Australian Open 1985, all of them partnering fellow countryman Mark Edmondson.

 Partnering with Evonne Goolagong, he won the French Open 1972, defeating Françoise Dürr and Jean-Claude Barclay in the final 6–2, 6–4.

 Evonne and Kim were finalists in 1972 at Wimbledon against Rosie Casals and Ilie Năstase who won 6–4, 6–4.
</item 7>

<item 8>
1982 French Open – Men's Doubles
The Men's Doubles tournament at the 1982 French Open was held from 24 May until 6 June 1982 on the outdoor clay courts at the Stade Roland Garros in Paris,
France.

 Sherwood Stewart and Ferdi Taygan won the title, defeating Hans Gildemeister and Belus Prajoux in the final.
</item 8>

<item 9>
1970 French Open
The 1970 French Open was a tennis tournament that was held at the Stade Roland Garros in Paris in France from 25 May through 7 June 1970.

 It was the 69th edition of the French Open, the 40th to be open to foreign competitors, and the second Grand Slam of the year.

 Jan Kodeš and Margaret Court won the singles titles.
</item 9>

<item 10>
1972 French Open – Men's Singles
Jan Kodes was the two-time defending champion, but lost in the quarterfinals to Patrick Proisy.
</item 10>
=== END ITEM id=5cf920869470315b ===

=== ITEM id=cc55633badfdc761 ===
Gold sentences and facts:
Sentence 0: Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.
  Fact 0.0: Jean-François André Clervoy was born on 19 November 1958.
  Fact 0.1: Jean-François André Clervoy is a French engineer.
  Fact 0.2: Jean-François André Clervoy is a CNES astronaut.
  Fact 0.3: Jean-François André Clervoy is an ESA astronaut.
Sentence 1: Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
  Fact 1.0: Prof. Dr. Ulrich Hans Walter was born on February 9, 1954.
  Fact 1.1: Prof. Dr. Ulrich Hans Walter is a German physicist.
  Fact 1.2: Prof. Dr. Ulrich Hans Walter is a German engineer.
  Fact 1.3: Prof. Dr. Ulrich Hans Walter is a former DFVLR astronaut.

Retrieved items:
<item 1>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 1>

<item 2>
The entries cover German rocket scientist Magnus “Mac” von Braun, Wernher’s brother; Wernher von Braun’s Ferry Rocket shuttle concept and the 1937 failed
Aggregate 3 launches (Operation Lighthouse); and Collier’s influential 1950s spaceflight series *Man Will Conquer Space Soon!*. They also profile German
astronauts Hans Schlegel, Ernst Messerschmid, Ulrich Walter, Sigmund Jähn, Ulf Merbold, and Soyuz
</item 2>

<item 3>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 3>

<item 4>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 4>

<item 5>
Ulrich Walter
Prof. Dr. Ulrich Hans Walter (born February 9, 1954) is a German physicist/engineer and a former DFVLR astronaut.
</item 5>

<item 6>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 6>

<item 7>
Walter Cunningham
Ronnie Walter Cunningham (born March 16, 1932), (Col, USMCR, Ret.)

, better known as Walter Cunningham, is a retired American astronaut.

 In 1968, he was a Lunar Module Pilot on the Apollo 7 mission.

 He was NASA's third civilian astronaut (after Neil Armstrong and Elliot See), and has also been a fighter pilot, physicist, entrepreneur, venture capitalist,
author of "The All-American Boys", lecturer, and host of the radio show "Lift-off to Logic".

 Walter Cunningham attended UCLA.
</item 7>

<item 8>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 8>

<item 9>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 9>

<item 10>
The profiles cover NASA computer scientist Annie Easley and space espresso: ISSpresso’s first in-space coffee, brewed by Samantha Cristoforetti in 2015. They
describe astrophysicist Jean-Loup Puget; missions Soyuz TM-7/T-6 and French astronaut Jean-Loup Chrétien; Patrick Baudry; Project Highwater; Apollo 7;
SpaceShipOne’s first privately funded human spaceflight; the Soyuz rocket; Lockheed test pilot John Christiansen
</item 10>
=== END ITEM id=cc55633badfdc761 ===

=== ITEM id=3312ad80b38be2a0 ===
Gold sentences and facts:
Sentence 0: Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
  Fact 0.0: Leycesteria is a genus of flowering plants.
  Fact 0.1: Leycesteria is in the honeysuckle family Caprifoliaceae.
  Fact 0.2: Leycesteria is native to temperate Asia in the Himalaya.
  Fact 0.3: Leycesteria is native to southwestern China.
Sentence 1: Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.
  Fact 1.0: Anigozanthos is a small genus of Australian plants.
  Fact 1.1: Anigozanthos is in the Bloodwort family Haemodoraceae.

Retrieved items:
<item 1>
Anigozanthos humilis
Anigozanthos humilis is a species of "Anigozanthos" in the family Haemodoraceae, known as common cat's paw or catspaw.

 This flowering perennial plant is endemic to Southwest Australia and widespread in its open forests.
</item 1>

<item 2>
Anigozanthos flavidus
Anigozanthos flavidus is a species of plant found in Southwest Australia.

 It is member of the Haemodoraceae family.

 It is commonly known as the tall, yellow, or evergreen, kangaroo paw.

 The specific epithet, "flavidus", refers to the yellow flowers of this plant.
</item 2>

<item 3>
Forests of Australia
Australia has many forests of importance due to significant features, despite being one of the driest continents.

 Australia has approximately 123 million hectares of native forest, which represents about 16% of Australia's land area.

 The majority of Australia's trees are hardwoods, typically eucalypts, rather than softwoods like pine.

 While softwoods dominate some native forests, their total area is judged insufficient to constitute a major forest type in Australia's National Forest
Inventory.

 The Forests Australia website provides up-to-date information on Australia's forests.

 Detailed information on Australia's forests is available from Australia's State of the Forests Reports that are published every five years.
</item 3>

<item 4>
Lepidozamia
Lepidozamia is a genus of two species of cycad, both endemic to Australia.

 The name, derived from the Greek word lepidos, meaning scaly, refers to the scale-like structure of the stem and leaf bases.

 They are native to rainforest climates in eastern Queensland and eastern New South Wales.

 They have a chromosome number of 2n = 18.
</item 4>

<item 5>
Leycesteria formosa
Leycesteria formosa (Himalayan honeysuckle, flowering nutmeg, Himalaya nutmeg or pheasant berry) is a deciduous shrub in the family Caprifoliaceae, native to
the Himalaya and southwestern China.

 It is considered a noxious invasive species in Australia, New Zealand, the neighboring islands of Micronesia, and some other places.
</item 5>

<item 6>
Alopecurus
Alopecurus, or foxtail grass, is a common and widespread genus of plants in the grass family.

 It is common across temperate and subtropical parts of Eurasia, northern Africa, and the Americas, as well as naturalized in Australia and on various islands.
</item 6>

<item 7>
Anigozanthos manglesii
Anigozanthos manglesii, commonly known as the red-and-green kangaroo paw, Mangles kangaroo paw, Kurulbrang(Noongar) is a plant species endemic to Western
Australia, and the floral emblem of that state.
</item 7>

<item 8>
Leycesteria
Leycesteria is a genus of flowering plants in the honeysuckle family Caprifoliaceae, native to temperate Asia in the Himalaya and southwestern China.
</item 8>

<item 9>
Anigozanthos
Anigozanthos is a small genus of Australian plants in the Bloodwort family Haemodoraceae.

 The 11 species and several subspecies are commonly known as kangaroo paw and catspaw depending on the shape of their flowers.

 A further species, previously identified as "Anigozanthos fuliginosus" and commonly known as the black kangaroo paw, has been transferred to its own monotypic
genus and is now known as "Macropidia fuliginosa".
</item 9>

<item 10>
Anigozanthos preissii
Anigozanthos preissii, the Albany cat's paw, is a herbaceous plant species in the family Haemodoraceae, endemic to Western Australia.
</item 10>
=== END ITEM id=3312ad80b38be2a0 ===

=== ITEM id=10ea89c31f967bda ===
Gold sentences and facts:
Sentence 0: Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.
  Fact 0.0: Elizabeth Austin was born in 1958.
  Fact 0.1: Elizabeth Austin is an American writer.
  Fact 0.2: Elizabeth Austin lives in Oak Park, Illinois.
Sentence 1:  She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".
  Fact 1.0: Elizabeth Austin has also written for The Washington Monthly.
  Fact 1.1: Elizabeth Austin has also written for Ladies Home Journal.
  Fact 1.2: Elizabeth Austin has also written for Good Housekeeping.
  Fact 1.3: Elizabeth Austin has also written for Self.
  Fact 1.4: Elizabeth Austin has also written for Shape.
Sentence 2: Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight fitness) and became the number one women's fitness magazine.
  Fact 2.0: Shape is a monthly English language fitness magazine.
  Fact 2.1: Shape was started by Weider Publications in 1981.
  Fact 2.2: Shape was founded by Christine MacIntyre.
  Fact 2.3: Christine MacIntyre is a pioneer in women's free weight fitness.
  Fact 2.4: Shape became the number one women's fitness magazine.

Retrieved items:
<item 1>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 1>

<item 2>
Memphis Sport
Memphis Sport is a sports and fitness magazine featuring articles on local and regional teams, players and events.

 In addition, "Memphis Sport" focuses on health and fitness for an active lifestyle.

 The magazine debuted in Memphis, Tennessee in July 2006.

 Local franchises covered include the University of Memphis Tigers, Memphis Redbirds, Mississippi Riverkings and Tennessee Titans.
</item 2>

<item 3>
Football World
Football World, later renamed "Athletic World", was an American magazine devoted to the coverage of inter-collegiate sports.

 Its masthead described it as "A Magazine With a Mission to Serve the College Man," a publication "devoted to Inter-collegiate Athletics and sports of Amateur
standing only."

 It was founded in 1921 by J. D. Fetzer.

 The name of the magazine was later changed to "Athletic World" as the coverage extended to a broader range of sports, including women's swimming.

 Unlike other sports magazines of the era, which focused on promoting a healthy lifestyle, "Football World/Athletic World" celebrated the entertainment value of
sports with a special emphasis on the personalities of famous athletes.

 The magazine was renamed "Outing" in December 1924, reflecting a change in its focus.
</item 3>

<item 4>
Wendy Austin
Dr Wendy Elizabeth Austin Hewitt, Wendy Austin is a BBC journalist and broadcaster in BBC Northern Ireland, and former presenter of BBC Radio Ulster’s current
affairs programme "Talkback".
</item 4>

<item 5>
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
</item 5>

<item 6>
Ingrid Romero
Ingrid Romero (born July 10, 1985) is an international fitness model, fitness competitor, and fitness coach.

 Training & Fitness Magazine called her "one of the most recognized faces in the fitness industry," and "one of the most accomplished female fitness stars
ever."
</item 6>

<item 7>
Anais Zanotti
Anais Zanotti (born January 8, 1986 in Saint-Tropez) is a French model and stunt woman of French and Italian ancestry who lives in Miami, Florida.

 She has been featured in magazines such as "Playboy", "FHM", "Maxim", "GQ", "Cosmopolitan", "Vogue", and "People", Men's Fitness magazine, Muscle and Fitness,
Women Fitness magazine and more.

 She has over 40 magazine covers.

 Zanotti is also a certified skydiver with over 1250 jumps.GS .
</item 7>

<item 8>
Muscle &amp; Fitness (British magazine)
Muscle & Fitness is a British fitness magazine, which contains articles on strength and fitness training, diet plans and strategies for men and women, and tips
and advice.
</item 8>

<item 9>
Elle (magazine)
Elle is a worldwide lifestyle magazine of French origin that focuses on fashion, beauty, health, and entertainment.

 "Elle" is also the world's best-selling fashion magazine.

 It was founded by Pierre Lazareff and his wife Hélène Gordon in 1945.

 The title, in French, means "she" or "her".
</item 9>

<item 10>
Elizabeth Austin (writer)
Elizabeth Austin (born 1958) is an American writer living in Oak Park, Illinois.

 Austin has lived in the Midwestern United States all her life.

 She began her journalism career at the now-defunct City News Bureau of Chicago and has written articles for national news magazines such as "Newsweek" and
"Time", in addition to a wide range of other national and regional publications, such as "People", and the "Chicago Tribune".

 She has also written for "The Washington Monthly, Ladies Home Journal, Good Housekeeping, Self", and "Shape".

 She has often written about medicine and health care.

 Austin also has been heard on WBEZ, the National Public Radio station in Chicago, as well as the BBC in the UK and Scotland.

 She has appeared on the Today Show and on ITV in Europe.
</item 10>
=== END ITEM id=10ea89c31f967bda ===

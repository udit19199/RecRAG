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

=== ITEM id=c5f8fbe1bf708d15 ===
Gold sentences and facts:
Sentence 0:  The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
  Fact 0.0: The ground was home to the Boston Reds.
  Fact 0.1: The Boston Reds played in the Players' League in 1890.
  Fact 0.2: The Boston Reds played in the American Association in 1891.
Sentence 1:  It is home to the Port of Boston, a major shipping facility in the northeastern United States.
  Fact 1.0: The ground is home to the Port of Boston.
  Fact 1.1: The Port of Boston is a major shipping facility.
  Fact 1.2: The Port of Boston is in the northeastern United States.

Retrieved items:
<item 1>
Transportation in Boston
The Boston transportation system includes roadway, subway, regional rail, air, and sea options for passenger and freight transit in Boston, Massachusetts.

 The Massachusetts Port Authority (Massport) operates the Port of Boston, which includes a container shipping facility in South Boston, and Logan International
Airport, in East Boston.

 The Massachusetts Bay Transportation Authority (MBTA) operates bus, subway, short distance rail, and water ferry passenger services throughout the city and
region.

 Amtrak operates passenger rail service to and from major northeastern cities.

 A major bus terminal at South Station is served by varied intercity bus companies.

 The city is bisected by major highways I-90 and I-93, the intersection of which has undergone a major renovation, nicknamed the Big Dig.
</item 1>

<item 2>
New Bedford Bay Sox
The New Bedford Bay Sox are a baseball team that plays in the New England Collegiate Baseball League, a collegiate summer baseball league located in the
northeastern United States region of New England.

 The team is located in New Bedford, Massachusetts.

 New Bedford's Paul Walsh Field serves as the home field of the Bay Sox.

 Th NECBL's players use wooden bats, as opposed to aluminum, and are college baseball players recruited from the top intercollegiate baseball programs in the
United States.

 The franchise has appeared in league finals four times in their history, all as the Torrington Twisters (1997, 1998, 2003, 2006), and have twice played host to
the NECBL All-Star Game as the Twisters (1998, 2008).

 Pat and Beth O'Connor, owners of Little Fenway and Little Wrigley joined the ownership team of the New Bedford Bay Sox in October, 2011, and signed a 3-year
lease to play at Paul Walsh Field.

 In November 2011, former MLB player Rick Miller was announced as the 2012 Bay Sox manager.
</item 2>

<item 3>
Braves Field
Braves Field was a baseball park in the Northeastern United States, located in Boston, Massachusetts.

 Today the site is home to Nickerson Field on the campus of Boston University.

 The stadium was home of the Boston Braves of the National League from 1915–1952, prior to the Braves' move to Milwaukee in 1953.

 The stadium hosted the 1936 Major League Baseball All-Star Game and Braves home games during the 1948 World Series.

 The Boston Red Sox used Braves Field for their home games in the 1915 and 1916 World Series since the stadium had a larger seating capacity than Fenway Park.

 Braves Field was the site of Babe Ruth's final season, playing for the Braves in 1935.

 From 1929 to 1932, the Boston Red Sox played select regular season games periodically at Braves Field.

 On May 1, 1926, Braves Field hosted the longest baseball game in history – 26 Innings.

 The game ended in a 1–1 tie.
</item 3>

<item 4>
Fisher Hill Reservoir
The Fisher Hill Reservoir and Gatehouse are a historic element of the public water supply for the Greater Boston area.

 The reservoir was located on Fisher Road between Hyslop and Channing Roads in Brookline, Massachusetts, and is now the site of Fisher Hill Reservoir Park.

 It was built in 1886-87 as an early component of the Boston Water Board's expansion of its high service system.

 The gatehouse may have been designed by Arthur Vinal, who also designed the high pumping station at Chestnut Hill Reservoir.

 It is a two-story Richardsonian Romanesque structure, with its first floor finished in stone and its second in brick.

 Brownstone trim is used on the windows and corner quoins, and the voussoirs which form the arches on the first floor.

 There are pipes to the reservoir and down to Chestnut Hill, and gates for controlling access to local the distribution network.

 The building was taken out of service in the 1950s.
</item 4>

<item 5>
Camden Yards Sports Complex
The Camden Yards Sports Complex is located in the center of Baltimore, Maryland.

 The complex is composed of multiple buildings and stadiums including Oriole Park at Camden Yards and M&T Bank Stadium.

 The two stadiums are home to the Baltimore Orioles of Major League Baseball and the Baltimore Ravens of the National Football League.

 The complex still houses the recently closed Sports Legends Museum at Camden Yards (a non-profit museum featuring Maryland sports teams).

 Along with the Sports Legends Museum, the Babe Ruth Birthplace and Museum is located approximately two blocks from the main entrance of Camden Yards at Eutaw
Street.

 Geppi's Entertainment Museum is also located in Camden Station, atop the Sports Legends at Camden Yards.

 The complex is more than museums and stadiums.

 It is a location for community events such as the Dew Tour's Panasonic Open in June 2007 and 2008, the Baltimore Marathon, and the African American Festival
which is held every year.
</item 5>

<item 6>
Congress Street Grounds
Congress Street Grounds is a former baseball ground located in Boston, Massachusetts.

 The ballpark, as the name implies, was along Congress Street, near the intersection of Thompson Place, and not far from the Fort Point Channel on South Boston
Flats, a newly filled in piece of land on Boston Harbor.

 The ground was home to the Boston Reds, that played in the Players' League in 1890 and the American Association in 1891.
</item 6>

<item 7>
New Bedford Bay Sox
Miller was succeeded as manager prior to the 2015 season by Westport, MA native and former Boston Red Sox draft pick Kyle Fernandes.

 Fernandes has led the Bay Sox to back-to-back postseason appearances in his two seasons as manager, most recently losing in a three-game series to the eventual
NECBL Champions the Mystic Schooners.
</item 7>

<item 8>
Boston Harbor
Boston Harbor is a natural harbor and estuary of Massachusetts Bay, and is located adjacent to the city of Boston, Massachusetts.

 It is home to the Port of Boston, a major shipping facility in the northeastern United States.
</item 8>

<item 9>
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
</item 9>

<item 10>
TD Garden
TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.

 It is named after its sponsor, TD Bank, a subsidiary of Canada's Toronto-Dominion Bank.

 It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.

 The arena is located directly above the MBTA's North Station.
</item 10>
=== END ITEM id=c5f8fbe1bf708d15 ===

=== ITEM id=71b3651e15fe35ca ===
Gold sentences and facts:
Sentence 0:  The show originally ran from 7 September 1977 to 20 November 1995, and was hosted by Gordon Burns and usually broadcast on the ITV network on Mondays at 7pm.
  Fact 0.0: The show originally ran from 7 September 1977 to 20 November 1995.
  Fact 0.1: The show was hosted by Gordon Burns.
  Fact 0.2: The show was usually broadcast on the ITV network.
  Fact 0.3: The show was usually broadcast on Mondays at 7pm.
Sentence 1:  He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West Tonight" from January 1997 to October 2011.
  Fact 1.0: The person referred to as 'He' was the host of 'The Krypton Factor' for its original 18-year run.
  Fact 1.1: The Krypton Factor's original run was 1977–1995.
  Fact 1.2: The person referred to as 'He' was the chief anchorman for the BBC regional news programme 'North West Tonight'.
  Fact 1.3: The person referred to as 'He' was the chief anchorman for 'North West Tonight' from January 1997 to October 2011.

Retrieved items:
<item 1>
The Week Show
The Week Show is a television late night talk show sketch comedy program on Bite TV in Canada.

 It originally ran from December 2008 to mid-2010.

 The show is hosted by Matt Chin and Ricky Thompson, who started on an internet late night talk show "My Show with Matt Chin" online in 2001.
</item 1>

<item 2>
The Krypton Factor
The Krypton Factor is a British game show produced by Granada for broadcast on ITV.

 The show originally ran from 7 September 1977 to 20 November 1995, and was hosted by Gordon Burns and usually broadcast on the ITV network on Mondays at 7pm.
</item 2>

<item 3>
18 Doughty Street
18 Doughty Street was a British political Internet-based broadcaster that hosted a webcast as its chief product.

 It began broadcasting at 18:55 on 10 October 2006, from its studio at 18 Doughty Street in the Bloomsbury area of London, and ceased broadcasting at 23:00 on
Thursday 8 November 2007.

 It claimed to be Britain's first Internet-based TV station.
</item 3>

<item 4>
Late-night talk show
A late-night talk show is a genre of talk show popular in the United States, where the format originated.

 It is generally structured around humorous monologues about the day's news, guest interviews, comedy sketches and music performances.

 The late-night talk show format was popularized, though not invented, by Johnny Carson with "The Tonight Show Starring Johnny Carson" on NBC.

 Typically the show's host conducts interviews from behind a desk, while the guest is seated on a couch.

 Many late night talk shows feature a house band which generally performs cover songs for the studio audience during commercial breaks and occasionally will
back up a guest artist.
</item 4>

<item 5>
Password (UK game show)
Password was a British panel game show based on the US version of the same name.

 It was originally aired on ITV produced by ATV from 12 March to 10 September 1963 hosted by Shaw Taylor, then it aired on BBC2 from 24 March to 28 April 1973
hosted by Brian Redhead before moving to its flagship channel BBC1 from 7 January 1974 to 1976 first hosted by Eleanor Summerfield then by Esther Rantzen, it
was then aired on Channel 4 produced by Thames in association with Talbot Television and Goodson-Todman Productions from 6 November 1982 to 14 May 1983 hosted
by Tom O'Connor and then finally aired back on ITV produced by Ulster from 22 July 1987 to 5 August 1988 hosted by Gordon Burns.
</item 5>

<item 6>
David Letterman
David Michael Letterman (born April 12, 1947) is an American television host, comedian, writer, and producer.

 He hosted a late night television talk show for 33 years, beginning with the February 1, 1982 debut of "Late Night with David Letterman" on NBC, and ending
with the May 20, 2015 broadcast of "Late Show with David Letterman" on CBS.

 In total, Letterman hosted 6,028 episodes of "Late Night" and "Late Show", surpassing friend and mentor Johnny Carson as the longest-serving late night talk
show host in American television history.

 In 1996 Letterman was ranked 45th on "TV Guide"' s 50 Greatest TV Stars of All Time.
</item 6>

<item 7>
Coast to Coast AM
Coast to Coast AM is an American late-night radio talk show that deals with a variety of topics.

 Most frequently the topics relate to either the paranormal or conspiracy theories.

 The program is distributed by Premiere Networks, both as part of its talk network and separately as a syndicated program.

 The program now airs seven nights a week 1:00 a.m. – 5:00 a.m. Eastern Time Zone.
</item 7>

<item 8>
A Word in Your Ear
A Word in Your Ear is a game show that originally aired BBC1 from 19 April 1993 to 14 October 1994 then on The Family Channel from 1995.

 It was hosted by Gordon Burns.

 The host presided over male and female pairs of celebrities as they participate in a few rounds of communication games.

 Celebrities appearing included Lynsey De Paul, Nick Owen, Philippa Kennedy and Bob Holness.
</item 8>

<item 9>
Gordon Burns (footballer)
Gordon Burns (born 2 December 1978) is a Scottish former professional footballer who is currently manager of Troon in the Scottish Junior Football Association,
West Region.

 He has previously played in the Scottish Championship for Ayr United.
</item 9>

<item 10>
Gordon Burns
Gordon Henry Burns (born 10 June 1942) is a Northern Irish journalist and broadcaster.

 He was the host of "The Krypton Factor" for its original 18-year run (1977–1995) and was the chief anchorman for the BBC regional news programme "North West
Tonight" from January 1997 to October 2011.

 In November 2011 Gordon moved back to Belfast were he was born.
</item 10>
=== END ITEM id=71b3651e15fe35ca ===

=== ITEM id=265c774fc7830ce9 ===
Gold sentences and facts:
Sentence 0: Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.
  Fact 0.0: Psychopsis is abbreviated Psychp in horticultural trade.
  Fact 0.1: Psychopsis is a genus of 5 known species of orchids.
  Fact 0.2: Psychopsis orchids are native to northern South America.
  Fact 0.3: Psychopsis orchids are native to Central America.
  Fact 0.4: Psychopsis orchids are native to Trinidad.
Sentence 1: Aubrieta (commonly known as Aubretia) is a genus of about 12 species of flowering plants in the cabbage family Brassicaceae.
  Fact 1.0: Aubrieta is commonly known as Aubretia.
  Fact 1.1: Aubrieta is a genus of about 12 species of flowering plants.
  Fact 1.2: Aubrieta is in the cabbage family Brassicaceae.

Retrieved items:
<item 1>
Arnebia
Arnebia is a genus of flowering plants in the family Boraginaceae.
</item 1>

<item 2>
Aubrieta
Aubrieta (commonly known as Aubretia) is a genus of about 12 species of flowering plants in the cabbage family Brassicaceae.

 The genus is named after Claude Aubriet, a French flower-painter.

 It originates from southern Europe east to central Asia but is now a common garden escape throughout Europe.

 It is a low, spreading plant, hardy, evergreen and perennial, with small violet, pink or white flowers, and inhabits rocks and banks.

 It prefers light, well-drained soil, is tolerant of a wide pH range, and can grow in partial shade or full sun.
</item 2>

<item 3>
Cassytha
Cassytha is a genus of some two dozen species of obligately parasitic vines in the family Lauraceae.

 Superficially, and in some aspects of their ecology, they closely resemble plants in the unrelated genus "Cuscuta", the dodders.

 When fruit and flowers are absent in the field, the physical resemblance is so close that few people without technical training can tell them apart.

 In this respect and in their ecology the two genera present a spectacular example of convergent evolution.

 In its divergence from habits typical of the Lauraceae, "Cassytha" also presents examples of mosaic evolution
</item 3>

<item 4>
Cassiope
Cassiope is a genus of 9-12 small shrubby species in the family Ericaceae.

 They are native to the Arctic and north temperate montane regions.

 The genus is named after Cassiopeia of Greek mythology.

 Common names, shared with several other similar related genera, include heather and heath.

 They have scale-like leaves lying against the stems, and produce solitary bell-shaped flowers in late spring.

 Though hardy, flowers can be damaged by late frosts.
</item 4>

<item 5>
Butea
Butea is a genus of flowering plants belonging to the pea family, Fabaceae.

 It is sometimes considered to have only two species, "B. monosperma" and "B. superba", or is expanded to include four or five species.
</item 5>

<item 6>
Genoplesium
Genoplesium commonly known as midge orchids, is a genus of about 50 species of flowering plants in the orchid family, Orchidaceae and is found in Australia, New
Zealand and New Caledonia.

 Midge orchids are terrestrial herbs with a single leaf at the base of the plant.

 They are similar to orchids in the genus "Prasophyllum" in that plants without flowers have a hollow, onion-like leaf.

 The flowers are small but often scented and attractive to their insect pollinators.

 There is disagreement about which species belong to this genus and some taxonomists suggest that most belong in the genus "Corunastylis".
</item 6>

<item 7>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 7>

<item 8>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 8>

<item 9>
Psychopsis
Psychopsis, abbreviated Psychp in horticultural trade, is a genus of 5 known species of orchids native to northern South America, Central America and Trinidad.

 It was formerly included in the massively paraphyletic "wastebin genus" "Oncidium".

 The genus as a whole is commonly called butterfly orchids, but some species of other orchid genera are also called thus.
</item 9>

<item 10>
Abrus
Abrus is a genus of flowering plants in the pea family, Fabaceae and the only genus found in the tribe Abreae.

 It contains, 13–18 species, but is best known for a single species, Jequirity ("A. precatorius").

 The highly toxic seeds of that species are used to make jewellery.
</item 10>
=== END ITEM id=265c774fc7830ce9 ===

=== ITEM id=14049a6ad06df810 ===
Gold sentences and facts:
Sentence 0: The Sports Museum (also known as The Sports Museum of New England) is a non-profit museum currently located in the TD Garden in Boston, Massachusetts.
  Fact 0.0: The Sports Museum is also known as The Sports Museum of New England.
  Fact 0.1: The Sports Museum is a non-profit museum.
  Fact 0.2: The Sports Museum is currently located in the TD Garden.
  Fact 0.3: The TD Garden is in Boston, Massachusetts.
Sentence 1: TD Garden, often called Boston Garden and "The Garden", is a multi-purpose arena in Boston, Massachusetts.
  Fact 1.0: TD Garden is often called Boston Garden.
  Fact 1.1: TD Garden is often called "The Garden".
  Fact 1.2: TD Garden is a multi-purpose arena.
  Fact 1.3: TD Garden is in Boston, Massachusetts.
Sentence 2:  It opened in 1995 as a replacement for the original Boston Garden and has been known as Shawmut Center, FleetCenter, and TD Banknorth Garden.
  Fact 2.0: TD Garden opened in 1995.
  Fact 2.1: TD Garden opened as a replacement for the original Boston Garden.
  Fact 2.2: TD Garden has been known as Shawmut Center.
  Fact 2.3: TD Garden has been known as FleetCenter.
  Fact 2.4: TD Garden has been known as TD Banknorth Garden.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
Graph query row:
entity: The Sports Museum
entity_labels: ['__KGBuilder__', '__Entity__', 'Organization']
kind: non-profit museum
founded_on: 1977
held_on: None
relationship: LOCATED_IN
related_entity: TD Garden
related_kind: building
related_founded_on: None
related_held_on: None
</item 2>

<item 3>
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
</item 3>

<item 4>
Estonian Sports Museum
The Estonian Sports and Olympic Museum, founded in 1963 and modernised in 2001, is the largest sports museum in the Baltic states.

 The museum is located on Rüütli street in Tartu, Estonia and the museum has a small outpost in Otepää that concentrates on the Winter sports.

 Before 2016, Estonian Sports and Olympic Museum was named Estonian Sports Museum.
</item 4>

<item 5>
Smith Fieldhouse
The George Albert Smith Fieldhouse is a 5,000 seat multi-purpose arena in Provo, Utah.

 Built in 1951, it is the home of the Brigham Young University Cougars volleyball teams and most home gymnastics meets.

 It was named for George Albert Smith, the eighth president of The Church of Jesus Christ of Latter-day Saints, who died the year the fieldhouse opened.

 Prior to the Marriott Center opening in 1971 it was home to the basketball teams.

 At that time, the arena held 10,500 people.

 Smith Fieldhouse also has a track and several offices used by BYU's athletic department.
</item 5>

<item 6>
Sports Museum of America
The Sports Museum of America (SmA) was the United States' first national sports museum dedicated to the history and cultural significance of sports in America.

 It opened in May 2008 and closed less than nine months later, in February 2009.
</item 6>

<item 7>
Sports Legends Museum at Camden Yards
Sports Legends Museum at Camden Yards was a non-profit sports museum in Baltimore, Maryland, United States, owned and operated by the Babe Ruth Birthplace &
Museum.

 It opened on May 14, 2005, with the daughter of celebrated baseball player Babe Ruth in attendance.

 After ten years of operation the museum closed abruptly on October 12, 2015 after failing to reach an agreement with the Maryland Stadium Authority for the
continued use of Camden Station.

 The 22000 sqft museum was adjacent to the main gate of Oriole Park at Camden Yards and has artifacts and interactive exhibits profiling Maryland’s sports
history.

 Exhibits included such area teams as the Baltimore Orioles, Baltimore Ravens, Baltimore Colts, Maryland Terrapins, Baltimore Elite Giants, Baltimore Black Sox,
and the Baltimore Blast.
</item 7>

<item 8>
Camden Yards Sports Complex
The Camden Yards Sports Complex is located in the center of Baltimore, Maryland.

 The complex is composed of multiple buildings and stadiums including Oriole Park at Camden Yards and M&T Bank Stadium.

 The two stadiums are home to the Baltimore Orioles of Major League Baseball and the Baltimore Ravens of the National Football League.

 The complex still houses the recently closed Sports Legends Museum at Camden Yards (a non-profit museum featuring Maryland sports teams).

 Along with the Sports Legends Museum, the Babe Ruth Birthplace and Museum is located approximately two blocks from the main entrance of Camden Yards at Eutaw
Street.

 Geppi's Entertainment Museum is also located in Camden Station, atop the Sports Legends at Camden Yards.

 The complex is more than museums and stadiums.

 It is a location for community events such as the Dew Tour's Panasonic Open in June 2007 and 2008, the Baltimore Marathon, and the African American Festival
which is held every year.
</item 8>

<item 9>
Fanattic Sports Museum
Fanattic Sports Museum (FSM) is a sports museum, dedicated to various types of sports and located in Ecospace Business Park, New Town, Kolkata, India.

 Established in January 2017, FSM is known as the first sports museum in Kolkata as well as in West Bengal.
</item 9>

<item 10>
Stadio del Ghiaccio Gianmario Scola
Stadio del Ghiaccio Gianmario Scola is an indoor sporting arena in Canazei, Italy.

 The capacity of the arena is 3,500 spectators.

 It is the home arena of the Sportiva Hockey Club Fassa ice hockey team of the Serie A ice hockey league.

 The arena also hosted games during the 1994 World Ice Hockey Championships along with the Forum di Assago in Milan and the PalaOnda in Bolzano.
</item 10>
=== END ITEM id=14049a6ad06df810 ===

=== ITEM id=be92ee284efae09c ===
Gold sentences and facts:
Sentence 0: Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.
  Fact 0.0: Heavenly Creatures is a 1994 New Zealand psychological drama.
  Fact 0.1: Heavenly Creatures was directed by Peter Jackson.
  Fact 0.2: Peter Jackson co-wrote the screenplay with his partner, Fran Walsh.
  Fact 0.3: Heavenly Creatures is about the notorious 1954 Parker-Hulme murder case.
  Fact 0.4: The Parker-Hulme murder case took place in Christchurch, New Zealand.
Sentence 1:  The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.
  Fact 1.0: The main premise of Heavenly Creatures deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme.
  Fact 1.1: Pauline Parker and Juliet Hulme murder Pauline Parker's mother.
Sentence 2: Anne Perry (born 28 October 1938 as Juliet Marion Hulme) is an English author of historical detective fiction, best known for her Thomas Pitt and William Monk series.
  Fact 2.0: Anne Perry was born on 28 October 1938 as Juliet Marion Hulme.
  Fact 2.1: Anne Perry is an English author of historical detective fiction.
  Fact 2.2: Anne Perry is best known for her Thomas Pitt and William Monk series.

Retrieved items:
<item 1>
Holly Hunter
Holly Hunter (born March 20, 1958) is an American actress and producer.

 For her performance as Ada McGrath in the 1993 film "The Piano", she won the Academy Award for Best Actress, BAFTA Award for Best Actress in a Leading Role,
AACTA Award for Best Actress in a Leading Role, Golden Globe Award for Best Actress – Motion Picture Drama, and the Cannes Best Actress Award.

 She was also nominated for the Academy Award for Best Actress for "Broadcast News" (1987), and the Academy Award for Best Supporting Actress for "The Firm"
(1993) and "Thirteen" (2003).
</item 1>

<item 2>
Heavenly Creatures
Heavenly Creatures is a 1994 New Zealand psychological drama directed by Peter Jackson, from a screenplay he co-wrote with his partner, Fran Walsh, about the
notorious 1954 Parker–Hulme murder case in Christchurch, New Zealand.

 The film features Melanie Lynskey and Kate Winslet in their screen debuts with supporting roles by Sarah Peirse, Diana Kent, Clive Merrison, and Simon
O'Connor.

 The main premise deals with the relationship between two teenage girls, Pauline Parker and Juliet Hulme, who murder Parker's mother.

 The events of the film cover the period from their meeting in 1952 to the murder in 1954.
</item 2>

<item 3>
Keri Hulme
Keri Hulme (born 9 March 1947) is a New Zealand writer.

 Her only novel, "The Bone People", won the Booker Prize in 1985.
</item 3>

<item 4>
Juliet Weber
Juliet Weber is a producer, director, writer, and an editor for the film industry.

 She co-produced films such as Fastpitch (2000) and Unchained Memories (2003).
</item 4>

<item 5>
Parker–Hulme murder case
The Parker–Hulme murder case began in the city of Christchurch, New Zealand, on 22 June 1954, when Honora Rieper (also known as Honora Parker, her legal name)
was killed by her teenaged daughter, Pauline Parker, and Pauline's close friend Juliet Hulme (later known as Anne Perry).

 Parker was 16 at the time, while Hulme was 15.
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
Peter Elliott (actor)
Peter Elliott is a leading New Zealand actor.

 He has appeared in numerous television shows including "Shortland Street", "Gloss" and .

 He has also appeared in several movies including "Heavenly Creatures".

 Peter has a daughter Lucy Elliott who is an actress, playing character Dayna Jenkins on "Shortland Street" in 2013.
</item 7>

<item 8>
Kate Winslet filmography
Kate Winslet is a British actress and singer who has appeared in numerous films and television series.

 Her film debut was as Juliet Hulme in the 1994 film "Heavenly Creatures".

 She went on to appear in the later 1990s films "Sense and Sensibility" as Marianne Dashwood (1995), which earned her a nomination for the Academy Award for
Best Supporting Actress, "Jude" as Sue Bridehead (1996), "Hamlet" as Ophelia (1996) and "Titanic" as Rose DeWitt Bukater (1997).

 For her role in "Titanic", it earned her Best Actress nominations for an Academy Award, Golden Globe and Screen Actors Guild Award.
</item 8>

<item 9>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 9>

<item 10>
Juno Temple
Juno Temple (born 21 July 1989) is an English actress.

 She has appeared in films such as " Black Mass", "The Other Boleyn Girl", "Wild Child", "Atonement", "Maleficent", "The Three Musketeers" and "The Dark Knight
Rises".

 She also had a starring role in the HBO period drama series "Vinyl".
</item 10>
=== END ITEM id=be92ee284efae09c ===

=== ITEM id=389e13ca39fce437 ===
Gold sentences and facts:
Sentence 0: Migene González-Wippler is a Puerto Rican new-age author and a leading expert on the Afro-Caribbean religion of Santería.
  Fact 0.0: Migene González-Wippler is a Puerto Rican.
  Fact 0.1: Migene González-Wippler is a new-age author.
  Fact 0.2: Migene González-Wippler is a leading expert on the Afro-Caribbean religion of Santería.
Sentence 1: Santería, also known as Regla de Ochá, La Regla de Ifá, or Lucumi, is an Afro-American religion of Caribbean origin that developed in the Spanish Empire among West African descendants.
  Fact 1.0: Santería is also known as Regla de Ochá.
  Fact 1.1: Santería is also known as La Regla de Ifá.
  Fact 1.2: Santería is also known as Lucumi.
  Fact 1.3: Santería is an Afro-American religion.
  Fact 1.4: Santería is a religion of Caribbean origin.
  Fact 1.5: Santería developed in the Spanish Empire.
  Fact 1.6: Santería developed among West African descendants.

Retrieved items:
<item 1>
Migene González-Wippler
Migene González-Wippler is a Puerto Rican new-age author and a leading expert on the Afro-Caribbean religion of Santería.

 González-Wippler was born in Puerto Rico and has degrees in psychology and anthropology from the University of Puerto Rico and from Columbia University.

 In addition to her solid background in social sciences she has also worked as a science editor for the Interscience Division of John Wiley, the American
Institute of Physics, and the American Museum of Natural History, and as an English editor for the United Nations in Vienna, where she resided for many years.

 She is a cultural anthropologist and lectures frequently at universities and other educational institutions.

 She also has contributed extensively to the Institute of Puerto Rican Culture's collection, especially when it comes to Santeria and its practices, beliefs and
organization.
</item 1>

<item 2>
Ewe people
The Ewe people (Ewe: "Eʋeawó" , lit.

 "Ewe people"; or " Eʋedukɔ́ ", lit.

 "Ewe nation","Eʋenyigba" Eweland;) are an African ethnic group.

 They are the largest ethnic group in Togo (32%), the third largest ethnic group in Ghana (14%), and are a minority ethnic group in southern Benin, Ivory Coast
and Nigeria.

 They speak the Ewe language (Ewe: "Eʋegbe" ) which belongs to the Niger-Congo family of languages.

 They are related to other speakers of Gbe languages, such as, the Fon, Gen, Phla Phera, and the Aja people of Togo and Benin.
</item 2>

<item 3>
Ghegs
The Ghegs or Gegs (Albanian: "Gegët" ) are one of two major ethnic subgroups of Albanians (the other being the Tosks) differentiated by their cultural,
linguistic, social and religious characteristics.

 The Ghegs live in Albania (north of the Shkumbin river), Kosovo, Macedonia and Montenegro.

 The name Gheg is derived from the term initially used by Orthodox population of pre-Ottoman Albania for confessional denotation when referring to their
Catholic neighbors who converted to Catholicism to better resist the Orthodox Serbs.

 The Ghegs speak Gheg Albanian, one of the two main dialects of Albanian language.

 The social organization of the Ghegs was traditionally tribal, with several distinct tribal groups of Ghegs.
</item 3>

<item 4>
Ethnic religion
In religious studies, ethnic religion (or indigenous religion) is the expression of religion associated with a particular ethnic group.

 Ethnic religions are often distinguished from religions which claim to not be limited in ethnic or national scope, such as Christianity or Islam.

 Ethnic religions do not have to be excluded to independent religions.

 Some localized denominations of global religions are practiced solely by certain ethnic groups, with ethnic groups like the Assyrians following a unique
denominational structure of Christianity known as the Assyrian Church of the East.
</item 4>

<item 5>
Mazahua language
The Mazahua language is an indigenous language of Mexico, spoken in the country's central states by the ethnic group that is widely known as the Mazahua but
calls itself the Hñatho.

 It is a Mesoamerican language and has many of the traits of the Mesoamerican Linguistic Area.

 In 2003, along with some 62 other indigenous languages, it was recognised by a statutory law of Mexico (General Law of Linguistic Rights of the Indigenous
Peoples) as an official language in the Federal District and the other administrative divisions in which it is spoken, and on an equal footing with Spanish.

 The largest concentration of Mazahua is found in the municipality of San Felipe del Progreso, State of México, near Toluca.
</item 5>

<item 6>
Santería
Santería, also known as Regla de Ochá, La Regla de Ifá, or Lucumi, is an Afro-American religion of Caribbean origin that developed in the Spanish Empire among
West African descendants.

 "Santeria" is a Spanish word that means the "worship of saints".

 Santería is influenced by and syncretized with Roman Catholicism.

 Its sacred language, a variety of Yoruba, is the Lucumí language.
</item 6>

<item 7>
Church of the Guanche People
The Church of the Guanche People (Spanish: "Iglesia del Pueblo Guanche" ) is a religious organisation, founded in 2001 in the city of San Cristóbal de La Laguna
(Tenerife, Canary Islands, Spain).

 It aims to perpetuate and spread the pagan religion of the ancient Guanche people.
</item 7>

<item 8>
Maléku language
The Maléku Jaíka language, also called Guatuso, Watuso-Wétar, and Guetar, is an indigenous language of north central Costa Rica.

 It is a Chibchan language and Votic language spoken by around 300 to 750 indigenous Maléku people.

 This language is considered to be endangered according to The Endangered Languages Project.

 Corobicí is possibly a dialect.
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
=== END ITEM id=389e13ca39fce437 ===

=== ITEM id=03a5ed94236d99e8 ===
Gold sentences and facts:
Sentence 0: Haringey Heartlands is a currently ongoing urban regeneration project in the London Borough of Haringey, in north London, England.
  Fact 0.0: Haringey Heartlands is an urban regeneration project.
  Fact 0.1: Haringey Heartlands is currently ongoing.
  Fact 0.2: Haringey Heartlands is in the London Borough of Haringey.
  Fact 0.3: The London Borough of Haringey is in north London, England.
Sentence 1: The Mall Wood Green is a large shopping centre and residential complex in Wood Green, north London.
  Fact 1.0: The Mall Wood Green is a large shopping centre.
  Fact 1.1: The Mall Wood Green is a residential complex.
  Fact 1.2: The Mall Wood Green is in Wood Green, north London.

Retrieved items:
<item 1>
Heartlands High School
Heartlands High School is a state secondary school located in Wood Green in the London Borough of Haringey.

 It welcomed its first students in September 2010.

 The headteacher is Mr. Simon Garrill.
</item 1>

<item 2>
London Skolars
London Skolars R.L.F.C. are a professional rugby league club based at the New River Stadium, Wood Green, Haringey in north London.

 They were founded in 1995 and have been professional since 2003, operating in the Kingstone Press League 1 in the 2015 season.

 They also run an A-team (formerly known as "Haringey Hornets") that play in the South Premier Division of the Rugby League Conference.
</item 2>

<item 3>
Haringey Heartlands
Haringey Heartlands is a currently ongoing urban regeneration project in the London Borough of Haringey, in north London, England.

 The site is located west of Shopping City in Wood Green.

 The aim of the development is "to create a vibrant and attractive new urban quarter which acts as the civic and cultural Heart of Haringey, integrating with
and benefiting wider communities."
</item 3>

<item 4>
Haringey London Borough Council
Haringey London Borough Council is the local authority for the London Borough of Haringey in Greater London, England.

 It is a London borough council, one of 32 in the United Kingdom capital of London.

 Haringey is divided into 19 wards, each electing three councillors.

 Haringey London Borough Council comprises 48 Labour Party councillors, and 9 Liberal Democrats.

 The council was created by the London Government Act 1963 and replaced three local authorities: Hornsey Borough Council, Tottenham Borough Council and Wood
Green Borough Council.

 The next election to the authority will be in 2018.
</item 4>

<item 5>
Wood Green War Memorial
The Wood Green War Memorial is located on the High Road, Wood Green, in the London Borough of Haringey.

 It was erected in 1920 and paid for by public subscription of the people of Wood Green to remember the men of the area who died during the First World War.

 It was subsequently adapted to include the dead of the Second World War.

 It is grade II listed with Historic England.
</item 5>

<item 6>
Wood Green tube station
Wood Green is a London Underground station on the Piccadilly line.

 The station is between Turnpike Lane and Bounds Green stations and is in Travelcard Zone 3.

 It is located at junction of High Road, Wood Green and Lordship Lane.

 It serves Wood Green Shopping City and the nearby Haringey Council administrative complex as well as a densely populated residential area.

 It is also the closest tube station to the Alexandra Palace Venue.
</item 6>

<item 7>
Wood Green Town F.C.
Wood Green Town Football Club was a football club based in Tottenham in the London Borough of Haringey, England.

 Formed in 1911 as a breakaway from Tufnell Park, they merged with Edmonton (a descendent of Tufnell Park) in 1973 to form Edmonton & Haringey.

 During their history the club were known as Tufnell Spartans, Wood Green, Wood Green Town and Haringey Borough.
</item 7>

<item 8>
White Hart Lane Community Sports Centre
White Hart Lane Community Sports Centre, also known as the New River Stadium is a rugby league and athletics stadium in Wood Green, Haringey, north London,
England that is home to London Skolars rugby league club, Enfield and Haringey Athletic Club, Wood Green Weightlifting Club, Hashtag United F.C. (YouTube based
football club), Haringey Rhinos rugby union club, Haringey Cycling Club, and the Next Level Football League.
</item 8>

<item 9>
The Mall Wood Green
The Mall Wood Green is a large shopping centre and residential complex in Wood Green, north London.

 It is generally still referred to by its former name of Wood Green Shopping City; the signage on the building still uses that name.
</item 9>

<item 10>
Alexandra Palace railway station
Alexandra Palace railway station (originally named Wood Green and later Wood Green (Alexandra Park)) is in the London Borough of Haringey in north London, and
is in Zone 3.

 The station, and all trains serving it, are operated by Great Northern.

 Trains mostly go to and from Moorgate on weekdays and weekends.
</item 10>
=== END ITEM id=03a5ed94236d99e8 ===

=== ITEM id=23d32f2049079f01 ===
Gold sentences and facts:
Sentence 0: Aram Saroyan (born September 25, 1943) is an American poet, novelist, biographer, memoirist and playwright, who is especially known for his minimalist poetry, famous examples of which include the one-word poem "lighght" and a one-letter poem comprising a four-legged version of the letter "m".
  Fact 0.0: Aram Saroyan was born on September 25, 1943.
  Fact 0.1: Aram Saroyan is an American poet.
  Fact 0.2: Aram Saroyan is an American novelist.
  Fact 0.3: Aram Saroyan is an American biographer.
  Fact 0.4: Aram Saroyan is an American memoirist.
  Fact 0.5: Aram Saroyan is an American playwright.
  Fact 0.6: Aram Saroyan is especially known for his minimalist poetry.
  Fact 0.7: A famous example of Aram Saroyan's minimalist poetry is the one-word poem "lighght".
  Fact 0.8: A famous example of Aram Saroyan's minimalist poetry is a one-letter poem comprising a four-legged version of the letter "m".
Sentence 1: Dr. Edward Keonjian (14 August 1909 – 6 September 1999) was a prominent engineer, an early leader in the field of low-power electronics, the father of microelectronics.
  Fact 1.0: Dr. Edward Keonjian was born on 14 August 1909.
  Fact 1.1: Dr. Edward Keonjian died on 6 September 1999.
  Fact 1.2: Dr. Edward Keonjian was a prominent engineer.
  Fact 1.3: Dr. Edward Keonjian was an early leader in the field of low-power electronics.
  Fact 1.4: Dr. Edward Keonjian was the father of microelectronics.

Retrieved items:
<item 1>
Edward Keonjian
Dr. Edward Keonjian (14 August 1909 – 6 September 1999) was a prominent engineer, an early leader in the field of low-power electronics, the father of
microelectronics.

 In 1954 Keonjian designed the world's first solar-powered, pocket-sized radio transmitter.

 In 1959 Keonjian designed the first prototype of integrated circuit.

 In 1963 he organized the world's first international symposium on low-power electronics.

 Later on Keonjian collaborated with NASA astronaut Neil Armstrong as chief of failure analysis on the Apollo 11 project.
</item 1>

<item 2>
Edmund Mortimer, 5th Earl of March
Edmund de Mortimer, 5th Earl of March and 7th Earl of Ulster (6 November 1391 – 18 January 1425) was an English nobleman.

 A great-great-grandson of King Edward III of England, he was heir presumptive to King Richard II of England, his first cousin twice removed, when Richard II
was deposed in favour of Henry IV.

 Edmund Mortimer's claim to the throne was the basis of rebellions and plots against Henry IV and his son Henry V, and was later taken up by the House of York
in the Wars of the Roses, though Mortimer himself was an important and loyal vassal of Henry V and Henry VI.

 Edmund Mortimer was the last Earl of March of the Mortimer family.
</item 2>

<item 3>
Daron Malakian
Daron Vartan Malakian (Armenian: Տարօն/Տարոն Վարդան Մալաքեան, born July 18, 1975) is an Armenian–American singer-songwriter, multi-instrumentalist, and record
producer.

 He is best known as the guitarist, songwriter and second vocalist of the rock band System of a Down and as the lead vocalist, lead guitarist and songwriter of
the band Scars on Broadway.

 Like the rest of the Hollywood-based band System of a Down, he is of Armenian ancestry, but he is the only member to have been born in the United States.

 Daron Malakian is known for his distinctive playing and is ranked 40th in Loudwire's list of "Top 50 Hard Rock + Metal Guitarists Of All Time" and #11 in
MusicRadar's poll, "The 20 Greatest Metal Guitarists Ever".

 He is placed 30th in Guitar World's List of The 100 Greatest Heavy Metal Guitarists of All Time.
</item 3>

<item 4>
Edmund Mortimer, 2nd Baron Mortimer
Edmund Mortimer, 2nd Lord Mortimer (1251 – 17 July 1304) was the second son and eventual heir of Roger Mortimer, 1st Baron Mortimer.

 His mother was Maud de Braose.

 As a younger son, Edmund had been intended for clerical or monastic life, and had been sent to study at Oxford University.
</item 4>

<item 5>
Strawberry Saroyan
Strawberry Saroyan (born 1970) is a journalist and author.

 The daughter of Aram Saroyan and granddaughter of playwright William Saroyan and actress Carol Matthau, she spent her childhood in Bolinas, California.

 She has a sister named Cream.

 She writes for the New York Times Style section and is the author of "".
</item 5>

<item 6>
Aram Saroyan
Aram Saroyan (born September 25, 1943) is an American poet, novelist, biographer, memoirist and playwright, who is especially known for his minimalist poetry,
famous examples of which include the one-word poem "lighght" and a one-letter poem comprising a four-legged version of the letter "m".
</item 6>

<item 7>
My Name Is Aram
My Name is Aram is a book of short stories by William Saroyan first published in 1940.

 The stories detail the exploits of Aram Garoghlanian, a boy of Armenian descent growing up in Fresno, California, and the various members of his large family.

 This book is assigned reading in some schools.
</item 7>

<item 8>
David M. Arden
David M. Arden (born September 6, 1949) is an American concert pianist whose performing and recording career has focused predominantly on contemporary and
American classical repertoire, including premiere performances and first recordings of piano works by a number of notable contemporary composers, such as Henryk
Górecki, Luciano Berio, Earle Brown, Carson Kievman and David Lang.

 The Russian composer Galina Ustvolskaya named Arden's recording of her 12 Preludes for Piano as her preferred recording of that work.
</item 8>

<item 9>
Edmund Mortimer (1302–1331)
Sir Edmund Mortimer (1302/1303 – 16 December 1331) was the eldest son of Roger Mortimer, 1st Earl of March and Joan de Geneville, 2nd Baroness Geneville.

 By his wife Elizabeth de Badlesmere he was the father of Roger Mortimer, 2nd Earl of March.

 Though Edmund survived his father by one year, he did not inherit his father's lands and titles as they were forfeited to the Crown and his son only reacquired
them gradually.
</item 9>

<item 10>
Ted Kooshian
Ted Kooshian (born October 8, 1961) is an American jazz pianist, keyboardist, and composer, who has performed with Aretha Franklin, Chuck Berry, Marvin
Hamlisch, Sarah Brightman, Il Divo, and Blood, Sweat, and Tears.

 Kooshian has played in many Broadway pit orchestras, and has been a member of the Ed Palermo Big Band since 1994.

 Originally from San Jose, California, Kooshian has been performing since the 1980s.
</item 10>
=== END ITEM id=23d32f2049079f01 ===

=== ITEM id=05f6bc9dddf0da3e ===
Gold sentences and facts:
Sentence 0: Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).
  Fact 0.0: Blinding Edge Pictures is an American film production company.
  Fact 0.1: Blinding Edge Pictures was founded in 2000 by M. Night Shyamalan.
  Fact 0.2: Blinding Edge Pictures is known for producing films written and directed by Shyamalan.
  Fact 0.3: Blinding Edge Pictures produced Unbreakable (2000), written and directed by Shyamalan.
  Fact 0.4: Blinding Edge Pictures produced Signs (2002), written and directed by Shyamalan.
  Fact 0.5: Blinding Edge Pictures produced The Village (2004), written and directed by Shyamalan.
  Fact 0.6: Blinding Edge Pictures produced The Happening (2008), written and directed by Shyamalan.
  Fact 0.7: Blinding Edge Pictures produced After Earth (2013), written and directed by Shyamalan.
  Fact 0.8: Blinding Edge Pictures produced The Visit (2015), written and directed by Shyamalan.
  Fact 0.9: Blinding Edge Pictures produced Split (2017), written and directed by Shyamalan.
Sentence 1: Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson, alongside Robin Wright and Spencer Treat Clark.
  Fact 1.0: Unbreakable is a 2000 American superhero thriller film.
  Fact 1.1: Unbreakable was written by M. Night Shyamalan.
  Fact 1.2: Unbreakable was produced by M. Night Shyamalan.
  Fact 1.3: Unbreakable was directed by M. Night Shyamalan.
  Fact 1.4: Unbreakable stars Bruce Willis.
  Fact 1.5: Unbreakable stars Samuel L. Jackson.
  Fact 1.6: Robin Wright is in the cast of Unbreakable.
  Fact 1.7: Spencer Treat Clark is in the cast of Unbreakable.

Retrieved items:
<item 1>
Cop Out (2010 film)
Cop Out is a 2010 American buddy cop action-comedy film directed and edited by Kevin Smith, written by Mark and Robb Cullen and starring Bruce Willis, Tracy
Morgan, Kevin Pollak and Seann William Scott.

 The plot revolves around two veteran NYPD partners (Willis and Morgan) on the trail of a stolen, rare, mint-condition baseball card who find themselves up
against a relentless, memorabilia-obsessed bloodthirsty gangster.

 This is the first film that Smith directed that he did not also write.

 Upon its release, the film was met with negative reviews by critics and underperformed at the box office.

 It is nevertheless Kevin Smith's highest grossing film.
</item 1>

<item 2>
Mercury Rising
Mercury Rising is a 1998 American political action thriller film starring Bruce Willis and Alec Baldwin.

 Directed by Harold Becker, the movie is based on Ryne Douglas Pearson's 1996 novel originally published as "Simple Simon".

 Willis plays Art Jeffries, an undercover FBI agent who protects a 9-year-old boy with autism who is targeted by government assassins after he cracks a top
secret government code.
</item 2>

<item 3>
Signs (film)
Signs is a 2002 American science fiction horror film written and directed by M. Night Shyamalan and executive produced by Shyamalan, Frank Marshall, Kathleen
Kennedy and Sam Mercer.

 A joint collective effort to commit to the film's production was made by Blinding Edge Pictures and The Kennedy/Marshall Company.

 It was commercially distributed by Touchstone Pictures theatrically, and by Touchstone Home Entertainment in home media format.

 Its story focuses on a former Episcopal priest named Graham Hess, played by Mel Gibson, who discovers a series of crop circles in his cornfield.

 Hess slowly discovers that the phenomena are a result of extraterrestrial life.

 It also stars Joaquin Phoenix, Rory Culkin, and Abigail Breslin.

 "Signs" explores faith, kinship, and extraterrestrials.
</item 3>

<item 4>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 4>

<item 5>
The Whole Nine Yards (film)
The Whole Nine Yards is a 2000 American-Canadian crime comedy film directed by Jonathan Lynn and starring Bruce Willis, Matthew Perry, Amanda Peet, Michael
Clarke Duncan and Natasha Henstridge.

 The title derives from a popular expression of uncertain origin.

 A sequel, "The Whole Ten Yards", was released in 2004.
</item 5>

<item 6>
Die Hard with a Vengeance
Die Hard with a Vengeance is a 1995 American action film and the third in the "Die Hard" film series.

 It was co-produced and directed by John McTiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and stars Bruce Willis as New York City Police
Department Lieutenant John McClane, Samuel L. Jackson as McClane's reluctant partner Zeus Carver, and Jeremy Irons as Simon Gruber.

 It was released on May 19, 1995, five years after "Die Hard 2", becoming the highest-grossing film at the worldwide box-office that year, but received mixed
reviews.

 It was followed by "Live Free or Die Hard" and "A Good Day to Die Hard" in 2007 and 2013, respectively.
</item 6>

<item 7>
Unbreakable (film)
Unbreakable is a 2000 American superhero thriller film written, produced, and directed by M. Night Shyamalan, and starring Bruce Willis and Samuel L. Jackson,
alongside Robin Wright and Spencer Treat Clark.

 The movie is the first installment in a trilogy.

 In "Unbreakable", a security guard named David Dunn survives a horrific train crash.

 After the incident, with the help of a manipulative disabled comic book shop owner named Elijah Price, he learns that he possesses superhuman powers.

 As Dunn explores and reluctantly confronts his powers while trying to navigate a difficult family life, he begins to fight crime and learns the true nature of
Elijah Price.
</item 7>

<item 8>
Blinding Edge Pictures
Blinding Edge Pictures is an American film production company, founded in 2000 by M. Night Shyamalan, which is known for producing films written and directed by
Shyamalan like "Unbreakable" (2000), "Signs" (2002), "The Village" (2004), "The Happening" (2008), "After Earth" (2013), "The Visit" (2015) and "Split" (2017).

 In 2015, the company released its first television series "Wayward Pines".
</item 8>

<item 9>
Last Man Standing (1996 film)
Last Man Standing is a 1996 American action thriller film written and directed by Walter Hill and starring Bruce Willis, Christopher Walken and Bruce Dern.

 It is a credited remake of Akira Kurosawa's "Yojimbo".
</item 9>

<item 10>
Striking Distance
Striking Distance is a 1993 American action thriller film starring Bruce Willis as Pittsburgh Police homicide detective Thomas Hardy.

 The film co-stars Sarah Jessica Parker, Dennis Farina, and Tom Sizemore.

 It was directed by Rowdy Herrington and written by Herrington and Marty Kaplan.

 The film was shot on location throughout Pittsburgh; its early title was "Three Rivers".
</item 10>
=== END ITEM id=05f6bc9dddf0da3e ===

=== ITEM id=3a3d7123af6199f5 ===
Gold sentences and facts:
Sentence 0: On February 25, 2017 he lost to Jarrett Hurd for the International Boxing Federation light middleweight world title.
  Fact 0.0: On February 25, 2017, this person lost to Jarrett Hurd.
  Fact 0.1: The February 25, 2017 loss was for the International Boxing Federation light middleweight world title.
Sentence 1: Jarrett Hurd (born August 31, 1990) is an American light middleweight boxer from Accokeek, Maryland.
  Fact 1.0: Jarrett Hurd was born on August 31, 1990.
  Fact 1.1: Jarrett Hurd is an American light middleweight boxer.
  Fact 1.2: Jarrett Hurd is from Accokeek, Maryland.

Retrieved items:
<item 1>
Tony Harrison (boxer)
On February 25, 2017 he lost to Jarrett Hurd for the International Boxing Federation light middleweight world title.

 The fight was originally supposed to be a title eliminator for the right to fight Jermall Charlo, but when Charlo vacated the title, the fight was elevated to
be for the vacant title.

 Harrison boxed well for majority of the first half of the fight but began to fade thereafter the finish came in round 9 when Hurd landed a right hand that
floored Harrison,
</item 1>

<item 2>
Steve Foster (boxer)
Steve "The Viking" Foster ( (1960--) 28 1960 (age 56 ) ) born in Salford is a retired English professional boxer of the 1980s and '90s who won the Commonwealth
light middleweight title and IBF Inter-Continental light middleweight title, and was a challenger for the World Boxing Association (WBA) Inter-Continental light
middleweight title against Shaun Cummins, BBBofC British light middleweight title against Robert McCracken, International Boxing Federation (IBF)
Inter-Continental light middleweight title against Bahre Ahmeti, World Boxing Organization (WBO) light middleweight title against Ronald "Winky" Wright, BBBofC
British middleweight title against Howard Eastman, World Boxing Federation (WBF) middleweight title against Cornelius Carr, and International Boxing
Organization (IBO) middleweight title against Mpush Makambi, his professional fighting weight varied from 146 lb , i.e. welterweight to 167 lb , i.e. super
middleweight.
</item 2>

<item 3>
Scott Dixon (boxer)
featherweight to 167+1/2 lb , i.e. Super middleweight.
</item 3>

<item 4>
Jarrett Hurd
Jarrett Hurd (born August 31, 1990) is an American light middleweight boxer from Accokeek, Maryland.
</item 4>

<item 5>
Nedal Hussein
Oriental and Pacific Boxing Federation (OPBF) featherweight title against Hiroyuki Enoki, and Oriental and Pacific Boxing Federation (OPBF) super featherweight
title against Takashi Uchiyama, his professional fighting weight varied from 117+1/4 lb , i.e. bantamweight to 132+3/4 lb , i.e. lightweight.
</item 5>

<item 6>
Scott Dixon (boxer)
"Super" Scott Dixon ( (1976--) 28 1976 (age (2017)-(1976)-((11)<(09)or(11)==(09)and(30)<(28)) ) ) born in Hamilton is a Scottish professional feather/super
feather/light/light welter/welter/light middle/middle/super middleweight boxer of the 1990s, 2000s and 2010s, who won the British Boxing Board of Control
(BBBofC) Scottish Area welterweight title, World Boxing Board (WBB) welterweight title, World Boxing Union (German Version) super middleweight title, and
Commonwealth welterweight title, and was a challenger for the World Athletic Association (WAA) welterweight title against Michael Carruth, BBBofC British
welterweight title against Derek Roche, World Boxing Federation (WBF) light middleweight title against Steve Roberts, World Boxing Organization (WBO)
Inter-Continental light middleweight title against Anthony Farnell, and World Boxing Union (WBU) light middleweight title against Mehrdad Takalobigashi, his
professional fighting weight varied from 125 lb , i.e.
</item 6>

<item 7>
Nedal Hussein
Nedal "Skinny" Hussein (born 1 December 1977) is an Australian professional bantam/super bantam/feather/super feather/Lightweight boxer of the 1990s and 2000s
who won the Australian super bantamweight title, Australian bantamweight title, International Boxing Federation (IBF) Pan Pacific featherweight title, World
Boxing Federation (WBF) featherweight title, World Boxing Union (WBU) super bantamweight title, World Boxing Organization (WBO) Asia Pacific super featherweight
title, International Boxing Federation (IBF) Pan Pacific super featherweight title, International Boxing Organization (IBO) Inter-Continental super
featherweight title, and Commonwealth super bantamweight title, and was a challenger for the World Boxing Council (WBC) International super bantamweight title
against Manny Pacquiao, World Boxing Council (WBC) super bantamweight title against Óscar Larios, World Boxing Organization (WBO) featherweight title against
Scott Harrison, Oriental and Pacific Boxing Federation (OPBF) featherweight title against Hiroyuki Enoki,
</item 7>

<item 8>
Kevin Kelly (boxer)
Kevin "Bones" Kelly (born August 7, 1969 in La Perouse, New South Wales) is an Australian retired professional welter/light middle/middleweight boxer of the
1990s and 2000s who won the South Australia State welterweight title, New South Wales (Australia) State welterweight title, New South Wales (Australia) State
light middleweight title, Australian light middleweight title, Pan Asian Boxing Association (PABA) light middleweight title, World Boxing Union (WBU)
middleweight title, and Commonwealth light middleweight title (twice), and was a challenger for the World Boxing Association (WBA) World light middleweight
title against David Reid, and World Boxing Organization (WBO) Inter-Continental light middleweight title against Gary Lockett, his professional fighting weight
varied from 142 lb , i.e. welterweight to 158+1/4 lb , i.e. middleweight.
</item 8>

<item 9>
Richard Williams (boxer)
Richard "The Secret" Williams ( (1971--) 09 1971 (age (2017)-(1971)-((11)<(05)or(11)==(05)and(30)<(09)) ) is an English professional light middle/middle/super
middleweight boxer of the 1990s, 2000s and 2010s who won the International Boxing Organization (IBO) light middleweight title, and Commonwealth light
middleweight title (twice), and was a challenger for the International Boxing Organization (IBO) light middleweight title against Sergio Gabriel Martinez, and
British Boxing Board of Control (BBBofC) British middleweight title against Howard Eastman, his professional fighting weight varied from 149+3/4 lb , i.e. light
middleweight to 162+1/4 lb , i.e. super middleweight.

 Richard Williams is trained by Brian Lawrence, and managed by Barry Hearn, and John Rooney.
</item 9>

<item 10>
Andre Thysse
Andre Thysse ( (1968--) 07 1968 (age (2017)-(1968)-((11)<(02)or(11)==(02)and(30)<(07)) ) ) born in Germiston, is a South African professional super middle/light
heavyweight boxer of the 1990s and 2000s who won the Gauteng super middleweight Title, South African super middleweight title, and Commonwealth super
middleweight title, and was a challenger for the International Boxing Organization (IBO) super middleweight title against Brian Magee, World Boxing Council
(WBC) super middleweight title against Markus Beyer, World Boxing Council (WBC) International super middleweight title against Mikkel Kessler, WBC International
super middleweight title against Jürgen Brähmer, World Boxing Council (WBC) Continental Americas super middleweight title against Lucian Bute, World Boxing
Council (WBC) International light heavyweight title against Adrian Diaconu, and International Boxing Federation (IBF) Australasian super middleweight title
against Sakio Bika, his professional fighting weight varied from 163+3/4 lb , i.e.
</item 10>
=== END ITEM id=3a3d7123af6199f5 ===

=== ITEM id=8eacd05b97dd257e ===
Gold sentences and facts:
Sentence 0: Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist Tayeb Salih.
  Fact 0.0: Season of Migration to the North is a classic post-colonial Sudanese novel.
  Fact 0.1: Season of Migration to the North is written by the novelist Tayeb Salih.
  Fact 0.2: The Arabic title of Season of Migration to the North is موسم الهجرة إلى الشمال, transliterated Mawsim al-Hiǧra ilā ash-Shamāl.
Sentence 1:  The novel is a counter narrative to "Heart of Darkness".
  Fact 1.0: Season of Migration to the North is a counter narrative to Heart of Darkness.
Sentence 2: Heart of Darkness (1899) is a novella by Polish-British novelist Joseph Conrad, about a voyage up the Congo River into the Congo Free State, in the heart of Africa, by the story's narrator Charles Marlow.
  Fact 2.0: Heart of Darkness is dated 1899.
  Fact 2.1: Heart of Darkness is a novella.
  Fact 2.2: Heart of Darkness is by Polish-British novelist Joseph Conrad.
  Fact 2.3: Heart of Darkness is about a voyage up the Congo River into the Congo Free State.
  Fact 2.4: The voyage in Heart of Darkness goes into the heart of Africa.
  Fact 2.5: The narrator of Heart of Darkness is Charles Marlow.

Retrieved items:
<item 1>
Robert Calder (writer)
Robert Lorin Calder, a Canadian writer and professor, won the Governor General's Award for English-language non-fiction in 1989 for his "Willie: The Life of W.
Somerset Maugham", a biography based on extensive archival work and interviews with surviving associates of Maugham, in particular Alan Searle.

 Unlike Ted Morgan, who had obtained permission from Maugham's executors to publish from Maugham's letters in his biography (1980), Calder was refused
permission to do so by the Royal Literary Fund and had to rely on paraphrase in referencing Maugham's unpublished correspondence.
</item 1>

<item 2>
Paradox (TV series)
Paradox is a 2009 British science fiction police drama, starring Tamzin Outhwaite as Detective Inspector Rebecca Flint.

 Written by Lizzie Mickery and produced by Clerkenwell Films for the BBC, it was filmed and set in Manchester, England.
</item 2>

<item 3>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 3>

<item 4>
First North Americans
First North Americans are a series of historical fiction novels published by Tor and written by husband and wife co-authors W. Michael Gear & Kathleen O'Neal
Gear.

 The series, which began with 1990's "People of the Wolf", explores various civilizations and cultures in prehistoric North America.

 It is comparable to Jean M. Auel's "Earth's Children" series, set in prehistoric Europe, but each of its books focuses on a different time period, location,
and set of characters.

 The first four novels form a coherent, more or less linear narrative, from the initial migration of Siberian peoples into what is now Canada and Alaska (People
of the Wolf) through the florescence of the Mississippian semi-urban mound-building culture, considered the "high-water mark" of North American pre-Columbian
civilization, around 1000 AD.

 The remaining novels cover a wide variety of times and settings, in no particular order, ranging from tropical Florida in the 6th millennium BC to the Chaco
Empire of the Southwest in the 13th century AD.
</item 4>

<item 5>
First North Americans
The novels take into account new developments in North American archaeology such as the discovery of Kennewick Man and the development of the coastal-route
model as a possible alternative or supplement to overland migration across Beringia.
</item 5>

<item 6>
Some Kind of Fairy Tale
Some Kind of Fairy Tale is a 2012 novel by the British author Graham Joyce.

 A work of speculative fiction, it won the British Fantasy Society's Fantasy Novel of the Year award (the Robert Holdstock Award) in 2013.

 Film rights to the novel have been sold and a potential movie is in the development stage.
</item 6>

<item 7>
Merlin's Wood
Merlin's Wood; or, The Vision of Magic is a short novel by British writer Robert Holdstock, first published in the United Kingdom in 1994.

 The novel is considered part of the "Mythago Wood" cycle, but takes place in Brittany, France instead of Herefordshire, England.

 The work has all new characters and focuses on the mythical birthplace and burial site of Merlin, the magical wood Brocéliande.

 Brocéliande is a smaller version of Ryhope wood where British myth predominates.
</item 7>

<item 8>
The Man from Snowy River (TV series)
The Man from Snowy River is an Australian adventure drama television series based on Banjo Paterson's poem "The Man from Snowy River".

 Released in Australia as "Banjo Paterson's The Man from Snowy River", the series was subsequently released in both the United States and the United Kingdom as
"Snowy River: The McGregor Saga".
</item 8>

<item 9>
Season of Migration to the North
Season of Migration to the North (Arabic: موسم الهجرة إلى الشمال‎ ‎ Mawsim al-Hiǧra ilā ash-Shamāl ) is a classic post-colonial Sudanese novel by the novelist
Tayeb Salih.

 Originally published in Arabic in 1966, it has since been translated into more than twenty languages.

 Salih was fluent in both English and Arabic, but chose to pen this novel in Arabic.

 The English translation was published in 1969 as part of the influential Heinemann African Writers Series.

 The novel is a counter narrative to "Heart of Darkness".

 It was described by Edward Said as one of the six great novels in Arabic literature.

 In 2001 it was selected by a panel of Arab writers and critics as the most important Arab novel of the twentieth century.
</item 9>

<item 10>
John Howard (author)
John Howard is an English author, born in London in 1961.

 His fiction has appeared in anthologies, magazines, and the collections "The Silver Voices", "Written by Daylight", and "Cities and Thrones and Powers".

 The majority of Howard's stories have central and eastern European settings; many are set in the fictional Romanian town of Steaua de Munte.

 "The Defeat of Grief" is a novella set in Steaua de Munte and the real Black Sea resort of Balcic; the novellas "The Fatal Vision" (in "Cities and Thrones and
Powers") and "The Lustre of Time" form part of an ongoing series with Steaua de Munte architect and academic Cristian Luca as protagonist.

 "Numbered as Sand or the Stars" attempts a 'secret history' of Hungary between the World Wars.
</item 10>
=== END ITEM id=8eacd05b97dd257e ===

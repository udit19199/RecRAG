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

=== ITEM id=7a73d3a04919247f ===
Gold sentences and facts:
Sentence 0: "White People Renovating Houses" is the first episode in the twenty-first season of the American animated television series "South Park".
  Fact 0.0: 'White People Renovating Houses' is the first episode of the twenty-first season of the American animated television series South Park.
Sentence 1:  It is the 278th episode of the series overall, and first aired on Comedy Central in the United States on September 13, 2017.
  Fact 1.0: 'White People Renovating Houses' is the 278th episode of South Park overall.
  Fact 1.1: 'White People Renovating Houses' first aired on Comedy Central in the United States.
  Fact 1.2: 'White People Renovating Houses' first aired on September 13, 2017.
Sentence 2: The twenty-first season of the American animated sitcom "South Park" premiered on Comedy Central on September 13, 2017, and will contain ten episodes.
  Fact 2.0: The twenty-first season of South Park is an American animated sitcom.
  Fact 2.1: The twenty-first season of South Park premiered on Comedy Central on September 13, 2017.
  Fact 2.2: The twenty-first season of South Park will contain ten episodes.

Retrieved items:
<item 1>
The Squirt and the Whale
"The Squirt and the Whale" is the nineteenth episode of "The Simpsons" twenty-first season.

 The 460th episode of the series overall, it originally aired on the Fox network in the United States on April 25, 2010.

 In the episode, the Simpson family attempt to save a beached whale that washed up onshore, only for it to die.
</item 1>

<item 2>
The Return of Chef
"The Return of Chef" is the first episode in the tenth season of the American animated television series "South Park".

 The 140th episode of the series overall, it first aired on Comedy Central in the United States on March 22, 2006.

 The episode was the first after the departure of actor Isaac Hayes, who voiced the character Chef.

 Hayes, a Scientologist, left after a falling-out with the creators over their treatment of Scientology in the previous season's episode "Trapped in the
Closet".

 Scientology has been accused of using brainwashing techniques, and the episode portrays Chef as having been brainwashed.
</item 2>

<item 3>
White People Renovating Houses
"White People Renovating Houses" is the first episode in the twenty-first season of the American animated television series "South Park".

 It is the 278th episode of the series overall, and first aired on Comedy Central in the United States on September 13, 2017.
</item 3>

<item 4>
Put It Down (South Park)
"Put It Down" is the second episode in the twenty-first season of the American animated television series "South Park".

 The 279th episode of the series overall, it first aired on Comedy Central in the United States on September 20, 2017.
</item 4>

<item 5>
Something Wall-Mart This Way Comes
"Something Wall-Mart This Way Comes" is the ninth episode of the eighth season of the American animated television series "South Park", and the 120th episode of
the series overall.

 It originally aired on Comedy Central in the United States on November 3, 2004.

 In the episode, a Wall-Mart is built in South Park, and the people start to get addicted to shopping from it, due to its irresistibly attractive bargains, thus
leading many businesses in South Park to close down.

 The four boys have to fight against Wall-Mart and to find a way to stop it from taking over the entire town.
</item 5>

<item 6>
Chickenlover
"Chickenlover" is the fourth episode in the second season of the American animated television series "South Park".

 The 16th episode of the series overall, it originally aired on Comedy Central in the United States on May 27, 1998.

 The episode was written by series co-creators Trey Parker and Matt Stone, along with David R. Goodman, and directed by Parker.

 In the episode, Officer Barbrady resigns as South Park's only police officer because of his illiteracy.

 Anarchy ensues, just as chickens are mysteriously being molested across South Park.

 Barbrady enlists the help of the boys to learn to read and discover who is molesting the chickens.

 Cartman, meanwhile, masquerades as a police officer.
</item 6>

<item 7>
South Park (season 21)
The twenty-first season of the American animated sitcom "South Park" premiered on Comedy Central on September 13, 2017, and will contain ten episodes.
</item 7>

<item 8>
South Park (season 10)
The tenth season of "South Park", an American animated television series created by Trey Parker and Matt Stone, began airing on March 22, 2006.

 The tenth season concluded after 14 episodes on November 15, 2006.

 This is the first season in which Kenny does not die and the last season featuring Isaac Hayes (the voice of Chef) as Hayes quit the show following the
backlash behind season nine's "Trapped in the Closet" episode.

 This season also had a minor controversy when the Halloween episode "Hell on Earth 2006" depicted "The Crocodile Hunter's" Steve Irwin with a stingray lodged
in his chest getting thrown out of Satan's Halloween party for not being in costume.

 Episode 2 in this season is the last one with the Braniff Airlines logo.

 All the episodes in this season were written and directed by Trey Parker.
</item 8>

<item 9>
Holiday Special (South Park)
"Holiday Special" is the third episode in the twenty-first season of the American animated television series "South Park".

 The 280th episode of the series overall, it first aired on Comedy Central in the United States on September 27, 2017.
</item 9>

<item 10>
Moho House
"Moho House" is the twenty first episode of the twenty-eighth season of the animated television series "The Simpsons", and the 617th episode of the series
overall.

 It aired in the United States on Fox on May 7, 2017.
</item 10>
=== END ITEM id=7a73d3a04919247f ===

=== ITEM id=a9f05bb4d01de3f6 ===
Gold sentences and facts:
Sentence 0: Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming pieces and a game system.
  Fact 0.0: Icehouse pieces are also called Icehouse Pyramids.
  Fact 0.1: Treehouse pieces are also called Treehouse Pyramids.
  Fact 0.2: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, and Treehouse Pyramids are officially called Looney Pyramids.
  Fact 0.3: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are nestable pyramid-shaped gaming pieces.
  Fact 0.4: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are stackable pyramid-shaped gaming pieces.
  Fact 0.5: Icehouse pieces, Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids, and Looney Pyramids are a game system.
Sentence 1:  The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
  Fact 1.0: The Looney Pyramids game system was invented by Andrew Looney.
  Fact 1.1: The Looney Pyramids game system was invented by John Cooper.
  Fact 1.2: The Looney Pyramids game system was invented in 1987.
  Fact 1.3: The Looney Pyramids game system was originally for use in the game of Icehouse.
Sentence 2:  In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
  Fact 2.0: "Kill Doctor Lucky" won the Origins Award in 1998.
  Fact 2.1: "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".

Retrieved items:
<item 1>
The Hunter (EP)
The Hunter is the first EP (and follow up to the 2010 solo album "The Boxer") by Kele Okereke (under the professional name Kele), lead singer and rhythm
guitarist of the British rock band Bloc Party.

 It was released on 7 November 2011 by Wichita Recordings in the UK, set back a week from the original release date, and 3 days earlier on 4 November 2011 by
Wichita Recordings and Liberator Music in Australia.

 The first single released from the EP was "What Did I Do?

", which features guest vocals from Lucy Taylor.

 The music video was released on 13 September 2011, with the song released as a digital single in Japan 10 days later.
</item 1>

<item 2>
Icehouse (song)
"Icehouse" is a song by the Australian rock band Flowers.

 It was released as a single in Europe in 1982 by Chrysalis Records from the band's first album, "Icehouse", after the band changed its name to Icehouse.

 In the United States, the song peaked at number 28 on the "Billboard" Top Tracks chart in 1981.
</item 2>

<item 3>
Fire and Ice (1983 film)
Fire and Ice is a 1983 American adult animated epic high fantasy adventure film directed by Ralph Bakshi.

 The film, a collaboration between Bakshi and Frank Frazetta, was distributed by 20th Century Fox, which also distributed Bakshi's 1977 release, "Wizards".

 The animated feature, based on characters Bakshi and Frazetta co-created, was made using the process of rotoscoping, in which scenes were shot in live action
and then traced onto animation cels.
</item 3>

<item 4>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 4>

<item 5>
Icehouse pieces
Icehouse pieces, or Icehouse Pyramids, Treehouse pieces, Treehouse Pyramids and officially Looney Pyramids, are nestable and stackable pyramid-shaped gaming
pieces and a game system.

 The game system was invented by Andrew Looney and John Cooper in 1987, originally for use in the game of Icehouse.
</item 5>

<item 6>
We Can Get Together
"We Can Get Together" is the second single released by the Australian rock band Flowers, later known as Icehouse.

 It was released in October 1980, on the independent label Regular Records from their first album, "Icehouse", two weeks before the album itself was released.

 It peaked at #16 on the Australian Kent Music Report Singles Charts.
</item 6>

<item 7>
Kill Doctor Lucky
Kill Doctor Lucky is a humorous board game designed by James Ernest and released in 1996 by Cheapass Games.

 In 1998, "Kill Doctor Lucky" won the Origins Award for "Best Abstract Board Game of 1997".
</item 7>

<item 8>
Boxes (Icehouse album)
Boxes is a soundtrack album by Australian band Icehouse, released by Festival Records / Chrysalis Records in November 1985.

 The work was originally conceived by its composers, Iva Davies and Robert Kretschmer, in collaboration with choreographer Graeme Murphy of the Sydney Dance
Company for performance as the ballet "Boxes".

 The first live performance of "Boxes" was given by the Sydney Dance Company together with Davies and Kretschmer of Icehouse and guest percussionist Masaki
Tanazawa in the Opera Theatre of the Sydney Opera House on 7 November 1985.

 "Boxes" was released in the US under the name Sydney Dance Company.

 The single "No Promises" was released as a 12" in November and a cassingle in December on Regular Records for Australian and New Zealand markets, it contained
other tracks from "Boxes"; two tracks from "Boxes" were later released on their next studio album "Measure for Measure" in April 1986.
</item 8>

<item 9>
Can't Help Myself
"Can't Help Myself" is the first single released by the Australian synthpop/rock band Flowers, later known as Icehouse.

 It was released in May 1980 as a 7" vinyl single on independent label, Regular Records, five months ahead of debut album "Icehouse".

 A 10" vinyl single was released in July and had a cover depicting individual images of band members diagonally across the band's name and the single's title
(see infobox at right middle).

 It peaked at #10 on the Australian Singles Charts.
</item 9>

<item 10>
No Promises (Icehouse song)
"No Promises" is the first single released by Australian band, Icehouse from the band's 1986 album, "Measure for Measure" and was released in November 1985 on
Regular Records as 7" Vinyl Single and 12" Vinyl Single formats.

 It was also released in the UK and Europe by Chrysalis Records again on 7" and 12" formats but with different track listings.

 It was subsequently released in the US by Chrysalis on 7" and 12" formats, again with different track listings.

 It peaked at #30 on the Australian singles charts.

 A remix version by [love] tatto was released on the Icehouse album "Meltdown" in 2002.
</item 10>
=== END ITEM id=a9f05bb4d01de3f6 ===

=== ITEM id=32e67c61de8b1474 ===
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
Aloha Dalire
Fabianne Pomaialoha Wong Dalire, known professionally as Aloha Dalire, (June 22, 1950 – August 6, 2014) was an American Hawaiian kumu hula, or master hula
teacher.

 She won the first Miss Aloha Hula as Aloha Wong, in 1971, the same year that the Merrie Monarch Festival was established.

 The Miss Aloha Hula title is hula's top solo wahine (women's) honor.
</item 1>

<item 2>
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
</item 2>

<item 3>
Suzanne Chun Oakland
Suzanne N.J. 'Susie' Chun Oakland is a Democratic member of the Hawaii Senate, representing the 13th District since 1996.

 Previously she was a member of the Hawaii House of Representatives from 1990-96.

 She is generally considered to have liberal political views.
</item 3>

<item 4>
Maryland General Assembly
The Maryland General Assembly is the state legislature of the U.S. state of Maryland that convenes within the State House in Annapolis.

 It is a bicameral body: the upper chamber, the Maryland State Senate, has 47 representatives and the lower chamber, the Maryland House of Delegates, has 141
representatives.

 Members of both houses serve four-year terms.

 Each house elects its own officers, judges the qualifications and election of its own members, establishes rules for the conduct of its business, and may
punish or expel its own members.
</item 4>

<item 5>
Charles Hastings Judd
Colonel Charles Hastings Judd (September 8, 1835 – April 18, 1890) was an American businessman, rancher, courtier and politician of the Kingdom of Hawaii.

 He served as Chamberlain and Colonel of the military staff of King Kalākaua and traveled with the monarch on his 1881 world tour.
</item 5>

<item 6>
Ophelia Ford
Ophelia Ford (born July 5, 1950) was a member of the Tennessee Senate.

 She is the younger sister of former state senator John Ford and former Congressman Harold Ford, Sr., and the aunt of former Congressman Harold Ford, Jr. She
represented Senate District 29, which covers South Memphis and North Memphis.
</item 6>

<item 7>
Hawaii State Capitol
The Hawaii State Capitol is the official statehouse or capitol building of the U.S. state of Hawaii.

 From its chambers, the executive and legislative branches perform the duties involved in governing the state.

 The Hawaii State Legislature—composed of the twenty-five member Hawaii State Senate led by the President of the Senate and the fifty-one member Hawaii State
House of Representatives led by the Speaker of the House—convenes in the building.

 Its principal tenants are the Governor of Hawaii and Lieutenant Governor of Hawaii, as well as all legislative offices and the Legislative Reference Bureau.
</item 7>

<item 8>
Hawaii State Legislature
The Hawaii State Legislature is the state legislature of the U.S. state of Hawaii.

 The state legislature is a bicameral body consisting of a lower house, the Hawaii State House of Representatives, with 51 representatives, and an upper house,
the 25-member Hawaii State Senate.

 There are a total of 76 representatives in the legislature, each representing single member districts across the islands.

 The powers of the legislature are granted under Article III of the Constitution of Hawaii.
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
Oregon Territorial Legislature
Oregon’s Territorial Legislature was a bicameral legislative body created by the United States Congress in 1848 as the legislative branch of the government of
the Oregon Territory.

 The upper chamber Council and lower chamber House of Representatives first met in July 1849; they served as the region's legislative body until Oregon became a
state in February 1859, when they were replaced by the bicameral Oregon State Legislature.
</item 10>
=== END ITEM id=32e67c61de8b1474 ===

=== ITEM id=963fdf21c4fd9887 ===
Gold sentences and facts:
Sentence 0: Lake Louisvilla is a neighborhood partially located in Louisville, Kentucky.
  Fact 0.0: Lake Louisvilla is a neighborhood.
  Fact 0.1: Lake Louisvilla is partially located in Louisville, Kentucky.
Sentence 1:  It is located between Westport Road in Louisville and KY 22 in Oldham County.
  Fact 1.0: Lake Louisvilla is located between Westport Road in Louisville.
  Fact 1.1: Lake Louisvilla is located between KY 22 in Oldham County.
Sentence 2: Oldham County is a county located in the commonwealth of Kentucky.
  Fact 2.0: Oldham County is a county.
  Fact 2.1: Oldham County is located in the commonwealth of Kentucky.
Sentence 3:  As of the 2010 census, the population was 60,316.
  Fact 3.0: As of the 2010 census, the population of Oldham County was 60,316.
Sentence 4:  Oldham County was a prohibition or completely dry county until January 2005 as the result of a 2004 'moist' vote, permitting sales of alcohol in restaurants that seat at least 100 patrons in which 70%+ of total revenue is derived from sales of food.
  Fact 4.0: Oldham County was a prohibition or completely dry county until January 2005.
  Fact 4.1: Oldham County stopped being completely dry in January 2005 as the result of a 2004 'moist' vote.
  Fact 4.2: The 2004 'moist' vote permitted sales of alcohol in restaurants that seat at least 100 patrons.
  Fact 4.3: The 2004 'moist' vote permitted alcohol sales only in restaurants where 70% or more of total revenue is derived from sales of food.

Retrieved items:
<item 1>
Oldham County, Kentucky
Oldham County is a county located in the commonwealth of Kentucky.

 As of the 2010 census, the population was 60,316.

 Its county seat is La Grange.

 The county is named for Colonel William Oldham.

 Oldham County was a prohibition or completely dry county until January 2005 as the result of a 2004 'moist' vote, permitting sales of alcohol in restaurants
that seat at least 100 patrons in which 70%+ of total revenue is derived from sales of food.

 After a vote in late 2015; Oldham county has become a completely wet county.
</item 1>

<item 2>
Henderson County, Kentucky
Henderson County is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 46,250.

 The county seat is Henderson.

 The county was formed in 1798 and named for Colonel Richard Henderson who purchased 17,000,000 acre of land from the Cherokee Indians, part of which would
eventually make up the county.
</item 2>

<item 3>
Lawrence, Nassau County, New York
Lawrence is a village in Nassau County, New York in the United States.

 As of the 2010 United States Census, the village population was 6,483.
</item 3>

<item 4>
Lake Worth, Florida
Lake Worth is a city in Palm Beach County, Florida, United States, which takes its name from the body of water along its eastern border, originally called "Lake
Worth", and now generally known as the Lake Worth Lagoon.

 The lake itself was named for General William J. Worth, who led U.S. forces during the last part of the Second Seminole War.

 s of 2010 , the population estimated by the U.S. Census Bureau was 34,910.

 The city is included in the Miami-Fort Lauderdale-Pompano Beach Metropolitan Area, which is home to approximately 5,563,857 people.
</item 4>

<item 5>
Lake Louisvilla, Louisville
Lake Louisvilla is a neighborhood partially located in Louisville, Kentucky.

 It is located between Westport Road in Louisville and KY 22 in Oldham County.

 Lake Louisvilla was developed in the 1920s as a summer resort for people living in the city of Louisville.

 The state of Kentucky drained the lake in the late 1980s due to safety concerns regarding the stability of a dam.
</item 5>

<item 6>
Squire Creek Country Club
The Squire Creek Lodge features two floors offering a combined total of seven bedrooms.

 The Squire Creek Pool overlooks the golf course and is served by the Waterside Cafe.

 The Squire Creek Development offers six residential estates including the Squire Creek Estates, Timberland Estates, Fairway Estates, The Fairways, Fairway
Villas, and The Park Homes.

 Squire Creek Country Club is the title sponsor of the Louisiana Peach Festival in Ruston.

 Squire Creek hosted the 2005 Western Athletic Conference Golf Championships.
</item 6>

<item 7>
Casey County, Kentucky
Casey County is a county located in the U.S. Commonwealth of Kentucky.

 As of the 2010 census, the population was 15,955.

 Its county seat is Liberty.

 The county was formed in 1806 from the western part of Lincoln County and named for Colonel William Casey, a pioneer settler who moved his family to Kentucky
in 1779.

 It is the only Kentucky county entirely in the Knobs region.

 Casey County is home to annual Casey County Apple Festival, and is a prohibition or dry county.

 It is considered part of the Appalachian region of Kentucky.
</item 7>

<item 8>
Garrard County, Kentucky
Garrard County ( ;) is a county located in the U.S. state of Kentucky.

 As of the 2010 census, the population was 16,912.

 Its county seat is Lancaster.

 The county was formed in 1796 and was named for James Garrard, Governor of Kentucky from 1796 to 1804.

 It is a prohibition or dry county but Lancaster is wet.

 Lancaster was founded as a settlement of log cabins in 1776 at a springs that later provided a constant source of water to early pioneers.

 It is one of the oldest cities in the Commonwealth.

 Boonesborough, 25 miles to the east, was founded by Daniel Boone in 1775.

 Lexington, 28 miles to the north, was founded in 1775.

 Stanford, originally known as St. Asaph, is 10 miles south of Lancaster.

 It too was founded in 1775.

 The oldest permanent settlemenet in Kentucky, Harrodsburg, was founded in 1774 and is 18 miles to the west.

 The present day courthouse is one of the oldest courthouses in Kentucky in continuous use.
</item 8>

<item 9>
Kentucky County, Virginia
Kentucky County (then alternately spelled Kentucke County) was formed by the Commonwealth of Virginia by dividing Fincastle County into three new counties:
Kentucky, Washington, and Montgomery, effective December 31, 1776.

 During the three and one-half years of Kentucky County's existence, its seat of government was Harrodstown (then also known as Oldtown, later renamed
Harrodsburg).
</item 9>

<item 10>
Wheeling Island
Wheeling Island is the most populated island in the Ohio River.

 It lies within the city of Wheeling in Ohio County, West Virginia, in the United States.

 The 2000 census showed a resident population of 3,142 people on the island, which has a land area of 1.514 km² (374 acres).

 Neville Island, PA is larger (3.4 km²) but has fewer people (1,232).
</item 10>
=== END ITEM id=963fdf21c4fd9887 ===

=== ITEM id=8a9b5bed3689c61d ===
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
HMS Monmouth
Seven ships of the Royal Navy have been named HMS "Monmouth".

 Monmouth was the name of a castle and is now the name of a town in Wales; the name also recognises James Scott, 1st Duke of Monmouth, the "Black Duke".

 The first HMS "Monmouth" was named after the town but whilst the second one was being used the Monmouth Rebellion took place and this caused some concern to
the crew.

 Since that time, all of "Monmouth"s have had their names painted in black and the name and the crest are covered because of the shame of the Duke of Monmouth's
execution.

 Today the officers of the latest HMS "Monmouth" carry black handkerchiefs and the ship is known as the Black Duke.

 However the town of Monmouth still gives the freedom of the town to the ship's crew and talks of the "enduring link" between the town and the ship.
</item 2>

<item 3>
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
</item 3>

<item 4>
Levitt &amp; Sons
Levitt & Sons was a real estate development company founded by Abraham Levitt and later managed by his son William Levitt.

 The company is most famous for having built the town of Levittown, New York.

 The company's designs and building practices totally revolutionized the entire Home Building Industry and altered America's landscape with massive suburban
communities.
</item 4>

<item 5>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 5>

<item 6>
Oakland Mills, New Jersey
Oakland Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The area is named for Oakland Mill, a mill along the Manalapan Brook operated by the Boo family.

 Some areas within the Manalapan Brook valley and surround forestland are preserved while the remainder of the area surrounding Oakland Mills has since been
developed into housing developments.
</item 6>

<item 7>
Monmouth Heights at Marlboro, New Jersey
Monmouth Heights at Marlboro is an unincorporated community located within Marlboro Township in Monmouth County, New Jersey, United States.

 The housing development was created in the 1960s by Levitt & Sons though some protests occurred by long-time residents of Marlboro Township at the time of its
creation.

 The community is located along New Jersey Route 79 across the highway from Marlboro High School.
</item 7>

<item 8>
Manhasset Stable
The Manhasset Stable was the "nom de course" for an American Thoroughbred horse racing stable established in the early 1930s by Joan Whitney Payson, founder of
the New York Mets baseball team and a member of the prominent New York City Whitney family who have been major figures in the sport for more than one hundred
years.

 Joan Payson named the stable for Manhasset, New York where she grew up.
</item 8>

<item 9>
Taylors Mills, New Jersey
Taylors Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The main road that runs through the area is named for the community, Taylors Mills Road.
</item 9>

<item 10>
Hanbys Corner, Delaware
Hanbys Corner is an unincorporated community in New Castle County, Delaware, United States.

 It is located at the intersection of Delaware Route 3 (Marsh Road) and Delaware Route 92 (Naamans Road), in Brandywine Hundred.

 The area is named for Richard G. Hanby, who first purchased the 125 acre parcel from the descendants of William Penn in 1753.

 His descendants included several notable figures in the local political scene including Samuel Winfield Hanby (1817-1892) who was elected as a State
Representative in 1874, Jacob Klose Hanby (1839-1932) who was Samuel's son and was elected State Representative in 1904, Robert Johnson Hanby (1834-1898), who
served in the 124th PA infantry during the Civil War and was elected State Senator in 1896, and Florence Wood Hanby (1870-1963), who was Robert's
daughter-in-law and was the first woman elected to the Delaware House of Representatives in 1924.

 In addition Wayne Hanby and James Hanby have both served as Justices of the Peace for New Castle County.
</item 10>
=== END ITEM id=8a9b5bed3689c61d ===

=== ITEM id=5ca2fb5811a778ac ===
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
Chico Hamilton Trio
Chico Hamilton Trio is an album by drummer and bandleader Chico Hamilton recorded at sessions in 1953, 1954 and 1956 released on the Pacific Jazz label.

 The album features Hamilton's first recordings for Pacific Jazz from 1953 and 1954, six tracks originally released on a 10-inch album, along with an additional
four recordings from 1956.
</item 2>

<item 3>
David M. Arden
David M. Arden (born September 6, 1949) is an American concert pianist whose performing and recording career has focused predominantly on contemporary and
American classical repertoire, including premiere performances and first recordings of piano works by a number of notable contemporary composers, such as Henryk
Górecki, Luciano Berio, Earle Brown, Carson Kievman and David Lang.

 The Russian composer Galina Ustvolskaya named Arden's recording of her 12 Preludes for Piano as her preferred recording of that work.
</item 3>

<item 4>
Fred Hersch
Fred Hersch (born October 21, 1955) is an American jazz pianist and educator.

 He has performed solo and led his own groups, including the Pocket Orchestra consisting of piano, trumpet, voice, and percussion.

 He was the first person to play weeklong engagements as a solo pianist at the Village Vanguard in New York City.

 He has recorded more than 70 of his jazz compositions.

 Hersch has been nominated for several Grammy Awards, and, as of December 2014, had been on the Jazz Studies faculty of the New England Conservatory since 1980
(with breaks).
</item 4>

<item 5>
Vladimir Ashkenazy
Vladimir Davidovich Ashkenazy (Russian: Влади́мир Дави́дович Ашкена́зи , "Vladimir Davidovich Ashkenazi"; born 6 July 1937) is an internationally recognized
solo pianist, chamber music performer, and conductor.

 He is originally from Russia and has held Icelandic citizenship since 1972.

 He has lived in Switzerland since 1978.

 Ashkenazy has collaborated with well-known orchestras and soloists.

 In addition, he has recorded a large storehouse of classical and romantic works.

 His recordings have earned him five Grammy awards plus Iceland's Order of the Falcon.
</item 5>

<item 6>
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
</item 6>

<item 7>
Janet Williams (soprano)
Janet Williams is an American soprano who has won international critical acclaim for performances at the Metropolitan Opera, Berlin Staatsoper, Paris Opera,
Théâtre des Champs-Élysées, Opera de Lyon, Nice Opera, Théâtre Royal de la Monnaie, Opera Geneva, Frankfurt Opera, Cologne Opera, Leipzig Opera, San Francisco
Opera, Washington Opera, Dallas Opera, and Michigan Opera Theatre as well as in concerts throughout Europe, North America, Canada, Israel and Japan with
conductors including Vladimir Ashkenazy, Daniel Barenboim, Myung-whun Chung, Philippe Herreweghe, René Jacobs, Marek Janowski, Neeme Järvi, Raymond Leppard,
Fabio Luisi, Sir Neville Marriner, Nicholas McGegan, Zubin Mehta, Kent Nagano, John Nelson, Donald Runnicles, Gerard Schwarz and Michael Tilson Thomas.
</item 7>

<item 8>
The Impeccable Mr. Wilson
The Impeccable Mr. Wilson is an album by American jazz pianist Teddy Wilson featuring performances recorded in 1956 for the Norgran label.
</item 8>

<item 9>
Marvin Hamlisch
Marvin Frederick Hamlisch (June 2, 1944August 6, 2012) was an American composer and conductor.

 Hamlisch was one of only twelve people to win Emmy, Grammy, Oscar and Tony awards.

 This collection of all four is referred to as an "EGOT".

 He is one of only two people (along with Richard Rodgers) to have won those four prizes and a Pulitzer Prize.
</item 9>

<item 10>
Don Williams discography
This is a detailed discography for American country music artist Don Williams.
</item 10>
=== END ITEM id=5ca2fb5811a778ac ===

=== ITEM id=3bc7771f7576316a ===
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
Yucca aloifolia
Yucca aloifolia is the type species for the genus Yucca.

 Common names include aloe yucca, dagger plant, and Spanish bayonet.

 It grows in sandy soils, especially on sand dunes along the coast.

 "Yucca aloifolia" is native to the Atlantic and Gulf Coasts of the United States from southern Virginia south to Florida and west to the Texas Gulf Coast, to
Mexico along the Yucatán coast, and to Bermuda, and parts of the Caribbean.

 Normally Yucca aloifolia is grown in USDA zones 8 through 11.

 Yucca aloifolia is a popular landscape plant in beach areas along the lower East Coast from Virginia to Florida.
</item 1>

<item 2>
Yucca × schottii
Yucca" × "schottii is a plant species in the genus "Yucca", native to southern Arizona, southwestern New Mexico, and the northern parts of Sonora and Chihuahua.

.

 The common names are Schott's yucca, hoary yucca, and mountain yucca.

 The "×" in the name indicates that this is a nothospecies, regarded as being a natural hybrid between two other species.

 In this case, "Yucca" × "schottii" is believed to have originated as a hybrid between "Y. baccata" and "Y. madrensis".

 "Yucca" × "schottii" is firmly established and does reproduce freely in the wild.
</item 2>

<item 3>
Rodgersia
Rodgersia is a genus of flowering plants in the Saxifragaceae family.

 "Rodgersia" are herbaceous perennials originating from east Asia.
</item 3>

<item 4>
Hesperoyucca whipplei
Hesperoyucca whipplei (syn.

 "Yucca whipplei" ) (chaparral yucca, our Lord's candle, Spanish bayonet, Quixote yucca or foothill yucca is a species of flowering plant closely related to,
and formerly usually included in, the genus "Yucca".

 It is native to southern California, United States and Baja California, Mexico, where it occurs mainly in chaparral, coastal sage scrub, and oak woodland plant
communities at altitudes of 0–2500 m.
</item 4>

<item 5>
Yucca
Yucca is a genus of perennial shrubs and trees in the family Asparagaceae, subfamily Agavoideae.

 Its 40-50 species are notable for their rosettes of evergreen, tough, sword-shaped leaves and large terminal panicles of white or whitish flowers.

 They are native to the hot and dry (arid) parts of the Americas and the Caribbean.

 Early reports of the species were confused with the cassava ("Manihot esculenta").

 Consequently, Linnaeus mistakenly derived the generic name from the Taíno word for the latter, "yuca" (spelled with a single "c").

 It is commonly found growing in rural graveyards and when in bloom the cluster of (usually pale) flowers on a thin stalk appear as floating apparitions.
</item 5>

<item 6>
The entries describe two Mexican semi-desert succulents, Echeveria setosa and E. elegans, and the genus Yucca: 40–50 arid-American and Caribbean shrubs and
trees with tough, sword-shaped leaves and flower panicles. Featured yuccas range from coastal U.S. and Caribbean habitats (Y. aloifolia) to California and Baja
(Hesperoyucca whipplei), Texas and northeastern Mexico (Y. rupic
</item 6>

<item 7>
Yucca glauca
Yucca glauca (syn.

 "Yucca angustifolia") is a species of perennial evergreen plant, adapted to xeric (dry)growth conditions.

 It is also known as small soapweed, soapweed yucca, Spanish bayonet, Great Plains yucca and beargrass.
</item 7>

<item 8>
Yucca brevifolia
Yucca brevifolia is a plant species belonging to the genus "Yucca".

 It is tree-like in habit, which is reflected in its common names: Joshua tree, yucca palm, tree yucca, and palm tree yucca.
</item 8>

<item 9>
Yucca rupicola
Yucca rupicola is a plant in the family Asparagaceae, known as the twistleaf yucca, twisted-leaf yucca, Texas yucca or twisted-leaf Spanish-dagger.

 The species was described by George Heinrich Adolf Scheele in 1850.

 This is a small, acaulescent plant with distinctive twisted leaves.

 It is native to the Edwards Plateau region of Texas and also to northeastern Mexico (Coahuila, Nuevo León).
</item 9>

<item 10>
Yucca gloriosa var. tristis
Yucca gloriosa" var.

 "tristis (syn.

 Yucca recurvifolia, "Yucca gloriosa" var.

 "recurvifolia"), known as curve-leaf yucca, curved-leaved Spanish-dagger or pendulous yucca, is a variety of "Yucca gloriosa".

 It is often grown as an ornamental plant, but is native to the southeastern United States.

 In contrast to "Y. gloriosa" var.

 "tristis", the leaves of "Y. gloriosa" var.

 "gloriosa" are hard stiff, erect and narrower.
</item 10>
=== END ITEM id=3bc7771f7576316a ===

=== ITEM id=354fe68dcfedde7c ===
Gold sentences and facts:
Sentence 0: Gasland is a 2010 American documentary written and directed by Josh Fox.
  Fact 0.0: Gasland is a documentary.
  Fact 0.1: Gasland is from 2010.
  Fact 0.2: Gasland is American.
  Fact 0.3: Gasland was written by Josh Fox.
  Fact 0.4: Gasland was directed by Josh Fox.
Sentence 1: To Shoot an Elephant is a 2009 documentary film about the 2008-2009 Gaza War directed by Alberto Arce and Mohammad Rujailahk.
  Fact 1.0: To Shoot an Elephant is a 2009 documentary film.
  Fact 1.1: To Shoot an Elephant is about the 2008-2009 Gaza War.
  Fact 1.2: To Shoot an Elephant was directed by Alberto Arce.
  Fact 1.3: To Shoot an Elephant was directed by Mohammad Rujailahk.

Retrieved items:
<item 1>
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
</item 1>

<item 2>
To Shoot an Elephant
To Shoot an Elephant is a 2009 documentary film about the 2008-2009 Gaza War directed by Alberto Arce and Mohammad Rujailahk.
</item 2>

<item 3>
First Kill
First Kill is a 2001 Dutch documentary film that revolves around the psychology of war.

 Important turning points in the Vietnam War are used to illustrate the effect of war on body and mind.

 The documentary consists of interviews with Michael Herr and several Vietnam veterans.

 The depth of the interviews provides insight into the feelings that accompany violence, fear, hate, seduction and pleasure.
</item 3>

<item 4>
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
</item 4>

<item 5>
Gasland
Gasland is a 2010 American documentary written and directed by Josh Fox.

 Nominated for an Academy Award for Best Documentary in 2011, the film focuses on communities in the United States affected by natural gas drilling and,
specifically, a method of horizontal drilling into shale formations known as hydraulic fracturing.

 The film was a key mobilizer for the anti-fracking movement, and "brought the term 'hydraulic fracturing' into the nation's living rooms" according to "The New
York Times".

 The film was nominated for the Academy Award for Best Documentary Feature.
</item 5>

<item 6>
South of the Border (2009 film)
South of the Border is a 2009 American documentary film directed by Oliver Stone.

 The documentary premiered at the 2009 Venice Film Festival.

 Writer for the project Tariq Ali calls the documentary "a political road movie".

 Stone stated that he hopes the film will help people better understand a leader who is wrongly ridiculed "as a strongman, as a buffoon, as a clown."
</item 6>

<item 7>
Rick Ray
Rick Ray is an American filmmaker best known for his 2006 documentary film 10 Questions for the Dalai Lama which he wrote, filmed and directed.

 His documentary "Lynching Charlie Lynch", about the trials of former medical marijuana dispensary owner Charles C. Lynch, premiered at the 2011 San Luis Obispo
International Film Festival on March 9.
</item 7>

<item 8>
The Last Lions
The Last Lions is a 2011 African nature documentary film by National Geographic Society, videotaped and directed by Dereck and Beverly Joubert.

 It was shot at the Okavango Delta in Botswana.

 The film premiered at the Palm Springs International Film Festival in January 2011 and was released in select theaters on February 18, 2011.

 The film follows in the tradition of other National Geographic big cat films, such as "India: Land of the Tiger" and "Eye of the Leopard".
</item 8>

<item 9>
Hunting Elephants
Hunting Elephants (Hebrew: לצוד פילים‎ ‎ , "Latzud Pilim", "lit."

 To Hunt Elephants) is a 2013 Israeli crime comedy film directed by Reshef Levi.

 It stars, among others, Sasson Gabai, Moni Moshonov and Patrick Stewart.

 It was released to theaters in Israel on July 4, 2013.
</item 9>

<item 10>
Baghdad ER
Baghdad ER is a documentary released by HBO on May 21, 2006.

 It shows the Iraq war from the perspective of a military hospital in Baghdad.

 It has some relatively disturbing scenes in it (e.g. amputations), therefore the U.S. Army is officially warning that military personnel watching it could
experience symptoms of post-traumatic stress disorder (PTSD).
</item 10>
=== END ITEM id=354fe68dcfedde7c ===

=== ITEM id=dfd1ecaa60a77a9a ===
Gold sentences and facts:
Sentence 0:  They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC lightweight champion Eddie Alvarez.
  Fact 0.0: This team is most well known for including Sugar Rashad Evans.
  Fact 0.1: Sugar Rashad Evans is a former UFC light heavyweight champion.
  Fact 0.2: This team is most well known for including Vitor Belfort.
  Fact 0.3: Vitor Belfort is a former UFC light heavyweight champion.
  Fact 0.4: This team is most well known for including Eddie Alvarez.
  Fact 0.5: Eddie Alvarez is a former UFC lightweight champion.
Sentence 1: Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.
  Fact 1.0: Vitor Vieira Belfort was born on April 1, 1977.
  Fact 1.1: Vitor Vieira Belfort is a Brazilian mixed martial artist.
  Fact 1.2: Vitor Vieira Belfort is currently competing in the Middleweight division of the Ultimate Fighting Championship.

Retrieved items:
<item 1>
Graph query row:
person: Vitor Belfort
connected_to: Blackzilians
relationship: MEMBER_OF
division: None
status: None
period: None
</item 1>

<item 2>
Sam Alvey
Samuel Alvey (born May 6, 1986) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.

 A professional MMA competitor since 2008, Alvey has also formerly competed for Bellator, King of the Cage, the MFC, and was a contestant on .

 As of March 2017, Alvey is #14 in the official UFC Middleweight Rankings.
</item 2>

<item 3>
David Branch (fighter)
David Branch (born September 26, 1981) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 A professional competitor since 2007, he has also formerly competed for the World Series of Fighting, Bellator, Shark Fights, and Titan FC.

 Branch was the inaugural World Series of Fighting Light Heavyweight and Middleweight Champion.

 He is currently ranked as the #9 Middleweight in the world according to Sherdog.
</item 3>

<item 4>
Jack Marshman
Jack "Hammer" Marshman (born December 19, 1989) is a Welsh mixed martial artist.

 He was the Middleweight Cage Warriors Fighting Championship (CWFC) champion and Lonsdale Middleweight Champion.

 He currently competes in Middleweight division in the Ultimate Fighting Championship(UFC).
</item 4>

<item 5>
Luke Zachrich
Luke Zachrich (born October 1, 1981) is an American mixed martial artist and former professional boxer who formerly competed in the Middleweight division of the
Ultimate Fighting Championship A professional mixed martial arts competitor since 2006, Zachrich was a member of Team Forrest on the seventh season of The
Ultimate Fighter, and has also fought for King of the Cage, Xtreme Fighting Organization, Bellator, and Ultimate Victory Challenge, the latter of which he was
the inaugural Middleweight Champion.
</item 5>

<item 6>
Kelvin Gastelum
Kelvin Gastelum (born October 24, 1991) is an American professional mixed martial artist who is currently signed to the Ultimate Fighting Championship (UFC).

 He was the winner of "The Ultimate Fighter 17".

 Gastelum is currently competing as a middleweight and is ranked #8 in the UFC official middleweight rankings.
</item 6>

<item 7>
Blackzilians
The Blackzilians is a defunct professional team of fighters in mixed martial arts, boxing, kickboxing, amateur wrestling and catch wrestling located in South
Florida.

 They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC
lightweight champion Eddie Alvarez.
</item 7>

<item 8>
Rashad Evans
Rashad Anton Evans (born September 25, 1979) is an American mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 A professional competitor since 2003, Evans was the Heavyweight winner of The Ultimate Fighter 2 and is also a former UFC Light Heavyweight Champion.
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
Vitor Belfort
Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting
Championship.

 He is the former UFC Light Heavyweight Champion, UFC 12 Heavyweight Tournament Champion, and the Cage Rage World Light Heavyweight Champion.

 As of June 16, 2017, he is the #11 contender in official UFC Middleweight rankings.
</item 10>
=== END ITEM id=dfd1ecaa60a77a9a ===

=== ITEM id=1ad9f861d4c5f02c ===
Gold sentences and facts:
Sentence 0: "I'll Keep the Lovelight Burning" is a popular song written in 1942 by Harry Tobias, Nick Kenny, and Harold Levey, popularized by Patti Page in 1949.
  Fact 0.0: "I'll Keep the Lovelight Burning" is a popular song.
  Fact 0.1: "I'll Keep the Lovelight Burning" was written in 1942.
  Fact 0.2: "I'll Keep the Lovelight Burning" was written by Harry Tobias.
  Fact 0.3: "I'll Keep the Lovelight Burning" was written by Nick Kenny.
  Fact 0.4: "I'll Keep the Lovelight Burning" was written by Harold Levey.
  Fact 0.5: "I'll Keep the Lovelight Burning" was popularized by Patti Page.
  Fact 0.6: Patti Page popularized "I'll Keep the Lovelight Burning" in 1949.
Sentence 1: Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country music.
  Fact 1.0: Clara Ann Fowler's date range is November 8, 1927 to January 1, 2013.
  Fact 1.1: Clara Ann Fowler is known by the professional name Patti Page.
  Fact 1.2: Patti Page was an American singer.
  Fact 1.3: Patti Page was a singer of traditional pop music.
  Fact 1.4: Patti Page was a singer of country music.

Retrieved items:
<item 1>
Jimmy O'Neill (DJ)
James Franklin O'Neill (January 8, 1940 – January 11, 2013) was an American DJ and broadcaster who hosted the ABC television show "Shindig!

" from 1964-1966.

 O'Neill was owner of Pandora's Box, an influential Sunset Strip music venue in West Hollywood, California that was the center of the 1966 Sunset Strip curfew
riots.
</item 1>

<item 2>
Live in Old Smokey
Live in Old Smokey is a live CD by English singer/songwriter Linda Lewis recorded at Ronnie Scott's in London and issued in 2006.
</item 2>

<item 3>
Elton John
Sir Elton Hercules John, (born Reginald Kenneth Dwight; 25 March 1947), is an English singer, pianist, and composer.

 He has worked with lyricist Bernie Taupin as his songwriting partner since 1967; they have collaborated on more than 30 albums to date.

 In his five-decade career Elton John has sold more than 300million records, making him one of the best-selling music artists in the world.

 He has more than fifty Top 40 hits, including seven consecutive No. 1 US albums, 58 "Billboard" Top 40 singles, 27 Top 10, four No. 2 and nine No. 1.

 For 31 consecutive years (1970–2000) he had at least one song in the "Billboard" Hot 100.

 His tribute single, re-penned in dedication to the late Princess Diana, "Candle in the Wind 1997" sold over 33million copies worldwide and is the best-selling
single in the history of the UK and US singles charts.

 He has also composed music, produced records, and has occasionally acted in films.
</item 3>

<item 4>
Willie and the Hand Jive
"Willie and the Hand Jive" is a song written by Johnny Otis and originally released as a single in 1958 by Johnny Otis Show, reaching #9 on the "Billboard" Hot
100 chart and #5 on the Billboard R&B chart.

 The song has a Bo Diddley beat and was partly inspired by the music sung by a chain gang Otis heard while he was touring.

 The lyrics are about a man who became famous for doing a dance with his hands, but the song has been accused of glorifying masturbation.

 It has since been covered by numerous artists, including The Strangeloves, Eric Clapton, Cliff Richard, Kim Carnes, George Thorogood and The Grateful Dead.

 Clapton's 1974 version was also released as a single and also reached the Billboard Top 40, peaking at #26.

 Thorogood's 1985 version reached #25 on the Hot Mainstream Rock Tracks chart.
</item 4>

<item 5>
Keep This Fire Burning
"Keep This Fire Burning" is a song by Swedish pop singer Robyn, released as the first single from her third album "Don't Stop the Music".

 It was released in Sweden on September 21, 2002, where it became her highest charting single since 1995's "Do You Really Want Me (Show Respect)".

 The song was also released as a single in Australia under the name "By Your Side", due to the Australian bush fires which were happening at the time.

 In 2008, a re-recorded version of the song appeared on the special edition of Robyn's eponymous album.
</item 5>

<item 6>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 6>

<item 7>
You Keep Me Hangin' On
"You Keep Me Hangin' On" is a 1966 song written and composed by Holland–Dozier–Holland.

 It first became a popular "Billboard" Hot 100 number one hit for the American Motown group The Supremes in late 1966.

 The rock band Vanilla Fudge covered the song a year later and had a top ten hit with their version.

 British pop singer Kim Wilde covered "You Keep Me Hangin' On" in 1986, bumping it back to number one on the "Billboard" Hot 100 in June 1987.

 The single reached number one by two different musical acts in America.

 In the first 32 years of the "Billboard" Hot 100 rock era, “You Keep Me Hangin' On” became one of only six songs to achieve this feat.

 In 1996, country music singer Reba McEntire's version reached number 2 on the US "Billboard" Hot Dance Club Play chart.
</item 7>

<item 8>
I'll Keep the Lovelight Burning
"I'll Keep the Lovelight Burning" is a popular song written in 1942 by Harry Tobias, Nick Kenny, and Harold Levey, popularized by Patti Page in 1949.

 Louis Armstrong also covered the song in 1949.
</item 8>

<item 9>
Patti Page
Clara Ann Fowler (November 8, 1927 – January 1, 2013), known by her professional name Patti Page, was an American singer of traditional pop music and country
music.

 She was the top-charting female vocalist and best-selling female artist of the 1950s, selling over 100 million records during a six-decade long career.

 She was often introduced as "the Singin' Rage, Miss Patti Page".

 New York WNEW disc-jockey William B. Williams introduced her as "A Page in my life called Patti".
</item 9>

<item 10>
Tony Hiller
Anthony Toby "Tony" Hiller (born 30 July 1927) is a British songwriter.

 He began his musical career as a member of the song and dance duo The Hiller Brothers, sharing the stage with his brother Irving.

 The Hiller Brothers appeared with many artists of the time including Alma Cogan, Tommy Cooper, Val Doonican, Matt Monro, The Shadows, Bernard Manning, Kathy
Kirby, Roger Whittaker, Rip Taylor, Gene Vincent, Lance Percival, Tessie O'Shea, Frank Ifield, Deep River Boys, The Dallas Boys, Clark Brothers, Paul Melba and
Ray Burns.
</item 10>
=== END ITEM id=1ad9f861d4c5f02c ===

=== ITEM id=4cae24c0e0d08997 ===
Gold sentences and facts:
Sentence 0:  It has its offices in New Roc City in New Rochelle, New York.
  Fact 0.0: An entity has its offices in New Roc City in New Rochelle, New York.
Sentence 1: New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in Westchester County, New York.
  Fact 1.0: New Roc City is also known as New Rochelle Center.
  Fact 1.1: New Roc City is an entertainment complex.
  Fact 1.2: New Roc City is a retail complex.
  Fact 1.3: New Roc City is a residential complex.
  Fact 1.4: New Roc City is in the Downtown section of the city of New Rochelle.
  Fact 1.5: New Rochelle is in Westchester County, New York.

Retrieved items:
<item 1>
New Rochelle Harbor (Long Island Sound)
New Rochelle Harbor is the name of a harbor located along Long Island Sound in the city of New Rochelle in Westchester County, New York.

 The Davenport Neck peninsula off the mainland divides New Rochelle's waterfront into two bays; the westerly referred to as New Rochelle Harbor and the easterly
as Echo Bay.
</item 1>

<item 2>
New Rochelle Walk of Fame
The New Rochelle Walk of Fame was installed in 2011 in Ruby Dee Park at Library Green, located in the downtown section of the city of New Rochelle in
Westchester County, New York.

 The "walk" is a tribute to some of New Rochelle's most notable residents from throughout its 325-year history.

 It was created and funded by former resident Roderick Kennedy, Jr., working in partnership with the City of New Rochelle and the New Rochelle Business
Improvement District.
</item 2>

<item 3>
Graph query row:
school: Japanese Weekend School of New York
connected_entity: New Roc City
connection_path: ['Japanese Weekend School of New York', 'New Roc City']
relationships: ['LOCATED_IN']
New_Roc_City_section: Downtown
section_kind: section
</item 3>

<item 4>
Standard Star Building
The Standard Star Building is a historic commercial structure located in the Downtown section of New Rochelle, Westchester County, New York.

 The building, designed by architect Lawrence J. Barnard, was completed in 1924 and is an architecturally significant example of the Italian Renaissance style
in New Rochelle.

 Although it has been altered, these changes occurred only on the Le Count Place façade and the interior of the building.

 The original Standard Star building exterior remains largely unchanged.

 It is further historically significant for its association with a long-published New Rochelle newspaper covering life in New Rochelle.
</item 4>

<item 5>
New Jersey Japanese School
The New Jersey Japanese School (ニュージャージー日本人学校 , "Nyūjājī Nihonjin Gakkō" , NJJS) is a Japanese school located in Oakland, New Jersey, United States in the New
York City metropolitan area.

 It is one of the two Japanese day schools operated by the Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi
Kai"), a nonprofit organization which also operates two Japanese weekend school systems in the New York City area.
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
Ware's Department Store
Ware’s Department Store is a historic building located in the city of New Rochelle in Westchester County, New York and is significant for both architectural and
commercial reasons.

 Ware's was Westchester’s first and, for many years, largest department store, and was prominently located on New Rochelle’s fashionable Main Street.

 The store operated from 1881 to the late 1930s, when the property was sold to the retailer Bloomingdale's to serve as their first suburban department store
location.

 Not only was Ware's a key place in New Rochelle during its 20th-century boom years, its founder and owner, Howard R. Ware was a leading figure in the rapidly
growing community as well.

 Ware first moved to New Rochelle from Massachusetts at the age of 13 and began to work as a clerk.

 In 1881 he became partner in the firm of Ware & Sheffield, which eventually became a stock company in 1913.

 renamed He was a director and vice president of the National City Bank of New Rochelle, a founder and first president of the local Y.M.C.A. from 1899 to 1916,
and an active member of St.
</item 7>

<item 8>
New Roc City
New Roc City, also known as New Rochelle Center, is an entertainment, retail and residential complex in the Downtown section of the city of New Rochelle in
Westchester County, New York.

 It is located at 33 LeCount Place, between Main Street North and Main Street South.

 The center was built on the site of the former New Rochelle Mall which closed in 1992.
</item 8>

<item 9>
Japanese Weekend School of New York
The Japanese Weekend School of New York (JWSNY; ニューヨーク "Nyūyōku Hoshū Jugyō Kō") is a Japanese supplementary school in the New York City metropolitan area.

 It has its offices in New Roc City in New Rochelle, New York.

 The Japanese Educational Institute of New York (JEI; ニューヨーク日本人教育審議会 "Nyūyōku Nihonjin Kyōiku Shingi Kai") manages the school system, and the JWSNY is one of
its two weekend school systems.

 The JEI also operates two Japanese day schools in the New York area.
</item 9>

<item 10>
Montefiore New Rochelle Hospital
Montefiore New Rochelle Hospital (formerly Sound Shore Medical Center) is a community-based, teaching hospital located at 16 Guion Place in the city of New
Rochelle, in Westchester County, New York, and affiliated with the Albert Einstein College of Medicine.

 The hospital opened in 1892 as New Rochelle Hospital.

 On November 6, 2013, Sound Shore was acquired by the Bronx-based Montefiore Medical Center, the University Hospital for the Albert Einstein College of
Medicine, was renamed Montefiore New Rochelle Hospital, and became part of the Montefiore Health System.
</item 10>
=== END ITEM id=4cae24c0e0d08997 ===

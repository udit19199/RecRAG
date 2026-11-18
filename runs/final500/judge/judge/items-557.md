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

=== ITEM id=e050e801c3cf9825 ===
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
Lake Louisvilla, Louisville
Lake Louisvilla is a neighborhood partially located in Louisville, Kentucky.

 It is located between Westport Road in Louisville and KY 22 in Oldham County.

 Lake Louisvilla was developed in the 1920s as a summer resort for people living in the city of Louisville.

 The state of Kentucky drained the lake in the late 1980s due to safety concerns regarding the stability of a dam.
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
Oldham County, Kentucky
Oldham County is a county located in the commonwealth of Kentucky.

 As of the 2010 census, the population was 60,316.

 Its county seat is La Grange.

 The county is named for Colonel William Oldham.

 Oldham County was a prohibition or completely dry county until January 2005 as the result of a 2004 'moist' vote, permitting sales of alcohol in restaurants
that seat at least 100 patrons in which 70%+ of total revenue is derived from sales of food.

 After a vote in late 2015; Oldham county has become a completely wet county.
</item 5>

<item 6>
Wheeling Island
Wheeling Island is the most populated island in the Ohio River.

 It lies within the city of Wheeling in Ohio County, West Virginia, in the United States.

 The 2000 census showed a resident population of 3,142 people on the island, which has a land area of 1.514 km² (374 acres).

 Neville Island, PA is larger (3.4 km²) but has fewer people (1,232).
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
Squire Creek Country Club
The Squire Creek Lodge features two floors offering a combined total of seven bedrooms.

 The Squire Creek Pool overlooks the golf course and is served by the Waterside Cafe.

 The Squire Creek Development offers six residential estates including the Squire Creek Estates, Timberland Estates, Fairway Estates, The Fairways, Fairway
Villas, and The Park Homes.

 Squire Creek Country Club is the title sponsor of the Louisiana Peach Festival in Ruston.

 Squire Creek hosted the 2005 Western Athletic Conference Golf Championships.
</item 10>
=== END ITEM id=e050e801c3cf9825 ===

=== ITEM id=af560bfeb8b5aceb ===
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
Diamond Rio
Diamond Rio is an American country and Christian country music band.

 The band was founded in 1982 as an attraction for the Opryland USA theme park in Nashville, Tennessee, and was originally known as the Grizzly River Boys, then
the Tennessee River Boys.

 It was founded by Matt Davenport, Danny Gregg, and Ty Herndon, the last of whom became a solo artist in the mid-1990s.

 After undergoing several membership changes in its initial years, the band has consisted of the same six members since 1989: Marty Roe (lead vocals, rhythm
guitar), Gene Johnson (mandolin, guitar, fiddle, tenor vocals), Jimmy Olander (lead guitar, Dobro, banjo), Brian Prout (drums), Dan Truman (keyboards, organ,
synthesizer), and Dana Williams (bass guitar, baritone vocals).
</item 1>

<item 2>
Progressive country
Progressive country is a subgenre of country music developed in the early 1970s.

 In the late 1960s and early 1970s, mainstream country music was dominated by the slick Nashville sound and the rock-influenced Bakersfield sound of artists
like Merle Haggard.

 A new generation of country artists emerged, influenced by contemporary rock music, singer-songwriters such as Bob Dylan, and the liberal politics of the 1960s
counterculture.
</item 2>

<item 3>
Hjärtats trakt
Hjärtats trakt was released on 20 April 1993 and is a compilation album from Swedish pop artist Per Gessle.

 The album includes songs taken from Gessle's solo albums "Per Gessle" and "Scener".

 It was also released under license by the British record company Pickwick and it is the only Gessle album released by another label than EMI.

 In 1998 the album was re-released by the Dutch record company Disky Communications B.V. which is a subsidiary of EMI.
</item 3>

<item 4>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 4>

<item 5>
Night Is Fallin' in My Heart
"Night Is Fallin' in My Heart" is a song written by Dennis Linde.

 The song was first recorded by Country music artist J.P. Pennington for his 1991 album, "Whatever It Takes".

 It was later recorded and released as a single in October 1994 by American country music group Diamond Rio.

 It was the second single released from their third album, "Love a Little Stronger".

 It peaked at number 9 in the United States, and number 6 in Canada.
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
Every Beat of My Heart (song)
"Every Beat of My Heart" is a rhythm and blues song by Johnny Otis.

 It was first recorded in 1954 by his group, The Royals (later to be known as The Midnighters).

 In 1961 Gladys Knight & the Pips recorded the song for their debut single on the Vee-Jay label.

 Credited to The Pips, it was the first of eleven releases by the group to make it to number one on the R&B/soul chart.

 It was also the group's first top ten on the "Billboard" Hot 100.

 "Every Beat of My Heart" was first recorded for the Huntom label, who later sold the master to Vee-Jay.

 At the time of the song's release, The Pips were on the Fury label where they re-recorded the song without piano.

 In an unusual occurrence, the Fury recording of the song also made the top twenty on the R&B Sides chart and also made the Hot 100.
</item 9>

<item 10>
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
</item 10>
=== END ITEM id=af560bfeb8b5aceb ===

=== ITEM id=e156e7a8246a3b98 ===
Gold sentences and facts:
Sentence 0:  In the category of pornographic websites, it was surpassed in the rankings by competitor sites xHamster, XVideos, and Pornhub, as well as the adult webcam site LiveJasmin.
  Fact 0.0: In the category of pornographic websites, this site was surpassed in the rankings by xHamster.
  Fact 0.1: In the category of pornographic websites, this site was surpassed in the rankings by XVideos.
  Fact 0.2: In the category of pornographic websites, this site was surpassed in the rankings by Pornhub.
  Fact 0.3: In the category of pornographic websites, this site was surpassed in the rankings by the adult webcam site LiveJasmin.
  Fact 0.4: xHamster, XVideos, and Pornhub are competitor sites.
Sentence 1: XVideos is the world's largest pornographic video sharing and viewing website.
  Fact 1.0: XVideos is the world's largest pornographic video sharing and viewing website.

Retrieved items:
<item 1>
XY TV
It launched in December 2003 and distribution was limited to select cable companies.

 As of May 31, 2006, XY.tv has ceased operations, and is no longer broadcasting.

 It was wholly owned by American radio personality John Garabedian.
</item 1>

<item 2>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 2>

<item 3>
Pravda.ru
Pravda.ru (formerly Pravda Online) is a Russian internet news website established in 1999 and owned by Pravda.ru Holding (headed by Vadim Gorshenin).
</item 3>

<item 4>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 4>

<item 5>
YouPorn
YouPorn is a free pornographic video sharing website and one of the 100 most accessed websites in the world.

 Since launching in August 2006, it grew to become the most popular pornographic website on the internet, and, in November 2007, it was reported to be the
largest free pornographic website as well.

 As of February 2013, it was the 83rd most popular website overall and the fifth most popular pornographic website.

 In the category of pornographic websites, it was surpassed in the rankings by competitor sites xHamster, XVideos, and Pornhub, as well as the adult webcam site
LiveJasmin.
</item 5>

<item 6>
18 Doughty Street
18 Doughty Street was a British political Internet-based broadcaster that hosted a webcast as its chief product.

 It began broadcasting at 18:55 on 10 October 2006, from its studio at 18 Doughty Street in the Bloomsbury area of London, and ceased broadcasting at 23:00 on
Thursday 8 November 2007.

 It claimed to be Britain's first Internet-based TV station.
</item 6>

<item 7>
XVideos
XVideos is the world's largest pornographic video sharing and viewing website.

 It's registered to the Polish company, WGCZ Holding.

 Alexa ranks it as the 50th most visited website in the world.
</item 7>

<item 8>
Wengie
Her YouTube channel also recently got awarded with "Best Channel" as well as "Overall Winner" for the Australian Online Video Awards.
</item 8>

<item 9>
XHamster
xHamster is a pornographic media and social networking site headquartered in Limassol, Cyprus.

 xHamster serves user-submitted pornographic videos, webcam models, pornographic photographs, and erotic literature, and incorporates social networking
features.

 xHamster was founded in 2007.

 With more than 10 million members, it is the third most popular pornography website on the Internet after XVideos and Pornhub.
</item 9>

<item 10>
Pioneer Pekao Investments
Pioneer Pekao is a Polish company managing several investment funds in Poland.

 It is the biggest and first investment funds manager on the Polish market.
</item 10>
=== END ITEM id=e156e7a8246a3b98 ===

=== ITEM id=9b4dc9dabef5c9d3 ===
Gold sentences and facts:
Sentence 0: Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct "Final Destination 5" six years later, and filmed in the IMAX 3D format.
  Fact 0.0: Aliens of the Deep is a 2005 documentary film.
  Fact 0.1: Aliens of the Deep was directed in part by James Cameron.
  Fact 0.2: James Cameron directed Aliens of the Deep alongside fellow cameraman and friend Steven Quale.
  Fact 0.3: Steven Quale would go on to direct Final Destination 5 six years later.
  Fact 0.4: Aliens of the Deep was filmed in the IMAX 3D format.
Sentence 1:  Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of the planet's more unusual forms of life.
  Fact 1.0: James Cameron teams with NASA scientists to explore the Mid-Ocean Ridges.
  Fact 1.1: The Mid-Ocean Ridges are submerged chains of mountains.
  Fact 1.2: The Mid-Ocean Ridges are in the Atlantic and Pacific oceans.
  Fact 1.3: The Mid-Ocean Ridges are home to some of the planet's more unusual forms of life.
Sentence 2: Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.
  Fact 2.0: Poetry in Motion is a 1982 Canadian documentary film.
  Fact 2.1: Poetry in Motion was directed by Ron Mann.
  Fact 2.2: Poetry in Motion features contemporary North American poetry.
  Fact 2.3: Poetry in Motion features contemporary North American music.
Sentence 3:  Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.
  Fact 3.0: Poetry in Motion features some Black Mountain poets.
  Fact 3.1: Poetry in Motion features some Beat poets.
  Fact 3.2: Poetry in Motion features some minimalist poets.
  Fact 3.3: Poetry in Motion features some avant-garde poets.
Sentence 4:  It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.
  Fact 4.0: Poetry in Motion was released in theaters.
  Fact 4.1: Poetry in Motion was later distributed on VHS.
  Fact 4.2: Poetry in Motion was later distributed on LaserDisc.
  Fact 4.3: Poetry in Motion was later distributed on DVD.
Sentence 5:  An extended CD-ROM version was also released.
  Fact 5.0: An extended CD-ROM version of Poetry in Motion was also released.

Retrieved items:
<item 1>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 1>

<item 2>
The entries profile filmmakers, producers, performers, and artists: Tim Bedore is a Chicago-born comedian; Rogue One (2016) depicts rebels stealing Death Star
plans; Frederick K. Keller, Peter Guber, Baltasar Kormákur, Steve Rash, Jonathan Kaplan, Kurt Voss, Jim Sharman, Barry Sandler, and Howard Rosenman are
directors or producers. Other highlights include Titan A.E.’s commercial failure; Vito Russo’s documentary; Scott Free’s founding by the
</item 2>

<item 3>
Titan A.E.
Titan A.E. is a 2000 American animated science fiction film directed by Don Bluth and Gary Goldman.

 Its title refers to the spacecraft central to the plot, with "A.E." meaning "After Earth".

 It stars Matt Damon, Bill Pullman, John Leguizamo, Nathan Lane, Janeane Garofalo and Drew Barrymore.

 The film's animation technique combines traditional hand-drawn animation and extensive use of computer generated imagery.

 Its working title was "Planet Ice".

 It was theatrically released on June 16, 2000, by 20th Century Fox and was the final film for Fox Animation Studios.

 The film grossed $36.8 million on a $75–$90 million budget, making a $100-million loss for the studio.
</item 3>

<item 4>
Poetry in Motion (film)
Poetry in Motion is a 1982 Canadian documentary film directed by Ron Mann featuring contemporary North American poetry and music.

 Featured are some of the Black Mountain poets, Beats, minimalist poets, and avant-garde poets.

 It was released in theaters, later being distributed on VHS, LaserDisc, and DVD.

 An extended CD-ROM version was also released.
</item 4>

<item 5>
The entries survey documentaries, films, TV, and film-related concepts: Los Angeles hip-hop, Aboriginal country music, Latin American politics, oceans, nerd
culture, Iraqi life and war, Bougainville, education, disability, poetry, surfing, and family relationships. Featured works include *Gasland*, *Baghdad ER*,
*War/Dance*, *First Life*, and *The Coconut Revolution*. Several received major awards or nominations. The material also profiles filmmaker Wayne Coles-Janess
and defines
</item 5>

<item 6>
Oceans (film)
Oceans (French: Océans ) is a 2009 French nature documentary film directed, produced, co-written, and narrated by Jacques Perrin, with Jacques Cluzaud as
co-director.

 The film explores the marine species of Earth's five oceans and reflects on the negative aspects of human activity on the environment, with Perrin (Pierce
Brosnan in English) providing narration.
</item 6>

<item 7>
We Are Poets
We Are Poets is a documentary film directed by Daniel Lucchesi and Alex Ramseyer-Bache.

 It follows six young poets from Leeds Young Authors performance poetry group in Leeds on their visit to the international Poetry Slam Competition in Washington
D.C, United States.

 It premièred at Europe's largest Documentary Film festival - Doc/Fest in Sheffield's Showroom Cinema on 11 June 2011, where it won the coveted Sheffield Youth
Jury Award.
</item 7>

<item 8>
Aliens of the Deep
Aliens of the Deep is a 2005 documentary film, directed in part by James Cameron alongside fellow cameraman and friend Steven Quale, who would go on to direct
"Final Destination 5" six years later, and filmed in the IMAX 3D format.

 It was produced by Walden Media and Walt Disney Pictures.

 Cameron teams with NASA scientists to explore the Mid-Ocean Ridges, submerged chains of mountains in the Atlantic and Pacific oceans that are home to some of
the planet's more unusual forms of life.
</item 8>

<item 9>
Deep Dark Canyon
Deep Dark Canyon is a 2012 American independent dramatic thriller film by husband and wife writer/director team Abe Levy and Silver Tree.

 The film stars Ted Levine, Nick Eversman and Spencer Treat Clark.

 The film was originally called "Lawless" and was shot on location in the rural town of Guerneville, California.
</item 9>

<item 10>
The Polar Bears
The Polar Bears is a 2012 3D computer animated short film presented by The Coca-Cola Company, produced by Ridley Scott, written by David Reynolds, and directed
by John Stevenson.

 The film features the voices of Lin-Manuel Miranda, Armie Hammer, Jonathan Adams, and Megyn Price.

 The film is based on Coca-Cola's iconic polar bears and was released on the company's website and YouTube channel on December 31, 2012.
</item 10>
=== END ITEM id=9b4dc9dabef5c9d3 ===

=== ITEM id=2ac2a558a64ac33c ===
Gold sentences and facts:
Sentence 0:  He is best known for his roles in "How To Be Indie" (2009–2011), "Wingin' It" (2010–2013), "" (2012–2013), and "Open Heart" (2015).
  Fact 0.0: The person described is best known for a role in 'How To Be Indie' (2009–2011).
  Fact 0.1: The person described is best known for a role in 'Wingin' It' (2010–2013).
  Fact 0.2: The person described is best known for a role in an untitled work (2012–2013).
  Fact 0.3: The person described is best known for a role in 'Open Heart' (2015).
Sentence 1: Open Heart is a 2015 American-Canadian mystery-drama television series produced by the Epitome Pictures unit of DHX Media in association with marblemedia.
  Fact 1.0: Open Heart is a 2015 television series.
  Fact 1.1: Open Heart is an American-Canadian television series.
  Fact 1.2: Open Heart is a mystery-drama television series.
  Fact 1.3: Open Heart was produced by the Epitome Pictures unit of DHX Media.
  Fact 1.4: Open Heart was produced in association with marblemedia.

Retrieved items:
<item 1>
Don't Stop Believing (TV series)
Don't Stop Believing is a British television talent show that aired on Channel 5 in summer 2010.

 It was inspired by the musical comedy-drama "Glee", which airs in the United States on the Fox network.

 The series featured live shows in which musical performance groups competed against each other, with viewers voting on the winner.

 Solo singers were also sought to join a group to represent the United Kingdom on the American glee club circuit.

 The show was hosted by Emma Bunton, and judged by Anastacia, Duncan James, Tamzin Outhwaite and Charles "Chucky" Klapow.

 The programme was shown in simulcast on Irish TV channel 3e and repeated a week later on parent channel TV3 Ireland.

 The show was not renewed for a second series due to low ratings.
</item 1>

<item 2>
Dan Flannery
Arthur Daniel Flannery (born March 10, 1944) is an American actor from Evanston, Illinois.

 He has played roles in several movies, short films, and T.V. shows from a variety of genres including drama, action, and thriller.

 His filmography includes The Straight Story (1999), and Contagion (2011) and T.V. series such as Empire (2015 TV series) and Boss (2011-2012).
</item 2>

<item 3>
Everest (2015 film)
Everest is a 2015 British-American biographical adventure film directed and co-produced by Baltasar Kormákur, co-produced by Tim Bevan, Eric Fellner, Nicky
Kentish Barnes, Tyler Thompson and Brian Oliver and written by William Nicholson and Simon Beaufoy, adapted from Beck Weathers' memoir "Left for Dead: My
Journey Home from Everest" (2000).

 It stars an ensemble cast of Jason Clarke, Josh Brolin, John Hawkes, Robin Wright, Michael Kelly, Sam Worthington, Keira Knightley, Emily Watson, and Jake
Gyllenhaal.

 It is based on the real events of the 1996 Mount Everest disaster, and focuses on the survival attempts of two expedition groups, one led by Rob Hall (Clarke)
and the other by Scott Fischer (Gyllenhaal).
</item 3>

<item 4>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 4>

<item 5>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 5>

<item 6>
Garcia Plays Dylan
Garcia Plays Dylan is an album composed of various live performances featuring Jerry Garcia playing covers of Bob Dylan songs.

 It is culled from performances from 1973–1995, and features Garcia playing with Grateful Dead, Legion of Mary, Jerry Garcia Band, and Garcia-Saunders.

 Garcia takes lead vocals on all tracks.
</item 6>

<item 7>
Dean Winchester
Dean Winchester is a fictional character and one of the two protagonists of the American drama television series "Supernatural".

 He is portrayed by Jensen Ackles.

 Other versions of the character having been portrayed by Hunter Brochu (toddler), Ridge Canipe (child), Nicolai Lawton-Giustra (pre-teen), Brock Kelly and
Dylan Everett (teen), and Chad Everett (elderly).
</item 7>

<item 8>
Wingin' It
Wingin' It (originally titled Angel on Campus prior to its debut) is a Canadian teen sitcom which aired on Family.

 The series was produced by Temple Street Productions in association with Family.

 It stars Demetrius Joyette and Dylan Everett.

 This show included guest-stars from other Family Channel shows such as "The Latest Buzz".

 It was announced that the series was renewed for a third season on June 13, 2011.

 Family announced that the series is not planned for a fourth season.
</item 8>

<item 9>
Dylan Everett
Dylan Everett (born January 24, 1995) is a Canadian actor.

 He is best known for his roles in "How To Be Indie" (2009–2011), "Wingin' It" (2010–2013), "" (2012–2013), and "Open Heart" (2015).
</item 9>

<item 10>
Open Heart (TV series)
Open Heart is a 2015 American-Canadian mystery-drama television series produced by the Epitome Pictures unit of DHX Media in association with marblemedia.

 It first aired on January 20, 2015, on TeenNick in the United States and previously on YTV; now currently on ABC Spark in Canada.
</item 10>
=== END ITEM id=2ac2a558a64ac33c ===

=== ITEM id=e58ef4e72eb11edc ===
Gold sentences and facts:
Sentence 0: Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and Vietnam.
  Fact 0.0: Aspidistra is a genus of flowering plants.
  Fact 0.1: Aspidistra is in the family Asparagaceae.
  Fact 0.2: Aspidistra is in the subfamily Nolinoideae.
  Fact 0.3: Aspidistra is native to eastern Asia.
  Fact 0.4: Aspidistra is native to southeastern Asia.
  Fact 0.5: Aspidistra is particularly native to China.
  Fact 0.6: Aspidistra is particularly native to Vietnam.
Sentence 1: Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
  Fact 1.0: Cyrtanthus is a genus of perennial plants.
  Fact 1.1: Cyrtanthus is a genus of herbaceous plants.
  Fact 1.2: Cyrtanthus is a genus of bulbous plants.
  Fact 1.3: Cyrtanthus is in the family Amaryllidaceae.
  Fact 1.4: Cyrtanthus is in the subfamily Amaryllidoideae.

Retrieved items:
<item 1>
Traubiinae
Traubiinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 1>

<item 2>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 2>

<item 3>
Ammocharis
Ammocharis is a small genus from sub-Saharan Africa, in the Amaryllidaceae family (subfamily Amaryllidoideae) which includes 7 species distributed in Africa.

 The plant grows as a succulent, above-ground bulb, preferring seasonally wet, hot, sandy soils and full sun.
</item 3>

<item 4>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 4>

<item 5>
Amarylloidinae
Amarylloidinae is a now obsolete informal name for an "infrafamily" within the Amaryllidaceae (Amaryllis) family, erected by Hamilton Traub.

 This grouping was designed to fill a perceived gap between the formal rank of subfamily and tribes.

 In his treatment of this family, he divided it first into four subfamiles.

 Within subfamily Amarylloideae he then divided his sixteen tribes into two infrafamilies, Amarylloidinae (12 tribes) and Pancratioidinae (4 tribes), both of
which were subsequently demonstrated to be polyphyletic, and hence were abandoned by Dahlgren, who used no rank between family and tribe.

 On the other hand, he also used a much more restricted Amaryllidaceae corresponding to Traub's subfamily Amarylloideae.

 Thus Traub's Amarylloideae most closely resembles subfamily Amaryllidoideae "sensu" APGIII.
</item 5>

<item 6>
Aspidistra
Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and
Vietnam.

 They grow in shade under trees and shrubs.

 Their leaves arise more or less directly from ground level, where their flowers also appear.

 The number of species known has increased considerably from the 1980s onwards, with around 100 accepted as of July 2013 . "

Aspidistra elatior" is common worldwide as a foliage house plant that is very tolerant of neglect.

 It and other species can also be grown in shade outside, where they are generally hardy to -5 C .
</item 6>

<item 7>
The entries describe diverse flowering plants and groups, chiefly from Africa. Ice plants (Aizoaceae) include South African Aloinopsis and Conophytum, and
southern African Conicosia, whose wet-opening capsules release tiny seeds. Other genera span many families and regions: African irises and bulbous amaryllids;
Asian toad lilies and Rodgersia; European and North African Coronilla; and aquatic Alismataceae. Vascular plants are defined by xylem
</item 7>

<item 8>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 8>

<item 9>
Clinantheae
Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was described in 2000 by Alan Meerow "et al." as a result of a molecular phylogenetic study of the American Amaryllidoideae.

 This demonstrated that the Stenomesseae tribe, including the type genus "Stenomesson" was polyphyletic.

 Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade.

 Since the type species of "Stenomesson" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the
other genera that remained.

 This was "Clinanthus", the oldest name for these species, and consequently the tribe Clinantheae.
</item 9>

<item 10>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 10>
=== END ITEM id=e58ef4e72eb11edc ===

=== ITEM id=b093cddabef351a1 ===
Gold sentences and facts:
Sentence 0:  Her favorite events are 800 metres and 1500 metres.
  Fact 0.0: This person's favorite events are the 800 metres.
  Fact 0.1: This person's favorite events are the 1500 metres.
Sentence 1:  The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.
  Fact 1.0: The distance has been contested at the Summer Olympics since 1896.
  Fact 1.1: The distance has been contested at the World Championships in Athletics since 1983.

Retrieved items:
<item 1>
Sinimole Paulose
Sinimol Paulose (born 24 June 1983) is an Indian middle distance runner.

 Her favorite events are 800 metres and 1500 metres.

 In 2009, she received Arjuna Award for her contribution to the Indian athletics.

 Sinimol was born in Piravom in Ernakulam district of Kerala state.

 She represented India at Beijing Olympics.
</item 1>

<item 2>
Ross MacKenzie (athlete)
Ross MacKenzie (born 18 July 1946) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1968 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Brian MacLaren, Bill Crothers, and Robert McLaren).

 MacKenzie also finished fifth in the 1967 Pan American Games 400 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Brian MacLaren and Bill Crothers.
</item 2>

<item 3>
Siegmar Ohlemann
Siegmar ("Sig") Ohlemann (born May 11, 1938) is a retired middle distance runner, who represented Canada in three events (800 metres, 4 × 100 metres and 4 × 400
metres) at the 1960 Summer Olympics in Rome, Italy.

 He claimed the silver medal in 800 metres at the 1963 Pan American Games in Brazil, behind fellow Canadian Don Bertoia.
</item 3>

<item 4>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 4>

<item 5>
Mary Chemweno
Mary Chemweno Koskei is a former Kenyan middle distance runner who specialised in 400 m, 800 m and 1500 m events.

 She won the gold medal in the inaugural 1979 African Championships in 800 metres, and another gold medal six years later in 1500 metres at the 1985 edition.

 She is also a three-time Kenyan national champion, having won both the 400 and 800 metres events in 1981 as well as the 800 m event again in 1986.

 Mid-career, in the early 1980s, she married long-distance runner Kipsubai Koskei.
</item 5>

<item 6>
K. M. Binu
Kalayathumkuzhi Mathews Binu (born 20 December 1980) is an Indian track and field athlete from Kerala who specializes in 400 metres and 800 metres.

 He holds the current 400 metres national record of 45.48 s set at the 2004 Athens Olympics on 20 August 2004.

 He broke the 44-year-old Olympics mark (by an Indian) held by Milkha Singh who set an Indian National Record with a timing of 45.73 s at the 1960 Rome
Olympics.

 He and his elder sister K. M. Beenamol made history when they became the first Indian siblings to win medals in a major international competition.

 They won medals at the Busan Asian Games (2002).

 While Binu won the men's 800 metres silver, his sister won the gold medal in the women's event.

 Binu received the Arjuna Award for the year 2006 for his achievements in the Indian athletics.
</item 6>

<item 7>
Caster Semenya
Mokgadi Caster Semenya Bronze OIB (born 7 January 1991) is a South African middle-distance runner and 2016 Olympic gold medallist.

 Semenya won gold in the women's 800 metres at the 2009 World Championships with a time of 1:55.45 and at the 2017 World Championships in her new personal best,
1:55.16.

 Semenya also won silver medals at the 2011 World Championships and the 2012 Summer Olympics, both in the 800 metres.

 She was the winner of the gold medal in the 800 metre event at the 2016 Summer Olympics.
</item 7>

<item 8>
John Lindsay (Paralympian)
John Lindsay, OAM (born 29 January 1970) is an Australian Paralympic athlete from Melbourne.

 He competed in the 1988 Seoul games in distances ranging from 100 m to 800 m, but did not win any medals.

 At the 1992 Barcelona Games, he won a gold medal in the Men's 200 m TW3 event, for which he received a Medal of the Order of Australia, a silver medal in the
Men's 100 m TW3 event and a bronze medal in the Men's 400 m TW3 event.

 That year, he had a Victorian Institute of Sport scholarship.
</item 8>

<item 9>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 9>

<item 10>
1500 metres
The 1500 metres or 1,500-metre run (typically pronounced 'fifteen-hundred metres') is the foremost middle distance track event in athletics.

 The distance has been contested at the Summer Olympics since 1896 and the World Championships in Athletics since 1983.

 It is equivalent to 1.5 kilometers or approximately ⁄  miles.
</item 10>
=== END ITEM id=b093cddabef351a1 ===

=== ITEM id=ccfd56495df74a67 ===
Gold sentences and facts:
Sentence 0:  After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.
  Fact 0.0: He spent some time in general practice.
  Fact 0.1: He became a Lecturer in animal behaviour at De Montfort University.
  Fact 0.2: He became a Senior Lecturer in animal behaviour at De Montfort University.
  Fact 0.3: He became a Principal Lecturer in animal behaviour at De Montfort University.
  Fact 0.4: He subsequently moved to the University of Lincoln.
  Fact 0.5: He has been Head of the Department of Biological Sciences at the University of Lincoln.
Sentence 1:  Individuals who have earned a Doctor of Philosophy degree may, in most jurisdictions, use the title of "Doctor" (often abbreviated "Dr") or, in non-English speaking countries, variants such as "Dr. phil."
  Fact 1.0: People who have earned a Doctor of Philosophy degree may, in most jurisdictions, use the title 'Doctor'.
  Fact 1.1: The title 'Doctor' is often abbreviated 'Dr'.
  Fact 1.2: In non-English speaking countries, people with a Doctor of Philosophy degree may use variants such as 'Dr. phil.'

Retrieved items:
<item 1>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 1>

<item 2>
David M. Sever
David M. Sever is an American herpetologist, histologist, anatomist and reproductive biologist.

 He has been a professor and department head in the Department of Biological Sciences at Southeastern Louisiana University since 2004, and held the Kenneth
Dyson Endowed Professorship in Biological Sciences from 2012 to 2015.

 He is well known for over 30 years of research on the secondary sexual characteristics of salamanders and more generally on comparative histoanatomy of the
urogenital systems of vertebrates.

 and was recognized as the 2013 Distinguished Herpetologist of the Year by the Herpetologists' League.
</item 2>

<item 3>
Richard F. Wilson
Richard F. Wilson was selected to be Illinois Wesleyan University's 18th President on April 6, 2004 and took office on July 1, 2004.

 He was inaugurated on April 9, 2005.

 He had spent the previous 26 years at the University of Illinois at Urbana-Champaign, leaving as associate chancellor for development and vice president of the
University of Illinois Foundation.

 Mr. Wilson is a Ph.D. graduate of the University of Michigan.

 Wilson announced his intended retirement from Illinois Wesleyan, to take place around June 2015 or whenever a successor is named and transitioned.
</item 3>

<item 4>
Ekhard Salje
Ekhard Karl Hermann Salje, FRS (born 1946) is Professor of Mineralogy and Petrology and former Head of the Department of Earth Sciences, Cambridge University.
</item 4>

<item 5>
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
</item 5>

<item 6>
Graph query row:
person: Daniel Simon Mills
professional_title: ['biologist', 'Professor of Veterinary Behavioural Medicine', 'Lecturer', 'Senior Lecturer', 'Principal Lecturer', 'Head of the Department
of Biological Sciences']
</item 6>

<item 7>
Richard W. Aldrich
Richard Warren Aldrich is an American neuroscientist who is currently the Karl Folkers Chair of Interdisciplinary Medical Research and Professor of Neurobiology
in the School of Biological Sciences at the University of Texas.

 Graduating from the University of Arizona with a BS in Biological Sciences (1975), Aldrich went on to earn a Ph. D. in Neuroscience from Stanford University
(1980).

 After completing a post-doctoral fellowship in physiology at Yale University under the direction of W. Knox Chandler and Charles F. Stevens, Aldrich was hired
to teach at Yale, instructing in the Department of Molecular Neurobiology.

 In 1985, Aldrich returned to Stanford to teach neurobiology and physiology, eventually serving as Chair of the Department of Molecular and Cellular Physiology
from 2001 to 2004.

 He was an Investigator in the Howard Hughes Medical Institute from 1990 until 2006.

 Aldrich's research has focused on the "molecular mechanisms of ion channel function and their role in electrical signaling."

 Aldrich has served as the President of the Society of General Physiologists and the Biophysical Society.
</item 7>

<item 8>
Alan Kamil
Alan C. "Al" Kamil is an American experimental psychologist.

 He is the Director, School of Biological Sciences and George Holmes Professor of Biological Sciences and Psychology at the University of Nebraska, Lincoln.

 Kamil's work focusses on the evolution of memory and adaptive specializations of learning in many animal species, especially the Clark's nutcracker and other
birds.

 Kamil has published peer reviewed articles on both theoretical aspects of comparative psychology and animal cognition, and on empirical studies of animal
learning and memory.

 In 2013 Kamil was honoured by the Comparative Cognition Society for his contributions to the study of animal cognition.
</item 8>

<item 9>
Daniel Mills (biologist)
Daniel Simon Mills, FRCVS (born 21 Aug 1966) is an English biologist and the UK's first Professor of Veterinary Behavioural Medicine based at the University of
Lincoln, United Kingdom.

 He attended St Edmund's College, Ware before studying at the University of Bristol where he received his BVSc degree and completed his Ph.D. degree in animal
behaviour from De Montfort University.

 After some time spent in general practice, he became a Lecturer, Senior Lecturer and Principal Lecturer in animal behaviour at De Montfort University, and
subsequently the University of Lincoln, where he has been Head of the Department of Biological Sciences.

 His research has focused on the assessment of emotion in companion animals and the use of semiochemicals to manage their problem behaviour (pheromonatherapy).
</item 9>

<item 10>
William McGinnis
William "Bill" McGinnis, Ph.D. is a molecular biologist and professor of biology at the University of California San Diego.

 At UC San Diego he has also served as the Chairman of the Department of Biology from July 1998 - June 1999, as Associate Dean of the Division of Natural
Sciences from July 1, 1999 - June 2000, and as Interim Dean of the newly established Division of Biological Sciences from July 1, 2000 - February 1, 2001.

 Dr. McGinnis was appointed Dean of the Divisional Biological Sciences on July 1, 2013
</item 10>
=== END ITEM id=ccfd56495df74a67 ===

=== ITEM id=8eac92a3f25e99a5 ===
Gold sentences and facts:
Sentence 0:  Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer Jimmy Chamberlin.
  Fact 0.0: The band was formed by Corgan and guitarist James Iha.
  Fact 0.1: The band was formed in Chicago, Illinois, in 1988.
  Fact 0.2: The band quickly gained steam with the addition of bassist D'arcy Wretzky.
  Fact 0.3: The band quickly gained steam with the addition of drummer Jimmy Chamberlin.
Sentence 1: Greek Fire is an American rock band from St. Louis, Missouri.
  Fact 1.0: Greek Fire is an American rock band.
  Fact 1.1: Greek Fire is from St. Louis, Missouri.

Retrieved items:
<item 1>
Greek Fire (band)
Greek Fire is an American rock band from St. Louis, Missouri.

 The band was formed in 2008 by members of Story of the Year and Maybe Today.

 Since formation, Greek Fire has released a self-titled EP, a single titled "Doesn't Matter Anyway", on August 16, 2011, they released their debut, full-length
album, "Deus Ex Machina", and have recently announced a new addition to the "Lost/Found" EPs titled "Broken" set to be released before "Found".
</item 1>

<item 2>
Mike Byrne (musician)
Michael William Byrne (born February 6, 1990) is a drummer who was a member of the band The Smashing Pumpkins.

 When Pumpkins frontman Billy Corgan called for rehearsals to replace Jimmy Chamberlin, Byrne auditioned and was chosen out of thousands of applicants.

 He played drums in Sky Saxon tribute band Spirits in the Sky for 6 shows in August 2009, along with Corgan, Kerry Brown, Kevin Dippold, Mark Tulin, Linda
Strawberry, Ysanne Spevack, Mark Weitz, and Dave Navarro, and also performs drums on the ongoing Pumpkins 44-song project, "Teargarden by Kaleidyscope".

 As a part of "Teargarden", Mike contributed drumming duties on "Oceania", the Pumpkins' eighth full-length album.

 On April 17, 2010, Byrne played his first show under the Smashing Pumpkins moniker in celebration of Record Store Day.
</item 2>

<item 3>
Jimmy Chamberlin
James Joseph Chamberlin (born June 10, 1964) is an American drummer and record producer.

 He is best known as the drummer for the alternative rock band The Smashing Pumpkins.

 Following the 2000 breakup of the band, Chamberlin joined Pumpkins frontman Billy Corgan in the supergroup Zwan and also formed his own group, the Jimmy
Chamberlin Complex.

 In 2005, Chamberlin joined Corgan in reforming The Smashing Pumpkins; he eventually left the group in March 2009, though he returned again in 2015 for a summer
tour.

 He performed in the group Skysaw until 2012.

 He is currently active under the Jimmy Chamberlin Complex name.

 In addition to his current work as CEO, Chamberlin has joined Chicago jazz saxophonist Frank Catalano for a string of 2013–15 performances in the Chicago area.

 An EP by Catalano and Chamberlin "Love Supreme Collective - EP" was released on the 29th of July 2014.
</item 3>

<item 4>
List of the Smashing Pumpkins band members
The Smashing Pumpkins are an alternative rock band formed in Chicago, Illinois in 1988.

 The band was formed by guitarist/vocalist Billy Corgan and guitarist James Iha after the demise of Corgan's first band, The Marked.

 Since its inception, The Smashing Pumpkins has gone through multiple line-up changes, with Corgan the only consistent member.
</item 4>

<item 5>
The Aeroplane Flies High
The Aeroplane Flies High is a five-disc box set released by American alternative rock band The Smashing Pumpkins in 1996.

 It contains expanded versions of the five singles from their album "Mellon Collie and the Infinite Sadness" and also included a 44-page booklet with pictures
and writings by the band's lead singer Billy Corgan, as well as lyrics.

 A limited edition release, the box reached number 42 on the "Billboard" charts, and sold 300,000 units (1.5 million discs in all), generating a platinum disc
for the band.

 Originally intended to be limited to 200,000 copies, Virgin Records produced more after the original run sold out due to overwhelming and unexpected demand.

 The album was remastered in 2013 under the supervision of frontman Billy Corgan and reissued on vinyl and as a CD/DVD box set.
</item 5>

<item 6>
Zeitgeist (The Smashing Pumpkins album)
Zeitgeist is the seventh album by American alternative rock band The Smashing Pumpkins, released on July 10, 2007 in the United States and Canada.

 It was the first album The Smashing Pumpkins released after their 2000 disbandment and 2005 reunion.

 The album was produced by Roy Thomas Baker, Billy Corgan, Jimmy Chamberlin, and Terry Date.

 The album would be Chamberlin's last with the band before his departure in 2009.

 The album debuted strongly, but sales soon decreased, and critical reception was mixed.

 It was certified Gold in the United States on February 1, 2008.
</item 6>

<item 7>
Muzzle (song)
"Muzzle" is a song by American alternative rock band The Smashing Pumpkins from their third album, "Mellon Collie and the Infinite Sadness".

 It was one of the last songs written by Billy Corgan for "Mellon Collie", with the song's lyrics referring to what Corgan thought the public's perception was
of him at the time.

 It was rumored to be the Smashing Pumpkins fifth and final single from this album, as is evidenced by the fact that a promotional single for the song was
issued to radio stations worldwide.

 However, the song "Thirty-Three" was released as the fifth and final single instead.
</item 7>

<item 8>
1979 (song)
"1979" is a song by American alternative rock band the Smashing Pumpkins.

 Released in 1996 as the second single from their third studio album, "Mellon Collie and the Infinite Sadness", "1979" was written by frontman Billy Corgan, and
features loops and samples that were uncharacteristic of previous Smashing Pumpkins songs.

 The song was written as a coming of age story by Corgan.

 In the year 1979, Corgan was 12 and this is what he considered his transition into adolescence.

 The song was popular with critics and fans; Allmusic's Amy Hanson called it a "somewhat surprising hit".

 The song was nominated for the Record of the Year and Best Rock Performance by a Duo or Group with Vocal at the 39th Annual Grammy Awards, and won an MTV Video
Music Award for Best Alternative Video.

 In 2012, it was voted the second-best Smashing Pumpkins song by Rolling Stone magazine readers.
</item 8>

<item 9>
Pinhead Gunpowder
Pinhead Gunpowder is an American punk rock band that formed in East Bay, California, in 1990.

 The band currently consists of Aaron Cometbus (drums, lyrics), Bill Schneider (bass), Billie Joe Armstrong (guitar, vocals) and Jason White (guitar, vocals).

 The band's name comes from a brand of "high octane" green tea served at the Arcata co-op and discovered by Aaron Cometbus during one of his many dumpster
diving adventures.
</item 9>

<item 10>
Billy Corgan
William Patrick "Billy" Corgan Jr. (born March 17, 1967) is an American musician, songwriter, producer, poet, and entrepreneur.

 He is best known as the lead singer, primary songwriter, guitarist, and sole permanent member of The Smashing Pumpkins.

 Formed by Corgan and guitarist James Iha in Chicago, Illinois, in 1988, the band quickly gained steam with the addition of bassist D'arcy Wretzky and drummer
Jimmy Chamberlin.

 Strong album sales and large-scale tours propelled the band's increasing fame in the 1990s until their break-up in 2000.

 Corgan started a new band called Zwan, and after their quick demise, he released a solo album ("TheFutureEmbrace") and a collection of poetry ("Blinking with
Fists") before setting his sights on reforming Smashing Pumpkins.
</item 10>
=== END ITEM id=8eac92a3f25e99a5 ===

=== ITEM id=56feff746e63d542 ===
Gold sentences and facts:
Sentence 0: Mulled wine is a beverage usually made with red wine along with various mulling spices and sometimes raisins.
  Fact 0.0: Mulled wine is a beverage.
  Fact 0.1: Mulled wine is usually made with red wine.
  Fact 0.2: Mulled wine is usually made with various mulling spices.
  Fact 0.3: Mulled wine is sometimes made with raisins.
Sentence 1:  It is served hot or warm and is alcoholic, although there are non-alcoholic versions of mulled wine.
  Fact 1.0: Mulled wine is served hot or warm.
  Fact 1.1: Mulled wine is alcoholic.
  Fact 1.2: There are non-alcoholic versions of mulled wine.
Sentence 2: Blue Lagoon is a popular summer cocktail featuring blue Curaçao.
  Fact 2.0: Blue Lagoon is a popular summer cocktail.
  Fact 2.1: Blue Lagoon features blue Curaçao.

Retrieved items:
<item 1>
Flaming volcano
The flaming volcano is a large tropical group cocktail made of rum, brandy, pineapple juice, orange juice, almond syrup, and sometimes other ingredients.

 It is usually a multi-user drink, served to a group in a special, ritual vessel known as a volcano bowl, which is a decorative ceramic bowl (typically of about
32 USoz capacity) designed with a rising central hub feature usually resembling a crude volcano cone.

 The cone includes a second, much smaller central pit or "crater" which is partially filled with rum or other flammable high-alcohol liquor.

 The "crater" liquor is then ignited, creating a mild volcanic ambience with its central blue flame.
</item 1>

<item 2>
Volcano bowl
Volcano bowls are ceramic drinkware originally associated with mid-20th century American tiki bars and tropical-themed restaurants.

 Drinks served in volcano bowls are typically rum-based, mixed with tropical fruit juices and other liquors such as brandy, vodka, and triple sec, and garnished
with fruit.

 The Flaming Volcano cocktail is especially associated with this drinkware.
</item 2>

<item 3>
The Blue Lagoon (novel)
The Blue Lagoon is a romance novel written by Henry De Vere Stacpoole and was first published by T. Fisher Unwin in 1908.

 It is the first novel of the "Blue Lagoon" trilogy, which also includes "The Garden of God" (1923) and "The Gates of Morning" (1925).

 The novel has inspired several film adaptations, most notably "The Blue Lagoon" starring Brooke Shields as Emmeline and Christopher Atkins as Richard ("Dicky"
in the book).
</item 3>

<item 4>
Flaming beverage
Flaming beverages include cocktails and other mixed drinks that contain flammable, high-proof alcohol, which is ignited prior to consumption.

 The alcohol may be an integral part of the drink, or it may be floated as a thin layer across the top of the drink.

 The flames are mostly for dramatic flair.

 However, in combination with certain ingredients, the flavor of the drink is altered.

 Some flavors are enhanced, and it may impart a toasted flavor to some drinks.
</item 4>

<item 5>
Feuerzangenbowle
Feuerzangenbowle (] ) is a traditional German alcoholic drink for which a rum-soaked sugarloaf is set on fire and drips into mulled wine.

 It is often part of a Christmas or New Year's Eve tradition.

 The name translates literally to "fire-tongs punch".
</item 5>

<item 6>
Mulled wine
Mulled wine is a beverage usually made with red wine along with various mulling spices and sometimes raisins.

 It is served hot or warm and is alcoholic, although there are non-alcoholic versions of mulled wine.

 It is a traditional drink during winter, especially around Christmas.
</item 6>

<item 7>
Blue Lagoon (cocktail)
Blue Lagoon is a popular summer cocktail featuring blue Curaçao.
</item 7>

<item 8>
Mulling spices
Mulling spices is a spice mixture used in drink recipes.

 The spices are usually used to spice hot apple cider, mulled wine, wassail, and other drinks (such as juices) during the autumn or winter.

 A "mulled" drink is a drink which has been prepared with these spices (usually through heating the drink in a pot with mulling spices and then straining).

 Mulling spices may also be added to the brewing process to make spiced beer.
</item 8>

<item 9>
Joulupöytä
Joulupöytä (translated "Yule table") is the traditional assortment of foods served at Christmas in Finland, similar to the Swedish julbord.

 It contains many different dishes, most of them typical for the season.

 The main dish is usually a large Christmas ham, which is eaten with mustard or bread along with the other dishes.

 Fish is also served (often lutefisk and gravlax), and the ham is served with "laatikko"s, casseroles made with swede, potato and carrot, occasionally liver.

 The traditional Christmas beverage is either alcoholic or non-alcoholic mulled wine ("glögi" in Finnish).
</item 9>

<item 10>
Oz and Hugh Drink to Christmas
Oz and Hugh Drink to Christmas is a BBC television programme in which wine personality Oz Clarke and comedian Hugh Dennis travel through Britain to sample a
wide array of seasonal Christmas beverages, including whisky, winter ales, mulled wine, wassail, sloe gin, Buck's Fizz, Port wine and Sherry.

 Upon its 20 December 2009 broadcast on BBC Two, it had a viewership of approximately 2.4 million with an audience share of 9%.
</item 10>
=== END ITEM id=56feff746e63d542 ===

=== ITEM id=4e3d4863ec72de00 ===
Gold sentences and facts:
Sentence 0:  His siblings were Abby, Nelson, Laurance, Winthrop, and David.
  Fact 0.0: Abby was his sibling.
  Fact 0.1: Nelson was his sibling.
  Fact 0.2: Laurance was his sibling.
  Fact 0.3: Winthrop was his sibling.
  Fact 0.4: David was his sibling.
Sentence 1: David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.
  Fact 1.0: David Rockefeller lived from June 12, 1915 to March 20, 2017.
  Fact 1.1: David Rockefeller was an American banker.
  Fact 1.2: David Rockefeller was chairman of Chase Manhattan Corporation.
  Fact 1.3: David Rockefeller was chief executive of Chase Manhattan Corporation.

Retrieved items:
<item 1>
David Rockefeller
David Rockefeller (June 12, 1915 – March 20, 2017) was an American banker who was chairman and chief executive of Chase Manhattan Corporation.

 He was the oldest living member of the Rockefeller family and family patriarch from August 2004 until his death in March 2017.

 Rockefeller was a son of John D. Rockefeller Jr. and Abby Aldrich Rockefeller, and a grandson of John D. Rockefeller and Laura Spelman Rockefeller.
</item 1>

<item 2>
Alida Rockefeller Messinger
Alida Ferry Rockefeller (born 1948) is an American philanthropist.

 She is the youngest daughter of John Davison Rockefeller III (1906–1978) and Blanchette Ferry Hooker (1909–1992), and a fourth-generation member of the
Rockefeller family.

 Her brother is former Senator John Davison "Jay" Rockefeller IV (born 1937).

 According to an account in "The New York Times", Alida was only five years old when her father began to teach her about philanthropy.

 She said:
</item 2>

<item 3>
Rockefeller family
The Rockefeller family ( ) is an American industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the
late 19th and early 20th centuries, with John D. Rockefeller and his brother William Rockefeller primarily through Standard Oil.

 The family is also known for its long association with and control of Chase Manhattan Bank.

 They are considered to be one of the most powerful families, if not the most powerful family, in the history of the United States.
</item 3>

<item 4>
Chase Bank
JPMorgan Chase Bank, N.A., doing business as Chase Bank, is a national bank that constitutes the consumer and commercial banking subsidiary of the U.S.
multinational banking and financial services holding company, JPMorgan Chase.

 The bank was known as Chase Manhattan Bank until it merged with J.P. Morgan & Co. in 2000.

 Chase Manhattan Bank was formed by the merger of the Chase National Bank and The Manhattan Company in 1955.

 The bank has been headquartered in Columbus, Ohio since its merger with Bank One Corporation in 2004.

 The bank acquired the deposits and most assets of Washington Mutual.
</item 4>

<item 5>
JPMorgan Chase
JPMorgan Chase & Co. is a U.S. multinational banking and financial services holding company headquartered in New York City.

 It is the largest bank in the United States, the world's sixth largest bank by total assets, with total assets of US$2.5 trillion, and the world's second most
valuable bank by market capitalization, after ICBC.

 It is a major provider of financial services, and according to "Forbes" magazine is the world's sixth largest public company based upon a composite ranking.

 The hedge fund unit of JPMorgan Chase is the second largest hedge fund in the United States.

 The company was formed in 2000, when Chase Manhattan Corporation merged with J.P. Morgan & Co.
</item 5>

<item 6>
John D. Rockefeller III
John Davison Rockefeller III (March 21, 1906 – July 10, 1978) was a philanthropist and third-generation member of the prominent Rockefeller family.

 He was the eldest son of philanthropists John D. Rockefeller Jr. and Abby Aldrich Rockefeller.

 His siblings were Abby, Nelson, Laurance, Winthrop, and David.
</item 6>

<item 7>
Hope Aldrich Rockefeller
Hope Aldrich Rockefeller (born May 17, 1938) is an American newspaper publisher.

 She is the eldest daughter of philanthropist John Davison Rockefeller III (1906–1978) and Blanchette Ferry Hooker (1909–1992).

 Her elder brother is former Senator John Davison "Jay" Rockefeller IV (born 1937).
</item 7>

<item 8>
William B. Harrison Jr.
William B. Harrison Jr., born August 12, 1943, in Rocky Mount, North Carolina, is the former CEO and chairman of JPMorgan Chase.

 He attended high school at Virginia Episcopal School, where he was a basketball star.

 He attended the University of North Carolina at Chapel Hill, where he was admitted to the Zeta Psi fraternity.

 Having risen through the ranks of Chemical Bank before succeeding Walter V. Shipley during the Chemical Bank takeover of the Chase Manhattan Corporation, he
and Douglas A. Warner III, then CEO of J.P. Morgan & Co., were the principal architects of the US$30.9 billion Chase and J.P. Morgan & Co. merger of 2000.

 Harrison has been a director of the Firm or a predecessor institution since 1991.

 Harrison is also a director of Merck & Co., Inc.
</item 8>

<item 9>
Rockefeller Brothers Fund
The Rockefeller Brothers Fund (RBF) is a philanthropic foundation created and run by members of the Rockefeller family.

 It was founded in New York City in 1940 as the primary philanthropic vehicle for the five third-generation Rockefeller brothers: John D. Rockefeller III,
Nelson, Laurance, Winthrop and David.

 It is distinct from the Rockefeller Foundation.

 The Rockefellers are an industrial, political, and banking family that made one of the world's largest fortunes in the oil business during the late 19th and
early 20th centuries.
</item 9>

<item 10>
Walter V. Shipley
Walter V. Shipley (born November 2, 1935) was the former Chairman and Chief Executive Officer of Chase Manhattan Bank and, before that, the company with which
it merged Chemical Bank.

 Shipley was named chief executive of Chemical in 1981 and held the position through 1999 and remained at the bank as chairman through January 2000, just prior
to the bank's merger with J.P. Morgan & Co..

 During his 18-year tenure, Shipley oversaw Chemical's mergers with Texas Commerce Bank in 1987, Manufacturers Hanover in 1991 and Chase Manhattan Bank in 1996.
</item 10>
=== END ITEM id=4e3d4863ec72de00 ===

=== ITEM id=06b0b9e809440d3c ===
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
University of Copenhagen
The University of Copenhagen (UCPH) (Danish: "Københavns Universitet" ) is the oldest university and research institution in Denmark.

 Founded in 1479 as a studium generale, it is the second oldest institution for higher education in Scandinavia after Uppsala University (1477).

 The university has 23,473 undergraduate students, 17,398 postgraduate students, 2,968 doctoral students and over 9,000 employees.

 The university has four campuses located in and around Copenhagen, with the headquarters located in central Copenhagen.

 Most courses are taught in Danish; however, many courses are also offered in English and a few in German.

 The university has several thousands of foreign students, about half of whom come from Nordic countries.
</item 1>

<item 2>
Akademisches Kunstmuseum
Akademisches Kunstmuseum (English:Academic Art Museum) is an art museum in Bonn, Germany.

 It is the oldest museum in Bonn and houses the antique collection of the University of Bonn with more than 500 antique statues and reliefs, and over 2,000
originals.

 It is located in a neoclassical building at the southern end of the Hofgarten, near the Electoral Palace.
</item 2>

<item 3>
National Museum, Kraków
The National Museum in Kraków (Polish: "Muzeum Narodowe w Krakowie" ), popularly abbreviated as MNK, established in 1879, is the main branch of Poland's
National Museum, which has several independent branches with permanent collections around the country.

 The Museum consists of 21 departments which are divided by art period; 11 galleries, 2 libraries, and 12 conservation workshops.

 It holds some 780,000 art objects, spanning from classical archeology to modern art, with special focus on Polish painting.

<ref name="culture.pl/muzea"> </ref><ref name="muzeum/Zbiory"> </ref>
</item 3>

<item 4>
Riverview, New Brunswick
Riverview is a Canadian town in Albert County, New Brunswick, Canada.

 Riverview is located on the south side of the Petitcodiac River, across from the larger cities of Moncton and Dieppe.

 Riverview has an area of 34 km2 , and a population density of 564.6 PD/km2 .

 Riverview's slogan is "A Great Place To Grow".

 With a population of 19,667, Riverview is the fifth largest municipality in New Brunswick, having a larger population than the cities of Edmundston, Bathurst,
Campbellton, and Miramichi, despite its designation of "town".
</item 4>

<item 5>
Bonn
The Federal City of Bonn (] ) is a city on the banks of the Rhine in the German state of North Rhine-Westphalia, with a population of over 300,000.

 About 24 km south-southeast of Cologne, Bonn is in the southernmost part of the Rhine-Ruhr region, Germany's largest metropolitan area, with over 11 million
inhabitants.
</item 5>

<item 6>
Ecclesiastical Province of Freiburg
The Ecclesiastical Province of Freiburg ("Kirchenprovinz Freiburg") or Upper Rhenish Ecclesiastical Province ("Oberrheinische Kirchenprovinz") is an
ecclesiastical province of the Roman Catholic Church in the Upper Rhine area of Germany, centring on Freiburg im Breisgau.

 It covers the Roman Catholic Archdiocese of Freiburg, the Roman Catholic Diocese of Mainz and the Roman Catholic Diocese of Rottenburg-Stuttgart, covering
large areas of Baden-Württemberg and Hesse and small parts of Rhineland-Palatinate.

 Its metropolitan bishop is the Archbishop of Freiburg - that Archdiocese and the Province were both set up in 1821 in the wake of the 1801 Concordat and the
1815 Congress of Vienna.
</item 6>

<item 7>
Croatian Natural History Museum
The Croatian Natural History Museum (Croatian: "Hrvatski prirodoslovni muzej" ) is the oldest and biggest natural history museum and the main body for natural
history research, preservation and collection in Croatia.

 Located on Dimitrije Demeter Street in Gornji Grad, one of the oldest neighbourhoods of the Croatian capital Zagreb, it owns one of the biggest museum
collections in Croatia, with over 2 million artefacts, including over 1.1 million animal specimens.

 It was founded in 1846 as the "National Museum".

 The National Museum was later split up into five museums, three of which were in 1986 merged as departments of the newly named Croatian Natural History Museum.

 The museum contains a large scientific library open to the public, and publishes the first Croatian natural history scientific journal, "Natura Croatica".
</item 7>

<item 8>
List of Kerala cities by population
The article lists major cities in Kerala based on the total population in city corporations or municipalities and their outgrowths.

 This list is based on the latest census data.

 Population of cities after major mergers post to latest census are also included.
</item 8>

<item 9>
National Museum of Poland
"National Museum of Poland" is the common name for several of the country's largest and most notable museums. Poland's National Museum comprises several
independent branches, each operating a number of smaller museums.

 The main branch is the National Museum in Kraków (Polish: "Muzeum Narodowe w Krakowie" ), established in 1879 with permanent collections consisting of several
hundred thousand items – kept in big part at the Main Building (along the 3 Maja St.), but also in the eight of its divisions around the city.
</item 9>

<item 10>
Harris Museum
The Harris Museum, Art Gallery & Preston Free Public Library is a Grade I listed museum building in Preston.
</item 10>
=== END ITEM id=06b0b9e809440d3c ===
